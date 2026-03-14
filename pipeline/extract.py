from __future__ import annotations

import re
from dataclasses import replace

from schemas import Transcript, WorkoutStep, WorkoutSummary
from .transcribe import TranscriptError, call_llm_json


TRANSITION_RE = re.compile(
    r"\b(next|now|then|move on|moving on|switch sides|switch side|onto|rest|round \d+|start with|begin with)\b",
    re.IGNORECASE,
)
REST_RE = re.compile(r"\b(rest|break|recover)\b", re.IGNORECASE)
SIDE_RE = re.compile(r"\b(left|right|both sides|other side)\b", re.IGNORECASE)
DURATION_PATTERNS = [
    re.compile(r"\b(\d+)\s*(seconds|second|secs|sec)\b", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*(minutes|minute|min)\b", re.IGNORECASE),
]
REPS_RE = re.compile(r"\b(\d+)\s*(reps|rep|times)\b", re.IGNORECASE)
SETS_RE = re.compile(r"\b(\d+)\s*(sets|set|rounds|round)\b", re.IGNORECASE)
FILLER_PREFIXES = [
    "okay",
    "all right",
    "alright",
    "and",
    "so",
    "next",
    "now",
    "then",
    "we have",
    "we are doing",
    "we're doing",
    "let's do",
    "lets do",
    "start with",
    "begin with",
    "moving into",
    "move into",
    "move on to",
    "onto",
    "for",
    "do",
]
SIDE_SWITCH_ONLY = {
    "switch side",
    "switch sides",
    "other side",
    "both side",
    "both sides",
    "left",
    "right",
}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def extract_duration_sec(text: str) -> float | None:
    for pattern in DURATION_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        value = int(match.group(1))
        unit = match.group(2).lower()
        return float(value * 60 if unit.startswith("min") else value)
    return None


def extract_reps(text: str) -> int | None:
    match = REPS_RE.search(text)
    return int(match.group(1)) if match else None


def extract_sets(text: str) -> int | None:
    match = SETS_RE.search(text)
    return int(match.group(1)) if match else None


def extract_notes(text: str) -> str | None:
    matches = SIDE_RE.findall(text)
    if not matches:
        return None
    ordered: list[str] = []
    for match in matches:
        lowered = match.lower()
        if lowered not in ordered:
            ordered.append(lowered)
    return ", ".join(ordered)


def _strip_metrics(text: str) -> str:
    cleaned = text
    for pattern in DURATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = REPS_RE.sub("", cleaned)
    cleaned = SETS_RE.sub("", cleaned)
    cleaned = re.sub(r"\bx\s*(\d+)\b", r" \1 ", cleaned, flags=re.IGNORECASE)
    return _normalize_whitespace(cleaned)


def extract_action_name(text: str) -> str | None:
    if not text:
        return None
    if REST_RE.search(text):
        return "Rest"

    cleaned = _strip_metrics(text.lower())
    cleaned = re.sub(r"\b(of|for|the|a|an)\b", " ", cleaned)
    cleaned = re.sub(r"[()!?,.:;/]", " ", cleaned)
    cleaned = re.sub(r"^switch sides and ", "", cleaned)
    cleaned = re.sub(r"^switch side and ", "", cleaned)
    cleaned = re.sub(r"^hold ", "", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    for prefix in FILLER_PREFIXES:
        if cleaned.startswith(prefix + " "):
            cleaned = cleaned[len(prefix) + 1 :]
            break
        if cleaned == prefix:
            cleaned = ""
            break
    cleaned = _normalize_whitespace(cleaned)
    cleaned = re.sub(r"\bseconds?\b|\bminutes?\b|\bsec\b|\bmin\b", "", cleaned)
    cleaned = _normalize_whitespace(cleaned)
    if cleaned in SIDE_SWITCH_ONLY:
        return None
    if not cleaned or len(cleaned) < 3:
        return None
    if len(cleaned.split()) > 8:
        cleaned = " ".join(cleaned.split()[:8])
    if re.fullmatch(r"[a-z0-9 '\-]+", cleaned):
        return cleaned.title()
    return cleaned


def _build_step(index: int, name: str, start_sec: float, text: str) -> WorkoutStep:
    return WorkoutStep(
        index=index,
        name=name,
        start_sec=start_sec,
        end_sec=start_sec,
        duration_sec=extract_duration_sec(text),
        reps=extract_reps(text),
        sets=extract_sets(text),
        notes=extract_notes(text),
        name_source="transcript_rule",
    )


def _merge_step(previous: WorkoutStep, current_text: str, segment_end_sec: float) -> WorkoutStep:
    merged = replace(previous)
    merged.end_sec = max(merged.end_sec, segment_end_sec)
    if merged.duration_sec is None:
        merged.duration_sec = extract_duration_sec(current_text)
    if merged.reps is None:
        merged.reps = extract_reps(current_text)
    if merged.sets is None:
        merged.sets = extract_sets(current_text)
    if merged.notes is None:
        merged.notes = extract_notes(current_text)
    return merged


def _finalize_step_boundaries(steps: list[WorkoutStep], total_duration_sec: float) -> list[WorkoutStep]:
    for index, step in enumerate(steps):
        next_start = steps[index + 1].start_sec if index + 1 < len(steps) else total_duration_sec
        if step.end_sec <= step.start_sec:
            step.end_sec = max(step.start_sec, next_start)
        else:
            step.end_sec = max(step.end_sec, min(next_start, total_duration_sec))
        if step.end_sec > next_start and index + 1 < len(steps):
            step.end_sec = next_start
        if step.duration_sec is None or step.duration_sec <= 0:
            step.duration_sec = max(0.0, step.end_sec - step.start_sec)
    return steps


def _rule_based_steps(transcript: Transcript, total_duration_sec: float) -> list[WorkoutStep]:
    steps: list[WorkoutStep] = []
    for segment in transcript.segments:
        text = _normalize_whitespace(segment.text)
        candidate_name = extract_action_name(text)
        is_transition = bool(TRANSITION_RE.search(text))
        has_metrics = any(
            value is not None
            for value in (extract_duration_sec(text), extract_reps(text), extract_sets(text))
        )
        should_start = candidate_name is not None and (is_transition or has_metrics or not steps)
        if not steps and candidate_name is not None:
            should_start = True

        if should_start:
            if steps:
                steps[-1].end_sec = segment.start_sec
            steps.append(_build_step(len(steps) + 1, candidate_name, segment.start_sec, text))
            steps[-1].end_sec = segment.end_sec
            continue

        if steps:
            same_action = candidate_name is not None and candidate_name == steps[-1].name
            if same_action or not is_transition or has_metrics or segment.end_sec - steps[-1].end_sec <= 4.0:
                steps[-1] = _merge_step(steps[-1], text, segment.end_sec)
                continue

        if candidate_name is not None and is_transition:
            if steps:
                steps[-1].end_sec = segment.start_sec
            steps.append(_build_step(len(steps) + 1, candidate_name, segment.start_sec, text))
            steps[-1].end_sec = segment.end_sec

    return _finalize_step_boundaries(steps, total_duration_sec)


def _llm_steps(transcript: Transcript, total_duration_sec: float) -> list[WorkoutStep]:
    compact_segments = [
        {
            "start_sec": round(segment.start_sec, 2),
            "end_sec": round(segment.end_sec, 2),
            "text": segment.text,
        }
        for segment in transcript.segments
    ]
    prompt = (
        "Extract workout steps from this transcript. "
        "Each step needs name, start_sec, end_sec, duration_sec, reps, sets, and notes. "
        "Use null for missing values. Keep original language for names.\n"
        f"Transcript: {compact_segments}"
    )
    schema_hint = (
        '{"steps":[{"name":"string","start_sec":0,"end_sec":0,'
        '"duration_sec":0,"reps":null,"sets":null,"notes":null}]}'
    )
    raw_steps = call_llm_json(prompt, schema_hint)
    if not raw_steps:
        return []
    steps: list[WorkoutStep] = []
    for index, item in enumerate(raw_steps, start=1):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        steps.append(
            WorkoutStep(
                index=index,
                name=name,
                start_sec=float(item.get("start_sec", 0.0)),
                end_sec=float(item.get("end_sec", 0.0)),
                duration_sec=(
                    float(item["duration_sec"]) if item.get("duration_sec") is not None else None
                ),
                reps=int(item["reps"]) if item.get("reps") is not None else None,
                sets=int(item["sets"]) if item.get("sets") is not None else None,
                notes=str(item["notes"]).strip() if item.get("notes") else None,
                name_source="transcript_llm",
            )
        )
    return _finalize_step_boundaries(steps, total_duration_sec)


def build_workout_summary(
    transcript: Transcript,
    title: str,
    source_url: str,
    total_duration_sec: float,
    language: str,
) -> WorkoutSummary:
    steps = _rule_based_steps(transcript, total_duration_sec)
    if not steps:
        steps = _llm_steps(transcript, total_duration_sec)
    if not steps:
        raise TranscriptError("Could not extract any workout steps from the transcript.")
    return WorkoutSummary(
        title=title,
        source_url=source_url,
        language=language,
        total_duration_sec=total_duration_sec,
        transcript_source=transcript.source,
        steps=steps,
    )


def summary_needs_visual_fallback(summary: WorkoutSummary, transcript: Transcript) -> bool:
    if not summary.steps:
        return True
    total_steps = len(summary.steps)
    rest_like = sum(
        1
        for step in summary.steps
        if step.name.lower() in {"rest", "foreign", "music", "applause"}
    )
    unique_non_rest = {
        step.name.lower()
        for step in summary.steps
        if step.name.lower() not in {"rest", "foreign", "music", "applause"}
    }
    noisy_segments = sum(
        1
        for segment in transcript.segments
        if re.search(r"\b(music|applause|foreign)\b", segment.text, re.IGNORECASE)
    )
    transcript_noise_ratio = noisy_segments / max(1, len(transcript.segments))
    if rest_like / total_steps >= 0.45:
        return True
    if len(unique_non_rest) < 3 and total_steps >= 5:
        return True
    return transcript_noise_ratio >= 0.35
