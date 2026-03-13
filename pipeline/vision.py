from __future__ import annotations

import csv
import difflib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops, ImageOps

from schemas import WorkoutStep, WorkoutSummary


class VisionExtractionError(RuntimeError):
    pass


FITNESS_TOKENS = [
    "to",
    "standing",
    "jumping",
    "jack",
    "jacks",
    "jump",
    "squat",
    "squats",
    "double",
    "crunch",
    "crunches",
    "push",
    "pushup",
    "pushups",
    "up",
    "ups",
    "towel",
    "pull",
    "pulls",
    "running",
    "run",
    "climber",
    "climbers",
    "mountain",
    "burpee",
    "burpees",
    "thruster",
    "lunge",
    "lunges",
    "plank",
    "oblique",
    "side",
    "cross",
    "combo",
    "plie",
    "hold",
    "pulse",
    "pulses",
    "scissor",
    "jackknife",
    "kick",
    "kickout",
    "kickouts",
    "kicks",
    "punch",
    "punches",
    "twist",
    "rotation",
    "toe",
    "touch",
    "touches",
    "tap",
    "taps",
    "step",
    "steps",
    "march",
    "raise",
    "raises",
    "press",
    "presses",
    "reach",
    "reaches",
    "reverse",
    "fly",
    "flies",
    "arm",
    "arms",
    "circle",
    "circles",
    "swimmer",
    "swimmers",
    "knee",
    "knees",
    "elbow",
    "elbows",
    "left",
    "right",
    "rest",
]
STOP_WORDS = {"rep", "goal", "pause", "video", "minute", "minutes"}
ACTION_NAME_BLACKLIST = {
    "chuckmeo",
    "chudmeo",
    "chudkmeo",
    "chuokmeo",
    "固動作",
    "固定动作",
    "下一個動作",
    "下一个动作",
    "分鐘",
    "分钟",
    "運動",
    "运动",
    "訓練",
    "训练",
    "感受部位",
}
ACTION_TEXT_NOISE = (
    "感受部位",
    "分鐘",
    "分钟",
    "運動",
    "运动",
    "站立式",
    "腹肌",
    "燃脂",
    "組間",
    "组间",
    "個動作",
    "个动作",
    "無跳",
    "无跳",
    "訓練",
    "训练",
    "开始",
    "開始",
    "固定动作",
    "固動作",
    "下一個動作",
    "下一个动作",
    "休息",
    "next",
    "rest",
    "workout",
    "beginner",
    "no jumping",
)
INSTRUCTION_TEXT_HINTS = (
    "接著",
    "接着",
    "然後",
    "然后",
    "保持",
    "雙手",
    "双手",
    "同時",
    "同时",
    "左右",
    "再",
)
CANONICAL_EXERCISES = [
    "Squat",
    "Split Squats",
    "Double Crunches",
    "Bicycle Crunches",
    "Push Up",
    "Towel Pull",
    "Running Climber",
    "Burpee",
    "Jumping Jacks",
    "Hi Low Planks",
    "Mountain Climber",
    "Lunge",
    "Plank",
    "Side Plank",
    "Sit Up",
    "Crunches",
    "High Knees",
    "Lateral Shuffle",
    "Dead Bug",
    "Glute Bridge",
]
ALIAS_SPLITS = {
    "PUSHUP": "PUSH UP",
    "PUSHUPS": "PUSH UPS",
    "DOUBLECRUNCHES": "DOUBLE CRUNCHES",
    "RUNNINGCLIMBER": "RUNNING CLIMBER",
    "TOWELPULL": "TOWEL PULL",
    "BUTTKICK": "BUTT KICK",
    "BUTTKICKS": "BUTT KICKS",
    "OBLIQUETO": "OBLIQUE TO",
    "SIDETO": "SIDE TO",
    "SIDEJACKKNIFE": "SIDE JACKKNIFE",
    "CRUNCHTO": "CRUNCH TO",
    "CRUNCHTOKICKOUT": "CRUNCH TO KICKOUT",
    "STANDINGCRUNCHES": "STANDING CRUNCHES",
    "SCISSORCRUNCHES": "SCISSOR CRUNCHES",
}
ALIAS_LABELS = {
    "TATERALSHUFFLE": "Lateral Shuffle",
    "LATERALSHUFFLE": "Lateral Shuffle",
    "SPUTSQUATS": "Split Squats",
    "SPLITSQUATS": "Split Squats",
    "HILOWPLANKS": "Hi Low Planks",
    "BICYCLECRUNCHES": "Bicycle Crunches",
    "HIGHKNEES": "High Knees",
}
START_ACTION_NUMBER_MIN = 20
START_PREP_NUMBER_MAX = 12
CJK_ACTION_MIN_CHARS = 2
TIMER_BOX_SPECS: dict[str, tuple[float, float, float, float]] = {
    "top_left": (0.0, 0.0, 0.34, 0.28),
    "top_center": (0.24, 0.0, 0.76, 0.24),
    "top_right": (0.66, 0.0, 1.0, 0.28),
    "upper_left_wide": (0.0, 0.0, 0.5, 0.3),
    "upper_right_wide": (0.5, 0.0, 1.0, 0.3),
    "upper_mid_left": (0.12, 0.0, 0.54, 0.28),
    "upper_mid_right": (0.46, 0.0, 0.88, 0.28),
    "mid_left": (0.0, 0.1, 0.38, 0.4),
    "mid_right": (0.62, 0.1, 1.0, 0.4),
}
PREVIEW_BOX_SPECS: dict[str, tuple[float, float, float, float]] = {
    "upper_left_card_header": (0.0, 0.0, 0.36, 0.1),
    "upper_left_card": (0.0, 0.0, 0.36, 0.18),
    "upper_band": (0.0, 0.0, 0.72, 0.22),
    "upper_left_wide": (0.0, 0.0, 0.72, 0.3),
    "upper_left_narrow": (0.0, 0.0, 0.5, 0.26),
    "upper_center": (0.1, 0.0, 0.75, 0.28),
}
SCENE_SIGNATURE_BOX = (0.02, 0.12, 0.78, 0.98)
SCENE_SIGNATURE_SIZE = (40, 40)
LONG_ACTION_SEGMENT_SEC = 10.0
VISUAL_FALLBACK_STRONG_MATCH = 0.04
VISUAL_FALLBACK_GOOD_MATCH = 0.06
VISUAL_FALLBACK_MARGIN = 0.006
LOCAL_NAME_VOTE = 3.0
BOUNDARY_NAME_VOTE = 5.0
REST_HINT_NAME_VOTE = 8.0
TIMER_OCR_SPARSE_STRIDE = 2
TIMER_OCR_SPARSE_HEAD = 12
TIMER_OCR_SPARSE_TAIL = 4
RETRY_WINDOW_STRIDE = 2
PREVIEW_RETRY_WINDOW_STRIDE = 3
PreviewNameSource = Literal["explicit", "candidate", "rest_hint", "visual"]
TRUSTED_VISUAL_REFERENCE_SOURCES = {"explicit"}


@dataclass(slots=True)
class SampleObservation:
    time_sec: float
    ratios: list[float]
    active_index: int | None
    is_rest: bool
    timer_state: str | None


@dataclass(slots=True)
class TimerSegment:
    kind: str
    start_sec: float
    end_sec: float
    duration_sec: float
    name: str | None = None
    name_source: str | None = None


@dataclass(slots=True)
class VisionProfile:
    style: str
    overlay_hits: int = 0
    timer_hits: int = 0
    next_card_hits: int = 0
    timer_box_id: str | None = None
    preview_box_id: str | None = None
    preview_box_ratios: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class OcrFrameInfo:
    time_sec: float
    label_texts: list[str]
    timer_texts: list[str]
    timer_number: int | None
    timer_kind: str | None
    explicit_preview_name: str | None
    preview_name: str | None
    frame_path: Path | None = None


@dataclass(slots=True)
class OcrDetection:
    text: str
    box: tuple[float, float, float, float]


@dataclass(slots=True)
class PreviewBoxCandidateScore:
    box_id: str
    box_ratios: tuple[float, float, float, float]
    score: float
    next_hits: int
    action_hits: int
    matching_frames: int
    matching_detections: int
    noise_detections: int
    median_fill_ratio: float
    refined_ratios: tuple[float, float, float, float]


@dataclass(slots=True)
class TimerBoxCandidateScore:
    box_id: str
    score: float
    recognized_count: int
    descending_pairs: int
    reset_hits: int
    kind_hits: int
    longest_streak: int
    texts_by_frame: list[list[str]] | None = None


@dataclass(slots=True)
class CountdownSegmentSpan:
    start_index: int
    end_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    rest_hits: int
    prep_hits: int
    action_hits: int
    max_number: int | None
    min_number: int | None


@dataclass(slots=True)
class TimerTimelineSegment:
    kind: str
    start_index: int
    end_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    preview_name: str | None = None
    preview_name_source: PreviewNameSource | None = None


def _run_command(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise VisionExtractionError(f"Command failed: {' '.join(args)}\n{stderr}")


def _extract_sample_frames(
    video_path: Path,
    frames_dir: Path,
    sample_interval_sec: int,
    total_duration_sec: float,
) -> list[tuple[float, Path]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frames_dir / "sample_%04d.jpg"
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{sample_interval_sec}",
            "-q:v",
            "3",
            str(output_pattern),
        ]
    )
    frames = sorted(frames_dir.glob("sample_*.jpg"))
    results: list[tuple[float, Path]] = []
    for index, frame_path in enumerate(frames):
        time_sec = index * sample_interval_sec
        if time_sec > total_duration_sec:
            break
        results.append((float(time_sec), frame_path))
    if not results:
        raise VisionExtractionError("No sample frames were extracted for vision fallback.")
    return results


def _prepare_crop(image: Image.Image, box: tuple[int, int, int, int], scale: int = 4) -> Image.Image:
    cropped = image.crop(box).convert("L")
    cropped = ImageOps.autocontrast(cropped)
    cropped = cropped.resize((cropped.width * scale, cropped.height * scale))
    return cropped.point(lambda pixel: 255 if pixel > 140 else 0, mode="1").convert("L")


def _run_tesseract_tsv(image_path: Path, psm: int = 6) -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "tesseract",
            str(image_path),
            "stdout",
            "--psm",
            str(psm),
            "tsv",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows = list(csv.DictReader(completed.stdout.splitlines(), delimiter="\t"))
    return [row for row in rows if row.get("text", "").strip()]


def _normalize_label(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z ]+", " ", text).upper()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or cleaned in STOP_WORDS:
        return None
    if "REST" in cleaned:
        return "Rest"
    words = cleaned.split()
    rebuilt: list[str] = []
    for word in words:
        split_word = _split_compound_token(word)
        rebuilt.extend(split_word.split())
    normalized = " ".join(rebuilt).strip()
    if not normalized:
        return None
    canonical = _canonicalize_label(normalized)
    if canonical is not None:
        canonical_compact = canonical.replace(" ", "").upper()
        normalized_compact = normalized.replace(" ", "").upper()
        if (
            normalized_compact != canonical_compact
            and canonical_compact in normalized_compact
            and len(normalized.split()) > len(canonical.split())
        ):
            return normalized.title()
        return canonical
    return normalized.title()


def _match_canonical_exercise(compact: str) -> str | None:
    match = _best_canonical_exercise(compact)
    return match[0] if match is not None else None


def _best_canonical_exercise(compact: str) -> tuple[str, float] | None:
    compact = re.sub(r"[^A-Z]+", "", compact.upper())
    if len(compact) < 4:
        return None

    best_label: str | None = None
    best_score = 0.0
    for label in CANONICAL_EXERCISES:
        canonical_compact = label.replace(" ", "").upper()
        score = difflib.SequenceMatcher(None, compact, canonical_compact).ratio()
        if canonical_compact in compact or compact in canonical_compact:
            score += 0.25
        for token in canonical_compact.split():
            if token and token[: min(4, len(token))] in compact:
                score += 0.06
        if score > best_score:
            best_score = score
            best_label = label
    if best_label is None or best_score < 0.62:
        return None
    return best_label, best_score


def _canonicalize_label(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z ]+", " ", text).upper()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or cleaned in STOP_WORDS:
        return None
    if "REST" in cleaned:
        return "Rest"
    compact = cleaned.replace(" ", "")
    if compact in ALIAS_LABELS:
        return ALIAS_LABELS[compact]
    match = _best_canonical_exercise(compact)
    return match[0] if match is not None else None


def _split_compound_token(token: str) -> str:
    token = token.upper()
    if " " in token:
        return " ".join(_split_compound_token(part) for part in token.split())
    if token in ALIAS_SPLITS:
        return ALIAS_SPLITS[token]
    if token in {part.upper() for part in FITNESS_TOKENS}:
        return token

    token_bank = sorted({part.upper() for part in FITNESS_TOKENS}, key=len, reverse=True)
    score: list[tuple[int, str] | None] = [None] * (len(token) + 1)
    score[0] = (0, "")
    for index in range(len(token)):
        current = score[index]
        if current is None:
            continue
        current_cost, current_text = current
        for part in token_bank:
            if token.startswith(part, index):
                next_index = index + len(part)
                next_text = (current_text + " " + part).strip()
                next_cost = current_cost + 1
                candidate = score[next_index]
                if candidate is None or next_cost < candidate[0]:
                    score[next_index] = (next_cost, next_text)
    result = score[len(token)]
    return result[1] if result is not None else token


def _ocr_strip_phrases(frame_path: Path) -> list[tuple[float, str]]:
    image = Image.open(frame_path)
    width, height = image.size
    strip = _prepare_crop(image, (0, int(height * 0.88), width, height))
    temp_path = frame_path.with_name(frame_path.stem + "_strip.png")
    strip.save(temp_path)
    rows = _run_tesseract_tsv(temp_path, psm=6)
    temp_path.unlink(missing_ok=True)

    phrases: list[tuple[float, str]] = []
    current_words: list[str] = []
    current_left = 0
    current_right = 0
    previous_right = None
    for row in rows:
        text = row["text"].strip()
        left = int(row["left"])
        width_px = int(row["width"])
        right = left + width_px
        gap = left - previous_right if previous_right is not None else 0
        if current_words and gap > 90:
            phrase_text = _canonicalize_label(" ".join(current_words))
            if phrase_text:
                center = (current_left + current_right) / 2
                phrases.append((center / strip.width, phrase_text))
            current_words = []
        if not current_words:
            current_left = left
        current_words.append(text)
        current_right = right
        previous_right = right

    if current_words:
        phrase_text = _canonicalize_label(" ".join(current_words))
        if phrase_text:
            center = (current_left + current_right) / 2
            phrases.append((center / strip.width, phrase_text))
    return phrases


def _orange_ratios(frame_path: Path, slot_count: int) -> list[float]:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    strip = image.crop((0, int(height * 0.90), width, height))
    ratios: list[float] = []
    for index in range(slot_count):
        x0 = int(index * strip.width / slot_count)
        x1 = int((index + 1) * strip.width / slot_count)
        region = strip.crop((x0, 0, x1, strip.height))
        pixels = list(region.getdata())
        orange = sum(
            1
            for red, green, blue in pixels
            if red > 150 and 70 < green < 175 and blue < 120 and red > green > blue
        )
        ratios.append(orange / max(1, len(pixels)))
    return ratios


def _is_rest_frame(frame_path: Path) -> bool:
    image = Image.open(frame_path)
    width, height = image.size
    focus = _prepare_crop(image, (0, int(height * 0.65), width, height), scale=3)
    temp_path = frame_path.with_name(frame_path.stem + "_rest.png")
    focus.save(temp_path)
    completed = subprocess.run(
        ["tesseract", str(temp_path), "stdout", "--psm", "6"],
        capture_output=True,
        text=True,
        check=False,
    )
    temp_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        return False
    text = completed.stdout.lower()
    return "rest" in text or "pause the video" in text


def _timer_state(frame_path: Path) -> str | None:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    crop = image.crop((0, 0, int(width * 0.18), int(height * 0.22)))
    red = 0
    green = 0
    orange = 0
    for red_channel, green_channel, blue_channel in crop.getdata():
        if red_channel > 120 and green_channel < 90 and blue_channel < 90:
            red += 1
        if green_channel > 120 and red_channel < 120 and blue_channel < 120:
            green += 1
        if (
            red_channel > 150
            and 80 < green_channel < 220
            and blue_channel < 140
            and red_channel > green_channel
        ):
            orange += 1
    if orange > max(300, green * 1.8):
        return "orange"
    if red > green * 1.3 and red > 80:
        return "red"
    if green > max(250, red * 1.3, orange * 1.2):
        return "green"
    return None


def detect_timer_state(frame_path: Path) -> str | None:
    return _timer_state(frame_path)


def _timer_state_top_right(frame_path: Path) -> str | None:
    image = Image.open(frame_path).convert("RGB")
    width, height = image.size
    crop = image.crop((int(width * 0.78), 0, width, int(height * 0.28)))
    orange = 0
    green = 0
    for red_channel, green_channel, blue_channel in crop.getdata():
        if (
            red_channel > 150
            and 80 < green_channel < 220
            and blue_channel < 140
            and red_channel > green_channel
        ):
            orange += 1
        if (
            green_channel > 130
            and red_channel < 180
            and blue_channel < 140
            and green_channel > red_channel
        ):
            green += 1
    if orange > 350 and orange > green * 1.3:
        return "orange"
    if green > 250 and green > orange * 1.1:
        return "green"
    return None


def _collect_observations(
    sample_frames: list[tuple[float, Path]],
    slot_count: int,
) -> list[SampleObservation]:
    observations: list[SampleObservation] = []
    for time_sec, frame_path in sample_frames:
        ratios = _orange_ratios(frame_path, slot_count)
        active_indexes = [index for index, ratio in enumerate(ratios) if ratio > 0.02]
        if active_indexes:
            observations.append(
                SampleObservation(
                    time_sec=time_sec,
                    ratios=ratios,
                    active_index=max(active_indexes),
                    is_rest=False,
                    timer_state=_timer_state(frame_path),
                )
            )
            continue
        observations.append(
            SampleObservation(
                time_sec=time_sec,
                ratios=ratios,
                active_index=None,
                is_rest=_is_rest_frame(frame_path),
                timer_state=_timer_state(frame_path),
            )
        )
    return observations


def _detect_round_ranges(
    observations: list[SampleObservation],
    gap_threshold_sec: int = 8,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start_index: int | None = None
    empty_run = 0
    for index, observation in enumerate(observations):
        has_active = observation.active_index is not None
        if has_active:
            empty_run = 0
            if start_index is None:
                start_index = index
            continue
        if start_index is not None:
            empty_run += 1
            if empty_run >= gap_threshold_sec:
                end_index = index - empty_run
                ranges.append((start_index, end_index))
                start_index = None
                empty_run = 0
    if start_index is not None:
        ranges.append((start_index, len(observations) - 1))
    return ranges


def _build_timer_segments(
    sample_frames: list[tuple[float, Path]],
    min_action_sec: int = 12,
    min_rest_sec: int = 4,
) -> list[TimerSegment]:
    states = [(time_sec, _timer_state_top_right(frame_path)) for time_sec, frame_path in sample_frames]
    segments: list[TimerSegment] = []
    current_state = None
    current_start = None
    for time_sec, state in states:
        if state == current_state:
            continue
        if current_state is not None and current_start is not None:
            duration = time_sec - current_start
            if (
                current_state == "green"
                and duration >= min_action_sec
            ) or (
                current_state == "orange"
                and duration >= min_rest_sec
            ):
                segments.append(
                    TimerSegment(
                        kind="action" if current_state == "green" else "rest",
                        start_sec=current_start,
                        end_sec=time_sec,
                        duration_sec=duration,
                        name_source="timer_color",
                    )
                )
        current_state = state
        current_start = time_sec
    if current_state is not None and current_start is not None:
        duration = sample_frames[-1][0] + 1.0 - current_start
        if (
            current_state == "green"
            and duration >= min_action_sec
        ) or (
            current_state == "orange"
            and duration >= min_rest_sec
        ):
            segments.append(
                TimerSegment(
                    kind="action" if current_state == "green" else "rest",
                    start_sec=current_start,
                    end_sec=sample_frames[-1][0] + 1.0,
                    duration_sec=duration,
                    name_source="timer_color",
                )
            )
    return segments


def _rapidocr_python() -> Path | None:
    configured = os.getenv("LOCAL_OCR_PYTHON")
    if configured:
        path = Path(configured).expanduser()
        return path if path.exists() else None
    candidate = Path.cwd() / ".venv-ocr" / "bin" / "python"
    if candidate.exists():
        return candidate
    current = Path(sys.executable).expanduser()
    return current if current.exists() else None


def _rapidocr_cache_path(path: Path) -> Path:
    return path.with_name(path.name + ".ocr.json")


def _load_cached_rapidocr_text(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    cache_path = _rapidocr_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    stat = path.stat()
    if (
        payload.get("source_size") != stat.st_size
        or payload.get("source_mtime_ns") != stat.st_mtime_ns
    ):
        return None
    texts = payload.get("texts")
    return texts if isinstance(texts, list) else None


def _write_cached_rapidocr_text(path: Path, texts: list[str]) -> None:
    if not path.exists():
        return
    stat = path.stat()
    payload = {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "texts": texts,
    }
    _rapidocr_cache_path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _rapidocr_detection_cache_path(path: Path) -> Path:
    return path.with_name(path.name + ".ocr_boxes.json")


def _load_cached_rapidocr_detections(path: Path) -> list[OcrDetection] | None:
    if not path.exists():
        return None
    cache_path = _rapidocr_detection_cache_path(path)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    stat = path.stat()
    if (
        payload.get("source_size") != stat.st_size
        or payload.get("source_mtime_ns") != stat.st_mtime_ns
    ):
        return None

    detections = payload.get("detections")
    if not isinstance(detections, list):
        return None

    parsed: list[OcrDetection] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        box = item.get("box")
        if not isinstance(text, str) or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            parsed.append(
                OcrDetection(
                    text=text,
                    box=tuple(float(value) for value in box),
                )
            )
        except (TypeError, ValueError):
            continue
    return parsed


def _write_cached_rapidocr_detections(path: Path, detections: list[OcrDetection]) -> None:
    if not path.exists():
        return
    stat = path.stat()
    payload = {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "detections": [
            {
                "text": item.text,
                "box": list(item.box),
            }
            for item in detections
        ],
    }
    _rapidocr_detection_cache_path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _rapidocr_text_map(paths: list[Path]) -> dict[str, list[str]]:
    if not paths:
        return {}

    results: dict[str, list[str]] = {}
    uncached_paths: list[Path] = []
    for path in paths:
        cached_texts = _load_cached_rapidocr_text(path)
        if cached_texts is not None:
            results[str(path)] = cached_texts
        else:
            uncached_paths.append(path)

    if not uncached_paths:
        return results

    python_path = _rapidocr_python()
    if python_path is None:
        return results
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "rapidocr_batch.py"
    completed = subprocess.run(
        [str(python_path), str(script_path), *[str(path) for path in uncached_paths]],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return results

    ocr_results = json.loads(completed.stdout)
    for path in uncached_paths:
        texts = ocr_results.get(str(path), [])
        results[str(path)] = texts
        _write_cached_rapidocr_text(path, texts)
    return results


def _rapidocr_detection_map(paths: list[Path]) -> dict[str, list[OcrDetection]]:
    if not paths:
        return {}

    results: dict[str, list[OcrDetection]] = {}
    uncached_paths: list[Path] = []
    for path in paths:
        cached_detections = _load_cached_rapidocr_detections(path)
        if cached_detections is not None:
            results[str(path)] = cached_detections
        else:
            uncached_paths.append(path)

    if not uncached_paths:
        return results

    python_path = _rapidocr_python()
    if python_path is None:
        return results
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "rapidocr_batch.py"
    completed = subprocess.run(
        [str(python_path), str(script_path), "--with-boxes", *[str(path) for path in uncached_paths]],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return results

    raw_results = json.loads(completed.stdout)
    for path in uncached_paths:
        detections: list[OcrDetection] = []
        for item in raw_results.get(str(path), []):
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            box = item.get("box")
            if not isinstance(text, str) or not isinstance(box, list) or len(box) != 4:
                continue
            try:
                detections.append(
                    OcrDetection(
                        text=text,
                        box=tuple(float(value) for value in box),
                    )
                )
            except (TypeError, ValueError):
                continue
        results[str(path)] = detections
        _write_cached_rapidocr_detections(path, detections)
    return results


def _rapidocr_texts(sample_frames: list[tuple[float, Path]]) -> dict[str, list[str]]:
    return _rapidocr_text_map([path for _, path in sample_frames])


def _prepare_rapidocr_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    scale: int = 3,
) -> Image.Image:
    cropped = ImageOps.autocontrast(image.crop(box).convert("L"))
    if scale > 1:
        cropped = cropped.resize((cropped.width * scale, cropped.height * scale))
    return cropped


def _build_ocr_crops(
    sample_frames: list[tuple[float, Path]],
    crops_dir: Path,
    box_fn,
    scale: int = 3,
) -> list[tuple[float, Path]]:
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_frames: list[tuple[float, Path]] = []
    for time_sec, frame_path in sample_frames:
        image = Image.open(frame_path)
        width, height = image.size
        crop = _prepare_rapidocr_crop(image, box_fn(width, height), scale=scale)
        crop_path = crops_dir / f"{frame_path.stem}.png"
        if crop_path.exists():
            try:
                existing_crop = Image.open(crop_path)
                same_size = existing_crop.size == crop.size
                same_pixels = same_size and ImageChops.difference(existing_crop, crop).getbbox() is None
                if same_pixels:
                    crop_frames.append((time_sec, crop_path))
                    continue
            except OSError:
                pass
        crop.save(crop_path)
        crop_frames.append((time_sec, crop_path))
    return crop_frames


def _normalized_box(
    width: int,
    height: int,
    ratios: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x0_ratio, y0_ratio, x1_ratio, y1_ratio = ratios
    return (
        int(width * x0_ratio),
        int(height * y0_ratio),
        int(width * x1_ratio),
        int(height * y1_ratio),
    )


def _box_fn(specs: dict[str, tuple[float, float, float, float]], box_id: str):
    ratios = specs[box_id]
    return lambda width, height: _normalized_box(width, height, ratios)


def _top_right_timer_box(width: int, height: int) -> tuple[int, int, int, int]:
    return _box_fn(TIMER_BOX_SPECS, "top_right")(width, height)


def _top_left_label_box(width: int, height: int) -> tuple[int, int, int, int]:
    return _box_fn(PREVIEW_BOX_SPECS, "upper_left_wide")(width, height)


def _default_preview_box_id(style_hint: str) -> str:
    return "upper_band" if style_hint == "timer_next_card" else "upper_left_wide"


def _default_preview_box_ratios(style_hint: str) -> tuple[float, float, float, float]:
    return PREVIEW_BOX_SPECS[_default_preview_box_id(style_hint)]


def _ocr_box_texts(
    sample_frames: list[tuple[float, Path]],
    crops_root: Path,
    specs: dict[str, tuple[float, float, float, float]],
    scale: int,
) -> dict[str, list[list[str]]]:
    texts_by_box: dict[str, list[list[str]]] = {}
    for box_id in specs:
        crops = _build_ocr_crops(
            sample_frames,
            crops_root / box_id,
            _box_fn(specs, box_id),
            scale=scale,
        )
        ocr_map = _rapidocr_text_map([path for _, path in crops])
        texts_by_box[box_id] = [ocr_map.get(str(crop_path), []) for _, crop_path in crops]
    return texts_by_box


def _normalize_detections(
    detections: list[OcrDetection],
    image_size: tuple[int, int],
) -> list[OcrDetection]:
    normalized: list[OcrDetection] = []
    for item in detections:
        normalized_box = _as_normalized_detection_box(item.box, image_size)
        if normalized_box is None:
            continue
        normalized.append(OcrDetection(text=item.text, box=normalized_box))
    return normalized


def _ocr_box_detections(
    sample_frames: list[tuple[float, Path]],
    crops_root: Path,
    specs: dict[str, tuple[float, float, float, float]],
    scale: int,
) -> dict[str, list[list[OcrDetection]]]:
    detections_by_box: dict[str, list[list[OcrDetection]]] = {}
    for box_id in specs:
        crops = _build_ocr_crops(
            sample_frames,
            crops_root / box_id,
            _box_fn(specs, box_id),
            scale=scale,
        )
        detection_map = _rapidocr_detection_map([path for _, path in crops])
        normalized_detections: list[list[OcrDetection]] = []
        for _time_sec, crop_path in crops:
            detections = detection_map.get(str(crop_path), [])
            try:
                image_size = Image.open(crop_path).size
            except OSError:
                image_size = (0, 0)
            normalized_detections.append(_normalize_detections(detections, image_size))
        detections_by_box[box_id] = normalized_detections
    return detections_by_box


def _find_contiguous_ranges(values: list[float], threshold: float) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value >= threshold:
            if start is None:
                start = index
            continue
        if start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(values)))
    return ranges


def _detect_dark_header_specs(
    sample_frames: list[tuple[float, Path]],
    max_candidates: int = 4,
) -> dict[str, tuple[float, float, float, float]]:
    if not sample_frames:
        return {}

    candidates: Counter[tuple[float, float, float, float]] = Counter()
    inspected_frames = sample_frames[: min(12, len(sample_frames))]
    for _, frame_path in inspected_frames:
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("L")
        width, height = image.size
        upper_height = max(1, int(height * 0.35))
        upper = image.crop((0, 0, width, upper_height))
        row_dark_ratios: list[float] = []
        for y in range(upper_height):
            dark_count = 0
            for x in range(width):
                if upper.getpixel((x, y)) <= 96:
                    dark_count += 1
            row_dark_ratios.append(dark_count / width)

        for row_start, row_end in _find_contiguous_ranges(row_dark_ratios, threshold=0.08):
            band_height = row_end - row_start
            if band_height < max(12, int(height * 0.025)) or band_height > int(height * 0.12):
                continue

            col_dark_ratios: list[float] = []
            for x in range(width):
                dark_count = 0
                for y in range(row_start, row_end):
                    if upper.getpixel((x, y)) <= 96:
                        dark_count += 1
                col_dark_ratios.append(dark_count / max(1, band_height))

            for col_start, col_end in _find_contiguous_ranges(col_dark_ratios, threshold=0.3):
                band_width = col_end - col_start
                if band_width < int(width * 0.14) or band_width > int(width * 0.55):
                    continue
                if col_start > int(width * 0.2):
                    continue

                x0 = round(col_start / width, 2)
                x1 = round(col_end / width, 2)
                y0 = round(max(0, row_start - band_height * 0.35) / height, 2)
                y1 = round(min(upper_height, row_end + band_height * 0.75) / height, 2)
                candidates[(x0, y0, x1, y1)] += 1

    specs: dict[str, tuple[float, float, float, float]] = {}
    for index, (ratios, _count) in enumerate(candidates.most_common(max_candidates)):
        specs[f"dynamic_header_{index + 1}"] = ratios
    return specs


def _expand_box_ratios(
    box: tuple[float, float, float, float],
    *,
    margin_x: float,
    margin_y: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    expand_x = max(0.01, width * margin_x)
    expand_y = max(0.01, height * margin_y)
    return (
        max(0.0, x0 - expand_x),
        max(0.0, y0 - expand_y),
        min(1.0, x1 + expand_x),
        min(1.0, y1 + expand_y),
    )


def _expand_box_ratios_asymmetric(
    box: tuple[float, float, float, float],
    *,
    margin_left: float,
    margin_top: float,
    margin_right: float,
    margin_bottom: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    expand_left = max(0.01, width * margin_left)
    expand_top = max(0.01, height * margin_top)
    expand_right = max(0.01, width * margin_right)
    expand_bottom = max(0.01, height * margin_bottom)
    return (
        max(0.0, x0 - expand_left),
        max(0.0, y0 - expand_top),
        min(1.0, x1 + expand_right),
        min(1.0, y1 + expand_bottom),
    )


def _union_detection_boxes(detections: list[OcrDetection]) -> tuple[float, float, float, float] | None:
    if not detections:
        return None
    x0 = min(item.box[0] for item in detections)
    y0 = min(item.box[1] for item in detections)
    x1 = max(item.box[2] for item in detections)
    y1 = max(item.box[3] for item in detections)
    return (x0, y0, x1, y1)


def _as_normalized_detection_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = box
    normalized = (
        max(0.0, min(1.0, x0 / width)),
        max(0.0, min(1.0, y0 / height)),
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
    )
    if normalized[2] - normalized[0] <= 0.02 or normalized[3] - normalized[1] <= 0.02:
        return None
    return normalized


def _lift_child_box(
    parent_box: tuple[float, float, float, float],
    child_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    parent_x0, parent_y0, parent_x1, parent_y1 = parent_box
    parent_width = parent_x1 - parent_x0
    parent_height = parent_y1 - parent_y0
    child_x0, child_y0, child_x1, child_y1 = child_box
    return (
        parent_x0 + parent_width * child_x0,
        parent_y0 + parent_height * child_y0,
        parent_x0 + parent_width * child_x1,
        parent_y0 + parent_height * child_y1,
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    center = _median(values)
    return sum(abs(value - center) for value in values) / len(values)


def _texts_from_detections(detections: list[OcrDetection]) -> list[str]:
    return [item.text for item in detections]


def _phrase_texts_from_detections(detections: list[OcrDetection]) -> list[str]:
    if not detections:
        return []

    ordered = sorted(
        (item for item in detections if _clean_text(item.text)),
        key=lambda item: ((item.box[1] + item.box[3]) / 2.0, item.box[0]),
    )
    if not ordered:
        return []

    heights = [max(0.0, item.box[3] - item.box[1]) for item in ordered]
    line_threshold = max(0.035, _median(heights) * 0.8)
    lines: list[list[OcrDetection]] = []
    for item in ordered:
        center_y = (item.box[1] + item.box[3]) / 2.0
        if not lines:
            lines.append([item])
            continue
        previous_line = lines[-1]
        previous_center_y = _median([(entry.box[1] + entry.box[3]) / 2.0 for entry in previous_line])
        if abs(center_y - previous_center_y) <= line_threshold:
            previous_line.append(item)
            continue
        lines.append([item])

    phrases: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item.box[0])
        token_texts = [_clean_text(item.text) for item in line if _clean_text(item.text)]
        if not token_texts:
            continue
        spaced = " ".join(token_texts)
        compact = "".join(token_texts)
        phrases.extend(token_texts)
        if spaced not in phrases:
            phrases.append(spaced)
        if compact != spaced and compact not in phrases:
            phrases.append(compact)
    return phrases


def _is_preview_detection_candidate(text: str) -> bool:
    if _extract_explicit_preview_name([text]) is not None:
        return True
    candidate, score = _score_action_candidate(text)
    return candidate is not None and score > 0


def _refine_preview_box_ratios(
    matching_boxes: list[tuple[float, float, float, float]],
    box_ratios: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not matching_boxes:
        return box_ratios

    refined = (
        _median([box[0] for box in matching_boxes]),
        _median([box[1] for box in matching_boxes]),
        _median([box[2] for box in matching_boxes]),
        _median([box[3] for box in matching_boxes]),
    )
    # Keep extra context around the detected title text, with a larger right margin
    # because workout labels are frequently truncated on the final word.
    refined = _expand_box_ratios_asymmetric(
        refined,
        margin_left=0.16,
        margin_top=0.4,
        margin_right=0.5,
        margin_bottom=0.75,
    )
    lifted = _lift_child_box(box_ratios, refined)
    lifted_width = lifted[2] - lifted[0]
    lifted_height = lifted[3] - lifted[1]
    base_width = box_ratios[2] - box_ratios[0]
    base_height = box_ratios[3] - box_ratios[1]
    if lifted_width < base_width * 0.14 or lifted_height < base_height * 0.18:
        return box_ratios

    # When detections hug the crop boundary, the parent candidate box was too tight.
    # Expand the final crop in global coordinates instead of staying trapped inside
    # the original candidate region.
    right_edge = _median([box[2] for box in matching_boxes])
    left_edge = _median([box[0] for box in matching_boxes])
    top_edge = _median([box[1] for box in matching_boxes])
    bottom_edge = _median([box[3] for box in matching_boxes])

    extra_left = 0.0
    extra_top = 0.0
    extra_right = 0.0
    extra_bottom = 0.0

    if right_edge >= 0.94:
        extra_right += max(base_width * 0.45, 0.045)
    elif right_edge >= 0.88:
        extra_right += max(base_width * 0.25, 0.025)

    if left_edge <= 0.06:
        extra_left += max(base_width * 0.18, 0.02)

    if top_edge <= 0.05:
        extra_top += max(base_height * 0.2, 0.01)

    if bottom_edge >= 0.95:
        extra_bottom += max(base_height * 0.2, 0.01)

    if extra_left or extra_top or extra_right or extra_bottom:
        lifted = (
            max(0.0, lifted[0] - extra_left),
            max(0.0, lifted[1] - extra_top),
            min(1.0, lifted[2] + extra_right),
            min(1.0, lifted[3] + extra_bottom),
        )
    return lifted


def _score_preview_box_candidate(
    box_id: str,
    box_ratios: tuple[float, float, float, float],
    detections_by_frame: list[list[OcrDetection]],
) -> PreviewBoxCandidateScore:
    next_hits = 0
    action_hits = 0
    matching_frames = 0
    matching_detections = 0
    noise_detections = 0
    matching_boxes: list[tuple[float, float, float, float]] = []
    fill_ratios: list[float] = []
    center_xs: list[float] = []
    center_ys: list[float] = []
    widths: list[float] = []
    heights: list[float] = []

    for detections in detections_by_frame:
        texts = _texts_from_detections(detections)
        if _extract_explicit_preview_name(texts):
            next_hits += 1
        if _extract_action_name(texts) is not None:
            action_hits += 1

        nonempty_detections = [item for item in detections if _clean_text(item.text)]
        matching = [item for item in nonempty_detections if _is_preview_detection_candidate(item.text)]
        matching_detections += len(matching)
        noise_detections += max(0, len(nonempty_detections) - len(matching))
        if not matching:
            continue

        union_box = _union_detection_boxes(matching)
        if union_box is None:
            continue
        matching_frames += 1
        matching_boxes.append(union_box)
        width = max(0.0, union_box[2] - union_box[0])
        height = max(0.0, union_box[3] - union_box[1])
        widths.append(width)
        heights.append(height)
        fill_ratios.append(width * height)
        center_xs.append((union_box[0] + union_box[2]) / 2.0)
        center_ys.append((union_box[1] + union_box[3]) / 2.0)

    median_fill_ratio = _median(fill_ratios)
    refined_ratios = _refine_preview_box_ratios(matching_boxes, box_ratios)

    score = 0.0
    score += next_hits * 8.0
    score += action_hits * 3.0
    score += matching_frames * 2.0
    score += matching_detections * 0.75
    score -= noise_detections * 0.3

    if median_fill_ratio >= 0.08:
        score += min(4.5, median_fill_ratio * 14.0)
    elif median_fill_ratio >= 0.04:
        score += 0.5
    elif matching_frames:
        score -= 2.5

    if matching_boxes:
        spread = (
            _mean_abs_deviation(center_xs)
            + _mean_abs_deviation(center_ys)
            + _mean_abs_deviation(widths)
            + _mean_abs_deviation(heights)
        )
        score += max(0.0, 5.0 - spread * 24.0)

    if matching_frames >= 2 and median_fill_ratio >= 0.1:
        score += 1.5

    return PreviewBoxCandidateScore(
        box_id=box_id,
        box_ratios=box_ratios,
        score=score,
        next_hits=next_hits,
        action_hits=action_hits,
        matching_frames=matching_frames,
        matching_detections=matching_detections,
        noise_detections=noise_detections,
        median_fill_ratio=median_fill_ratio,
        refined_ratios=refined_ratios,
    )


def _select_probe_frames(
    sample_frames: list[tuple[float, Path]],
    max_count: int = 36,
    keep_head: int = 8,
) -> list[tuple[float, Path]]:
    if len(sample_frames) <= max_count:
        return sample_frames

    selected_indexes = set(range(min(keep_head, len(sample_frames))))
    remaining_indexes = list(range(min(keep_head, len(sample_frames)), len(sample_frames)))
    remaining_slots = max(0, max_count - len(selected_indexes))
    if remaining_slots > 0 and remaining_indexes:
        stride = max(1, len(remaining_indexes) // remaining_slots)
        for index in remaining_indexes[::stride]:
            selected_indexes.add(index)
            if len(selected_indexes) >= max_count:
                break

    return [sample_frames[index] for index in sorted(selected_indexes)]


def _count_descending_pairs(numbers: list[int | None]) -> tuple[int, int, int]:
    descending_pairs = 0
    reset_hits = 0
    longest_streak = 0
    current_streak = 0
    previous_number: int | None = None
    for number in numbers:
        if not isinstance(number, int):
            current_streak = 0
            previous_number = None
            continue
        if previous_number is not None:
            difference = previous_number - number
            if 1 <= difference <= 2:
                descending_pairs += 1
                current_streak += 1
            elif number >= previous_number + 6 or (previous_number <= 3 and number >= 28):
                reset_hits += 1
                current_streak = 0
            else:
                current_streak = 0
        previous_number = number
        longest_streak = max(longest_streak, current_streak)
    return descending_pairs, reset_hits, longest_streak


def _score_timer_box_candidate(box_id: str, texts_by_frame: list[list[str]]) -> TimerBoxCandidateScore:
    numbers = [_extract_timer_number(texts) for texts in texts_by_frame]
    kinds = [_extract_timer_kind(texts) for texts in texts_by_frame]
    recognized_count = sum(
        1 for number, kind in zip(numbers, kinds) if number is not None or kind is not None
    )
    kind_hits = sum(1 for kind in kinds if kind is not None)
    descending_pairs, reset_hits, longest_streak = _count_descending_pairs(numbers)
    score = (
        recognized_count * 1.0
        + kind_hits * 1.5
        + descending_pairs * 3.0
        + reset_hits * 6.0
        + longest_streak * 4.0
    )
    return TimerBoxCandidateScore(
        box_id=box_id,
        score=score,
        recognized_count=recognized_count,
        descending_pairs=descending_pairs,
        reset_hits=reset_hits,
        kind_hits=kind_hits,
        longest_streak=longest_streak,
        texts_by_frame=texts_by_frame,
    )


def _detect_timer_box(
    sample_frames: list[tuple[float, Path]],
    workdir: Path,
) -> TimerBoxCandidateScore | None:
    probe_frames = _select_probe_frames(sample_frames)
    candidate_texts = _ocr_box_texts(
        probe_frames,
        workdir / "profile_timer_candidates",
        TIMER_BOX_SPECS,
        scale=4,
    )
    candidate_scores = [
        _score_timer_box_candidate(box_id, texts_by_frame)
        for box_id, texts_by_frame in candidate_texts.items()
    ]
    candidate_scores.sort(
        key=lambda item: (
            item.score,
            item.longest_streak,
            item.reset_hits,
            item.recognized_count,
        ),
        reverse=True,
    )
    if not candidate_scores:
        return None
    best = candidate_scores[0]
    if best.recognized_count < max(6, len(probe_frames) // 6):
        return None
    if best.descending_pairs == 0 and best.reset_hits == 0 and best.kind_hits < 3:
        return None
    return best


def _detect_preview_box(
    sample_frames: list[tuple[float, Path]],
    target_indexes: set[int],
    workdir: Path,
) -> tuple[str | None, tuple[float, float, float, float] | None, int, int]:
    if not target_indexes:
        return None, None, 0, 0

    target_frames = [sample_frames[index] for index in sorted(target_indexes)]
    preview_specs = dict(PREVIEW_BOX_SPECS)
    preview_specs.update(_detect_dark_header_specs(target_frames))
    candidate_detections = _ocr_box_detections(
        target_frames,
        workdir / "profile_preview_candidates",
        preview_specs,
        scale=3,
    )

    candidate_scores = [
        _score_preview_box_candidate(box_id, preview_specs[box_id], detections_by_frame)
        for box_id, detections_by_frame in candidate_detections.items()
    ]
    candidate_scores.sort(
        key=lambda item: (
            item.score,
            item.next_hits,
            item.action_hits,
            item.matching_frames,
            item.median_fill_ratio,
            -item.noise_detections,
        ),
        reverse=True,
    )
    if not candidate_scores:
        return None, None, 0, 0
    best = candidate_scores[0]
    return best.box_id, best.refined_ratios, best.next_hits, best.action_hits


def _detect_visual_style(sample_frames: list[tuple[float, Path]]) -> VisionProfile:
    timer_candidate = _detect_timer_box(sample_frames, sample_frames[0][1].parent.parent)
    if timer_candidate is None:
        overlay_hits = sum(
            1
            for _, frame_path in sample_frames
            if max(_orange_ratios(frame_path, 6), default=0.0) > 0.02
        )
        style = "overlay_bar" if overlay_hits else "timer_only"
        return VisionProfile(style=style, overlay_hits=overlay_hits)

    overlay_hits = 0
    timer_texts_by_frame = timer_candidate.texts_by_frame or []
    target_indexes = _select_full_ocr_target_indexes(timer_texts_by_frame)
    preview_box_id, preview_box_ratios, next_card_hits, _ = _detect_preview_box(
        sample_frames,
        target_indexes,
        sample_frames[0][1].parent.parent,
    )
    style = "timer_next_card" if next_card_hits >= 2 else "timer_only"
    return VisionProfile(
        style=style,
        overlay_hits=overlay_hits,
        timer_hits=timer_candidate.recognized_count,
        next_card_hits=next_card_hits,
        timer_box_id=timer_candidate.box_id,
        preview_box_id=preview_box_id or _default_preview_box_id(style),
        preview_box_ratios=preview_box_ratios or _default_preview_box_ratios(style),
    )


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_english_candidate(text: str) -> str | None:
    cleaned = re.sub(r"[^A-Za-z ]+", " ", text).upper()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    normalized = _normalize_label(cleaned)
    if normalized is not None:
        return normalized

    rebuilt_words: list[str] = []
    for word in cleaned.split():
        rebuilt_words.extend(_split_compound_token(word).split())
    rebuilt = " ".join(rebuilt_words).strip()
    if not rebuilt:
        return None

    letters_only = re.sub(r"[^A-Z]+", "", rebuilt)
    if len(letters_only) < 4:
        return None
    return rebuilt.title()


def _extract_timer_number(texts: list[str]) -> int | None:
    candidates: list[int] = []
    for text in texts:
        for match in re.findall(r"\d+", text):
            value = int(match)
            if 0 <= value <= 60:
                candidates.append(value)
    return candidates[-1] if candidates else None


def _extract_timer_kind(texts: list[str]) -> str | None:
    joined = " ".join(texts)
    joined_upper = joined.upper()
    number = _extract_timer_number(texts)
    if "休息" in joined or "REST" in joined_upper or "NEXT:" in joined_upper:
        return "rest"
    if "開始" in joined or "开始" in joined or "START" in joined_upper:
        if number is not None and number >= START_ACTION_NUMBER_MIN:
            return "action"
        if number is not None and number <= START_PREP_NUMBER_MAX:
            return "prep"
        return None
    if "WORK" in joined_upper or "GO" in joined_upper:
        return "action"
    return None


def _score_action_candidate(text: str) -> tuple[str | None, float]:
    cleaned = _clean_text(text)
    if not cleaned:
        return None, -100.0

    lowered = cleaned.lower()
    if any(token in lowered for token in ACTION_TEXT_NOISE):
        return None, -100.0

    if re.search(r"[\u4e00-\u9fff]", cleaned):
        candidate = re.sub(r"[：:，,。.！？!?0-9A-Za-z ]+", "", cleaned).strip()
        if not candidate or candidate in ACTION_NAME_BLACKLIST:
            return None, -100.0
        if len(candidate) < CJK_ACTION_MIN_CHARS:
            return None, -100.0
        score = 0.0
        if 2 <= len(candidate) <= 6:
            score += 4.0
        elif len(candidate) <= 8:
            score += 1.5
        else:
            score -= 3.0
        if any(token in cleaned for token in INSTRUCTION_TEXT_HINTS):
            score -= 4.0
        return candidate, score

    candidate = _normalize_english_candidate(cleaned)
    if candidate is None or candidate.lower() in ACTION_NAME_BLACKLIST:
        return None, -100.0
    if "meo" in candidate.lower():
        return None, -100.0
    word_count = len(candidate.split())
    trailing_direction = candidate.split()[-1].upper() if candidate.split() else None
    letter_count = len(re.sub(r"[^A-Za-z]+", "", candidate))
    if letter_count < 4:
        return None, -100.0
    score = 0.0
    if 1 <= word_count <= 4:
        score += 3.0
    elif word_count == 5 and trailing_direction in {"L", "R", "LEFT", "RIGHT"}:
        score += 2.5
    if letter_count <= 2:
        score -= 6.0
    if word_count == 1 and letter_count <= 5:
        score -= 1.5
    if len(candidate) > 28:
        score -= 2.0
    if candidate.upper() in {"REST", "START"}:
        score -= 5.0
    return candidate, score


def _extract_explicit_preview_name(texts: list[str]) -> str | None:
    candidate_texts = texts[:]
    joined = _clean_text(" ".join(texts))
    compact_joined = _clean_text("".join(texts))
    if joined:
        candidate_texts.append(joined)
    if compact_joined and compact_joined != joined:
        candidate_texts.append(compact_joined)

    for text in candidate_texts:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        if re.search(r"NEXT\s*:", cleaned, re.IGNORECASE):
            name_part = re.split(r"NEXT\s*:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)[1]
            name = _normalize_english_candidate(name_part)
            if name:
                return name
        if "下一個動作" in cleaned or "下一个动作" in cleaned:
            parts = re.split(r"下一個動作|下一个动作", cleaned, maxsplit=1)
            if len(parts) == 2:
                candidate, score = _score_action_candidate(parts[1])
                if candidate and score > 0:
                    return candidate
    return None


def _extract_action_name(texts: list[str]) -> str | None:
    explicit = _extract_explicit_preview_name(texts)
    if explicit is not None:
        return explicit

    candidates: list[tuple[str, float]] = []
    for text in texts:
        candidate, score = _score_action_candidate(text)
        if candidate and score > 0:
            candidates.append((candidate, score))
    if not candidates:
        return None
    totals: dict[str, float] = {}
    for candidate, score in candidates:
        totals[candidate] = totals.get(candidate, 0.0) + score
    return max(totals.items(), key=lambda item: (item[1], -len(item[0])))[0]


def _expand_index_window(indexes: set[int], index: int, total_count: int, radius: int = 1) -> None:
    for offset in range(-radius, radius + 1):
        candidate = index + offset
        if 0 <= candidate < total_count:
            indexes.add(candidate)


def _add_retry_window(
    indexes: set[int],
    total_count: int,
    start_index: int,
    end_index: int,
    *,
    anchor_index: int | None = None,
    stride: int = RETRY_WINDOW_STRIDE,
    dense_radius: int = 1,
) -> None:
    window_start = max(0, start_index)
    window_end = min(total_count, end_index)
    if window_start >= window_end:
        return
    if window_end - window_start <= 4:
        indexes.update(range(window_start, window_end))
        return

    anchor = anchor_index if anchor_index is not None else window_start
    for candidate in range(window_start, window_end):
        if abs(candidate - anchor) <= dense_radius:
            indexes.add(candidate)
            continue
        if (candidate - window_start) % max(1, stride) == 0:
            indexes.add(candidate)


def _compress_target_indexes(
    target_indexes: set[int],
    timer_kinds: list[str | None],
    timer_numbers: list[int | None],
    max_count: int,
    min_gap: int,
) -> set[int]:
    if len(target_indexes) <= max_count:
        return target_indexes

    reset_indexes: set[int] = set()
    for index in range(1, len(timer_numbers)):
        previous_number = timer_numbers[index - 1]
        current_number = timer_numbers[index]
        if (
            isinstance(previous_number, int)
            and isinstance(current_number, int)
            and (
                current_number >= previous_number + 6
                or (previous_number <= 3 and current_number >= 24)
            )
        ):
            reset_indexes.update({index - 1, index})

    ranked_indexes: list[tuple[float, int]] = []
    for index in sorted(target_indexes):
        score = 0.0
        if index < 6:
            score += 10.0
        kind = timer_kinds[index]
        number = timer_numbers[index]
        if kind == "rest":
            score += 9.0
        elif kind == "prep":
            score += 7.0
        elif kind == "action":
            score += 2.0
        if isinstance(number, int):
            if number <= 2:
                score += 8.0
            elif number <= 4:
                score += 6.0
            elif number <= 8:
                score += 2.0
        if index in reset_indexes:
            score += 8.0
        ranked_indexes.append((score, index))

    kept_indexes: list[int] = []
    protected_indexes = {index for index in sorted(target_indexes) if index < 6 or index in reset_indexes}
    for index in sorted(protected_indexes):
        if index < 6:
            kept_indexes.append(index)
        elif all(abs(index - kept_index) >= min_gap for kept_index in kept_indexes):
            kept_indexes.append(index)
        if len(kept_indexes) >= max_count:
            return set(sorted(kept_indexes))

    for _, index in sorted(ranked_indexes, key=lambda item: (-item[0], item[1])):
        if index in kept_indexes:
            continue
        if any(abs(index - kept_index) < min_gap for kept_index in kept_indexes):
            continue
        kept_indexes.append(index)
        if len(kept_indexes) >= max_count:
            break

    if len(kept_indexes) < max_count:
        for index in sorted(target_indexes):
            if index in kept_indexes:
                continue
            kept_indexes.append(index)
            if len(kept_indexes) >= max_count:
                break
    return set(sorted(kept_indexes))


def _select_full_ocr_target_indexes(timer_texts_by_frame: list[list[str]]) -> set[int]:
    total_count = len(timer_texts_by_frame)
    if total_count == 0:
        return set()

    target_indexes = set(range(min(6, total_count)))
    timer_kinds = [_extract_timer_kind(texts) for texts in timer_texts_by_frame]
    timer_numbers = [_extract_timer_number(texts) for texts in timer_texts_by_frame]

    for index, kind in enumerate(timer_kinds):
        if kind in {"rest", "prep"}:
            _expand_index_window(target_indexes, index, total_count)
        if isinstance(timer_numbers[index], int) and timer_numbers[index] <= 4:
            _expand_index_window(target_indexes, index, total_count)
        if (
            index > 0
            and isinstance(timer_numbers[index - 1], int)
            and isinstance(timer_numbers[index], int)
            and (
                timer_numbers[index] >= timer_numbers[index - 1] + 6
                or (timer_numbers[index - 1] <= 3 and timer_numbers[index] >= 24)
            )
        ):
            _expand_index_window(target_indexes, index - 1, total_count, radius=2)
            _expand_index_window(target_indexes, index, total_count, radius=2)

    run_start: int | None = None
    for index in range(total_count + 1):
        is_missing = (
            index < total_count
            and timer_kinds[index] is None
            and timer_numbers[index] is None
        )
        if is_missing:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue

        run_end = index
        run_length = run_end - run_start
        if run_length >= 2:
            for target in range(run_start, run_end):
                _expand_index_window(target_indexes, target, total_count)
        else:
            previous_number = timer_numbers[run_start - 1] if run_start > 0 else None
            next_number = timer_numbers[run_end] if run_end < total_count else None
            if (
                isinstance(previous_number, int)
                and isinstance(next_number, int)
                and next_number >= previous_number + 6
            ):
                _expand_index_window(target_indexes, run_start, total_count)
        run_start = None

    return _compress_target_indexes(
        target_indexes,
        timer_kinds,
        timer_numbers,
        max_count=min(48, max(18, total_count // 6)),
        min_gap=2,
    )


def _select_sparse_timer_ocr_target_indexes(total_count: int) -> set[int]:
    if total_count <= 0:
        return set()

    target_indexes = set(range(min(TIMER_OCR_SPARSE_HEAD, total_count)))
    for index in range(TIMER_OCR_SPARSE_HEAD, total_count, TIMER_OCR_SPARSE_STRIDE):
        target_indexes.add(index)
    for index in range(max(0, total_count - TIMER_OCR_SPARSE_TAIL), total_count):
        target_indexes.add(index)
    return target_indexes


def _find_previous_timer_signal(
    timer_numbers: list[int | None],
    timer_kinds: list[str | None],
    index: int,
    max_lookback: int = 4,
) -> tuple[int | None, str | None, int | None]:
    for candidate in range(index - 1, max(-1, index - max_lookback - 1), -1):
        if isinstance(timer_numbers[candidate], int) or timer_kinds[candidate] is not None:
            return timer_numbers[candidate], timer_kinds[candidate], candidate
    return None, None, None


def _select_boundary_timer_retry_indexes(
    timer_texts_by_frame: list[list[str]],
    sparse_target_indexes: set[int],
) -> set[int]:
    total_count = len(timer_texts_by_frame)
    if total_count == 0:
        return set()

    retry_indexes: set[int] = set()
    timer_kinds = [_extract_timer_kind(texts) for texts in timer_texts_by_frame]
    timer_numbers = [_extract_timer_number(texts) for texts in timer_texts_by_frame]

    for index in sorted(sparse_target_indexes):
        if index < 6:
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 1,
                index + 2,
                anchor_index=index,
                dense_radius=1,
            )

    for index in range(1, total_count):
        previous_number, previous_kind, previous_index = _find_previous_timer_signal(
            timer_numbers,
            timer_kinds,
            index,
        )
        current_number = timer_numbers[index]
        current_kind = timer_kinds[index]

        if (
            isinstance(previous_number, int)
            and isinstance(current_number, int)
            and (
                current_number >= previous_number + 6
                or (previous_number <= 3 and current_number >= 24)
            )
        ):
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 3,
                index + 4,
                anchor_index=index,
                dense_radius=1,
            )

        if current_kind is not None and previous_kind is not None and current_kind != previous_kind:
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 3,
                index + 4,
                anchor_index=index,
                dense_radius=1,
            )

        if (
            isinstance(current_number, int)
            and current_number >= 24
            and (
                previous_index is None
                or not isinstance(previous_number, int)
                or previous_number <= 4
                or previous_kind in {"rest", "prep"}
            )
        ):
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 5,
                index + 3,
                anchor_index=index,
                dense_radius=2,
            )

    return retry_indexes - sparse_target_indexes


def _has_sufficient_timer_signal(timer_texts_by_frame: list[list[str]]) -> bool:
    total_count = len(timer_texts_by_frame)
    if total_count == 0:
        return False

    observed_texts = [texts for texts in timer_texts_by_frame if texts]
    observed_count = len(observed_texts)
    minimum_observed = max(8, min(80, total_count // 16))
    if observed_count < minimum_observed:
        return False

    score = _score_timer_box_candidate("timeline", observed_texts)
    if score.descending_pairs >= max(3, min(18, observed_count // 12)):
        return True
    if score.longest_streak >= 4:
        return True
    return score.kind_hits >= 4 or score.reset_hits >= 1


def _build_label_ocr_map(
    sample_frames: list[tuple[float, Path]],
    workdir: Path,
    style_hint: str,
    preview_box_id: str | None,
    preview_box_ratios: tuple[float, float, float, float] | None,
    target_indexes: set[int] | None,
) -> dict[str, list[str]]:
    if target_indexes is None:
        target_frames = sample_frames
    else:
        target_frames = [sample_frames[index] for index in sorted(target_indexes)]
    if not target_frames:
        return {}

    label_crops = _build_ocr_crops(
        target_frames,
        workdir / "ocr_label",
        _box_fn(
            {
                preview_box_id or "resolved_preview": (
                    preview_box_ratios or _default_preview_box_ratios(style_hint)
                )
            },
            preview_box_id or "resolved_preview",
        ),
        scale=3,
    )
    detection_map = _rapidocr_detection_map([path for _, path in label_crops])
    return {
        str(frame_path): _phrase_texts_from_detections(detection_map.get(str(label_crop_path), []))
        for (_, frame_path), (_, label_crop_path) in zip(target_frames, label_crops)
    }


def _count_preview_name_hits(label_frame_map: dict[str, list[str]]) -> tuple[int, int]:
    non_empty_count = 0
    named_count = 0
    for label_texts in label_frame_map.values():
        if label_texts:
            non_empty_count += 1
        explicit_name = _extract_explicit_preview_name(label_texts)
        preview_name = _extract_action_name(label_texts)
        if _is_plausible_preview_name(explicit_name) or _is_plausible_preview_name(preview_name):
            named_count += 1
    return non_empty_count, named_count


def _should_retry_dense_preview_ocr(
    total_count: int,
    sparse_target_count: int,
    label_frame_map: dict[str, list[str]],
) -> bool:
    if total_count <= sparse_target_count or total_count <= 0:
        return False
    non_empty_count, named_count = _count_preview_name_hits(label_frame_map)
    if named_count >= max(2, min(8, total_count // 120)):
        return False
    if non_empty_count >= max(4, min(12, sparse_target_count // 2)):
        return False
    return True


def _select_boundary_preview_retry_indexes(
    timer_texts_by_frame: list[list[str]],
    sparse_target_indexes: set[int],
) -> set[int]:
    total_count = len(timer_texts_by_frame)
    if total_count == 0:
        return set()

    retry_indexes = set(range(min(8, total_count)))
    timer_kinds = [_extract_timer_kind(texts) for texts in timer_texts_by_frame]
    timer_numbers = [_extract_timer_number(texts) for texts in timer_texts_by_frame]
    reset_indexes: set[int] = set()

    for index in range(1, total_count):
        previous_number, _previous_kind, previous_index = _find_previous_timer_signal(
            timer_numbers,
            timer_kinds,
            index,
        )
        current_number = timer_numbers[index]
        if (
            previous_index is not None
            and isinstance(previous_number, int)
            and isinstance(current_number, int)
            and (
                current_number >= previous_number + 6
                or (previous_number <= 3 and current_number >= 24)
            )
        ):
            reset_indexes.add(index)

    for index in sorted(sparse_target_indexes):
        should_expand = (
            index < 6
            or timer_kinds[index] in {"rest", "prep"}
            or (isinstance(timer_numbers[index], int) and timer_numbers[index] <= 4)
            or index in reset_indexes
            or (index - 1) in reset_indexes
            or (index + 1) in reset_indexes
        )
        if not should_expand:
            continue
        _add_retry_window(
            retry_indexes,
            total_count,
            index - 3,
            index + 4,
            anchor_index=index,
            stride=PREVIEW_RETRY_WINDOW_STRIDE,
            dense_radius=1,
        )

    for index in range(1, total_count):
        previous_number, previous_kind, previous_index = _find_previous_timer_signal(
            timer_numbers,
            timer_kinds,
            index,
        )
        current_number = timer_numbers[index]
        if index in reset_indexes:
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 4,
                index + 11,
                anchor_index=index,
                stride=PREVIEW_RETRY_WINDOW_STRIDE,
                dense_radius=1,
            )
        if (
            isinstance(current_number, int)
            and current_number >= 24
            and (
                previous_index is None
                or not isinstance(previous_number, int)
                or previous_number <= 4
                or previous_kind in {"rest", "prep"}
            )
        ):
            _add_retry_window(
                retry_indexes,
                total_count,
                index - 8,
                index + 4,
                anchor_index=index,
                stride=PREVIEW_RETRY_WINDOW_STRIDE,
                dense_radius=4,
            )

    run_start: int | None = None
    for index in range(total_count + 1):
        kind = timer_kinds[index] if index < total_count else None
        if kind in {"rest", "prep"}:
            if run_start is None:
                run_start = index
            continue
        if run_start is None:
            continue
        _add_retry_window(
            retry_indexes,
            total_count,
            index - 4,
            index + 11,
            anchor_index=index,
            stride=PREVIEW_RETRY_WINDOW_STRIDE,
            dense_radius=1,
        )
        run_start = None

    return retry_indexes


def _build_ocr_frame_infos(
    sample_frames: list[tuple[float, Path]],
    workdir: Path,
    style_hint: str,
    timer_box_id: str,
    preview_box_id: str | None,
    preview_box_ratios: tuple[float, float, float, float] | None = None,
) -> list[OcrFrameInfo]:
    timer_crops = _build_ocr_crops(
        sample_frames,
        workdir / "ocr_timer",
        _box_fn(TIMER_BOX_SPECS, timer_box_id),
        scale=4,
    )
    sparse_timer_target_indexes = _select_sparse_timer_ocr_target_indexes(len(timer_crops))
    sparse_timer_paths = [timer_crops[index][1] for index in sorted(sparse_timer_target_indexes)]
    timer_ocr_map = _rapidocr_text_map(sparse_timer_paths)
    timer_texts_by_frame = [timer_ocr_map.get(str(timer_crop_path), []) for _, timer_crop_path in timer_crops]

    if not _has_sufficient_timer_signal(timer_texts_by_frame):
        timer_ocr_map = _rapidocr_text_map([path for _, path in timer_crops])
        timer_texts_by_frame = [
            timer_ocr_map.get(str(timer_crop_path), [])
            for _, timer_crop_path in timer_crops
        ]
    else:
        retry_timer_target_indexes = _select_boundary_timer_retry_indexes(
            timer_texts_by_frame,
            sparse_timer_target_indexes,
        )
        retry_timer_paths = [timer_crops[index][1] for index in sorted(retry_timer_target_indexes)]
        if retry_timer_paths:
            timer_ocr_map.update(_rapidocr_text_map(retry_timer_paths))
            timer_texts_by_frame = [
                timer_ocr_map.get(str(timer_crop_path), [])
                for _, timer_crop_path in timer_crops
            ]
        if not _has_sufficient_timer_signal(timer_texts_by_frame):
            timer_ocr_map = _rapidocr_text_map([path for _, path in timer_crops])
            timer_texts_by_frame = [
                timer_ocr_map.get(str(timer_crop_path), [])
                for _, timer_crop_path in timer_crops
            ]

    if not any(timer_texts_by_frame):
        return []
    target_indexes = _select_full_ocr_target_indexes(timer_texts_by_frame)
    label_frame_map = _build_label_ocr_map(
        sample_frames,
        workdir,
        style_hint,
        preview_box_id,
        preview_box_ratios,
        target_indexes,
    )
    if _should_retry_dense_preview_ocr(len(sample_frames), len(target_indexes), label_frame_map):
        label_frame_map = _build_label_ocr_map(
            sample_frames,
            workdir,
            style_hint,
            preview_box_id,
            preview_box_ratios,
            _select_boundary_preview_retry_indexes(timer_texts_by_frame, target_indexes),
        )

    infos: list[OcrFrameInfo] = []
    for (time_sec, frame_path), (_, timer_crop_path) in zip(sample_frames, timer_crops):
        timer_texts = timer_ocr_map.get(str(timer_crop_path), [])
        label_texts = label_frame_map.get(str(frame_path), [])
        infos.append(
            OcrFrameInfo(
                time_sec=time_sec,
                label_texts=label_texts,
                timer_texts=timer_texts,
                timer_number=_extract_timer_number(timer_texts),
                timer_kind=_extract_timer_kind(timer_texts),
                explicit_preview_name=_extract_explicit_preview_name(label_texts),
                preview_name=_extract_action_name(label_texts),
                frame_path=frame_path,
            )
        )
    return infos


def _has_timer_signal(info: OcrFrameInfo) -> bool:
    return info.timer_number is not None or info.timer_kind is not None


def _find_previous_timer_number(
    frame_infos: list[OcrFrameInfo],
    index: int,
    max_lookback: int = 3,
) -> tuple[int | None, int | None]:
    for candidate in range(index - 1, max(-1, index - max_lookback - 1), -1):
        if frame_infos[candidate].timer_number is not None:
            return frame_infos[candidate].timer_number, candidate
    return None, None


def _has_recent_timer_kind(
    frame_infos: list[OcrFrameInfo],
    index: int,
    kind: str,
    max_lookback: int = 2,
) -> bool:
    start_index = max(0, index - max_lookback)
    return any(frame_infos[candidate].timer_kind == kind for candidate in range(start_index, index))


def _is_timer_reset(frame_infos: list[OcrFrameInfo], index: int) -> bool:
    current_number = frame_infos[index].timer_number
    if current_number is None:
        return False

    previous_number, previous_index = _find_previous_timer_number(frame_infos, index)
    if previous_number is not None and previous_index is not None:
        if current_number >= previous_number + 6:
            return True
        if previous_number <= 4 and current_number >= 8:
            return True
        return False

    return current_number >= 20 and _has_recent_timer_kind(frame_infos, index, "rest")


def _has_kind_flip(frame_infos: list[OcrFrameInfo], index: int) -> bool:
    current_kind = frame_infos[index].timer_kind
    if current_kind is None:
        return False
    for candidate in range(index - 1, max(-1, index - 3), -1):
        previous_kind = frame_infos[candidate].timer_kind
        if previous_kind is None:
            continue
        return previous_kind != current_kind
    return False


def _build_countdown_spans(
    frame_infos: list[OcrFrameInfo],
    max_signal_gap: int = 3,
) -> list[CountdownSegmentSpan]:
    spans: list[CountdownSegmentSpan] = []
    current_start: int | None = None
    last_signal_index: int | None = None

    def flush_span(end_index: int) -> None:
        nonlocal current_start, last_signal_index
        if current_start is None or end_index <= current_start:
            current_start = None
            last_signal_index = None
            return

        chunk = frame_infos[current_start:end_index]
        if not chunk:
            current_start = None
            last_signal_index = None
            return

        recognized_numbers = sum(1 for info in chunk if info.timer_number is not None)
        known_numbers = [info.timer_number for info in chunk if info.timer_number is not None]
        duration_sec = chunk[-1].time_sec + 1.0 - chunk[0].time_sec
        if recognized_numbers < 2 and duration_sec < 3:
            current_start = None
            last_signal_index = None
            return

        spans.append(
            CountdownSegmentSpan(
                start_index=current_start,
                end_index=end_index,
                start_sec=chunk[0].time_sec,
                end_sec=chunk[-1].time_sec + 1.0,
                duration_sec=duration_sec,
                rest_hits=sum(1 for info in chunk if info.timer_kind == "rest"),
                prep_hits=sum(1 for info in chunk if info.timer_kind == "prep"),
                action_hits=sum(1 for info in chunk if info.timer_kind == "action"),
                max_number=max(known_numbers) if known_numbers else None,
                min_number=min(known_numbers) if known_numbers else None,
            )
        )
        current_start = None
        last_signal_index = None

    for index, info in enumerate(frame_infos):
        if not _has_timer_signal(info):
            if (
                current_start is not None
                and last_signal_index is not None
                and index - last_signal_index > max_signal_gap
            ):
                flush_span(last_signal_index + 1)
            continue

        if current_start is None:
            current_start = index
            last_signal_index = index
            continue

        if _is_timer_reset(frame_infos, index) or _has_kind_flip(frame_infos, index):
            flush_span(index)
            current_start = index

        last_signal_index = index

    if current_start is not None:
        flush_span((last_signal_index or current_start) + 1)

    return spans


def _opposite_kind(kind: str) -> str:
    return "rest" if kind == "action" else "action"


def _build_alternating_assignment(
    spans: list[CountdownSegmentSpan],
    anchor_index: int,
    anchor_kind: str,
) -> list[str]:
    assignment: list[str] = []
    for index in range(len(spans)):
        if abs(index - anchor_index) % 2 == 0:
            assignment.append(anchor_kind)
        else:
            assignment.append(_opposite_kind(anchor_kind))
    return assignment


def _score_kind_assignment(
    spans: list[CountdownSegmentSpan],
    assignment: list[str],
) -> float:
    score = 0.0
    for index, (span, kind) in enumerate(zip(spans, assignment)):
        if kind == "rest":
            score += span.rest_hits * 6.0
            score += span.prep_hits * 2.0 if span.max_number is not None and span.max_number <= 20 else 0.0
            score -= span.action_hits * 4.0
            if span.max_number is not None and span.max_number <= 20:
                score += 2.5
            if span.duration_sec <= 20:
                score += 1.5
            if span.duration_sec >= 28:
                score -= 1.0
            if index == 0 and span.max_number is not None and span.max_number >= 24 and span.duration_sec >= 18:
                score -= 0.5
        else:
            score += span.action_hits * 6.0
            score -= span.rest_hits * 4.0
            if span.max_number is not None and span.max_number >= 24:
                score += 2.5
            if span.prep_hits > 0 and span.max_number is not None and span.max_number >= 24:
                score += 1.0
            if span.duration_sec >= 18:
                score += 1.5
            if span.max_number is not None and span.max_number <= 15:
                score -= 2.0
            if index == 0 and span.max_number is not None and span.max_number >= 24 and span.duration_sec >= 18:
                score += 0.5
    return score


def _resolve_countdown_kinds(spans: list[CountdownSegmentSpan]) -> list[str]:
    if not spans:
        return []

    candidate_assignments = {
        tuple(_build_alternating_assignment(spans, 0, "rest")),
        tuple(_build_alternating_assignment(spans, 0, "action")),
    }
    for index, span in enumerate(spans):
        if span.rest_hits > 0:
            candidate_assignments.add(tuple(_build_alternating_assignment(spans, index, "rest")))
        if span.action_hits > 0:
            candidate_assignments.add(tuple(_build_alternating_assignment(spans, index, "action")))
        if span.prep_hits > 0 and span.max_number is not None and span.max_number <= 20:
            candidate_assignments.add(tuple(_build_alternating_assignment(spans, index, "rest")))

    best_assignment = max(
        candidate_assignments,
        key=lambda assignment: _score_kind_assignment(spans, list(assignment)),
    )
    return list(best_assignment)


def _classify_countdown_span(span: CountdownSegmentSpan) -> str:
    if span.rest_hits > 0:
        return "rest"
    if span.prep_hits > 0 and (span.max_number is None or span.max_number <= 20):
        return "rest"
    if span.max_number is not None:
        if span.max_number >= 20:
            return "action"
        if span.max_number >= 18 and span.duration_sec >= 18:
            return "action"
        if span.max_number <= 15:
            return "rest"
    return "rest" if span.duration_sec <= 12 else "action"


def _is_strong_action_span(span: CountdownSegmentSpan) -> bool:
    if span.max_number is None:
        return False
    if span.rest_hits > 0:
        return False
    if span.max_number < 24 or span.duration_sec < 24:
        return False
    return (span.action_hits + span.prep_hits) >= 8


def _is_strong_rest_span(span: CountdownSegmentSpan) -> bool:
    if span.max_number is None:
        return False
    if span.max_number > 15 or span.duration_sec > 18:
        return False
    return span.rest_hits >= 2


def _source_priority(source: PreviewNameSource | None) -> int:
    priorities: dict[PreviewNameSource, int] = {
        "visual": 0,
        "rest_hint": 1,
        "candidate": 2,
        "explicit": 3,
    }
    if source is None:
        return -1
    return priorities[source]


def _merge_name_vote(
    votes: dict[str, float],
    sources: dict[str, PreviewNameSource],
    source_scores: dict[str, float],
    name: str | None,
    score: float,
    source: PreviewNameSource | None,
) -> None:
    if not _is_plausible_preview_name(name) or source is None:
        return
    votes[name] = votes.get(name, 0.0) + score
    previous_score = source_scores.get(name, float("-inf"))
    previous_source = sources.get(name)
    if score > previous_score or (
        score == previous_score and _source_priority(source) > _source_priority(previous_source)
    ):
        sources[name] = source
        source_scores[name] = score


def _best_scored_name(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda item: (item[1], len(item[0])))[0]


def _best_scored_name_choice(
    scores: dict[str, float],
    sources: dict[str, PreviewNameSource],
) -> tuple[str | None, PreviewNameSource | None]:
    name = _best_scored_name(scores)
    if name is None:
        return None, None
    return name, sources.get(name)


def _step_name_source_from_preview(source: PreviewNameSource | None) -> str | None:
    if source is None:
        return None
    return {
        "explicit": "timer_explicit",
        "candidate": "timer_candidate",
        "rest_hint": "timer_rest_hint",
        "visual": "timer_visual",
    }[source]


def _extend_timeline_segment(
    previous: TimerTimelineSegment,
    segment: TimerTimelineSegment,
) -> None:
    previous.end_index = max(previous.end_index, segment.end_index)
    previous.end_sec = max(previous.end_sec, segment.end_sec)
    previous.duration_sec = previous.end_sec - previous.start_sec


def _build_timeline_segments(
    spans: list[CountdownSegmentSpan],
) -> list[TimerTimelineSegment]:
    timeline_segments: list[TimerTimelineSegment] = []
    resolved_kinds = _resolve_countdown_kinds(spans)
    local_kinds = [_classify_countdown_span(span) for span in spans]
    for index, span in enumerate(spans):
        if _is_strong_rest_span(span):
            resolved_kinds[index] = "rest"
        elif _is_strong_action_span(span):
            resolved_kinds[index] = "action"
    for index in range(len(spans) - 1):
        gap_duration = spans[index + 1].start_sec - spans[index].end_sec
        if gap_duration >= 4 and local_kinds[index] == local_kinds[index + 1] == "action":
            resolved_kinds[index] = "action"
            resolved_kinds[index + 1] = "action"
    for index, (span, kind) in enumerate(zip(spans, resolved_kinds)):
        timeline_segments.append(
            TimerTimelineSegment(
                kind=kind,
                start_index=span.start_index,
                end_index=span.end_index,
                start_sec=span.start_sec,
                end_sec=span.end_sec,
                duration_sec=span.duration_sec,
            )
        )
        if index + 1 >= len(spans):
            continue
        next_span = spans[index + 1]
        gap_duration = next_span.start_sec - span.end_sec
        if gap_duration < 4:
            continue
        timeline_segments.append(
            TimerTimelineSegment(
                kind="rest",
                start_index=span.end_index,
                end_index=next_span.start_index,
                start_sec=span.end_sec,
                end_sec=next_span.start_sec,
                duration_sec=gap_duration,
            )
        )
    return timeline_segments


def _merge_timeline_segments(
    segments: list[TimerTimelineSegment],
) -> list[TimerTimelineSegment]:
    merged: list[TimerTimelineSegment] = []
    for segment in segments:
        if segment.duration_sec <= 0:
            continue
        if merged:
            previous = merged[-1]
            contiguous = segment.start_sec - previous.end_sec <= 2.0
            if previous.kind == segment.kind and contiguous:
                if (
                    previous.kind == "action"
                    and min(previous.duration_sec, segment.duration_sec) >= LONG_ACTION_SEGMENT_SEC
                ):
                    merged.append(segment)
                    continue
                _extend_timeline_segment(previous, segment)
                continue
            if (
                previous.kind == segment.kind == "action"
                and segment.start_sec - previous.end_sec <= 3.0
                and min(previous.duration_sec, segment.duration_sec) < LONG_ACTION_SEGMENT_SEC
            ):
                _extend_timeline_segment(previous, segment)
                continue
        merged.append(segment)
    return merged


def _assign_timeline_preview_names(
    frame_infos: list[OcrFrameInfo],
    segments: list[TimerTimelineSegment],
) -> None:
    for index, segment in enumerate(segments):
        local_name, local_source = _aggregate_preview_name_with_source(
            frame_infos,
            segment.start_index,
            segment.end_index,
        )
        if segment.kind == "rest":
            segment.preview_name = local_name
            segment.preview_name_source = local_source
            continue

        votes: dict[str, float] = {}
        vote_sources: dict[str, PreviewNameSource] = {}
        source_scores: dict[str, float] = {}
        _merge_name_vote(
            votes,
            vote_sources,
            source_scores,
            local_name,
            LOCAL_NAME_VOTE,
            local_source,
        )
        boundary_name, boundary_source = _aggregate_preview_name_with_source(
            frame_infos,
            max(0, segment.start_index - 6),
            min(len(frame_infos), segment.start_index + 3),
        )
        _merge_name_vote(
            votes,
            vote_sources,
            source_scores,
            boundary_name,
            BOUNDARY_NAME_VOTE,
            boundary_source,
        )
        if index > 0 and segments[index - 1].kind == "rest":
            previous_segment = segments[index - 1]
            hint_source = (
                "explicit"
                if previous_segment.preview_name_source == "explicit"
                else "rest_hint"
                if previous_segment.preview_name_source is not None
                else None
            )
            _merge_name_vote(
                votes,
                vote_sources,
                source_scores,
                previous_segment.preview_name,
                REST_HINT_NAME_VOTE,
                hint_source,
            )
        segment.preview_name, segment.preview_name_source = _best_scored_name_choice(
            votes,
            vote_sources,
        )


def _is_plausible_preview_name(name: str | None) -> bool:
    if name is None:
        return False
    cleaned = _clean_text(name)
    if not cleaned:
        return False
    if cleaned.lower() in ACTION_NAME_BLACKLIST:
        return False
    if re.search(r"[\u4e00-\u9fff]", cleaned) and len(cleaned) < CJK_ACTION_MIN_CHARS:
        return False
    if re.fullmatch(r"[A-Za-z]", cleaned):
        return False
    if len(cleaned.replace(" ", "")) <= 2 and re.search(r"[A-Za-z]", cleaned):
        return False
    return "meo" not in cleaned.lower()


def _aggregate_preview_name(
    frame_infos: list[OcrFrameInfo],
    start_index: int,
    end_index: int,
) -> str | None:
    name, _source = _aggregate_preview_name_with_source(frame_infos, start_index, end_index)
    return name


def _aggregate_preview_name_with_source(
    frame_infos: list[OcrFrameInfo],
    start_index: int,
    end_index: int,
) -> tuple[str | None, PreviewNameSource | None]:
    explicit_scores: dict[str, float] = {}
    candidate_scores: dict[str, float] = {}
    for info in frame_infos[start_index:end_index]:
        explicit_name = info.explicit_preview_name
        if _is_plausible_preview_name(explicit_name):
            explicit_scores[explicit_name] = explicit_scores.get(explicit_name, 0.0) + REST_HINT_NAME_VOTE
        preview_name = info.preview_name
        if _is_plausible_preview_name(preview_name):
            candidate_scores[preview_name] = candidate_scores.get(preview_name, 0.0) + LOCAL_NAME_VOTE

    explicit_name = _best_scored_name(explicit_scores)
    if explicit_name is not None:
        return explicit_name, "explicit"
    candidate_name = _best_scored_name(candidate_scores)
    if candidate_name is not None:
        return candidate_name, "candidate"
    return None, None


def _sample_segment_indexes(segment: TimerTimelineSegment) -> list[int]:
    span = max(1, segment.end_index - segment.start_index)
    if span <= 3:
        offsets = {0, span // 2, span - 1}
    else:
        offsets = {span // 4, span // 2, (span * 3) // 4}
    return [segment.start_index + offset for offset in sorted(offsets)]


def _scene_signature(frame_path: Path) -> list[float]:
    image = Image.open(frame_path).convert("L")
    width, height = image.size
    box = (
        int(width * SCENE_SIGNATURE_BOX[0]),
        int(height * SCENE_SIGNATURE_BOX[1]),
        int(width * SCENE_SIGNATURE_BOX[2]),
        int(height * SCENE_SIGNATURE_BOX[3]),
    )
    crop = ImageOps.autocontrast(image.crop(box).resize(SCENE_SIGNATURE_SIZE))
    values = list(crop.tobytes())
    mean_value = sum(values) / max(1, len(values))
    return [(value - mean_value) / 255.0 for value in values]


def _signature_distance(left: list[float], right: list[float]) -> float:
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) / max(
        1, len(left)
    )


def _pick_visual_fallback_name(ranked_matches: list[tuple[float, str]]) -> str | None:
    if not ranked_matches:
        return None
    ranked_matches.sort()
    best_distance, best_name = ranked_matches[0]
    second_distance = ranked_matches[1][0] if len(ranked_matches) > 1 else 1.0
    if best_distance <= VISUAL_FALLBACK_STRONG_MATCH or (
        best_distance <= VISUAL_FALLBACK_GOOD_MATCH
        and second_distance - best_distance >= VISUAL_FALLBACK_MARGIN
    ):
        return best_name
    return None


def _assign_visual_fallback_names(
    frame_infos: list[OcrFrameInfo],
    segments: list[TimerTimelineSegment],
) -> None:
    signature_cache: dict[Path, list[float]] = {}
    references: dict[str, list[list[float]]] = {}

    def signature_for(index: int) -> list[float] | None:
        if not (0 <= index < len(frame_infos)):
            return None
        frame_path = frame_infos[index].frame_path
        if frame_path is None or not frame_path.exists():
            return None
        if frame_path not in signature_cache:
            signature_cache[frame_path] = _scene_signature(frame_path)
        return signature_cache[frame_path]

    for segment in segments:
        if (
            segment.kind != "action"
            or not _is_plausible_preview_name(segment.preview_name)
            or segment.preview_name_source not in TRUSTED_VISUAL_REFERENCE_SOURCES
        ):
            continue
        references.setdefault(segment.preview_name, [])
        for index in _sample_segment_indexes(segment):
            signature = signature_for(index)
            if signature is not None:
                references[segment.preview_name].append(signature)

    if not references:
        return

    for segment in segments:
        if segment.kind != "action" or _is_plausible_preview_name(segment.preview_name):
            continue
        target_signatures = [
            signature
            for signature in (signature_for(index) for index in _sample_segment_indexes(segment))
            if signature is not None
        ]
        if not target_signatures:
            continue

        ranked_matches: list[tuple[float, str]] = []
        for name, reference_signatures in references.items():
            if not reference_signatures:
                continue
            best_distance = min(
                _signature_distance(target_signature, reference_signature)
                for target_signature in target_signatures
                for reference_signature in reference_signatures
            )
            ranked_matches.append((best_distance, name))

        if not ranked_matches:
            continue

        best_name = _pick_visual_fallback_name(ranked_matches)
        if best_name is not None:
            segment.preview_name = best_name
            segment.preview_name_source = "visual"


def _build_timer_segments_with_rapidocr(
    sample_frames: list[tuple[float, Path]],
    workdir: Path,
    style_hint: str,
    timer_box_id: str,
    preview_box_id: str | None,
    preview_box_ratios: tuple[float, float, float, float] | None = None,
) -> list[TimerSegment]:
    frame_infos = _build_ocr_frame_infos(
        sample_frames,
        workdir,
        style_hint,
        timer_box_id,
        preview_box_id,
        preview_box_ratios,
    )
    if not frame_infos:
        return []

    spans = _build_countdown_spans(frame_infos)
    if not spans:
        return []
    timeline_segments = _merge_timeline_segments(_build_timeline_segments(spans))
    _assign_timeline_preview_names(frame_infos, timeline_segments)
    _assign_visual_fallback_names(frame_infos, timeline_segments)

    segments: list[TimerSegment] = []
    for segment in timeline_segments:
        if segment.kind == "action" and segment.duration_sec < 8:
            continue
        if segment.kind == "rest" and segment.duration_sec < 2:
            continue
        segments.append(
            TimerSegment(
                kind=segment.kind,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                duration_sec=segment.duration_sec,
                name=segment.preview_name,
                name_source=_step_name_source_from_preview(segment.preview_name_source),
            )
        )

    while segments and segments[-1].kind == "rest" and segments[-1].duration_sec > 10:
        segments.pop()
    return segments


def _ocr_slot_label(frame_path: Path, slot_index: int, slot_count: int) -> str | None:
    image = Image.open(frame_path)
    width, height = image.size
    strip = image.crop((0, int(height * 0.91), width, height))
    x0 = int(slot_index * strip.width / slot_count) + 2
    x1 = int((slot_index + 1) * strip.width / slot_count) - 2
    crop = strip.crop((x0, int(strip.height * 0.25), x1, strip.height))
    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray).resize((gray.width * 10, gray.height * 10))
    gray = gray.point(lambda pixel: 255 if pixel > 120 else 0, mode="1").convert("L")

    temp_path = frame_path.with_name(f"{frame_path.stem}_slot_{slot_index}.png")
    gray.save(temp_path)
    best_candidate: tuple[str, float] | None = None
    try:
        for psm in ("7", "8", "13"):
            completed = subprocess.run(
                [
                    "tesseract",
                    str(temp_path),
                    "stdout",
                    "--psm",
                    psm,
                    "-c",
                    "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            text = " ".join(completed.stdout.split())
            compact = re.sub(r"[^A-Za-z ]+", " ", text).upper()
            compact = re.sub(r"\s+", " ", compact).strip()
            if not compact:
                continue
            if compact.replace(" ", "") in ALIAS_LABELS:
                label = ALIAS_LABELS[compact.replace(" ", "")]
                score = 1.5
            else:
                match = _best_canonical_exercise(compact)
                if match is None:
                    continue
                label, score = match
            if best_candidate is None or score > best_candidate[1]:
                best_candidate = (label, score)
    finally:
        temp_path.unlink(missing_ok=True)
    return best_candidate[0] if best_candidate is not None else None


def _derive_cycle_labels(
    cycle_frames: list[tuple[float, Path]],
    slot_count: int,
) -> list[str]:
    labels_by_slot: list[list[str]] = [[] for _ in range(slot_count)]
    for _, frame_path in cycle_frames:
        for slot_index in range(slot_count):
            label = _ocr_slot_label(frame_path, slot_index, slot_count)
            if label is not None:
                labels_by_slot[slot_index].append(label)
        empty_slots = {
            slot_index for slot_index, labels in enumerate(labels_by_slot) if not labels
        }
        if empty_slots:
            for center, label in _ocr_strip_phrases(frame_path):
                slot_index = min(slot_count - 1, max(0, int(center * slot_count)))
                if slot_index in empty_slots:
                    labels_by_slot[slot_index].append(label)

    ordered_labels: list[str] = []
    for slot_index, labels in enumerate(labels_by_slot):
        if labels:
            ordered_labels.append(Counter(labels).most_common(1)[0][0])
        else:
            ordered_labels.append(f"Exercise {slot_index + 1}")
    if (
        ordered_labels
        and ordered_labels[0] == "Crunches"
        and "Jumping Jacks" in ordered_labels
        and "Split Squats" in ordered_labels
    ):
        ordered_labels[0] = "Bicycle Crunches"
    return ordered_labels


def _steps_from_round(
    round_observations: list[SampleObservation],
    labels: list[str],
    total_duration_sec: float,
    ratio_threshold: float = 0.05,
) -> list[WorkoutStep]:
    slot_count = len(labels)
    preview_starts: list[tuple[int, int]] = []
    search_from = 0
    min_preview_gap = 18
    for slot_index in range(slot_count):
        if preview_starts:
            search_from = max(search_from, preview_starts[-1][1] + min_preview_gap)
        for index in range(search_from, len(round_observations)):
            window = round_observations[index : index + 3]
            if len(window) < 2:
                continue
            hits = sum(1 for item in window if item.active_index == slot_index)
            if hits >= 2:
                preview_starts.append((slot_index, index))
                search_from = index + 1
                break

    preview_starts.sort(key=lambda item: round_observations[item[1]].time_sec)
    actual_starts: list[tuple[int, int]] = []
    def stable_timer(index: int, color: str) -> bool:
        window = round_observations[index : index + 3]
        return len(window) >= 2 and sum(1 for item in window if item.timer_state == color) >= 2

    def stable_slot(index: int, slot: int, threshold: float, require_active_index: bool = False) -> bool:
        window = round_observations[index : index + 3]
        if require_active_index:
            return len(window) >= 2 and sum(1 for item in window if item.active_index == slot) >= 2
        return len(window) >= 2 and sum(
            1
            for item in window
            if item.active_index == slot or item.ratios[slot] > threshold
        ) >= 2

    for order_index, (slot_index, preview_index) in enumerate(preview_starts):
        next_preview_index = (
            preview_starts[order_index + 1][1]
            if order_index + 1 < len(preview_starts)
            else len(round_observations) - 1
        )
        search_start = preview_index
        actual_index = None
        if slot_index == 0:
            for index in range(search_start, next_preview_index + 1):
                if stable_timer(index, "red") and stable_slot(
                    index, slot_index, ratio_threshold, require_active_index=True
                ):
                    actual_index = index
                    break
        else:
            green_index = None
            green_search_end = min(next_preview_index, search_start + 12)
            for index in range(search_start, green_search_end + 1):
                if stable_timer(index, "green") and stable_slot(
                    index, slot_index, 0.02, require_active_index=True
                ):
                    green_index = index
                    break
            red_search_start = green_index + 1 if green_index is not None else search_start
            for index in range(red_search_start, next_preview_index + 1):
                if stable_timer(index, "red") and stable_slot(
                    index, slot_index, ratio_threshold, require_active_index=True
                ):
                    actual_index = index
                    break
        if actual_index is None:
            actual_index = search_start
        actual_starts.append((slot_index, actual_index))

    actual_starts.sort(key=lambda item: round_observations[item[1]].time_sec)
    steps: list[WorkoutStep] = []
    for order_index, (slot_index, start_index) in enumerate(actual_starts):
        start_sec = round_observations[start_index].time_sec
        next_start_sec = (
            round_observations[actual_starts[order_index + 1][1]].time_sec
            if order_index + 1 < len(actual_starts)
            else min(total_duration_sec, round_observations[-1].time_sec + 1.0)
        )
        if next_start_sec <= start_sec:
            continue
        steps.append(
            WorkoutStep(
                index=len(steps) + 1,
                name=labels[slot_index],
                start_sec=start_sec,
                end_sec=next_start_sec,
                duration_sec=next_start_sec - start_sec,
                name_source="vision_overlay",
            )
        )
    return steps


def build_visual_workout_summary(
    video_path: Path,
    workdir: Path,
    title: str,
    source_url: str,
    total_duration_sec: float,
    language: str,
    sample_interval_sec: int = 1,
    overlay_slot_count: int = 6,
) -> WorkoutSummary:
    timeline_frames = _extract_sample_frames(
        video_path=video_path,
        frames_dir=workdir / "vision_timeline",
        sample_interval_sec=sample_interval_sec,
        total_duration_sec=total_duration_sec,
    )
    observations = _collect_observations(timeline_frames, overlay_slot_count)
    round_ranges = _detect_round_ranges(observations)
    steps: list[WorkoutStep] = []
    for round_index, (start_index, end_index) in enumerate(round_ranges):
        round_observations = observations[start_index : end_index + 1]
        if not round_observations:
            continue
        cycle_start = round_observations[0].time_sec
        cycle_end = round_observations[-1].time_sec + 1.0
        label_frames = [
            item
            for item in timeline_frames
            if cycle_start <= item[0] <= cycle_end and int(item[0] - cycle_start) % 15 == 0
        ]
        if not label_frames:
            label_frames = [(round_observations[0].time_sec, timeline_frames[start_index][1])]
        labels = _derive_cycle_labels(label_frames, overlay_slot_count)
        round_steps = _steps_from_round(round_observations, labels, total_duration_sec)
        for step in round_steps:
            step.index = len(steps) + 1
            steps.append(step)
        if round_index + 1 < len(round_ranges):
            next_start = observations[round_ranges[round_index + 1][0]].time_sec
            rest_duration = next_start - cycle_end
            if rest_duration >= 5:
                steps.append(
                    WorkoutStep(
                        index=len(steps) + 1,
                        name="Rest",
                        start_sec=cycle_end,
                        end_sec=next_start,
                        duration_sec=rest_duration,
                        name_source="vision_overlay_gap",
                    )
                )
    if len(steps) < 3:
        raise VisionExtractionError("Vision fallback produced too few workout steps.")
    return WorkoutSummary(
        title=title,
        source_url=source_url,
        language=language,
        total_duration_sec=total_duration_sec,
        transcript_source="vision_overlay",
        steps=steps,
    )


def build_timer_workout_summary(
    video_path: Path,
    workdir: Path,
    title: str,
    source_url: str,
    total_duration_sec: float,
    language: str,
    sample_interval_sec: int = 1,
    style_hint: str = "auto",
    timer_box_id: str = "top_right",
    preview_box_id: str | None = None,
    preview_box_ratios: tuple[float, float, float, float] | None = None,
) -> WorkoutSummary:
    sample_frames = _extract_sample_frames(
        video_path=video_path,
        frames_dir=workdir / "timer_timeline",
        sample_interval_sec=sample_interval_sec,
        total_duration_sec=total_duration_sec,
    )
    timer_segments = _build_timer_segments_with_rapidocr(
        sample_frames,
        workdir,
        style_hint,
        timer_box_id,
        preview_box_id,
        preview_box_ratios,
    )
    if not timer_segments:
        timer_segments = _build_timer_segments(sample_frames)
    if not timer_segments:
        raise VisionExtractionError("Timer fallback could not detect stable action/rest segments.")

    steps: list[WorkoutStep] = []
    action_index = 0
    pending_action_name: str | None = None
    pending_action_source: str | None = None
    for segment in timer_segments:
        if segment.kind == "action":
            action_index += 1
            chosen_name = segment.name or pending_action_name or f"Exercise {action_index:02d}"
            chosen_source = segment.name_source or pending_action_source or "timer_fallback"
            steps.append(
                WorkoutStep(
                    index=len(steps) + 1,
                    name=chosen_name,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    duration_sec=segment.duration_sec,
                    name_source=chosen_source,
                )
            )
            pending_action_name = None
            pending_action_source = None
        else:
            if segment.name:
                pending_action_name = segment.name
                pending_action_source = segment.name_source
            steps.append(
                WorkoutStep(
                    index=len(steps) + 1,
                    name="Rest",
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    duration_sec=segment.duration_sec,
                    name_source="timer_rest",
                )
            )

    if action_index == 0:
        raise VisionExtractionError("Timer fallback did not find any action segments.")

    return WorkoutSummary(
        title=title,
        source_url=source_url,
        language=language,
        total_duration_sec=total_duration_sec,
        transcript_source="vision_timer_next_card" if style_hint == "timer_next_card" else "vision_timer",
        steps=steps,
    )


def build_general_visual_workout_summary(
    video_path: Path,
    workdir: Path,
    title: str,
    source_url: str,
    total_duration_sec: float,
    language: str,
    sample_interval_sec: int = 1,
) -> WorkoutSummary:
    profile_duration_sec = min(total_duration_sec, 180.0)
    profile_frames = _extract_sample_frames(
        video_path=video_path,
        frames_dir=workdir / "vision_profile",
        sample_interval_sec=2,
        total_duration_sec=profile_duration_sec,
    )
    profile = _detect_visual_style(profile_frames)

    if profile.timer_box_id is None and profile.style == "overlay_bar":
        try:
            return build_visual_workout_summary(
                video_path=video_path,
                workdir=workdir,
                title=title,
                source_url=source_url,
                total_duration_sec=total_duration_sec,
                language=language,
                sample_interval_sec=sample_interval_sec,
            )
        except VisionExtractionError:
            if profile.timer_hits > 0 or profile.next_card_hits > 0:
                return build_timer_workout_summary(
                    video_path=video_path,
                    workdir=workdir,
                    title=title,
                    source_url=source_url,
                    total_duration_sec=total_duration_sec,
                    language=language,
                    sample_interval_sec=sample_interval_sec,
                    style_hint="timer_only",
                    timer_box_id=profile.timer_box_id or "top_right",
                    preview_box_id=profile.preview_box_id,
                    preview_box_ratios=profile.preview_box_ratios,
                )
            raise

    return build_timer_workout_summary(
        video_path=video_path,
        workdir=workdir,
        title=title,
        source_url=source_url,
        total_duration_sec=total_duration_sec,
        language=language,
        sample_interval_sec=sample_interval_sec,
        style_hint=profile.style,
        timer_box_id=profile.timer_box_id or "top_right",
        preview_box_id=profile.preview_box_id,
        preview_box_ratios=profile.preview_box_ratios,
    )
