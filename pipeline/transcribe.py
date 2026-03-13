from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from schemas import Transcript, TranscriptSegment, write_json


class TranscriptError(RuntimeError):
    pass


TIMECODE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})"
)


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_vtt(vtt_text: str, source: str, language: str) -> Transcript:
    segments: list[TranscriptSegment] = []
    lines = vtt_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line == "WEBVTT" or line.startswith("NOTE"):
            index += 1
            continue
        time_match = TIMECODE_RE.match(line)
        if time_match is None:
            index += 1
            continue
        start_sec = _parse_timestamp(time_match.group("start"))
        end_sec = _parse_timestamp(time_match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        text = " ".join(text_lines).strip()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        if text:
            if segments and segments[-1].text == text and abs(segments[-1].end_sec - start_sec) < 0.2:
                segments[-1].end_sec = end_sec
            else:
                segments.append(
                    TranscriptSegment(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        text=text,
                        source=source,
                    )
                )
    if not segments:
        raise TranscriptError("Subtitle file did not contain any parseable transcript segments.")
    return Transcript(language=language, source=source, segments=segments)


def load_transcript_from_subtitle(subtitle_path: Path, language: str, source: str) -> Transcript:
    vtt_text = subtitle_path.read_text(encoding="utf-8", errors="replace")
    transcript = parse_vtt(vtt_text, source=source, language=language)
    write_json(subtitle_path.with_suffix(".transcript.json"), transcript.to_dict())
    return transcript


def _run_ffmpeg_extract_audio(video_path: Path, audio_path: Path) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise TranscriptError(f"Audio extraction failed.\n{stderr}")


def _extract_audio(video_path: Path, workdir: Path) -> Path:
    audio_path = workdir / "audio.wav"
    _run_ffmpeg_extract_audio(video_path, audio_path)
    return audio_path


def _transcript_from_segments(
    raw_segments: list[dict],
    language: str,
    source: str,
) -> Transcript:
    segments = [
        TranscriptSegment(
            start_sec=float(segment.get("start", 0.0)),
            end_sec=float(segment.get("end", 0.0)),
            text=str(segment.get("text", "")).strip(),
            source=source,
        )
        for segment in raw_segments
        if str(segment.get("text", "")).strip()
    ]
    if not segments:
        raise TranscriptError(f"{source} transcription returned no usable segments.")
    return Transcript(language=language or "unknown", source=source, segments=segments)


def _transcribe_with_local_whisper(audio_path: Path, workdir: Path, language: str) -> Transcript:
    model_name = os.getenv("WHISPER_MODEL", "base")
    args = [
        sys.executable,
        "-m",
        "whisper",
        str(audio_path),
        "--output_dir",
        str(workdir),
        "--output_format",
        "json",
        "--model",
        model_name,
        "--verbose",
        "False",
    ]
    if language != "auto":
        args.extend(["--language", language])
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise TranscriptError(f"Local Whisper transcription failed.\n{stderr}")
    output_json = workdir / f"{audio_path.stem}.json"
    if not output_json.exists():
        raise TranscriptError("Local Whisper reported success but no JSON output was produced.")
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    transcript = _transcript_from_segments(
        payload.get("segments") or [],
        language=payload.get("language") or language,
        source="whisper",
    )
    write_json(workdir / "transcript.whisper.json", transcript.to_dict())
    return transcript


def _encode_multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"codex-{uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_path.name}"\r\n'
        ).encode()
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), boundary


def _post_json(url: str, headers: dict[str, str], payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise TranscriptError(f"API request failed: {exc.code} {message}") from exc


def _transcribe_with_api(audio_path: Path, language: str, workdir: Path) -> Transcript:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise TranscriptError("OPENAI_API_KEY is not set for API transcription fallback.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    fields = {
        "model": model,
        "response_format": "verbose_json",
    }
    if language != "auto":
        fields["language"] = language
    payload, boundary = _encode_multipart(fields, "file", audio_path)
    request = urllib.request.Request(
        f"{base_url}/audio/transcriptions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise TranscriptError(f"API transcription failed: {exc.code} {message}") from exc

    transcript = _transcript_from_segments(
        result.get("segments") or [],
        language=result.get("language") or language,
        source="api",
    )
    write_json(workdir / "transcript.api.json", transcript.to_dict())
    return transcript


def call_llm_json(prompt: str, schema_hint: str) -> list[dict] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-4.1-mini")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract structured workout steps from transcripts. "
                    "Return JSON only. No markdown."
                ),
            },
            {"role": "user", "content": f"{prompt}\n\nSchema:\n{schema_hint}"},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    result = _post_json(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        payload=payload,
    )
    content = result["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"LLM extraction returned invalid JSON: {content}") from exc
    items = parsed.get("steps")
    return items if isinstance(items, list) else None


def transcribe_video(video_path: Path, workdir: Path, language: str, backend: str) -> Transcript:
    audio_path = _extract_audio(video_path, workdir)
    backends = {
        "auto": ["whisper", "api"],
        "whisper": ["whisper"],
        "api": ["api"],
    }.get(backend)
    if backends is None:
        raise TranscriptError(f"Unsupported transcription backend: {backend}")

    errors: list[str] = []
    for backend_name in backends:
        try:
            if backend_name == "whisper":
                return _transcribe_with_local_whisper(audio_path, workdir, language)
            if backend_name == "api":
                return _transcribe_with_api(audio_path, language, workdir)
        except TranscriptError as exc:
            errors.append(f"{backend_name}: {exc}")
    raise TranscriptError("All transcription backends failed.\n" + "\n".join(errors))

