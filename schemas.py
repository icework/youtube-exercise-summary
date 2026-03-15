from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


@dataclass(slots=True)
class TranscriptSegment:
    start_sec: float
    end_sec: float
    text: str
    source: str = "subtitle"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start_sec"] = _round_float(self.start_sec)
        data["end_sec"] = _round_float(self.end_sec)
        return data


@dataclass(slots=True)
class Transcript:
    language: str
    source: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(slots=True)
class WorkoutStep:
    index: int
    name: str
    start_sec: float
    end_sec: float
    duration_sec: float | None = None
    reps: int | None = None
    sets: int | None = None
    notes: str | None = None
    name_source: str | None = None
    clip_start_sec: float | None = None
    clip_duration_sec: float | None = None
    clip_path: str | None = None
    screenshot_time_sec: float | None = None
    screenshot_path: str | None = None

    def finalize(self) -> None:
        if self.duration_sec is None and self.end_sec > self.start_sec:
            self.duration_sec = self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        data = asdict(self)
        data["start_sec"] = _round_float(self.start_sec)
        data["end_sec"] = _round_float(self.end_sec)
        data["duration_sec"] = _round_float(self.duration_sec)
        data["clip_start_sec"] = _round_float(self.clip_start_sec)
        data["clip_duration_sec"] = _round_float(self.clip_duration_sec)
        data["screenshot_time_sec"] = _round_float(self.screenshot_time_sec)
        return data


@dataclass(slots=True)
class WorkoutSummary:
    title: str
    source_url: str
    language: str
    total_duration_sec: float
    transcript_source: str
    steps: list[WorkoutStep] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def finalize(self) -> None:
        for step in self.steps:
            step.finalize()

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "title": self.title,
            "source_url": self.source_url,
            "language": self.language,
            "total_duration_sec": _round_float(self.total_duration_sec) or 0.0,
            "transcript_source": self.transcript_source,
            "generated_at": self.generated_at,
            "steps": [step.to_dict() for step in self.steps],
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
