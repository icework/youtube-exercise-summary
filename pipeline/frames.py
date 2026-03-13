from __future__ import annotations

import subprocess
from pathlib import Path

from schemas import WorkoutStep


class FrameCaptureError(RuntimeError):
    pass


def _compute_screenshot_time(step: WorkoutStep) -> float:
    duration = max(0.0, step.end_sec - step.start_sec)
    if duration <= 0.0:
        return step.start_sec
    screenshot_time = step.start_sec + duration * 0.3
    if screenshot_time >= step.end_sec:
        screenshot_time = step.start_sec + duration * 0.5
    return min(max(step.start_sec, screenshot_time), step.end_sec)


def capture_step_frames(video_path: Path, steps: list[WorkoutStep], workdir: Path) -> list[WorkoutStep]:
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        screenshot_time = _compute_screenshot_time(step)
        output_path = frames_dir / f"step_{step.index:03d}.jpg"
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{screenshot_time:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise FrameCaptureError(
                f"Failed to capture screenshot for step {step.index} ({step.name}).\n{stderr}"
            )
        step.screenshot_time_sec = screenshot_time
        step.screenshot_path = str(output_path)
    return steps

