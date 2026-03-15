from __future__ import annotations

import subprocess
from pathlib import Path

from schemas import WorkoutStep


class FrameCaptureError(RuntimeError):
    pass


def _compute_anchor_time(step: WorkoutStep) -> float:
    duration = max(0.0, step.end_sec - step.start_sec)
    if duration <= 0.0:
        return step.start_sec
    anchor_time = step.start_sec + duration * 0.3
    if anchor_time >= step.end_sec:
        anchor_time = step.start_sec + duration * 0.5
    return min(max(step.start_sec, anchor_time), step.end_sec)


def _probe_video_duration_sec(video_path: Path) -> float | None:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        duration_sec = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration_sec if duration_sec > 0 else None


def _compute_clip_window(
    step: WorkoutStep,
    anchor_sec: float,
    video_duration_sec: float | None,
    clip_target_sec: float = 5.0,
) -> tuple[float, float]:
    step_start = max(0.0, float(step.start_sec))
    step_end = max(step_start, float(step.end_sec))
    if video_duration_sec is not None:
        step_end = min(step_end, video_duration_sec)

    available_sec = max(0.0, step_end - step_start)
    if available_sec <= 0.0:
        # Keep output stable even when a malformed step has no usable range.
        fallback_start = max(0.0, anchor_sec)
        if video_duration_sec is not None:
            fallback_start = min(fallback_start, max(0.0, video_duration_sec - 0.5))
            fallback_sec = max(0.5, min(clip_target_sec, video_duration_sec - fallback_start))
            return fallback_start, fallback_sec
        return fallback_start, clip_target_sec

    clip_duration_sec = min(clip_target_sec, available_sec)
    min_start_sec = step_start
    max_start_sec = step_end - clip_duration_sec
    clip_start_sec = anchor_sec - clip_duration_sec * 0.5
    clip_start_sec = min(max_start_sec, max(min_start_sec, clip_start_sec))
    return clip_start_sec, clip_duration_sec


def capture_step_clips(video_path: Path, steps: list[WorkoutStep], workdir: Path) -> list[WorkoutStep]:
    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_duration_sec = _probe_video_duration_sec(video_path)

    for step in steps:
        anchor_sec = _compute_anchor_time(step)
        clip_start_sec, clip_duration_sec = _compute_clip_window(
            step=step,
            anchor_sec=anchor_sec,
            video_duration_sec=video_duration_sec,
        )
        output_path = clips_dir / f"step_{step.index:03d}.gif"
        completed = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{clip_start_sec:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{clip_duration_sec:.3f}",
                "-an",
                "-filter_complex",
                (
                    "[0:v]fps=12,scale=480:-1:flags=lanczos:force_original_aspect_ratio=decrease,"
                    "split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
                    "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
                ),
                "-loop",
                "0",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not output_path.exists():
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise FrameCaptureError(
                f"Failed to capture clip for step {step.index} ({step.name}).\n{stderr}"
            )
        step.clip_start_sec = clip_start_sec
        step.clip_duration_sec = clip_duration_sec
        step.clip_path = str(output_path)
    return steps


def capture_step_frames(video_path: Path, steps: list[WorkoutStep], workdir: Path) -> list[WorkoutStep]:
    # Backward-compat alias: default media is now short video clips.
    return capture_step_clips(video_path, steps, workdir)
