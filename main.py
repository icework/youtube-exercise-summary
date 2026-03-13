from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pipeline import (
    DownloadError,
    FrameCaptureError,
    RenderError,
    VisionExtractionError,
    build_general_visual_workout_summary,
    build_workout_summary,
    capture_step_frames,
    download_best_subtitle,
    download_video,
    fetch_video_metadata,
    load_transcript_from_subtitle,
    render_summary_html,
    summary_needs_visual_fallback,
    transcribe_video,
)
from pipeline.transcribe import TranscriptError
from schemas import write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a single-file HTML workout summary from a YouTube video."
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--output",
        required=True,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--workdir",
        help="Optional working directory for intermediate artifacts",
    )
    parser.add_argument(
        "--transcribe-backend",
        choices=["auto", "whisper", "api"],
        default="auto",
        help="Fallback transcription backend when subtitles are unavailable",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Preferred subtitle/transcription language. Defaults to auto.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep intermediate files when using a temporary workdir.",
    )
    return parser.parse_args()


def _prepare_workdir(requested_workdir: str | None) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if requested_workdir:
        workdir = Path(requested_workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir, None
    temp_dir = tempfile.TemporaryDirectory(prefix="yt-workout-")
    return Path(temp_dir.name), temp_dir


def _claim_artifacts_destination(output_path: Path) -> Path:
    base_name = f"{output_path.stem}_artifacts"
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter + 1}"
        destination = output_path.parent / f"{base_name}{suffix}"
        try:
            destination.mkdir(parents=False, exist_ok=False)
            return destination
        except FileExistsError:
            counter += 1


def _rewrite_copied_summary_paths(destination: Path) -> None:
    summary_path = destination / "workout_summary.json"
    if not summary_path.exists():
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    frames_dir = destination / "frames"
    for step in payload.get("steps", []):
        screenshot_path = step.get("screenshot_path")
        if screenshot_path:
            frame_name = Path(str(screenshot_path)).name
            step["screenshot_path"] = str((frames_dir / frame_name).resolve())
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_kept_artifacts(temp_dir: tempfile.TemporaryDirectory[str], output_path: Path) -> Path:
    kept_dir = Path(temp_dir.name).resolve()
    destination = _claim_artifacts_destination(output_path)
    shutil.copytree(kept_dir, destination, dirs_exist_ok=True)
    _rewrite_copied_summary_paths(destination)
    return destination


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


def _load_transcript_with_fallback(
    *,
    url: str,
    metadata: dict,
    video_path: Path,
    workdir: Path,
    preferred_language: str,
    transcribe_backend: str,
):
    subtitle = download_best_subtitle(
        url,
        metadata=metadata,
        workdir=workdir,
        preferred_language=preferred_language,
    )
    if subtitle is not None:
        try:
            return load_transcript_from_subtitle(
                subtitle.path,
                language=subtitle.language,
                source=subtitle.source,
            )
        except TranscriptError as exc:
            print(f"Subtitle parsing failed. Trying transcription fallback: {exc}")

    print("No usable subtitles found. Starting transcription fallback...")
    try:
        return transcribe_video(
            video_path,
            workdir=workdir,
            language=preferred_language,
            backend=transcribe_backend,
        )
    except TranscriptError as exc:
        print(f"Transcription unavailable, falling back to video-only analysis: {exc}")
        return None


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workdir, temp_dir = _prepare_workdir(args.workdir)

    try:
        try:
            print("Fetching YouTube metadata...")
            metadata = fetch_video_metadata(args.url)
            title = metadata.get("title") or "Workout Summary"
            total_duration_sec = float(metadata.get("duration") or 0.0)

            print("Downloading video...")
            video_path = download_video(args.url, workdir)
            probed_duration_sec = _probe_video_duration_sec(video_path)
            if probed_duration_sec is not None:
                total_duration_sec = probed_duration_sec

            print("Downloading subtitles if available...")
            transcript = _load_transcript_with_fallback(
                url=args.url,
                metadata=metadata,
                video_path=video_path,
                workdir=workdir,
                preferred_language=args.language,
                transcribe_backend=args.transcribe_backend,
            )

            print("Extracting workout steps...")
            if transcript is not None:
                transcript_json_path = workdir / "transcript.final.json"
                write_json(transcript_json_path, transcript.to_dict())
                try:
                    summary = build_workout_summary(
                        transcript=transcript,
                        title=title,
                        source_url=args.url,
                        total_duration_sec=total_duration_sec,
                        language=transcript.language if args.language == "auto" else args.language,
                    )
                except TranscriptError as exc:
                    print(f"Transcript extraction failed. Trying visual analysis fallback: {exc}")
                    summary = None
            else:
                summary = None

            if summary is not None and summary_needs_visual_fallback(summary, transcript):
                print("Transcript quality is low. Trying visual analysis fallback...")
                try:
                    summary = build_general_visual_workout_summary(
                        video_path=video_path,
                        workdir=workdir,
                        title=title,
                        source_url=args.url,
                        total_duration_sec=total_duration_sec,
                        language=transcript.language if args.language == "auto" else args.language,
                    )
                except VisionExtractionError as exc:
                    print(f"Visual fallback failed, keeping transcript-based summary: {exc}")
            elif summary is None:
                print("Trying visual analysis fallback...")
                summary = build_general_visual_workout_summary(
                    video_path=video_path,
                    workdir=workdir,
                    title=title,
                    source_url=args.url,
                    total_duration_sec=total_duration_sec,
                    language=args.language if args.language != "auto" else "unknown",
                )

            print("Capturing representative screenshots...")
            capture_step_frames(video_path, summary.steps, workdir)

            summary_json_path = workdir / "workout_summary.json"
            write_json(summary_json_path, summary.to_dict())

            print("Rendering HTML summary...")
            render_summary_html(summary, output_path)
            print(f"HTML summary written to {output_path}")
            if args.keep_artifacts or args.workdir:
                print(f"Intermediate files kept in {workdir}")
            return 0
        except (
            DownloadError,
            FileNotFoundError,
            FrameCaptureError,
            RenderError,
            TranscriptError,
            VisionExtractionError,
        ) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    finally:
        if temp_dir is not None and not args.keep_artifacts:
            temp_dir.cleanup()
        elif temp_dir is not None and args.keep_artifacts:
            destination = _copy_kept_artifacts(temp_dir, output_path)
            print(f"Copied temporary artifacts to {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
