from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class DownloadError(RuntimeError):
    pass


@dataclass(slots=True)
class SubtitleDownload:
    path: Path
    language: str
    source: str


def _run_command(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise DownloadError(f"Command failed: {' '.join(args)}\n{stderr}")
    return completed


def fetch_video_metadata(url: str) -> dict:
    completed = _run_command(
        [
            "yt-dlp",
            "--skip-download",
            "--no-warnings",
            "--dump-single-json",
            "--no-playlist",
            url,
        ]
    )
    return json.loads(completed.stdout)


def download_video(url: str, workdir: Path) -> Path:
    output_template = str(workdir / "video.%(ext)s")
    _run_command(
        [
            "yt-dlp",
            "--no-playlist",
            "--no-update",
            "--format",
            "18/b[ext=mp4][vcodec!=none][acodec!=none]/"
            "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
            "--merge-output-format",
            "mp4",
            "--output",
            output_template,
            url,
        ]
    )
    video_candidates = sorted(
        path
        for path in workdir.glob("video.*")
        if path.suffix not in {".part", ".vtt", ".json"}
    )
    if not video_candidates:
        raise DownloadError("Video download completed but no video file was found.")
    return video_candidates[0]


def _candidate_languages(metadata: dict, preferred_language: str) -> list[tuple[str, str]]:
    manual = metadata.get("subtitles") or {}
    automatic = metadata.get("automatic_captions") or {}
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def enqueue(source_name: str, language: str) -> None:
        key = (source_name, language)
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    metadata_language = metadata.get("language")
    base_languages: list[str] = []
    if preferred_language != "auto":
        base_languages.append(preferred_language)
    if isinstance(metadata_language, str) and metadata_language:
        base_languages.append(metadata_language)
        if "-" in metadata_language:
            base_languages.append(metadata_language.split("-", 1)[0])
    base_languages.extend(["en", "en-US", "zh", "zh-Hans", "zh-Hant"])

    for language in base_languages:
        if language in manual:
            enqueue("subtitles", language)
        if language in automatic:
            enqueue("automatic_captions", language)

    for language in manual:
        enqueue("subtitles", language)
    for language in automatic:
        enqueue("automatic_captions", language)
    return candidates


def _download_subtitle_variant(
    url: str,
    workdir: Path,
    language: str,
    source: str,
) -> Path | None:
    for file_path in workdir.glob("subtitle*"):
        if file_path.is_file():
            file_path.unlink()

    args = [
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--output",
        str(workdir / "subtitle.%(ext)s"),
        "--sub-format",
        "vtt",
        "--sub-langs",
        language,
    ]
    if source == "subtitles":
        args.append("--write-subs")
    else:
        args.append("--write-auto-subs")
    args.append(url)

    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    matches = sorted(workdir.glob("subtitle*.vtt"))
    return matches[0] if matches else None


def download_best_subtitle(
    url: str,
    metadata: dict,
    workdir: Path,
    preferred_language: str = "auto",
) -> SubtitleDownload | None:
    for source, language in _candidate_languages(metadata, preferred_language):
        subtitle_path = _download_subtitle_variant(url, workdir, language, source)
        if subtitle_path is not None:
            return SubtitleDownload(path=subtitle_path, language=language, source=source)
    return None
