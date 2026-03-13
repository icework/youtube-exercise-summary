# YouTube Exercise Summary

Generate a single-file HTML summary from a workout-style YouTube video. The output includes one representative screenshot per action plus any detected duration, reps, sets, and name provenance.

## What It Does

The pipeline downloads a YouTube workout video, tries transcript-based extraction first, and falls back to visual analysis when subtitles are weak or missing. The current visual path is timer-centric and tuned for videos that show recurring countdown timers and preview cards.

The project is still heuristic-driven. It works best on:

- follow-along workout videos with repeated `30s work / 10-15s rest` structure
- videos that show a visible countdown timer
- videos that show the next exercise name during rest or transition frames

## Requirements

System tools:

- `python3` 3.11+
- `yt-dlp`
- `ffmpeg`
- `ffprobe`
- `tesseract` for the legacy overlay path

Python packages are defined in [`pyproject.toml`](./pyproject.toml).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras:

- local Whisper transcription: `python -m pip install -e '.[whisper]'`
- isolated OCR environment: create `.venv-ocr` and install `rapidocr-onnxruntime` there if you want OCR separated from the main environment

The visual pipeline now also falls back to the current Python interpreter, so a normal single-environment install works without a dedicated `.venv-ocr`.

## Usage

```bash
youtube-exercise-summary \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output ./summary.html
```

Direct module usage still works:

```bash
python3 main.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output ./summary.html
```

Useful flags:

- `--workdir ./artifacts/run-001`
- `--transcribe-backend auto|whisper|api`
- `--language auto|en|zh`
- `--keep-artifacts`

## Environment Variables

- `OPENAI_API_KEY`: required for API transcription or transcript extraction through OpenAI-compatible endpoints
- `OPENAI_BASE_URL`: optional override for OpenAI-compatible APIs
- `OPENAI_TRANSCRIPTION_MODEL`: optional transcription model override
- `OPENAI_EXTRACTION_MODEL`: optional transcript extraction model override
- `WHISPER_MODEL`: optional local Whisper model name, default `base`
- `LOCAL_OCR_PYTHON`: optional Python executable used to run `scripts/rapidocr_batch.py`

## Output

Each run produces:

- one standalone HTML file with embedded screenshots
- intermediate transcript and summary JSON files inside the workdir
- frame captures under `frames/`
- OCR sidecar caches such as `*.ocr.json` and `*.ocr_boxes.json`

The main machine-readable artifact is `workout_summary.json`.

## Extraction Strategy

The pipeline currently has three extraction paths:

- `subtitle / transcript`: preferred when captions clearly describe the routine
- `vision_timer`: primary visual fallback for countdown-based workout videos
- `vision_overlay`: legacy fallback for videos with a strong bottom exercise bar

The timer path is intentionally timer-centric instead of color-centric:

- detect the timer region from multiple candidate boxes in the first `2-3` minutes
- segment the workout from countdown resets and rest/prep countdowns
- OCR only sparse preview frames plus dense windows around likely boundaries
- keep track of whether an action name came from explicit OCR, weaker OCR, rest hints, or visual fallback
- generate the final HTML from captured source frames

The current design notes and explored trade-offs are documented in [`docs/iteration-notes.md`](./docs/iteration-notes.md).

## Development

Run the test suite:

```bash
python -m unittest discover -s tests -v
python -m py_compile main.py schemas.py pipeline/*.py
```

Contributor notes live in [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Limitations

- It is not a general video understanding system; it is optimized for timer-driven workout videos.
- Action naming still depends on OCR quality and overlay layout.
- Very long videos can still be OCR-heavy, especially during the initial timer-box detection pass.
- Generated HTML embeds screenshots as base64, so large runs produce large output files.
