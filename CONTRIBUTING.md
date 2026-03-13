# Contributing

## Development Setup

1. Create a virtual environment.
2. Install the package in editable mode.
3. Install the optional Whisper extra if you want local transcription.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -e '.[whisper]'
```

System tools used by the pipeline:

- `yt-dlp`
- `ffmpeg`
- `ffprobe`
- `tesseract` for the legacy overlay path

## Tests

```bash
python -m unittest discover -s tests -v
python -m py_compile main.py schemas.py pipeline/*.py
```

## Notes

- Generated summaries and artifact folders should stay out of commits.
- The timer-first visual path is still heuristic and should be validated on new video families before large refactors.
