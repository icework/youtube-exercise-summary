import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import main as main_module
from schemas import Transcript, TranscriptSegment, WorkoutStep, WorkoutSummary


class MainTests(unittest.TestCase):
    def test_load_transcript_with_fallback_uses_transcription_when_subtitle_parse_fails(self) -> None:
        original_download_best_subtitle = main_module.download_best_subtitle
        original_load_transcript_from_subtitle = main_module.load_transcript_from_subtitle
        original_transcribe_video = main_module.transcribe_video

        main_module.download_best_subtitle = lambda *args, **kwargs: SimpleNamespace(
            path=Path("subtitle.vtt"),
            language="en",
            source="subtitles",
        )
        main_module.load_transcript_from_subtitle = lambda *args, **kwargs: (_ for _ in ()).throw(
            main_module.TranscriptError("bad subtitle")
        )
        fallback_transcript = Transcript(
            language="en",
            source="whisper",
            segments=[TranscriptSegment(start_sec=0.0, end_sec=1.0, text="jumping jacks")],
        )
        main_module.transcribe_video = lambda *args, **kwargs: fallback_transcript
        try:
            transcript = main_module._load_transcript_with_fallback(
                url="https://youtube.com/watch?v=test",
                metadata={},
                video_path=Path("video.mp4"),
                workdir=Path("workdir"),
                preferred_language="en",
                transcribe_backend="auto",
            )
        finally:
            main_module.download_best_subtitle = original_download_best_subtitle
            main_module.load_transcript_from_subtitle = original_load_transcript_from_subtitle
            main_module.transcribe_video = original_transcribe_video

        self.assertIs(transcript, fallback_transcript)

    def test_main_falls_back_to_visual_when_transcript_extraction_fails(self) -> None:
        original_parse_args = main_module._parse_args
        original_fetch_video_metadata = main_module.fetch_video_metadata
        original_download_video = main_module.download_video
        original_probe_video_duration_sec = main_module._probe_video_duration_sec
        original_load_transcript_with_fallback = main_module._load_transcript_with_fallback
        original_build_workout_summary = main_module.build_workout_summary
        original_build_general_visual_workout_summary = main_module.build_general_visual_workout_summary
        original_capture_step_frames = main_module.capture_step_frames
        original_render_summary_html = main_module.render_summary_html
        original_write_json = main_module.write_json

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "summary.html"
            workdir = temp_path / "artifacts"
            visual_summary = WorkoutSummary(
                title="Visual Summary",
                source_url="https://youtube.com/watch?v=test",
                language="en",
                total_duration_sec=30.0,
                transcript_source="vision_timer",
                steps=[
                    WorkoutStep(
                        index=1,
                        name="Jumping Jacks",
                        start_sec=0.0,
                        end_sec=30.0,
                        duration_sec=30.0,
                    )
                ],
            )
            visual_called = {"value": False}

            main_module._parse_args = lambda: SimpleNamespace(
                url="https://youtube.com/watch?v=test",
                output=str(output_path),
                workdir=str(workdir),
                transcribe_backend="auto",
                language="en",
                keep_artifacts=False,
            )
            main_module.fetch_video_metadata = lambda _url: {"title": "Test Video", "duration": 30.0}
            main_module.download_video = lambda _url, _workdir: Path("video.mp4")
            main_module._probe_video_duration_sec = lambda _video_path: 30.0
            main_module._load_transcript_with_fallback = lambda **kwargs: Transcript(
                language="en",
                source="subtitles",
                segments=[TranscriptSegment(start_sec=0.0, end_sec=5.0, text="music")],
            )
            main_module.build_workout_summary = lambda **kwargs: (_ for _ in ()).throw(
                main_module.TranscriptError("no steps")
            )

            def fake_build_general_visual_workout_summary(**kwargs):
                visual_called["value"] = True
                return visual_summary

            main_module.build_general_visual_workout_summary = fake_build_general_visual_workout_summary
            main_module.capture_step_frames = lambda video_path, steps, workdir: steps
            main_module.render_summary_html = lambda summary, path: path.write_text("ok", encoding="utf-8")
            main_module.write_json = lambda path, payload: path.write_text(json.dumps(payload), encoding="utf-8")
            try:
                exit_code = main_module.main()
            finally:
                main_module._parse_args = original_parse_args
                main_module.fetch_video_metadata = original_fetch_video_metadata
                main_module.download_video = original_download_video
                main_module._probe_video_duration_sec = original_probe_video_duration_sec
                main_module._load_transcript_with_fallback = original_load_transcript_with_fallback
                main_module.build_workout_summary = original_build_workout_summary
                main_module.build_general_visual_workout_summary = original_build_general_visual_workout_summary
                main_module.capture_step_frames = original_capture_step_frames
                main_module.render_summary_html = original_render_summary_html
                main_module.write_json = original_write_json

        self.assertEqual(exit_code, 0)
        self.assertTrue(visual_called["value"])

    def test_probe_video_duration_sec_parses_ffprobe_output(self) -> None:
        original_subprocess_run = main_module.subprocess.run
        main_module.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="1248.374422\n",
            stderr="",
        )
        try:
            duration_sec = main_module._probe_video_duration_sec(Path("video.mp4"))
        finally:
            main_module.subprocess.run = original_subprocess_run

        self.assertEqual(duration_sec, 1248.374422)

    def test_probe_video_duration_sec_returns_none_on_invalid_output(self) -> None:
        original_subprocess_run = main_module.subprocess.run
        main_module.subprocess.run = lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="not-a-number\n",
            stderr="",
        )
        try:
            duration_sec = main_module._probe_video_duration_sec(Path("video.mp4"))
        finally:
            main_module.subprocess.run = original_subprocess_run

        self.assertIsNone(duration_sec)

    def test_claim_artifacts_destination_avoids_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.html"
            first = main_module._claim_artifacts_destination(output_path)
            second = main_module._claim_artifacts_destination(output_path)

            self.assertEqual(first.name, "summary_artifacts")
            self.assertEqual(second.name, "summary_artifacts_2")
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_copy_kept_artifacts_rewrites_screenshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.html"
            temp_dir_wrapper = tempfile.TemporaryDirectory(dir=temp_dir)
            try:
                kept_dir = Path(temp_dir_wrapper.name)
                frames_dir = kept_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                (frames_dir / "step_001.jpg").write_bytes(b"jpg")
                (kept_dir / "workout_summary.json").write_text(
                    json.dumps(
                        {
                            "steps": [
                                {
                                    "index": 1,
                                    "name": "Jumping Jacks",
                                    "screenshot_path": str(frames_dir / "step_001.jpg"),
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

                copied_path = main_module._copy_kept_artifacts(temp_dir_wrapper, output_path)
                payload = json.loads((copied_path / "workout_summary.json").read_text(encoding="utf-8"))
                screenshot_path = Path(payload["steps"][0]["screenshot_path"])

                self.assertEqual(copied_path.name, "summary_artifacts")
                self.assertEqual(screenshot_path, (copied_path / "frames" / "step_001.jpg").resolve())
                self.assertTrue(screenshot_path.exists())
            finally:
                temp_dir_wrapper.cleanup()


if __name__ == "__main__":
    unittest.main()
