import unittest
import tempfile
from pathlib import Path

from pipeline.frames import _compute_anchor_time, _compute_clip_window, capture_step_clips
from schemas import WorkoutStep


class ClipWindowTests(unittest.TestCase):
    def test_compute_anchor_time_uses_thirty_percent_position(self) -> None:
        step = WorkoutStep(index=1, name="Jumping Jacks", start_sec=10.0, end_sec=30.0)
        self.assertEqual(_compute_anchor_time(step), 16.0)

    def test_compute_clip_window_centers_five_seconds_for_long_step(self) -> None:
        step = WorkoutStep(index=1, name="Jumping Jacks", start_sec=0.0, end_sec=30.0)
        clip_start_sec, clip_duration_sec = _compute_clip_window(
            step=step,
            anchor_sec=9.0,
            video_duration_sec=40.0,
            clip_target_sec=5.0,
        )
        self.assertAlmostEqual(clip_start_sec, 6.5)
        self.assertAlmostEqual(clip_duration_sec, 5.0)

    def test_compute_clip_window_uses_full_step_for_short_step(self) -> None:
        step = WorkoutStep(index=1, name="Hold", start_sec=10.0, end_sec=13.0)
        clip_start_sec, clip_duration_sec = _compute_clip_window(
            step=step,
            anchor_sec=11.0,
            video_duration_sec=20.0,
            clip_target_sec=5.0,
        )
        self.assertAlmostEqual(clip_start_sec, 10.0)
        self.assertAlmostEqual(clip_duration_sec, 3.0)

    def test_compute_clip_window_clamps_near_start_boundary(self) -> None:
        step = WorkoutStep(index=1, name="Sprint", start_sec=0.0, end_sec=30.0)
        clip_start_sec, clip_duration_sec = _compute_clip_window(
            step=step,
            anchor_sec=1.0,
            video_duration_sec=40.0,
            clip_target_sec=5.0,
        )
        self.assertAlmostEqual(clip_start_sec, 0.0)
        self.assertAlmostEqual(clip_duration_sec, 5.0)

    def test_compute_clip_window_clamps_to_video_end(self) -> None:
        step = WorkoutStep(index=1, name="Sprint", start_sec=58.0, end_sec=64.0)
        clip_start_sec, clip_duration_sec = _compute_clip_window(
            step=step,
            anchor_sec=60.0,
            video_duration_sec=60.0,
            clip_target_sec=5.0,
        )
        self.assertAlmostEqual(clip_start_sec, 58.0)
        self.assertAlmostEqual(clip_duration_sec, 2.0)

    def test_capture_step_clips_outputs_gif(self) -> None:
        from pipeline import frames as frames_module

        original_subprocess_run = frames_module.subprocess.run
        ffmpeg_args: list[str] = []

        def fake_run(args, **kwargs):
            nonlocal ffmpeg_args
            if args[0] == "ffprobe":
                class Result:
                    returncode = 0
                    stdout = "30.0\n"
                    stderr = ""

                return Result()
            if args[0] == "ffmpeg":
                ffmpeg_args = args
                Path(args[-1]).write_bytes(b"GIF89a")

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()
            raise AssertionError(f"unexpected command: {args[0]}")

        frames_module.subprocess.run = fake_run
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                video_path = temp_path / "video.mp4"
                video_path.write_bytes(b"video")
                step = WorkoutStep(index=1, name="Jumping Jacks", start_sec=0.0, end_sec=30.0)
                capture_step_clips(video_path, [step], temp_path)

                self.assertIsNotNone(step.clip_path)
                assert step.clip_path is not None
                self.assertTrue(step.clip_path.endswith(".gif"))
                self.assertIn("-filter_complex", ffmpeg_args)
                self.assertIn("-loop", ffmpeg_args)
        finally:
            frames_module.subprocess.run = original_subprocess_run


if __name__ == "__main__":
    unittest.main()
