import base64
import tempfile
import unittest
from pathlib import Path

from pipeline.render import render_summary_html
from schemas import WorkoutStep, WorkoutSummary


JPEG_BASE64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBUQEBUQEBUQEBUQEBUVEBUVEBUVFRUWFhUV"
    "FRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGhAQGi0lICUtLS0tLS0tLS0t"
    "LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAAEAAgMBIgACEQEDEQH/"
    "xAAXAAADAQAAAAAAAAAAAAAAAAAAAQMC/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEAMQ"
    "AAAB6AAAAP/EABgQAQEAAwAAAAAAAAAAAAAAAAEAEQIS/9oACAEBAAEFAo7lH//EABQRAQAAAAAAAA"
    "AAAAAAAAAAABD/2gAIAQMBAT8BP//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8BP//Z"
)


class RenderTests(unittest.TestCase):
    def test_render_summary_html_embeds_gif_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            clip_path = temp_path / "step_001.gif"
            clip_path.write_bytes(b"fake-gif")
            output_path = temp_path / "summary.html"

            summary = WorkoutSummary(
                title="Sample Workout",
                source_url="https://youtube.com/watch?v=test",
                language="en",
                total_duration_sec=90.0,
                transcript_source="subtitles",
                steps=[
                    WorkoutStep(
                        index=1,
                        name="Jumping Jacks",
                        start_sec=0.0,
                        end_sec=30.0,
                        duration_sec=30.0,
                        clip_start_sec=8.0,
                        clip_duration_sec=5.0,
                        clip_path=str(clip_path),
                    )
                ],
            )

            render_summary_html(summary, output_path)
            html = output_path.read_text(encoding="utf-8")

            self.assertIn("data:image/gif;base64", html)
            self.assertIn("<img", html)
            self.assertNotIn("<video", html)

    def test_render_summary_html_falls_back_to_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "step_001.jpg"
            image_path.write_bytes(base64.b64decode(JPEG_BASE64))
            output_path = temp_path / "summary.html"

            summary = WorkoutSummary(
                title="Sample Workout",
                source_url="https://youtube.com/watch?v=test",
                language="en",
                total_duration_sec=90.0,
                transcript_source="subtitles",
                steps=[
                    WorkoutStep(
                        index=1,
                        name="Jumping Jacks",
                        start_sec=0.0,
                        end_sec=30.0,
                        duration_sec=30.0,
                        reps=None,
                        sets=None,
                        notes="left",
                        name_source="timer_visual",
                        screenshot_time_sec=9.0,
                        screenshot_path=str(image_path),
                    )
                ],
            )

            render_summary_html(summary, output_path)
            html = output_path.read_text(encoding="utf-8")

            self.assertIn("data:image/jpeg;base64", html)
            self.assertIn("<img", html)
            self.assertIn("Jumping Jacks", html)
            self.assertIn("Sample Workout", html)
            self.assertNotIn("Name Source:", html)

    def test_render_summary_html_shows_media_fallback_for_missing_clip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "summary.html"

            summary = WorkoutSummary(
                title="Sample Workout",
                source_url="https://youtube.com/watch?v=test",
                language="en",
                total_duration_sec=90.0,
                transcript_source="subtitles",
                steps=[
                    WorkoutStep(
                        index=1,
                        name="Jumping Jacks",
                        start_sec=0.0,
                        end_sec=30.0,
                        duration_sec=30.0,
                        clip_path=str(temp_path / "missing.gif"),
                    )
                ],
            )

            render_summary_html(summary, output_path)
            html = output_path.read_text(encoding="utf-8")

            self.assertIn("Media unavailable", html)
            self.assertNotIn("<video", html)


if __name__ == "__main__":
    unittest.main()
