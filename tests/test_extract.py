import unittest

from pipeline.extract import (
    build_workout_summary,
    extract_action_name,
    summary_needs_visual_fallback,
)
from schemas import Transcript, TranscriptSegment


class ExtractTests(unittest.TestCase):
    def test_extract_action_name_strips_filler_and_metrics(self) -> None:
        self.assertEqual(
            extract_action_name("Next, 30 seconds of jumping jacks"),
            "Jumping Jacks",
        )

    def test_extract_action_name_ignores_side_switch_transition(self) -> None:
        self.assertIsNone(extract_action_name("Switch sides"))

    def test_build_workout_summary_extracts_steps_metrics(self) -> None:
        transcript = Transcript(
            language="en",
            source="subtitles",
            segments=[
                TranscriptSegment(
                    start_sec=0.0,
                    end_sec=4.0,
                    text="Start with 30 seconds of jumping jacks",
                ),
                TranscriptSegment(
                    start_sec=4.0,
                    end_sec=8.0,
                    text="Keep breathing and stay light on your feet",
                ),
                TranscriptSegment(
                    start_sec=8.0,
                    end_sec=12.0,
                    text="Next, 12 reps of push ups",
                ),
                TranscriptSegment(
                    start_sec=12.0,
                    end_sec=16.0,
                    text="Switch sides and hold a side plank for 20 seconds",
                ),
            ],
        )

        summary = build_workout_summary(
            transcript=transcript,
            title="Quick Workout",
            source_url="https://youtube.com/watch?v=test",
            total_duration_sec=20.0,
            language="en",
        )

        self.assertEqual([step.name for step in summary.steps], ["Jumping Jacks", "Push Ups", "Side Plank"])
        self.assertEqual(summary.steps[0].duration_sec, 30.0)
        self.assertEqual(summary.steps[1].reps, 12)
        self.assertEqual(summary.steps[2].duration_sec, 20.0)
        self.assertTrue(all(step.name_source == "transcript_rule" for step in summary.steps))

    def test_build_workout_summary_does_not_promote_side_switch_cue_to_step(self) -> None:
        transcript = Transcript(
            language="en",
            source="subtitles",
            segments=[
                TranscriptSegment(
                    start_sec=0.0,
                    end_sec=4.0,
                    text="Start with side plank for 20 seconds",
                ),
                TranscriptSegment(
                    start_sec=4.0,
                    end_sec=5.0,
                    text="Switch sides",
                ),
                TranscriptSegment(
                    start_sec=5.0,
                    end_sec=9.0,
                    text="Hold for 20 seconds",
                ),
            ],
        )

        summary = build_workout_summary(
            transcript=transcript,
            title="Quick Workout",
            source_url="https://youtube.com/watch?v=test",
            total_duration_sec=10.0,
            language="en",
        )

        self.assertEqual([step.name for step in summary.steps], ["Side Plank"])
        self.assertEqual(summary.steps[0].end_sec, 10.0)

    def test_summary_needs_visual_fallback_when_noise_dominates(self) -> None:
        transcript = Transcript(
            language="en",
            source="automatic_captions",
            segments=[
                TranscriptSegment(start_sec=0.0, end_sec=1.0, text="[Music]"),
                TranscriptSegment(start_sec=1.0, end_sec=2.0, text="foreign"),
                TranscriptSegment(start_sec=2.0, end_sec=3.0, text="rest"),
                TranscriptSegment(start_sec=3.0, end_sec=4.0, text="[Applause]"),
            ],
        )
        summary = build_workout_summary(
            transcript=Transcript(
                language="en",
                source="automatic_captions",
                segments=[
                    TranscriptSegment(start_sec=0.0, end_sec=2.0, text="rest"),
                    TranscriptSegment(start_sec=2.0, end_sec=4.0, text="foreign"),
                    TranscriptSegment(start_sec=4.0, end_sec=6.0, text="rest"),
                    TranscriptSegment(start_sec=6.0, end_sec=8.0, text="music"),
                    TranscriptSegment(start_sec=8.0, end_sec=10.0, text="go"),
                ],
            ),
            title="Noisy Workout",
            source_url="https://youtube.com/watch?v=test",
            total_duration_sec=10.0,
            language="en",
        )
        self.assertTrue(summary_needs_visual_fallback(summary, transcript))



if __name__ == "__main__":
    unittest.main()
