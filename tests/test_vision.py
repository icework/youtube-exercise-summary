import json
import tempfile
import time
import unittest
from pathlib import Path
import sys

from PIL import Image

from pipeline.vision import (
    CountdownSegmentSpan,
    OcrDetection,
    OcrFrameInfo,
    TimerSegment,
    _build_ocr_crops,
    _build_ocr_frame_infos,
    _aggregate_preview_name,
    _assign_visual_fallback_names,
    _build_countdown_spans,
    _build_timer_segments,
    _build_timer_segments_with_rapidocr,
    _build_timeline_segments,
    _detect_dark_header_specs,
    _detect_preview_box,
    _extract_action_name,
    _extract_timer_kind,
    _merge_timeline_segments,
    _normalize_label,
    _rapidocr_python,
    _rapidocr_text_map,
    _refine_preview_box_ratios,
    _resolve_countdown_kinds,
    _score_timer_box_candidate,
    _select_boundary_preview_retry_indexes,
    _select_boundary_timer_retry_indexes,
    _select_probe_frames,
    _select_full_ocr_target_indexes,
    _select_sparse_timer_ocr_target_indexes,
    _split_compound_token,
    TimerTimelineSegment,
    build_timer_workout_summary,
)


class VisionTests(unittest.TestCase):
    def test_rapidocr_python_falls_back_to_current_interpreter(self) -> None:
        from pipeline import vision as vision_module

        original_getenv = vision_module.os.getenv
        original_cwd = vision_module.Path.cwd
        try:
            vision_module.os.getenv = lambda name: None if name == "LOCAL_OCR_PYTHON" else original_getenv(name)
            vision_module.Path.cwd = classmethod(lambda cls: Path("/tmp/does-not-exist"))
            self.assertEqual(_rapidocr_python(), Path(sys.executable))
        finally:
            vision_module.os.getenv = original_getenv
            vision_module.Path.cwd = original_cwd

    def test_split_compound_token_uses_fitness_dictionary(self) -> None:
        self.assertEqual(_split_compound_token("DOUBLECRUNCHES"), "DOUBLE CRUNCHES")
        self.assertEqual(_split_compound_token("RUNNINGCLIMBER"), "RUNNING CLIMBER")
        self.assertEqual(_split_compound_token("PUSHUP"), "PUSH UP")
        self.assertEqual(_split_compound_token("BUTTKICK"), "BUTT KICK")
        self.assertEqual(_split_compound_token("OBLIQUETO"), "OBLIQUE TO")
        self.assertEqual(_split_compound_token("SIDEJACKKNIFE"), "SIDE JACKKNIFE")
        self.assertEqual(_split_compound_token("CRUNCHTO"), "CRUNCH TO")

    def test_normalize_label_title_cases_result(self) -> None:
        self.assertEqual(_normalize_label("## TOWELPULL"), "Towel Pull")
        self.assertEqual(_normalize_label("1-2 MINUTE REST"), "Rest")
        self.assertEqual(_normalize_label("TATERAL SHUFFLE"), "Lateral Shuffle")
        self.assertEqual(_normalize_label("SpUTSQUATS"), "Split Squats")
        self.assertEqual(_normalize_label("OBLIQUETO CROSS CRUNCH R"), "Oblique To Cross Crunch R")
        self.assertEqual(_normalize_label("SCISSORCRUNCHES R"), "Scissor Crunches R")

    def test_build_timer_segments_from_color_states(self) -> None:
        sample_frames = [(float(i), None) for i in range(9)]
        states = [None, "orange", "orange", "green", "green", "green", "orange", "orange", None]

        from pipeline import vision as vision_module

        original_timer_state = vision_module._timer_state_top_right
        vision_module._timer_state_top_right = lambda _frame: states.pop(0)
        try:
            segments = _build_timer_segments(sample_frames, min_action_sec=2, min_rest_sec=2)
        finally:
            vision_module._timer_state_top_right = original_timer_state

        self.assertEqual(
            segments,
            [
                TimerSegment(
                    kind="rest",
                    start_sec=1.0,
                    end_sec=3.0,
                    duration_sec=2.0,
                    name_source="timer_color",
                ),
                TimerSegment(
                    kind="action",
                    start_sec=3.0,
                    end_sec=6.0,
                    duration_sec=3.0,
                    name_source="timer_color",
                ),
                TimerSegment(
                    kind="rest",
                    start_sec=6.0,
                    end_sec=8.0,
                    duration_sec=2.0,
                    name_source="timer_color",
                ),
            ],
        )

    def test_detect_dark_header_specs_finds_top_left_title_bar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_path = Path(temp_dir) / "frame.jpg"
            image = Image.new("L", (400, 240), color=235)
            for y in range(8, 30):
                for x in range(12, 150):
                    image.putpixel((x, y), 24)
            image.save(frame_path)

            specs = _detect_dark_header_specs([(0.0, frame_path)])

        self.assertTrue(specs)
        ratios = next(iter(specs.values()))
        self.assertLessEqual(ratios[0], 0.06)
        self.assertLessEqual(ratios[1], 0.06)
        self.assertGreaterEqual(ratios[2], 0.34)
        self.assertLessEqual(ratios[2], 0.42)

    def test_extract_action_name_prefers_short_action_label(self) -> None:
        texts = [
            "7 分鐘站立式腹肌訓",
            "12個動作+組間5秒休息",
            "高抬腿",
            "感受部位：腹肌",
            "首先我們會用高抬腿來開始今天的訓",
        ]
        self.assertEqual(_extract_action_name(texts), "高抬腿")

    def test_extract_action_name_ignores_single_character_noise(self) -> None:
        self.assertIsNone(_extract_action_name(["意"]))

    def test_extract_action_name_from_next_card(self) -> None:
        texts = ["NEXT:STANDINGJACKS+PUNCH", "REST:8"]
        self.assertEqual(_extract_action_name(texts), "Standing Jacks Punch")

    def test_extract_action_name_from_split_next_card_tokens(self) -> None:
        texts = ["NEXT:", "STANDINGJACKS", "+", "PUNCH"]
        self.assertEqual(_extract_action_name(texts), "Standing Jacks Punch")

    def test_extract_action_name_accepts_directional_five_word_title(self) -> None:
        self.assertEqual(
            _extract_action_name(["OBLIQUE TO CROSS CRUNCH (R)"]),
            "Oblique To Cross Crunch R",
        )

    def test_extract_timer_kind_treats_start_as_prep(self) -> None:
        self.assertEqual(_extract_timer_kind(["開始", "03"]), "prep")
        self.assertEqual(_extract_timer_kind(["REST", "10"]), "rest")
        self.assertEqual(_extract_timer_kind(["開始", "30"]), "action")

    def test_select_full_ocr_target_indexes_skips_single_action_miss(self) -> None:
        timer_texts = [[str(30 - index)] for index in range(12)]
        timer_texts[9] = []
        targets = _select_full_ocr_target_indexes(timer_texts)

        self.assertNotIn(9, targets)
        self.assertTrue(set(range(6)).issubset(targets))

    def test_select_probe_frames_caps_detector_work(self) -> None:
        frames = [(float(index), f"frame-{index}") for index in range(90)]
        probe_frames = _select_probe_frames(frames, max_count=36, keep_head=8)

        self.assertEqual(len(probe_frames), 36)
        self.assertEqual([frame for _, frame in probe_frames[:8]], [f"frame-{index}" for index in range(8)])

    def test_select_sparse_timer_ocr_target_indexes_keeps_head_tail_and_stride(self) -> None:
        targets = _select_sparse_timer_ocr_target_indexes(20)

        self.assertTrue(set(range(12)).issubset(targets))
        self.assertIn(16, targets)
        self.assertIn(18, targets)
        self.assertIn(19, targets)
        self.assertNotIn(13, targets)
        self.assertNotIn(15, targets)

    def test_build_ocr_crops_reuses_identical_existing_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame_path = root / "sample_0001.jpg"
            Image.new("RGB", (32, 24), color=(255, 255, 255)).save(frame_path)

            sample_frames = [(0.0, frame_path)]
            box_fn = lambda width, height: (0, 0, width, height)

            first_crops = _build_ocr_crops(sample_frames, root / "ocr", box_fn, scale=2)
            crop_path = first_crops[0][1]
            first_mtime = crop_path.stat().st_mtime_ns

            time.sleep(0.01)
            second_crops = _build_ocr_crops(sample_frames, root / "ocr", box_fn, scale=2)
            second_mtime = crop_path.stat().st_mtime_ns

        self.assertEqual(second_crops[0][1], crop_path)
        self.assertEqual(second_mtime, first_mtime)

    def test_select_full_ocr_target_indexes_caps_preview_ocr_count(self) -> None:
        timer_texts: list[list[str]] = []
        for block in range(20):
            timer_texts.extend([["REST", "10"], ["REST", "09"]])
            timer_texts.extend([[str(value)] for value in range(30, 24, -1)])

        targets = _select_full_ocr_target_indexes(timer_texts)

        self.assertLessEqual(len(targets), 48)
        self.assertTrue(set(range(6)).issubset(targets))

    def test_rapidocr_text_map_uses_sidecar_cache(self) -> None:
        from pipeline import vision as vision_module

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            image_path.write_bytes(b"fake-image")
            stat = image_path.stat()
            cache_path = image_path.with_name(image_path.name + ".ocr.json")
            cache_path.write_text(
                json.dumps(
                    {
                        "source_size": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "texts": ["NEXT:THRUSTER"],
                    }
                ),
                encoding="utf-8",
            )

            original_subprocess_run = vision_module.subprocess.run
            vision_module.subprocess.run = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("subprocess.run should not be called when cache is valid")
            )
            try:
                result = _rapidocr_text_map([image_path])
            finally:
                vision_module.subprocess.run = original_subprocess_run

        self.assertEqual(result, {str(image_path): ["NEXT:THRUSTER"]})

    def test_build_timer_workout_summary_keeps_initial_rest(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        original_extract_sample_frames = vision_module._extract_sample_frames
        original_build_timer_segments = vision_module._build_timer_segments_with_rapidocr
        original_build_color_segments = vision_module._build_timer_segments
        vision_module._extract_sample_frames = lambda **_: [(0.0, Path("frame0.jpg"))]
        vision_module._build_timer_segments_with_rapidocr = lambda *_: [
            TimerSegment(
                kind="rest",
                start_sec=0.0,
                end_sec=10.0,
                duration_sec=10.0,
                name="High Knees",
                name_source="timer_explicit",
            ),
            TimerSegment(kind="action", start_sec=10.0, end_sec=40.0, duration_sec=30.0, name=None),
        ]
        vision_module._build_timer_segments = lambda *_args, **_kwargs: []
        try:
            summary = build_timer_workout_summary(
                video_path=Path("video.mp4"),
                workdir=Path("workdir"),
                title="Sample",
                source_url="https://example.com",
                total_duration_sec=40.0,
                language="en",
            )
        finally:
            vision_module._extract_sample_frames = original_extract_sample_frames
            vision_module._build_timer_segments_with_rapidocr = original_build_timer_segments
            vision_module._build_timer_segments = original_build_color_segments

        self.assertEqual(summary.steps[0].name, "Rest")
        self.assertEqual(summary.steps[0].start_sec, 0.0)
        self.assertEqual(summary.steps[1].name, "High Knees")
        self.assertEqual(summary.steps[0].name_source, "timer_rest")
        self.assertEqual(summary.steps[1].name_source, "timer_explicit")

    def test_build_timer_segments_with_rapidocr_uses_reset_and_rest_preview_names(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        frame_infos: list[OcrFrameInfo] = []
        for second in range(4):
            frame_infos.append(
                OcrFrameInfo(
                    time_sec=float(second),
                    label_texts=["NEXT: STANDING JACKS"],
                    timer_texts=["開始", str(3 - second)],
                    timer_number=3 - second,
                    timer_kind="prep",
                    explicit_preview_name="Standing Jacks",
                    preview_name="Standing Jacks",
                )
            )
        for second in range(4, 34):
            frame_infos.append(
                OcrFrameInfo(
                    time_sec=float(second),
                    label_texts=[],
                    timer_texts=[str(34 - second)],
                    timer_number=34 - second,
                    timer_kind=None,
                    explicit_preview_name=None,
                    preview_name=None,
                )
            )
        for second in range(34, 44):
            frame_infos.append(
                OcrFrameInfo(
                    time_sec=float(second),
                    label_texts=["NEXT:", "STANDINGJACKS", "+", "PUNCH"],
                    timer_texts=["REST", str(44 - second)],
                    timer_number=44 - second,
                    timer_kind="rest",
                    explicit_preview_name="Standing Jacks Punch",
                    preview_name="Standing Jacks Punch",
                )
            )
        for second in range(44, 74):
            frame_infos.append(
                OcrFrameInfo(
                    time_sec=float(second),
                    label_texts=[],
                    timer_texts=[str(74 - second)],
                    timer_number=74 - second,
                    timer_kind=None,
                    explicit_preview_name=None,
                    preview_name=None,
                )
            )

        original_build_ocr_frame_infos = vision_module._build_ocr_frame_infos
        vision_module._build_ocr_frame_infos = lambda *_args, **_kwargs: frame_infos
        try:
            segments = _build_timer_segments_with_rapidocr(
                sample_frames=[(float(index), Path(f"frame-{index}.jpg")) for index in range(len(frame_infos))],
                workdir=Path("workdir"),
                style_hint="timer_next_card",
                timer_box_id="top_right",
                preview_box_id="upper_band",
            )
        finally:
            vision_module._build_ocr_frame_infos = original_build_ocr_frame_infos

        self.assertEqual(
            segments[:4],
            [
                TimerSegment(
                    kind="rest",
                    start_sec=0.0,
                    end_sec=4.0,
                    duration_sec=4.0,
                    name="Standing Jacks",
                    name_source="timer_explicit",
                ),
                TimerSegment(
                    kind="action",
                    start_sec=4.0,
                    end_sec=34.0,
                    duration_sec=30.0,
                    name="Standing Jacks",
                    name_source="timer_explicit",
                ),
                TimerSegment(
                    kind="rest",
                    start_sec=34.0,
                    end_sec=44.0,
                    duration_sec=10.0,
                    name="Standing Jacks Punch",
                    name_source="timer_explicit",
                ),
                TimerSegment(
                    kind="action",
                    start_sec=44.0,
                    end_sec=74.0,
                    duration_sec=30.0,
                    name="Standing Jacks Punch",
                    name_source="timer_explicit",
                ),
            ],
        )

    def test_build_ocr_frame_infos_retries_boundary_preview_ocr_when_sparse_preview_is_empty(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        sample_frames = [(float(index), Path(f"sample_{index + 1:04d}.jpg")) for index in range(40)]
        timer_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.timer.png")
            for _, frame_path in sample_frames
        }
        label_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.label.png")
            for _, frame_path in sample_frames
        }

        original_build_ocr_crops = vision_module._build_ocr_crops
        original_rapidocr_text_map = vision_module._rapidocr_text_map
        original_rapidocr_detection_map = vision_module._rapidocr_detection_map
        original_select_targets = vision_module._select_full_ocr_target_indexes

        calls: list[tuple[str, int]] = []

        def fake_build_ocr_crops(frames, crops_dir, _box_fn, scale=3):
            if crops_dir.name == "ocr_timer":
                return [(time_sec, timer_crop_paths[frame_path]) for time_sec, frame_path in frames]
            if crops_dir.name == "ocr_label":
                return [(time_sec, label_crop_paths[frame_path]) for time_sec, frame_path in frames]
            raise AssertionError(f"unexpected crops dir: {crops_dir}")

        def fake_rapidocr_text_map(paths):
            if paths and all(str(path).endswith(".timer.png") for path in paths):
                timer_texts = []
                for index in range(len(paths)):
                    if index < 20:
                        timer_texts.append(["REST", f"{20 - index:02d}"])
                    else:
                        timer_texts.append([str(60 - index)])
                result = {}
                for path, texts in zip(paths, timer_texts):
                    result[str(path)] = texts
                return result
            raise AssertionError("label OCR should use detection map")

        def fake_rapidocr_detection_map(paths):
            calls.append(("label", len(paths)))
            if len(paths) == 2:
                return {str(path): [] for path in paths}

            result = {str(path): [] for path in paths}
            result[str(label_crop_paths[sample_frames[22][1]])] = [
                OcrDetection("NEXT:", (0.0, 0.0, 10.0, 10.0)),
                OcrDetection("SQUAT", (12.0, 0.0, 36.0, 10.0)),
            ]
            return result

        vision_module._build_ocr_crops = fake_build_ocr_crops
        vision_module._rapidocr_text_map = fake_rapidocr_text_map
        vision_module._rapidocr_detection_map = fake_rapidocr_detection_map
        vision_module._select_full_ocr_target_indexes = lambda _texts: {0, 1}
        try:
            infos = _build_ocr_frame_infos(
                sample_frames=sample_frames,
                workdir=Path("workdir"),
                style_hint="timer_next_card",
                timer_box_id="top_right",
                preview_box_id="upper_band",
            )
        finally:
            vision_module._build_ocr_crops = original_build_ocr_crops
            vision_module._rapidocr_text_map = original_rapidocr_text_map
            vision_module._rapidocr_detection_map = original_rapidocr_detection_map
            vision_module._select_full_ocr_target_indexes = original_select_targets

        self.assertEqual(calls[0], ("label", 2))
        self.assertGreater(calls[1][1], 2)
        self.assertLess(calls[1][1], len(sample_frames))
        self.assertEqual(infos[22].explicit_preview_name, "Squat")

    def test_build_ocr_frame_infos_uses_sparse_timer_ocr_before_boundary_retry(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        sample_frames = [(float(index), Path(f"sample_{index + 1:04d}.jpg")) for index in range(40)]
        timer_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.timer.png")
            for _, frame_path in sample_frames
        }
        label_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.label.png")
            for _, frame_path in sample_frames
        }

        original_build_ocr_crops = vision_module._build_ocr_crops
        original_rapidocr_text_map = vision_module._rapidocr_text_map
        original_rapidocr_detection_map = vision_module._rapidocr_detection_map

        calls: list[tuple[str, int]] = []

        def fake_build_ocr_crops(frames, crops_dir, _box_fn, scale=3):
            if crops_dir.name == "ocr_timer":
                return [(time_sec, timer_crop_paths[frame_path]) for time_sec, frame_path in frames]
            if crops_dir.name == "ocr_label":
                return [(time_sec, label_crop_paths[frame_path]) for time_sec, frame_path in frames]
            raise AssertionError(f"unexpected crops dir: {crops_dir}")

        def fake_rapidocr_text_map(paths):
            if paths and all(str(path).endswith(".timer.png") for path in paths):
                calls.append(("timer", len(paths)))
                result = {}
                for path in paths:
                    frame_index = int(Path(path).stem.split("_")[1].split(".")[0]) - 1
                    if frame_index < 10:
                        texts = ["REST", f"{10 - frame_index:02d}"]
                    else:
                        texts = [str(max(0, 40 - frame_index))]
                    result[str(path)] = texts
                return result
            raise AssertionError("label OCR should use detection map")

        def fake_rapidocr_detection_map(paths):
            calls.append(("label", len(paths)))
            return {str(path): [] for path in paths}

        vision_module._build_ocr_crops = fake_build_ocr_crops
        vision_module._rapidocr_text_map = fake_rapidocr_text_map
        vision_module._rapidocr_detection_map = fake_rapidocr_detection_map
        try:
            _build_ocr_frame_infos(
                sample_frames=sample_frames,
                workdir=Path("workdir"),
                style_hint="timer_next_card",
                timer_box_id="top_right",
                preview_box_id="upper_band",
            )
        finally:
            vision_module._build_ocr_crops = original_build_ocr_crops
            vision_module._rapidocr_text_map = original_rapidocr_text_map
            vision_module._rapidocr_detection_map = original_rapidocr_detection_map

        timer_calls = [count for kind, count in calls if kind == "timer"]
        self.assertGreaterEqual(len(timer_calls), 2)
        self.assertLess(timer_calls[0], len(sample_frames))
        self.assertLess(timer_calls[1], len(sample_frames))
        self.assertNotIn(len(sample_frames), timer_calls)

    def test_select_boundary_preview_retry_indexes_covers_initial_action_lead_in(self) -> None:
        timer_texts = (
            [[] for _ in range(16)]
            + [["30"], ["29"], ["28"], ["27"]]
            + [[] for _ in range(20)]
        )

        targets = _select_boundary_preview_retry_indexes(timer_texts, {0, 1})

        self.assertIn(12, targets)
        self.assertIn(13, targets)
        self.assertIn(14, targets)
        self.assertIn(15, targets)

    def test_select_boundary_timer_retry_indexes_targets_resets_without_full_scan(self) -> None:
        timer_texts = [
            ["REST", "10"],
            [],
            ["REST", "08"],
            [],
            ["30"],
            [],
            ["28"],
            [],
            ["26"],
            [],
            ["02"],
            [],
            ["30"],
            [],
            ["28"],
        ]

        targets = _select_boundary_timer_retry_indexes(timer_texts, {0, 2, 4, 6, 8, 10, 12, 14})

        self.assertIn(11, targets)
        self.assertIn(13, targets)
        self.assertLess(len(targets), len(timer_texts))

    def test_select_boundary_timer_retry_indexes_does_not_expand_every_sparse_action_frame(self) -> None:
        timer_texts = [[] for _ in range(20)]
        for index, number in zip(range(12, 20, 2), [30, 28, 26, 24]):
            timer_texts[index] = [str(number)]

        targets = _select_boundary_timer_retry_indexes(timer_texts, {0, 1, 2, 12, 14, 16, 18})

        self.assertNotIn(15, targets)
        self.assertNotIn(17, targets)

    def test_score_timer_box_candidate_prefers_real_countdown_sequence(self) -> None:
        strong = _score_timer_box_candidate(
            "candidate_a",
            [["開始", str(value)] for value in range(30, 24, -1)] + [["休息", "10"], ["休息", "09"]],
        )
        weak = _score_timer_box_candidate(
            "candidate_b",
            [["20MIN"], ["WORKOUT"], ["BEGINNER"], ["NO"], ["JUMPING"], []],
        )

        self.assertGreater(strong.score, weak.score)
        self.assertGreaterEqual(strong.longest_streak, 4)

    def test_detect_preview_box_prefers_next_card_hits(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        original_ocr_box_detections = vision_module._ocr_box_detections
        vision_module._ocr_box_detections = lambda *_args, **_kwargs: {
            "upper_band": [
                [OcrDetection("NEXT:THRUSTER", (0.2, 0.2, 0.7, 0.5))],
                [OcrDetection("NEXT:STANDINGJACKS", (0.18, 0.22, 0.72, 0.54))],
            ],
            "upper_left_wide": [
                [OcrDetection("感受部位：腹肌", (0.1, 0.1, 0.6, 0.3)), OcrDetection("高抬腿", (0.12, 0.42, 0.32, 0.56))],
                [OcrDetection("高抬腿", (0.1, 0.45, 0.28, 0.58))],
            ],
            "upper_left_narrow": [[], []],
            "upper_center": [[], []],
        }
        try:
            preview_box_id, preview_box_ratios, next_hits, action_hits = _detect_preview_box(
                sample_frames=[(0.0, Path("f0.jpg")), (1.0, Path("f1.jpg"))],
                target_indexes={0, 1},
                workdir=Path("workdir"),
            )
        finally:
            vision_module._ocr_box_detections = original_ocr_box_detections

        self.assertEqual(preview_box_id, "upper_band")
        self.assertIsNotNone(preview_box_ratios)
        self.assertEqual(next_hits, 2)
        self.assertGreaterEqual(action_hits, 2)

    def test_detect_preview_box_prefers_tighter_stable_bbox_candidate(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        original_ocr_box_detections = vision_module._ocr_box_detections
        original_detect_dark_header_specs = vision_module._detect_dark_header_specs
        vision_module._detect_dark_header_specs = lambda *_args, **_kwargs: {
            "dynamic_header_1": (0.0, 0.0, 0.34, 0.12)
        }
        vision_module._ocr_box_detections = lambda *_args, **_kwargs: {
            "upper_left_wide": [
                [
                    OcrDetection("OBLIQUE", (0.03, 0.06, 0.12, 0.13)),
                    OcrDetection("CRUNCH", (0.14, 0.06, 0.24, 0.13)),
                    OcrDetection("7 MIN", (0.68, 0.02, 0.86, 0.08)),
                ],
                [
                    OcrDetection("OBLIQUE", (0.04, 0.07, 0.13, 0.14)),
                    OcrDetection("CRUNCH", (0.15, 0.07, 0.25, 0.14)),
                    OcrDetection("BEGINNER", (0.6, 0.02, 0.88, 0.08)),
                ],
            ],
            "dynamic_header_1": [
                [
                    OcrDetection("OBLIQUE", (0.08, 0.2, 0.4, 0.72)),
                    OcrDetection("CRUNCH", (0.42, 0.2, 0.78, 0.72)),
                ],
                [
                    OcrDetection("OBLIQUE", (0.09, 0.22, 0.41, 0.74)),
                    OcrDetection("CRUNCH", (0.43, 0.22, 0.79, 0.74)),
                ],
            ],
        }
        try:
            preview_box_id, preview_box_ratios, next_hits, action_hits = _detect_preview_box(
                sample_frames=[(0.0, Path("f0.jpg")), (1.0, Path("f1.jpg"))],
                target_indexes={0, 1},
                workdir=Path("workdir"),
            )
        finally:
            vision_module._ocr_box_detections = original_ocr_box_detections
            vision_module._detect_dark_header_specs = original_detect_dark_header_specs

        self.assertEqual(preview_box_id, "dynamic_header_1")
        self.assertEqual(next_hits, 0)
        self.assertGreaterEqual(action_hits, 2)
        self.assertIsNotNone(preview_box_ratios)
        assert preview_box_ratios is not None
        self.assertLessEqual(preview_box_ratios[2] - preview_box_ratios[0], 0.34)

    def test_refine_preview_box_ratios_expands_when_text_hits_right_edge(self) -> None:
        refined = _refine_preview_box_ratios(
            matching_boxes=[
                (0.08, 0.2, 0.96, 0.72),
                (0.09, 0.22, 0.97, 0.74),
            ],
            box_ratios=(0.0, 0.02, 0.23, 0.09),
        )

        self.assertGreater(refined[2], 0.23)

    def test_build_ocr_frame_infos_uses_dynamic_preview_box_ratios_for_label_crop(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        sample_frames = [(float(index), Path(f"sample_{index + 1:04d}.jpg")) for index in range(4)]
        timer_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.timer.png")
            for _, frame_path in sample_frames
        }
        label_crop_paths = {
            frame_path: Path(f"/tmp/{frame_path.stem}.label.png")
            for _, frame_path in sample_frames
        }

        original_build_ocr_crops = vision_module._build_ocr_crops
        original_rapidocr_text_map = vision_module._rapidocr_text_map
        original_rapidocr_detection_map = vision_module._rapidocr_detection_map
        original_select_targets = vision_module._select_full_ocr_target_indexes

        seen_label_boxes: list[tuple[int, int, int, int]] = []

        def fake_build_ocr_crops(frames, crops_dir, box_fn, scale=3):
            if crops_dir.name == "ocr_timer":
                return [(time_sec, timer_crop_paths[frame_path]) for time_sec, frame_path in frames]
            if crops_dir.name == "ocr_label":
                seen_label_boxes.append(box_fn(200, 100))
                return [(time_sec, label_crop_paths[frame_path]) for time_sec, frame_path in frames]
            raise AssertionError(f"unexpected crops dir: {crops_dir}")

        def fake_rapidocr_text_map(paths):
            if paths and all(str(path).endswith(".timer.png") for path in paths):
                return {
                    str(path): ["REST", f"{10 - index:02d}"] if index < 2 else [str(32 - index)]
                    for index, path in enumerate(paths)
                }
            raise AssertionError("label OCR should use detection map")

        def fake_rapidocr_detection_map(paths):
            return {str(path): [] for path in paths}

        vision_module._build_ocr_crops = fake_build_ocr_crops
        vision_module._rapidocr_text_map = fake_rapidocr_text_map
        vision_module._rapidocr_detection_map = fake_rapidocr_detection_map
        vision_module._select_full_ocr_target_indexes = lambda _texts: {0}
        try:
            _build_ocr_frame_infos(
                sample_frames=sample_frames,
                workdir=Path("workdir"),
                style_hint="timer_next_card",
                timer_box_id="top_right",
                preview_box_id="dynamic_header_1",
                preview_box_ratios=(0.1, 0.2, 0.5, 0.6),
            )
        finally:
            vision_module._build_ocr_crops = original_build_ocr_crops
            vision_module._rapidocr_text_map = original_rapidocr_text_map
            vision_module._rapidocr_detection_map = original_rapidocr_detection_map
            vision_module._select_full_ocr_target_indexes = original_select_targets

        self.assertEqual(seen_label_boxes[0], (20, 20, 100, 60))

    def test_build_countdown_spans_splits_on_reset(self) -> None:
        infos = [
            OcrFrameInfo(0.0, [], ["REST", "03"], 3, "rest", None, None),
            OcrFrameInfo(1.0, [], ["REST", "02"], 2, "rest", None, None),
            OcrFrameInfo(2.0, [], ["REST", "01"], 1, "rest", None, None),
            OcrFrameInfo(3.0, [], ["開始", "30"], 30, "prep", None, None),
            OcrFrameInfo(4.0, [], ["29"], 29, None, None, None),
            OcrFrameInfo(5.0, [], ["28"], 28, None, None, None),
        ]

        spans = _build_countdown_spans(infos)

        self.assertEqual(
            spans,
            [
                CountdownSegmentSpan(0, 3, 0.0, 3.0, 3.0, 3, 0, 0, 3, 1),
                CountdownSegmentSpan(3, 6, 3.0, 6.0, 3.0, 0, 1, 0, 30, 28),
            ],
        )

    def test_resolve_countdown_kinds_uses_alternation(self) -> None:
        self.assertEqual(
            _resolve_countdown_kinds(
                [
                    CountdownSegmentSpan(0, 12, 0.0, 12.0, 12.0, 2, 0, 0, 12, 1),
                    CountdownSegmentSpan(12, 42, 12.0, 42.0, 30.0, 0, 1, 0, 30, 1),
                    CountdownSegmentSpan(42, 48, 42.0, 48.0, 6.0, 1, 0, 0, 6, 1),
                    CountdownSegmentSpan(48, 78, 48.0, 78.0, 30.0, 0, 1, 0, 30, 1),
                ]
            ),
            ["rest", "action", "rest", "action"],
        )

    def test_build_timeline_segments_inserts_rest_gap(self) -> None:
        segments = _build_timeline_segments(
            [
                CountdownSegmentSpan(0, 30, 0.0, 30.0, 30.0, 0, 1, 0, 30, 1),
                CountdownSegmentSpan(35, 65, 35.0, 65.0, 30.0, 0, 1, 0, 30, 1),
            ]
        )

        self.assertEqual(
            [(segment.kind, segment.start_sec, segment.end_sec) for segment in segments],
            [("action", 0.0, 30.0), ("rest", 30.0, 35.0), ("action", 35.0, 65.0)],
        )

    def test_build_timeline_segments_keeps_strong_action_spans_without_forced_alternation(self) -> None:
        segments = _merge_timeline_segments(
            _build_timeline_segments(
                [
                    CountdownSegmentSpan(0, 11, 0.0, 11.0, 11.0, 11, 0, 0, 10, 0),
                    CountdownSegmentSpan(11, 42, 11.0, 42.0, 31.0, 0, 13, 11, 30, 0),
                    CountdownSegmentSpan(42, 73, 42.0, 73.0, 31.0, 0, 13, 11, 30, 0),
                ]
            )
        )

        self.assertEqual(
            [(segment.kind, segment.start_sec, segment.end_sec) for segment in segments],
            [("rest", 0.0, 11.0), ("action", 11.0, 42.0), ("action", 42.0, 73.0)],
        )

    def test_aggregate_preview_name_prefers_explicit_name_over_noise(self) -> None:
        infos = [
            OcrFrameInfo(
                0.0,
                ["NEXT:STANDINGJACKS"],
                [],
                None,
                None,
                "Standing Jacks",
                "Standing Jacks",
            ),
            OcrFrameInfo(1.0, ["chuckmeo"], [], None, None, None, None),
            OcrFrameInfo(2.0, ["M"], [], None, None, None, None),
        ]

        self.assertEqual(_aggregate_preview_name(infos, 0, 3), "Standing Jacks")

    def test_assign_visual_fallback_names_uses_known_action_signatures(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frame_paths = {
                "high-1": temp_path / "high-1.jpg",
                "high-2": temp_path / "high-2.jpg",
                "wind-1": temp_path / "wind-1.jpg",
                "wind-2": temp_path / "wind-2.jpg",
                "unknown-a": temp_path / "unknown-a.jpg",
                "unknown-b": temp_path / "unknown-b.jpg",
            }
            for path in frame_paths.values():
                path.write_bytes(b"frame")

            original_scene_signature = vision_module._scene_signature
            signatures = {
                frame_paths["high-1"]: [0.0, 0.1, 0.2],
                frame_paths["high-2"]: [0.0, 0.1, 0.21],
                frame_paths["wind-1"]: [0.8, 0.9, 1.0],
                frame_paths["wind-2"]: [0.79, 0.9, 1.0],
                frame_paths["unknown-a"]: [0.01, 0.1, 0.2],
                frame_paths["unknown-b"]: [0.0, 0.11, 0.2],
            }
            vision_module._scene_signature = lambda frame_path: signatures[frame_path]
            try:
                frame_infos = [
                    OcrFrameInfo(0.0, [], [], None, None, None, None, frame_paths["high-1"]),
                    OcrFrameInfo(1.0, [], [], None, None, None, None, frame_paths["high-2"]),
                    OcrFrameInfo(2.0, [], [], None, None, None, None, frame_paths["wind-1"]),
                    OcrFrameInfo(3.0, [], [], None, None, None, None, frame_paths["wind-2"]),
                    OcrFrameInfo(4.0, [], [], None, None, None, None, frame_paths["unknown-a"]),
                    OcrFrameInfo(5.0, [], [], None, None, None, None, frame_paths["unknown-b"]),
                ]
                segments = [
                    TimerTimelineSegment("action", 0, 2, 0.0, 2.0, 2.0, "High Knees", "explicit"),
                    TimerTimelineSegment("action", 2, 4, 2.0, 4.0, 2.0, "Big Windmill", "candidate"),
                    TimerTimelineSegment("action", 4, 6, 4.0, 6.0, 2.0, None),
                ]
                _assign_visual_fallback_names(frame_infos, segments)
            finally:
                vision_module._scene_signature = original_scene_signature

        self.assertEqual(segments[-1].preview_name, "High Knees")
        self.assertEqual(segments[-1].preview_name_source, "visual")

    def test_assign_visual_fallback_names_does_not_chain_fallback_matches(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frame_paths = {
                "known-a": temp_path / "known-a.jpg",
                "known-b": temp_path / "known-b.jpg",
                "fallback-a": temp_path / "fallback-a.jpg",
                "fallback-b": temp_path / "fallback-b.jpg",
                "chained-a": temp_path / "chained-a.jpg",
                "chained-b": temp_path / "chained-b.jpg",
            }
            for path in frame_paths.values():
                path.write_bytes(b"frame")

            original_scene_signature = vision_module._scene_signature
            signatures = {
                frame_paths["known-a"]: [0.0, 0.0, 0.0],
                frame_paths["known-b"]: [0.0, 0.0, 0.0],
                frame_paths["fallback-a"]: [0.06, 0.06, 0.06],
                frame_paths["fallback-b"]: [0.06, 0.06, 0.06],
                frame_paths["chained-a"]: [0.11, 0.11, 0.11],
                frame_paths["chained-b"]: [0.11, 0.11, 0.11],
            }
            vision_module._scene_signature = lambda frame_path: signatures[frame_path]
            try:
                frame_infos = [
                    OcrFrameInfo(0.0, [], [], None, None, None, None, frame_paths["known-a"]),
                    OcrFrameInfo(1.0, [], [], None, None, None, None, frame_paths["known-b"]),
                    OcrFrameInfo(2.0, [], [], None, None, None, None, frame_paths["fallback-a"]),
                    OcrFrameInfo(3.0, [], [], None, None, None, None, frame_paths["fallback-b"]),
                    OcrFrameInfo(4.0, [], [], None, None, None, None, frame_paths["chained-a"]),
                    OcrFrameInfo(5.0, [], [], None, None, None, None, frame_paths["chained-b"]),
                ]
                segments = [
                    TimerTimelineSegment("action", 0, 2, 0.0, 2.0, 2.0, "High Knees", "explicit"),
                    TimerTimelineSegment("action", 2, 4, 2.0, 4.0, 2.0, None),
                    TimerTimelineSegment("action", 4, 6, 4.0, 6.0, 2.0, None),
                ]
                _assign_visual_fallback_names(frame_infos, segments)
            finally:
                vision_module._scene_signature = original_scene_signature

        self.assertEqual(segments[1].preview_name, "High Knees")
        self.assertEqual(segments[1].preview_name_source, "visual")
        self.assertIsNone(segments[2].preview_name)

    def test_assign_visual_fallback_names_ignores_candidate_references(self) -> None:
        from pathlib import Path
        from pipeline import vision as vision_module

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frame_paths = {
                "candidate-a": temp_path / "candidate-a.jpg",
                "candidate-b": temp_path / "candidate-b.jpg",
                "unknown-a": temp_path / "unknown-a.jpg",
                "unknown-b": temp_path / "unknown-b.jpg",
            }
            for path in frame_paths.values():
                path.write_bytes(b"frame")

            original_scene_signature = vision_module._scene_signature
            signatures = {
                frame_paths["candidate-a"]: [0.0, 0.1, 0.2],
                frame_paths["candidate-b"]: [0.0, 0.1, 0.21],
                frame_paths["unknown-a"]: [0.01, 0.1, 0.2],
                frame_paths["unknown-b"]: [0.0, 0.11, 0.2],
            }
            vision_module._scene_signature = lambda frame_path: signatures[frame_path]
            try:
                frame_infos = [
                    OcrFrameInfo(0.0, [], [], None, None, None, None, frame_paths["candidate-a"]),
                    OcrFrameInfo(1.0, [], [], None, None, None, None, frame_paths["candidate-b"]),
                    OcrFrameInfo(2.0, [], [], None, None, None, None, frame_paths["unknown-a"]),
                    OcrFrameInfo(3.0, [], [], None, None, None, None, frame_paths["unknown-b"]),
                ]
                segments = [
                    TimerTimelineSegment("action", 0, 2, 0.0, 2.0, 2.0, "High Knees", "candidate"),
                    TimerTimelineSegment("action", 2, 4, 2.0, 4.0, 2.0, None),
                ]
                _assign_visual_fallback_names(frame_infos, segments)
            finally:
                vision_module._scene_signature = original_scene_signature

        self.assertIsNone(segments[1].preview_name)


if __name__ == "__main__":
    unittest.main()
