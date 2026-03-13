# Iteration Notes

## Goal

Input a workout-style YouTube URL and generate a single-file HTML summary with:

- one screenshot per action
- action name when detectable
- duration / reps / sets when detectable

The recent work focused on videos where subtitles are weak or useless and the summary must be derived from on-screen overlays.

## Current Architecture

The pipeline now has three extraction paths:

1. `transcript/subtitle`
   - Use subtitles first.
   - Fall back to transcription when possible.
   - Best when spoken cues clearly describe the workout steps.

2. `vision_overlay`
   - Designed for videos with a bottom exercise bar.
   - Uses bottom-bar highlight detection plus OCR for slot labels.

3. `vision_timer`
   - Designed for timer-driven videos without a reliable bottom bar.
   - Detects the timer region from multiple candidate boxes.
   - Segments from countdown spans, resets, and rest/prep countdowns.
   - Uses sparse preview OCR for naming.

The entry point routes through `build_general_visual_workout_summary()` instead of hard-coding timer/overlay retry order.

## Video Families Observed

### Family A: Bottom Exercise Bar

Example:

- `hRlp66B-lx4`

Signals:

- bottom overlay slots
- orange highlight moves across the bar
- timer color changes are still useful

Best strategy:

- overlay-first segmentation
- OCR slot labels only for sampled label frames

### Family B: Timer Only, Chinese Labels

Example:

- `L1BTJsDAWjo`

Signals:

- top-right `開始/休息 + countdown`
- no bottom bar
- some clips have `下一個動作`
- action name appears in the upper-left area during preview / low countdown

Best strategy:

- segment from timer countdown only
- name actions from upper-left preview frames near rest or `countdown <= 4`

### Family C: Timer + NEXT Card

Example:

- `-hS70qmClnI`

Signals:

- top-right countdown
- explicit `NEXT: ...`
- explicit `REST: n`
- transcript mostly music noise

Best strategy:

- segment from timer countdown
- name actions from `NEXT:` cards using upper-band preview OCR

## What Was Tried

### 1. Transcript-First Only

Pros:

- cheap
- simple

Cons:

- failed on music-heavy videos
- auto subtitles often contained noise instead of exercise names

Outcome:

- kept as the first branch, but not sufficient

### 2. Full-Frame OCR on Sampled Frames

Pros:

- easy to prototype
- could recover `NEXT:` / `REST:` / Chinese labels from some frames

Cons:

- too slow on longer videos
- OCR quality depended too much on unrelated text regions
- naming and segmentation became entangled

Outcome:

- replaced with timer-first segmentation plus sparse preview OCR

### 3. Color-State Timer Detection

Pros:

- fast
- no OCR dependency

Cons:

- brittle across layouts
- failed when timer color or graphics changed
- weak on videos where the countdown was numeric but color cues were inconsistent

Outcome:

- still kept as a fallback, not the primary timer path

### 4. Timer OCR With Full-Frame Naming

Pros:

- better than pure color heuristics
- worked reasonably on some timer videos

Cons:

- still too expensive on long videos
- naming degraded when title text / instructional sentences outweighed the real action label
- led to regressions like `Exercise 01` or merged multi-action spans

Outcome:

- replaced with a two-stage timer algorithm

## Current Timer Algorithm

The current implementation is timer-first and split into four stages:

1. Timer box detection
   - OCR several candidate regions from the first `2-3` minutes.
   - Score each region by countdown behavior:
     - descending pairs such as `30, 29, 28`
     - reset hits such as `03 -> 30`
     - rest/prep keyword hits such as `REST`, `開始`, `休息`

2. Countdown-span segmentation
   - Build contiguous spans from timer OCR, tolerating short OCR gaps.
   - Split spans on countdown resets and explicit timer-kind flips.
   - Insert rest gaps between separated action-like spans when OCR misses the rest card itself.

3. Timeline kind resolution
   - Resolve each countdown span as `action` or `rest` mainly from:
     - timer keywords such as `REST` / `START`
     - observed timer scale (`30`-like vs `10/15`-like)
     - span duration
   - Treat `START / 開始` countdowns as prep/rest instead of action.

4. Sparse preview naming
   - OCR only likely-useful preview frames instead of every frame.
   - Prefer explicit `NEXT:` / `下一個動作` names.
   - Fall back to short action labels from the detected preview region.
   - Reject common watermark / noise tokens and very short OCR fragments.

This keeps:

- segmentation centered on the countdown timer
- naming centered on preview overlays

That separation is the main algorithmic improvement over previous iterations.

## Important Fixes Landed

### Artifact Handling

Problems fixed:

- kept artifact JSON used to point screenshots back to the original temp directory
- concurrent runs with the same output stem could delete each other’s artifact directory

Current behavior:

- copied artifact directories get a unique suffix when needed
- copied `workout_summary.json` has screenshot paths rewritten to the copied `frames/` directory

### Timer Summary Behavior

Problems fixed:

- the opening rest/countdown block was being dropped
- `START / 開始` countdowns were being treated as action
- `NEXT:` cards could collapse into partial names such as single letters

Current behavior:

- the initial rest segment is preserved
- its preview name can seed the following action
- split `NEXT:` OCR tokens are re-joined before normalization
- short English OCR fragments are rejected instead of becoming action names

## Current Quality Snapshot

### Good

- Family A still works reasonably well.
- Family C structure is well-understood.
- Artifact persistence is now safer and more reusable.

### Still Weak

- Family B is still not stable enough in end-to-end reruns.
- Video 2 previously regressed from named steps to `Exercise 01/02/...` and merged later steps into longer spans.
- Detector profiling is still OCR-heavy on full reruns because it evaluates several timer candidates across the first `2-3` minutes.

## Trade-offs

### Accuracy vs. Runtime

- Full-frame OCR improves recall but scales poorly.
- Timer-only segmentation is much cheaper and more stable.
- Sparse preview OCR is the current compromise.

### Generality vs. Style-Specific Logic

- A fully generic vision pipeline was not reliable enough.
- Routing by video family adds complexity but improves correctness.
- The practical trade-off is a small detector plus family-specific extractors.

### Naming vs. Segmentation

- When naming is allowed to influence segmentation, errors compound quickly.
- The newer approach intentionally lets segmentation come first.

## Recommended Next Steps

1. Validate on more timer layouts where the countdown is not in a corner.
2. Cache OCR results by crop path so repeated local reruns are cheaper.
3. Add a small benchmark script that compares generated summaries against known local baselines for videos 1, 2, and 3.
4. If runtime is still too high, reduce preview OCR further before touching timer OCR.

## Files Most Relevant To Continue

- `main.py`
- `pipeline/vision.py`
- `pipeline/frames.py`
- `tests/test_main.py`
- `tests/test_vision.py`
