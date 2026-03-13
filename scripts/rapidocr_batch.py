from __future__ import annotations

import json
import sys

from rapidocr_onnxruntime import RapidOCR


def _normalize_box(points: list[list[float]] | tuple[tuple[float, float], ...]) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> int:
    args = sys.argv[1:]
    include_boxes = False
    if args and args[0] == "--with-boxes":
        include_boxes = True
        args = args[1:]

    engine = RapidOCR()
    payload: dict[str, list[str] | list[dict[str, object]]] = {}
    for image_path in args:
        result, _ = engine(image_path)
        if include_boxes:
            payload[image_path] = [
                {
                    "text": item[1],
                    "box": _normalize_box(item[0]),
                }
                for item in (result or [])
            ]
            continue
        payload[image_path] = [item[1] for item in (result or [])]
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
