from __future__ import annotations

import base64
import html
import mimetypes
from importlib import resources
from pathlib import Path
from string import Template

from schemas import WorkoutStep, WorkoutSummary


class RenderError(RuntimeError):
    pass


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = int(round(seconds))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def _file_to_data_url(path_value: str | None) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime_type, _encoding = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    return f"data:{mime_type};base64,{encoded}"


def _metric_line(step: WorkoutStep) -> str:
    metrics: list[str] = []
    if step.duration_sec:
        metrics.append(f"Duration: {_format_seconds(step.duration_sec)}")
    if step.reps is not None:
        metrics.append(f"Reps: {step.reps}")
    if step.sets is not None:
        metrics.append(f"Sets: {step.sets}")
    return " | ".join(metrics) if metrics else "Duration / reps / sets not detected"


def _render_card(step: WorkoutStep) -> str:
    clip_url = _file_to_data_url(step.clip_path)
    image_url = _file_to_data_url(step.screenshot_path)
    clip_extension = Path(step.clip_path).suffix.lower() if step.clip_path else ""
    if clip_url and clip_extension == ".gif":
        media_html = f'<img src="{clip_url}" alt="{html.escape(step.name)} clip" class="card-image" />'
    elif clip_url:
        media_html = (
            f'<video src="{clip_url}" class="card-image" autoplay muted loop playsinline preload="metadata"></video>'
        )
    elif image_url:
        media_html = f'<img src="{image_url}" alt="{html.escape(step.name)} screenshot" class="card-image" />'
    else:
        media_html = '<div class="card-image card-image--empty">Media unavailable</div>'

    notes_html = ""
    if step.notes:
        notes_html = (
            f'<p class="card-notes"><strong>Notes:</strong> {html.escape(step.notes)}</p>'
        )
    if step.clip_start_sec is not None and step.clip_duration_sec is not None:
        media_label = (
            f'<p class="card-shot">Clip: {_format_seconds(step.clip_start_sec)} '
            f'({step.clip_duration_sec:.1f}s)</p>'
        )
    elif step.screenshot_time_sec is not None:
        media_label = f'<p class="card-shot">Screenshot at {_format_seconds(step.screenshot_time_sec)}</p>'
    else:
        media_label = ""

    return f"""
    <article class="card">
      {media_html}
      <div class="card-body">
        <div class="card-index">Step {step.index:02d}</div>
        <h2>{html.escape(step.name)}</h2>
        <p class="card-time">{_format_seconds(step.start_sec)} - {_format_seconds(step.end_sec)}</p>
        <p class="card-metrics">{html.escape(_metric_line(step))}</p>
        {notes_html}
        {media_label}
      </div>
    </article>
    """.strip()


def render_summary_html(summary: WorkoutSummary, output_path: Path) -> Path:
    template = Template(
        resources.files("pipeline").joinpath("templates/summary.html.j2").read_text(encoding="utf-8")
    )
    cards_html = "\n".join(_render_card(step) for step in summary.steps)
    rendered = template.safe_substitute(
        page_title=html.escape(summary.title),
        video_title=html.escape(summary.title),
        source_url=html.escape(summary.source_url),
        language=html.escape(summary.language),
        total_duration=_format_seconds(summary.total_duration_sec),
        transcript_source=html.escape(summary.transcript_source),
        generated_at=html.escape(summary.generated_at),
        step_count=str(len(summary.steps)),
        cards=cards_html,
    )
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
