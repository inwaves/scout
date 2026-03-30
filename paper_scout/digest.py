from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from .config import DEFAULT_SUBJECT_TEMPLATE
from .models import DigestContext, RenderedDigest


class DigestRenderer:
    """Render digest output in both Markdown and HTML."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        if template_dir is None:
            template_root = Path(__file__).resolve().parent / "templates"
        else:
            template_root = Path(template_dir).expanduser()

        self._env = Environment(
            loader=FileSystemLoader(str(template_root)),
            autoescape=select_autoescape(enabled_extensions=("html", "xml")),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["fmt_score"] = _fmt_score

    def render(
        self,
        context: DigestContext,
        subject_template: str | None = None,
    ) -> RenderedDigest:
        render_context = {
            "generated_at": context.generated_at,
            "date_display": context.generated_at.strftime("%B %d, %Y"),
            "total_reviewed": context.total_reviewed,
            "selected_count": len(context.entries),
            "threshold": context.threshold,
            "entries": context.entries,
        }

        try:
            markdown = self._env.get_template("digest.md.j2").render(**render_context).strip() + "\n"
            html = self._env.get_template("digest.html.j2").render(**render_context)
        except TemplateNotFound as exc:
            raise RuntimeError(
                f"Digest template missing: {exc.name}. Expected templates under paper_scout/templates/"
            ) from exc

        template = subject_template or DEFAULT_SUBJECT_TEMPLATE
        try:
            subject = template.format(
                date=context.generated_at.strftime("%Y-%m-%d"),
                count=len(context.entries),
            )
        except Exception:
            subject = DEFAULT_SUBJECT_TEMPLATE.format(
                date=context.generated_at.strftime("%Y-%m-%d"),
                count=len(context.entries),
            )

        return RenderedDigest(subject=subject, markdown=markdown, html=html)


def _fmt_score(value: float) -> str:
    formatted = f"{value:.1f}"
    return formatted.rstrip("0").rstrip(".")