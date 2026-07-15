"""ReportWriter: assembles a single self-contained report.html (spec §6.7).

All CSS, JS and the Plotly library are inlined so the report opens by double-clicking,
works fully offline, and can be shared as a single attachment. plotly.js is included
exactly once; each figure is embedded as an inert JSON payload that the template's
lazy-render runtime draws on first reveal (see charts._fig_to_div).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportWriter:
    def __init__(self, display_timezone: str = "Europe/Zurich") -> None:
        self.tz = ZoneInfo(display_timezone)
        self.env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        self.env.filters["localtime"] = self._localtime

    def _localtime(self, dt: datetime) -> str:
        return dt.astimezone(self.tz).strftime("%Y-%m-%d %H:%M %Z")

    def render(self, context: dict) -> str:
        template = self.env.get_template("report.html.j2")
        return template.render(plotlyjs=get_plotlyjs(), **context)

    def write(self, context: dict, output_dir: str | Path) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.html"
        out_path.write_text(self.render(context), encoding="utf-8")
        return out_path
