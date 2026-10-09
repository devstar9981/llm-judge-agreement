"""Draws results/agreement.png (and .svg) from results/summary.json.

    python -m judge.chart

Left: each judge's Pearson r with the human scores, with its 95% interval; small local judges
in blue, the large judges from the dataset in gray. Right: how often the judge lands within one
point of the human, for the large judges and the best small judge before and after calibration. Rendered as SVG in a page and screenshotted with headless Chrome or Edge.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from judge import data

ROOT = data.ROOT
RESULTS = ROOT / "results"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SMALL, LARGE = "#2a78d6", "#6f6e69"
W, H = 1200, 720
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LO, HI = 0.0, 0.8


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;")


def panel(
    x0: int, y0: int, w: int, title: str, note: str, rows: list[tuple[str, dict, str]],
    label_w: int = 250, lo: float = LO, hi: float = HI, step: float = 0.2, percent: bool = False,
) -> list[str]:
    """rows: (label, metric {value, ci}, color). Dot-and-interval plot on a lo-hi axis."""
    LO, HI = lo, hi  # noqa: N806
    fmt = (lambda v: f"{v:.0%}") if percent else (lambda v: f"{v:.2f}")  # noqa: E731
    row_h, top = 46, 74
    pw = w - label_w - 30
    px = lambda v: x0 + label_w + pw * (v - LO) / (HI - LO)  # noqa: E731
    bottom = y0 + top + row_h * len(rows)
    out = [
        f'<text x="{x0}" y="{y0 + 22}" font-size="22" font-weight="600" fill="{INK}">{esc(title)}</text>',
        f'<text x="{x0}" y="{y0 + 48}" font-size="16" fill="{INK_2}">{esc(note)}</text>',
    ]
    tick = LO
    while tick <= HI + 1e-9:
        x = px(tick)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 + top - 10}" y2="{bottom}" stroke="{GRID}" stroke-width="1"/>')
        label = f"{tick:.0%}" if percent else f"{tick:.1f}"
        out.append(f'<text x="{x:.1f}" y="{bottom + 24}" font-size="15" fill="{MUTED}" text-anchor="middle">{label}</text>')
        tick += step
    out.append(f'<line x1="{px(LO):.1f}" x2="{px(HI):.1f}" y1="{bottom}" y2="{bottom}" stroke="{AXIS}" stroke-width="1"/>')
    for i, (label, metric, color) in enumerate(rows):
        y = y0 + top + row_h * i + row_h / 2
        lo, hi = metric["ci"]
        out.append(f'<text x="{x0}" y="{y + 6:.1f}" font-size="16" fill="{INK}">{esc(label)}</text>')
        out.append(f'<line x1="{px(lo):.1f}" x2="{px(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round" opacity="0.45"/>')
        out.append(f'<circle cx="{px(metric["value"]):.1f}" cy="{y:.1f}" r="6.5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(
            f'<text x="{px(hi) + 10:.1f}" y="{y + 6:.1f}" font-size="15" font-weight="600" fill="{INK}" '
            f'style="font-variant-numeric: tabular-nums">{fmt(metric["value"])}</text>'
        )
    return out


def main() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf8"))
    judges = summary["judges"]
    left = [(c, j) for c, j in judges.items() if c.endswith("__full") or c in data.REFERENCE_JUDGES]
    left.sort(key=lambda cj: -cj[1]["pearson"]["value"])
    rows = [
        (j["name"].replace(" (full)", ""), j["pearson"], LARGE if c in data.REFERENCE_JUDGES else SMALL)
        for c, j in left
    ]
    small = [c for c in judges if c.endswith("__full") and not c.startswith("ensemble")]
    close = []
    if small:
        best = max(small, key=lambda c: judges[c]["pearson"]["value"])
        picks = [*sorted(data.REFERENCE_JUDGES, key=lambda c: -judges[c]["within1"]["value"]), best, f"{best}__calibrated"]
        close = [
            (judges[c]["name"].replace(" (full)", "").replace(" (full, calibrated)", ", calibrated"),
             judges[c]["within1"], LARGE if c in data.REFERENCE_JUDGES else SMALL)
            for c in picks if c in judges
        ]  # fmt: skip

    n = summary["sample"]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family=\'{FONT}\'>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="48" y="64" font-size="30" font-weight="600" fill="{INK}">How closely LLM judges match human rubric grades</text>',
        f'<text x="48" y="98" font-size="19" fill="{INK_2}">Pearson r with the human 1-5 score on {n["answers"]} answers to {n["prompts"]} BiGGen-Bench prompts. Bars are 95% intervals.</text>',
        f'<circle cx="55" cy="134" r="6" fill="{SMALL}"/><text x="70" y="140" font-size="18" fill="{INK}">Small open model, run locally (4-bit)</text>',
        f'<circle cx="405" cy="134" r="6" fill="{LARGE}"/><text x="420" y="140" font-size="18" fill="{INK}">Large judge (scores from the dataset)</text>',
    ]
    svg += panel(48, 176, 600, "Every judge, full rubric", "Higher means closer to the human grader", rows)
    if close:
        svg += panel(
            690, 176, 470, "Within one point of the human", "Calibrated: scores remapped on other prompts",
            close, label_w=210, lo=0.5, hi=1.0, step=0.1, percent=True,
        )  # fmt: skip
    svg.append("</svg>")

    html = f'<!doctype html><meta charset="utf-8"><style>html,body{{margin:0;background:{SURFACE}}}</style>{"".join(svg)}'
    (RESULTS / "agreement.svg").write_text("".join(svg), encoding="utf8")
    browser = next(
        p
        for p in [
            os.environ.get("CHROME_PATH", ""),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/usr/bin/google-chrome",
            "/usr/bin/chromium",
        ]
        if p and Path(p).exists()
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "chart.html"
        page.write_text(html, encoding="utf8")
        subprocess.run(
            [
                browser, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--user-data-dir={Path(tmp) / 'profile'}",
                f"--screenshot={RESULTS / 'agreement.png'}", page.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )  # fmt: skip
    print("wrote results/agreement.png and .svg")


if __name__ == "__main__":
    main()
