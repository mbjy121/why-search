"""
build_pdf.py

Converts TUTORIAL.md into a nicely-styled PROJECT_REPORT.pdf.
Runs once; not needed at app runtime.

    python build_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa


HERE = Path(__file__).parent
IN_PATH = HERE / "TUTORIAL.md"
OUT_PATH = HERE / "PROJECT_REPORT.pdf"


CSS = """
@page {
    size: A4;
    margin: 22mm 20mm 22mm 20mm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        left: 20mm; right: 20mm; bottom: 10mm; height: 8mm;
    }
}
body {
    font-family: Helvetica, Arial, sans-serif;
    color: #1f2937;
    font-size: 10.5pt;
    line-height: 1.55;
}
h1 {
    font-size: 22pt;
    color: #111827;
    margin-top: 18pt;
    margin-bottom: 8pt;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 4pt;
}
h2 {
    font-size: 15pt;
    color: #111827;
    margin-top: 20pt;
    margin-bottom: 6pt;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 3pt;
}
h3 {
    font-size: 12pt;
    color: #1f2937;
    margin-top: 14pt;
    margin-bottom: 4pt;
}
h4 { font-size: 11pt; color: #374151; margin-top: 10pt; }
p { margin: 4pt 0 8pt 0; }
strong { color: #111827; }
em { color: #374151; }
a { color: #2563eb; text-decoration: none; }

ul, ol { margin: 4pt 0 8pt 16pt; }
li { margin-bottom: 3pt; }

blockquote {
    margin: 8pt 0 8pt 0;
    padding: 6pt 10pt;
    background: #f8fafc;
    border-left: 3px solid #2563eb;
    color: #374151;
    font-size: 10pt;
}
blockquote p { margin: 2pt 0; }

code {
    font-family: Courier, monospace;
    background: #f3f4f6;
    color: #b91c1c;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9.5pt;
}
pre {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 6pt 8pt;
    font-family: Courier, monospace;
    font-size: 9pt;
    color: #111827;
    white-space: pre-wrap;
}
pre code {
    background: transparent;
    color: #111827;
    padding: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 12pt 0;
    font-size: 9.5pt;
}
th {
    background: #eef2ff;
    color: #111827;
    text-align: left;
    padding: 5pt 8pt;
    border: 1px solid #c7d2fe;
    font-weight: bold;
}
td {
    padding: 5pt 8pt;
    border: 1px solid #e5e7eb;
    vertical-align: top;
}
tr:nth-child(even) td { background: #fafafa; }

hr { border: none; border-top: 1px solid #e5e7eb; margin: 14pt 0; }

.doc-header {
    text-align: center;
    margin-bottom: 12pt;
    padding-bottom: 10pt;
    border-bottom: 1px solid #e5e7eb;
}
.doc-header .doc-title {
    font-size: 28pt;
    font-weight: bold;
    color: #111827;
    margin: 0;
}
.doc-header .doc-title em {
    color: #2563eb;
    font-style: normal;
}
.doc-header .doc-subtitle {
    color: #6b7280;
    font-size: 11pt;
    margin-top: 6pt;
}

#footer_content {
    font-size: 8pt;
    color: #9ca3af;
    text-align: center;
}
"""


def preprocess(md_text: str) -> str:
    """A couple of small tweaks so xhtml2pdf renders the doc cleanly."""
    # Strip anchor links like "#the-one-sentence-version" that xhtml2pdf
    # doesn't do a great job with, and drop the top-level table of contents
    # (each section still shows up with big headings).
    lines = md_text.splitlines()
    out = []
    skip_toc = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Table of contents"):
            skip_toc = True
            continue
        if skip_toc:
            if stripped.startswith("## ") and "Table of contents" not in stripped:
                skip_toc = False
            else:
                continue
        out.append(line)
    text = "\n".join(out)
    # Drop the very first H1 — we'll replace it with a styled header block
    text = re.sub(r"^# .*\n", "", text, count=1)

    # xhtml2pdf's default fonts don't cover most emoji or fancy Unicode.
    # Replace them with text-safe equivalents so the PDF stays crisp.
    replacements = {
        "✅": "[yes]",
        "❌": "[no]",
        "⚠️": "[partial]",
        "⚠": "[partial]",
        "🟠": "*",
        "🟣": "*",
        "👋": "!",
        "→": "->",
        "←": "<-",
        "—": " - ",
        "–": " - ",
        "…": "...",
        "·": "-",
        "‑": "-",  # non-breaking hyphen
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "️": "",   # emoji variation selector
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def build_html(md_text: str) -> str:
    body_html = md_lib.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
        output_format="html5",
    )
    header = (
        '<div class="doc-header">'
        '<div class="doc-title"><em>WHY</em> search</div>'
        '<div class="doc-subtitle">A tutorial for humans - plain-English walkthrough of what we built, and why</div>'
        '</div>'
    )
    footer = (
        '<div id="footer_content">'
        'WHY search &nbsp;|&nbsp; portfolio project &nbsp;|&nbsp; '
        'TF-IDF + semantic search over arXiv + GitHub'
        '</div>'
    )
    return f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
{header}
{body_html}
{footer}
</body></html>
""".strip()


def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing {IN_PATH}")
    md_text = IN_PATH.read_text(encoding="utf-8")
    md_text = preprocess(md_text)
    html = build_html(md_text)

    with open(OUT_PATH, "wb") as f:
        result = pisa.CreatePDF(src=html, dest=f, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF conversion failed with {result.err} errors.")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
