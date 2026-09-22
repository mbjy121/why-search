"""
app.py — WHY Search Streamlit UI.

Landing: big centered "WHY search" wordmark + Google-style rounded input
with the filter icon nested inside the input's right edge.

After a query: the wordmark shrinks to the top-left and shares a single row
with a smaller version of the same input. Results appear underneath.

Filter popover:
    * Search mode: Keyword (TF-IDF), Meaning (semantic), Hybrid (both)
    * Source:      Both / Papers / Repos
    * Results:     up to 20

When the user picks Hybrid, results render side-by-side in two columns
(Keyword | Meaning) so they can eyeball the difference. Score is shown as
a small circular gauge — color goes red → orange → yellow → green as the
match gets stronger relative to the top hit.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from search import SearchIndex, load_dataset


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WHY search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide default Streamlit chrome
HIDE_CHROME = """
<style>
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
[data-testid="stAppViewContainer"] { padding-top: 0; }
.block-container { padding-top: 0.5rem !important; max-width: 1200px; }
</style>
"""
st.markdown(HIDE_CHROME, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Design tokens + all component CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
    --why-fg: #1f2937;
    --why-muted: #6b7280;
    --why-bd: #dfe1e5;                    /* google's input border */
    --why-bd-focus: #4285f4;
    --why-bg: #ffffff;
    --why-accent: #2563eb;
    --why-mark: #fde68a;
    --why-mark-fg: #1f2937;
    --why-card-bg: #ffffff;
    --why-card-bd: #e5e7eb;
    --why-chip-bg: #fef3c7;
    --why-chip-fg: #92400e;
    --why-chip-bd: #fde68a;
    --why-shadow-input: 0 1px 6px rgba(32,33,36,0.08);
    --why-shadow-input-focus: 0 1px 12px rgba(32,33,36,0.18);
    --why-shadow-card: 0 1px 3px rgba(0,0,0,0.04);
}
@media (prefers-color-scheme: dark) {
    :root {
        --why-fg: #f3f4f6;
        --why-muted: #9ca3af;
        --why-bd: #3f4147;
        --why-bd-focus: #60a5fa;
        --why-bg: #202124;
        --why-accent: #60a5fa;
        --why-mark: #facc15;
        --why-mark-fg: #111827;
        --why-card-bg: #24262b;
        --why-card-bd: #3f4147;
        --why-chip-bg: #78350f55;
        --why-chip-fg: #fde68a;
        --why-chip-bd: #92400e;
    }
}

/* -------- Landing hero (before any search) -------- */
.why-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding-top: 12vh;
    animation: whyHeroIn 0.6s ease both;
}
.why-hero .why-brand {
    font-size: 5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0 0 10px 0;
    color: var(--why-fg);
}
.why-hero .why-brand em {
    font-style: normal;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.why-hero .why-tag {
    color: var(--why-muted);
    font-size: 1rem;
    margin: 0 0 26px 0;
}
@keyframes whyHeroIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* -------- Landing search wrapper: Google-style pill + inline filter -------- */
.why-searchwrap-landing [data-testid="stTextInput"] input,
.why-searchwrap-results  [data-testid="stTextInput"] input {
    border-radius: 999px !important;
    background: var(--why-bg) !important;
    color: var(--why-fg) !important;
    border: 1px solid var(--why-bd) !important;
    box-shadow: var(--why-shadow-input) !important;
    padding-left: 24px !important;
    padding-right: 60px !important;
    font-size: 1rem !important;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
}
.why-searchwrap-landing [data-testid="stTextInput"] input {
    height: 56px !important;
    font-size: 1.1rem !important;
    padding-left: 28px !important;
    padding-right: 68px !important;
}
.why-searchwrap-landing [data-testid="stTextInput"] input:hover,
.why-searchwrap-results  [data-testid="stTextInput"] input:hover {
    box-shadow: var(--why-shadow-input-focus) !important;
    border-color: rgba(66,133,244,0.5) !important;
}
.why-searchwrap-landing [data-testid="stTextInput"] input:focus,
.why-searchwrap-results  [data-testid="stTextInput"] input:focus {
    outline: none !important;
    border-color: var(--why-bd-focus) !important;
    box-shadow: var(--why-shadow-input-focus) !important;
}
/* Kill Streamlit's own input container border/bg so ours shows through */
.why-searchwrap-landing [data-testid="stTextInput"] > div,
.why-searchwrap-results  [data-testid="stTextInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Tighten the input+filter horizontal block so the input and gear feel like one pill */
[data-testid="stHorizontalBlock"]:has([data-testid="stPopover"]) {
    gap: 6px !important;
    align-items: center !important;
}

/* The one and only popover button on the page = the filter icon */
[data-testid="stPopoverButton"] {
    border-radius: 999px !important;
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    padding: 0 !important;
    background-color: var(--why-bg) !important;
    border: 1px solid var(--why-bd) !important;
    box-shadow: var(--why-shadow-input) !important;
    color: var(--why-muted) !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 3H2l8 9.46V19l4 2v-8.54L22 3z'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 22px 22px !important;
}
[data-testid="stPopoverButton"]:hover {
    box-shadow: var(--why-shadow-input-focus) !important;
    color: var(--why-accent) !important;
    border-color: rgba(66,133,244,0.5) !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%232563eb' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M22 3H2l8 9.46V19l4 2v-8.54L22 3z'/></svg>") !important;
}
/* Hide the button's own text label AND the Material Icons chevron */
[data-testid="stPopoverButton"] p,
[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
    display: none !important;
}

/* -------- Compact header row (after search) -------- */
.why-header-row {
    display: flex;
    align-items: center;
    gap: 26px;
    padding: 14px 0 10px 0;
    animation: whyHeaderIn 0.5s cubic-bezier(.2,.7,.2,1) both;
}
.why-brand-mini {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--why-fg);
    white-space: nowrap;
}
.why-brand-mini em {
    font-style: normal;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
@keyframes whyHeaderIn {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}

.why-results-meta {
    color: var(--why-muted);
    font-size: 0.85rem;
    margin: 6px 0 16px 0;
    animation: whyHeaderIn 0.4s ease both;
}

/* -------- "How it's built" video card (top-right of landing) -------- */
.why-topbar {
    display: flex;
    justify-content: flex-end;
    padding: 8px 4px 0 4px;
    animation: whyHeaderIn 0.5s ease both;
}
/* Style the underlying Streamlit button as a card. Streamlit adds
   `st-key-<button_key>` to the element container; we target it. */
.st-key-btn_how_built button {
    display: inline-flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 8px 16px 8px 8px !important;
    border-radius: 999px !important;
    border: 1px solid var(--why-card-bd) !important;
    background: var(--why-bg) !important;
    color: var(--why-fg) !important;
    box-shadow: var(--why-shadow-card) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
    width: auto !important;
    min-height: 40px !important;
    line-height: 1 !important;
    float: right;
}
.st-key-btn_how_built button:hover {
    transform: translateY(-1px);
    box-shadow: var(--why-shadow-input-focus) !important;
    border-color: rgba(66,133,244,0.5) !important;
    color: var(--why-accent) !important;
}
/* Streamlit's button text lives inside <div class="stMarkdown"><p>...</p></div>.
   We add the play-icon circle as ::before on the button itself. */
.st-key-btn_how_built button::before {
    content: "";
    display: inline-block;
    width: 26px;
    height: 26px;
    border-radius: 999px;
    background-image:
        url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'><polygon points='8,5 19,12 8,19'/></svg>"),
        linear-gradient(135deg, #2563eb, #7c3aed);
    background-repeat: no-repeat, no-repeat;
    background-position: 60% center, 0 0;
    background-size: 11px 11px, cover;
    flex-shrink: 0;
}
.st-key-btn_how_built {
    display: flex;
    justify-content: flex-end;
}

/* Video dialog frame */
.why-video-dialog iframe {
    width: 100%;
    border: 0;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    background: #000;
}

/* -------- Result cards -------- */
.why-card {
    border: 1px solid var(--why-card-bd);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    background: var(--why-card-bg);
    box-shadow: var(--why-shadow-card);
    animation: whyCardIn 0.4s ease both;
    display: grid;
    grid-template-columns: 46px 1fr;
    gap: 14px;
    align-items: start;
}
.why-card:nth-child(1) { animation-delay: 0.02s; }
.why-card:nth-child(2) { animation-delay: 0.06s; }
.why-card:nth-child(3) { animation-delay: 0.10s; }
.why-card:nth-child(4) { animation-delay: 0.14s; }
.why-card:nth-child(5) { animation-delay: 0.18s; }
.why-card:nth-child(n+6) { animation-delay: 0.22s; }
@keyframes whyCardIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.why-card-body { min-width: 0; }        /* let overflow work in flex/grid */

.why-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-right: 8px;
    vertical-align: middle;
}
.why-badge.paper { background: #2563eb22; color: #2563eb; border: 1px solid #2563eb55; }
.why-badge.repo  { background: #16a34a22; color: #16a34a; border: 1px solid #16a34a55; }

.why-title   {
    font-size: 1.02rem; font-weight: 700; margin: 4px 0 6px 0; color: var(--why-fg);
    display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
}
.why-title a { color: var(--why-accent); text-decoration: none; font-size: 0.82rem; font-weight: 500; }
.why-title a:hover { text-decoration: underline; }

/* 3-line description clamp */
.why-snippet {
    font-size: 0.92rem;
    line-height: 1.5;
    color: var(--why-fg);
    opacity: 0.88;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
.why-meta    { color: var(--why-muted); font-size: 0.8rem; margin-top: 8px; }
.why-chip {
    display: inline-block;
    padding: 2px 8px;
    margin: 2px 4px 2px 0;
    border-radius: 6px;
    font-size: 0.76rem;
    background: var(--why-chip-bg);
    color: var(--why-chip-fg);
    border: 1px solid var(--why-chip-bd);
}

mark.why-hl {
    background: var(--why-mark);
    color: var(--why-mark-fg);
    padding: 0 3px;
    border-radius: 3px;
}

/* Column headers when we render Hybrid side-by-side */
.why-col-heading {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--why-muted);
    padding: 4px 4px 10px 4px;
    border-bottom: 1px solid var(--why-card-bd);
    margin-bottom: 12px;
}
.why-col-heading .why-col-dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 999px;
    vertical-align: middle; margin-right: 8px;
}
.why-col-heading.keyword .why-col-dot { background: #f59e0b; }
.why-col-heading.meaning .why-col-dot { background: #7c3aed; }

/* Circular score gauge */
.why-gauge {
    width: 44px; height: 44px; position: relative;
    flex-shrink: 0;
}
.why-gauge svg { transform: rotate(-90deg); display: block; }
.why-gauge .why-gauge-num {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700; color: var(--why-fg);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load index once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_index() -> SearchIndex:
    entries = load_dataset()
    return SearchIndex(entries)


with st.spinner("Warming up the search engine ..."):
    index = get_index()

SEMANTIC_ON = bool(getattr(index, "semantic_available", lambda: False)())
N_PAPERS = sum(1 for e in index.entries if e["source_type"] == "paper")
N_REPOS = sum(1 for e in index.entries if e["source_type"] == "repo")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def highlight(text: str, terms: list[str]) -> str:
    if not text:
        return ""
    escaped = html.escape(text)
    for term in sorted(terms, key=len, reverse=True):
        if not term:
            continue
        if " " in term:
            pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
        else:
            pattern = re.compile(r"\b" + re.escape(html.escape(term)) + r"\b", re.IGNORECASE)
        escaped = pattern.sub(
            lambda m: f'<mark class="why-hl">{m.group(0)}</mark>', escaped
        )
    return escaped


def gauge_svg(fraction: float) -> str:
    """A tiny circular progress gauge; color goes red → orange → yellow → green."""
    f = max(0.0, min(1.0, fraction))
    # Color by strength (relative to top result in the set)
    if   f >= 0.75: color = "#16a34a"   # green
    elif f >= 0.50: color = "#84cc16"   # lime
    elif f >= 0.25: color = "#f59e0b"   # amber
    else:            color = "#ef4444"  # red
    pct_label = f"{int(round(f * 100))}"
    r = 18
    c = 2 * 3.14159265 * r
    dash = c * f
    gap = c - dash
    return (
        f'<div class="why-gauge">'
        f'<svg width="44" height="44" viewBox="0 0 44 44">'
        f'<circle cx="22" cy="22" r="{r}" stroke="#e5e7eb" stroke-width="4" fill="none"/>'
        f'<circle cx="22" cy="22" r="{r}" stroke="{color}" stroke-width="4" fill="none" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.2f} {gap:.2f}"/>'
        f'</svg>'
        f'<div class="why-gauge-num">{pct_label}</div>'
        f'</div>'
    )


def render_card(r, top_score: float) -> str:
    badge_cls = "paper" if r.source_type == "paper" else "repo"
    badge_txt = "PAPER" if r.source_type == "paper" else "REPO"
    title_html = html.escape(r.title or "(untitled)")
    snippet_html = highlight(r.snippet or "", r.matched_words)
    chips = "".join(
        f'<span class="why-chip">{html.escape(w)}</span>' for w in r.matched_words
    ) or '<span class="why-meta">matched by meaning</span>'
    url = html.escape(r.url or "")
    link = f'<a href="{url}" target="_blank" rel="noopener noreferrer">open ↗</a>' if url else ""
    frac = (r.score / top_score) if top_score > 0 else 0.0
    return (
        f'<div class="why-card">'
        f'{gauge_svg(frac)}'
        f'<div class="why-card-body">'
        f'<div><span class="why-badge {badge_cls}">{badge_txt}</span></div>'
        f'<div class="why-title">{title_html} {link}</div>'
        f'<div class="why-snippet">{snippet_html}</div>'
        f'<div style="margin-top:8px;">'
        f'<em style="color:var(--why-muted);font-size:0.76rem;">why it matched:</em> {chips}'
        f'</div>'
        f'</div>'
        f'</div>'
    )


def render_column(results: list, heading: str, dot_class: str) -> str:
    top = max((r.score for r in results), default=0.0)
    cards = "".join(render_card(r, top) for r in results) if results else \
        '<div class="why-meta">No matches with this mode.</div>'
    return (
        f'<div class="why-col-heading {dot_class}">'
        f'<span class="why-col-dot"></span>{html.escape(heading)}'
        f'</div>'
        f'<div>{cards}</div>'
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULT_MODE = "Hybrid" if SEMANTIC_ON else "Keyword"
state = st.session_state
state.setdefault("query", "")
state.setdefault("source_choice", "Both")
state.setdefault("mode", DEFAULT_MODE)
state.setdefault("top_k", 10)
# Seed the input widget's own state so it renders with state.query when the
# script is invoked mid-session (widget state is separate from state.query).
state.setdefault("_search_input", state.query)


def _submit_query():
    """Fired when the search input loses focus (Tab or Enter).
    Reads whatever is currently in the input widget and stores it as
    state.query. Streamlit reruns automatically after this returns."""
    state.query = state.get("_search_input", "").strip()


VIDEO_URL = "https://whysearch-film.vercel.app"


@st.dialog("How it's built", width="large")
def _show_video_dialog():
    """In-page modal showing the 'how it was built' walkthrough video."""
    st.markdown('<div class="why-video-dialog">', unsafe_allow_html=True)
    st.components.v1.iframe(VIDEO_URL, height=560, scrolling=True)
    st.markdown(
        f"<div style='margin-top:8px;color:var(--why-muted);font-size:0.8rem;'>"
        f"Trouble loading the player? "
        f"<a href='{VIDEO_URL}' target='_blank' rel='noopener'>Open in a new tab -></a>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_filter_popover(key: str):
    with st.popover("⚙", use_container_width=True, help="Filter and mode"):
        st.markdown("**Search mode**")
        mode_opts = ["Keyword", "Meaning", "Hybrid"] if SEMANTIC_ON else ["Keyword"]
        current_mode = state.mode if state.mode in mode_opts else mode_opts[0]
        state.mode = st.radio(
            "mode",
            options=mode_opts,
            index=mode_opts.index(current_mode),
            horizontal=True,
            label_visibility="collapsed",
            key=f"mode_{key}",
            help="Keyword = TF-IDF (word overlap). Meaning = sentence-transformers (semantic). Hybrid = both, side by side.",
        )
        if not SEMANTIC_ON:
            st.caption("Semantic mode is disabled — run `python build_embeddings.py` to enable it.")
        st.markdown("**Source**")
        state.source_choice = st.radio(
            "source",
            options=["Both", "Papers", "Repos"],
            index=["Both", "Papers", "Repos"].index(state.source_choice),
            horizontal=True,
            label_visibility="collapsed",
            key=f"src_{key}",
        )
        st.markdown("**Results**")
        state.top_k = st.slider(
            "top_k",
            min_value=1,
            max_value=20,
            value=state.top_k,
            label_visibility="collapsed",
            key=f"topk_{key}",
        )


has_query = bool(state.query.strip())


# ---------------------------------------------------------------------------
# Big hero — landing only
# ---------------------------------------------------------------------------
if not has_query:
    # Top-right "How it's built" card. Sits above the hero, floats to the right.
    st.markdown('<div class="why-topbar"><div class="why-video-slot">', unsafe_allow_html=True)
    _tl, _tr = st.columns([5, 2])
    with _tr:
        if st.button("How it's built", key="btn_how_built", use_container_width=False):
            _show_video_dialog()
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="why-hero">'
        '<div class="why-brand"><em>WHY</em> search</div>'
        '<div class="why-tag">Search across arXiv papers &amp; GitHub repos — see <em>why</em> every result matched.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Search bar row — SAME single input widget in both states.
# Only its surrounding layout changes. Using one widget with one key across
# both states means Streamlit can never render two copies at once.
# ---------------------------------------------------------------------------
wrap_class = "why-searchwrap-results" if has_query else "why-searchwrap-landing"

if has_query:
    c_brand, c_input, _c_pad = st.columns([2, 6, 1])
    with c_brand:
        st.markdown(
            '<div class="why-header-row"><div class="why-brand-mini"><em>WHY</em> search</div></div>',
            unsafe_allow_html=True,
        )
    input_container = c_input
    top_margin = "margin-top:14px;"
else:
    _l, input_container, _r = st.columns([1, 3, 1])
    top_margin = ""

with input_container:
    st.markdown(f'<div class="{wrap_class}" style="{top_margin}">', unsafe_allow_html=True)
    c_field, c_gear = st.columns([10, 1])
    with c_field:
        st.text_input(
            "search",
            placeholder="Ask anything — try 'conversational search' or 'vector database'",
            label_visibility="collapsed",
            key="_search_input",
            on_change=_submit_query,
        )
    with c_gear:
        _render_filter_popover("main")
    st.markdown('</div>', unsafe_allow_html=True)

if not has_query:
    # Stats line under the landing input
    _l, mid, _r = st.columns([1, 3, 1])
    with mid:
        st.markdown(
            f"<div style='text-align:center;color:var(--why-muted);"
            f"font-size:0.82rem;margin-top:14px;'>"
            f"{len(index.entries):,} documents indexed · "
            f"{N_PAPERS:,} papers · {N_REPOS:,} repos"
            f"{' · semantic ✓' if SEMANTIC_ON else ' · semantic ✗ (run build_embeddings.py)'}"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# Run search(es)
src_filter = {"Both": None, "Papers": "paper", "Repos": "repo"}[state.source_choice]

if state.mode == "Keyword":
    kw_results = index.search(state.query, top_k=state.top_k, source_type=src_filter)
    sem_results = []
elif state.mode == "Meaning":
    kw_results = []
    sem_results = index.semantic_search(state.query, top_k=state.top_k, source_type=src_filter)
else:  # Hybrid — show both side by side
    kw_results = index.search(state.query, top_k=state.top_k, source_type=src_filter)
    sem_results = index.semantic_search(state.query, top_k=state.top_k, source_type=src_filter)

# Meta line
total = len(kw_results) + len(sem_results)
mode_label = {"Keyword": "keyword only", "Meaning": "meaning only", "Hybrid": "keyword + meaning"}[state.mode]
st.markdown(
    f"<div class='why-results-meta'>{total} result{'s' if total != 1 else ''} for "
    f"<b>{html.escape(state.query)}</b> · {mode_label} · source: {state.source_choice.lower()}</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Render single- or two-column layout
# ---------------------------------------------------------------------------
if state.mode == "Hybrid":
    col_kw, col_sem = st.columns(2, gap="large")
    with col_kw:
        st.markdown(render_column(kw_results, "Keyword match — TF-IDF", "keyword"), unsafe_allow_html=True)
    with col_sem:
        st.markdown(render_column(sem_results, "Meaning match — semantic", "meaning"), unsafe_allow_html=True)
else:
    results = kw_results if state.mode == "Keyword" else sem_results
    if not results:
        st.warning(
            f"No matches for {state.query!r} with these filters. Try broader words or switch source to Both."
        )
        st.stop()
    top = max(r.score for r in results)
    cards = "".join(render_card(r, top) for r in results)
    st.markdown(cards, unsafe_allow_html=True)
