"""
app.py — T9.3 Tech News Tracker (Phase 1 + 2 + on-demand Phase 3)
"""

import base64
import html as _html
import os
import streamlit as st

# ─── Runtime switches ─────────────────────────────────────────────────────────
USE_MOCK     = False
FORCE_HYBRID = False
MAX_ARTICLES = 30
CACHE_TTL    = 1800

CATEGORIES = ["Technology", "Business", "Politics", "Sports", "Entertainment"]

SOURCE_COLORS = {
    "TechCrunch": ("#FF6B35", "#FF6B3514"),
    "The Verge":  ("#9B59B6", "#9B59B614"),
    "YourStory":  ("#1ABC9C", "#1ABC9C14"),
}

CATEGORY_ICONS = {
    "Technology": "⚡",
    "Business": "📈",
    "Politics": "🏛",
    "Sports": "🏆",
    "Entertainment": "🎬",
    "All": "◈",
}

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech News Tracker",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #07071a !important;
}
[data-testid="stMain"] { 
    background: #07071a;
    width: 75% !important;
    max-width: 75% !important;
    padding-right: 0 !important;
}
[data-testid="stAppViewContainer"] {
    width: 100% !important;
    display: flex;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0c22 !important;
    border-right: 1px solid #ffffff0a !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.8rem; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #11013d 0%, #06163b 55%, #040e22 100%);
    border: 1px solid #ffffff0c;
    border-radius: 18px;
    padding: 30px 38px 28px;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, #7c3aed28 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -50px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, #0ea5e920 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    background: linear-gradient(100deg, #c4b5fd 0%, #7dd3fc 55%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px;
    letter-spacing: -0.6px;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.82rem;
    color: #64748b;
    letter-spacing: 0.03em;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7c3aed;
    background: #7c3aed14;
    border: 1px solid #7c3aed30;
    border-radius: 20px;
    padding: 3px 10px;
    margin-bottom: 10px;
}

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
}
.metric-card {
    flex: 1;
    background: #0e0e28;
    border: 1px solid #ffffff08;
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color 0.25s, background 0.25s;
}
.metric-card:hover { border-color: #7c3aed30; background: #100e2a; }
.metric-val {
    font-size: 1.75rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #c4b5fd, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.metric-lbl {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
}

/* ── Article card ── */
.art-card {
    background: #0c0c26;
    border: 1px solid #ffffff07;
    border-radius: 16px;
    padding: 20px 24px 18px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
    position: relative;
}
.art-card:hover { border-color: #7c3aed25; }
.art-title {
    font-size: 0.97rem;
    font-weight: 600;
    line-height: 1.45;
    color: #e2e8f0;
    text-decoration: none;
    display: block;
    margin-bottom: 7px;
    transition: color 0.15s;
}
.art-title:hover { color: #c4b5fd; }

/* ── Badges ── */
.source-badge {
    display: inline-block;
    font-size: 9.5px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 20px;
    margin-right: 6px;
    vertical-align: middle;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.cat-chip {
    display: inline-block;
    font-size: 9.5px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    background: #ffffff08;
    color: #64748b;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    vertical-align: middle;
    margin-right: 4px;
}
.date-text {
    font-size: 10.5px;
    color: #475569;
    vertical-align: middle;
}

/* ── Confidence pill ── */
.conf-wrap { text-align: right; padding-top: 2px; }
.conf-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.conf-cat {
    font-size: 10px;
    color: #475569;
    margin-top: 4px;
    font-weight: 500;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Summary ── */
.summary-wrap {
    background: linear-gradient(135deg, #0a1535 0%, #0d1a3e 100%);
    border-left: 3px solid #7c3aed;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 12px 0 8px;
    overflow: visible !important;
    width: 100% !important;
    max-width: none !important;
}
.summary-para {
    font-size: 0.87rem;
    line-height: 1.75;
    color: #94a3b8;
    margin: 0 0 9px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    word-break: break-word;
    white-space: normal;
    overflow: visible !important;
    max-width: none !important;
}
.summary-para:last-child { margin-bottom: 0; }
.summary-meta {
    display: flex;
    gap: 14px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #ffffff08;
}
.summary-tag {
    font-size: 10px;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Score bars ── */
.score-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
.score-label { width:110px; font-size:11.5px; color:#64748b; font-weight:400; }
.score-label.active { color:#c4b5fd; font-weight:700; }
.score-track { flex:1; height:5px; background:#ffffff08; border-radius:4px; overflow:hidden; }
.score-fill { height:100%; border-radius:4px; }
.score-pct { font-size:11px; width:34px; text-align:right; color:#475569; }

/* ── Sidebar logo & sections ── */
.sb-logo {
    font-size: 1.15rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c4b5fd, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
    display: block;
    margin-bottom: 4px;
}
.sb-tagline {
    font-size: 0.7rem;
    color: #334155;
    letter-spacing: 0.04em;
    margin-bottom: 18px;
}
.sb-section {
    background: #ffffff04;
    border: 1px solid #ffffff07;
    border-radius: 12px;
    padding: 13px 15px;
    margin: 10px 0;
}
.sb-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #334155;
    margin-bottom: 10px;
}
.pipeline-row {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 5px 0;
}
.pipe-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.pipe-text { font-size: 0.78rem; color: #64748b; }
.pipe-text strong { color: #94a3b8; font-weight: 600; }
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
}
.info-key { font-size: 0.73rem; color: #334155; }
.info-val { font-size: 0.73rem; color: #64748b; font-weight: 500; }

/* ── Tab strip ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid #ffffff08;
    padding-bottom: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.8rem;
    font-weight: 500;
    color: #475569 !important;
    border-radius: 8px 8px 0 0;
    padding: 7px 16px;
    transition: color 0.15s, background 0.15s;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #94a3b8 !important;
    background: #ffffff06;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c4b5fd !important;
    font-weight: 700;
    background: #7c3aed12 !important;
}

/* ── Article count caption ── */
.art-count {
    font-size: 0.73rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 14px;
}

/* ── Spinner / info ── */
[data-testid="stInfo"] {
    background: #0a1535 !important;
    border-color: #7c3aed44 !important;
    border-radius: 10px;
    color: #64748b !important;
    font-size: 0.83rem;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.18s;
}

/* ── Dividers ── */
hr { border-color: #ffffff07 !important; margin: 18px 0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #ffffff07 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.78rem !important;
    color: #475569 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────
if "summaries" not in st.session_state:
    st.session_state.summaries = {}
if "summarising" not in st.session_state:
    st.session_state.summarising = set()


# ─── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource
def _get_gemini_client():
    from summariser import get_gemini_client
    return get_gemini_client()


@st.cache_data(ttl=CACHE_TTL)
def load_articles() -> list[dict]:
    if USE_MOCK:
        from mock_feeds import fetch_articles
    else:
        from rss_parser import fetch_articles
    from classifier import classify_articles
    articles = fetch_articles(max_articles=MAX_ARTICLES)
    articles = classify_articles(articles, force_hybrid=FORCE_HYBRID)
    return articles


# ─── On-demand summarisation ──────────────────────────────────────────────────

def fetch_summary_for(article: dict) -> None:
    url = article.get("url", article["title"])
    from summariser import summarise_article, get_gemini_client
    client = _get_gemini_client()
    scratch = {
        "title":       article.get("title", ""),
        "description": article.get("description", ""),
        "url":         article.get("url", ""),
        "source":      article.get("source", ""),
    }
    summarise_article(scratch, client=client)
    st.session_state.summaries[url] = {
        "text":        scratch.get("summary", ""),
        "source":      scratch.get("summary_source", "fallback"),
        "body_source": scratch.get("body_source", "rss_description"),
    }


# ─── Render helpers ───────────────────────────────────────────────────────────

def _conf_color(conf: float):
    if conf >= 0.70: return "#10b981", "#10b98120"   # green
    if conf >= 0.50: return "#f59e0b", "#f59e0b20"   # amber
    return "#ef4444", "#ef444420"                     # red


def _source_badge(source: str) -> str:
    fg, bg = SOURCE_COLORS.get(source, ("#64748b", "#64748b14"))
    return (
        f'<span class="source-badge" '
        f'style="background:{bg};color:{fg};border:1px solid {fg}44;">'
        f'{source}</span>'
    )


def render_article(article: dict, idx) -> None:
    title      = article.get("title", "Untitled")
    url        = article.get("url", title)
    link       = article.get("url", "#")
    source     = article.get("source", "")
    published  = article.get("published")
    category   = article.get("category", "Unknown")
    confidence = article.get("confidence", 0.0)
    all_scores = article.get("all_scores", {})

    text_col, bg_col = _conf_color(confidence)
    date_str  = published.strftime("%d %b %Y · %H:%M UTC") if published else ""
    cat_icon  = CATEGORY_ICONS.get(category, "")

    cached_summary = st.session_state.summaries.get(url)
    is_summarising = url in st.session_state.summarising

    with st.container(border=False):
        st.markdown('<div class="art-card">', unsafe_allow_html=True)

        # Title + meta row
        col_title, col_conf = st.columns([5, 1])
        with col_title:
            st.markdown(
                f'<a class="art-title" href="{link}" target="_blank">{title}</a>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _source_badge(source)
                + f'<span class="cat-chip">{cat_icon} {category}</span>'
                + f'<span class="date-text">{date_str}</span>',
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(
                f'<div class="conf-wrap">'
                f'<span class="conf-pill" style="background:{bg_col};color:{text_col};">'
                f'{confidence:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Summary / button
        if cached_summary:
            src_tag  = "Gemini" if cached_summary["source"] == "gemini" else "Fallback"
            body_tag = {
                "full_article":    "Full article",
                "rss_description": "RSS description",
            }.get(cached_summary.get("body_source", ""), "")

            paragraphs = [p.strip() for p in cached_summary["text"].split("\n\n") if p.strip()]
            paras_html = "".join(f'<p class="summary-para">{_html.escape(p)}</p>' for p in paragraphs)

            meta_html = (
                f'<div class="summary-meta">'
                f'<span class="summary-tag">★ {src_tag}</span>'
                + (f'<span class="summary-tag">· {body_tag}</span>' if body_tag else "")
                + f'</div>'
            )

            st.markdown(
                f'<div class="summary-wrap">{paras_html}{meta_html}</div>',
                unsafe_allow_html=True,
            )

        elif is_summarising:
            st.info("Fetching full article + calling Gemini…", icon="⏳")

        else:
            if st.button(
                "✦ Summarise",
                key=f"sum_btn_{idx}",
                help="Fetch the full article and generate a Gemini summary",
            ):
                st.session_state.summarising.add(url)
                with st.spinner("Fetching article + calling Gemini…"):
                    fetch_summary_for(article)
                st.session_state.summarising.discard(url)
                st.rerun()

        # Score breakdown
        if all_scores:
            with st.expander("Score breakdown", expanded=False):
                for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    pct   = int(score * 100)
                    is_top = label == category
                    fill  = "linear-gradient(90deg,#7c3aed,#6d28d9)" if is_top else "#1e293b"
                    cls   = "score-label active" if is_top else "score-label"
                    st.markdown(
                        f'<div class="score-row">'
                        f'<span class="{cls}">{label}</span>'
                        f'<div class="score-track">'
                        f'<div class="score-fill" style="width:{pct}%;background:{fill};"></div>'
                        f'</div>'
                        f'<span class="score-pct">{score:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown('</div>', unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<span class="sb-logo">📡 News Tracker</span>'
        '<div class="sb-tagline">T9.3 · Real-time RSS + AI Classification</div>',
        unsafe_allow_html=True,
    )

    # Pipeline status
    st.markdown(
        '<div class="sb-section">'
        '<div class="sb-section-title">Pipeline</div>'
        '<div class="pipeline-row">'
        '<div class="pipe-dot" style="background:#10b981;box-shadow:0 0 6px #10b98160;"></div>'
        '<span class="pipe-text"><strong>Phase 1</strong> · RSS parsed</span></div>'
        '<div class="pipeline-row">'
        '<div class="pipe-dot" style="background:#10b981;box-shadow:0 0 6px #10b98160;"></div>'
        '<span class="pipe-text"><strong>Phase 2</strong> · Classified</span></div>'
        '<div class="pipeline-row">'
        '<div class="pipe-dot" style="background:#7c3aed;box-shadow:0 0 6px #7c3aed60;"></div>'
        '<span class="pipe-text"><strong>Phase 3</strong> · On-demand / Gemini</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Config info
    st.markdown(
        '<div class="sb-section">'
        '<div class="sb-section-title">Config</div>'
        f'<div class="info-row"><span class="info-key">Source</span>'
        f'<span class="info-val">{"Mock" if USE_MOCK else "Live RSS"}</span></div>'
        f'<div class="info-row"><span class="info-key">Classifier</span>'
        f'<span class="info-val">{"Hybrid" if FORCE_HYBRID else "BART"}</span></div>'
        f'<div class="info-row"><span class="info-key">Cache TTL</span>'
        f'<span class="info-val">{CACHE_TTL // 60} min</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⟳  Refresh articles", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

# Hero header
st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">● Live</div>'
    '<div class="hero-title">Tech News Tracker</div>'
    '<div class="hero-sub">Real-time RSS · AI Classification · Gemini Summaries</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Top right pane image (full height)
if os.path.exists("modi.jpeg"):
    with open("modi.jpeg", "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f'<div style="position: fixed; top: 0; right: 0; z-index: 999; width: 25%; height: 100vh; overflow: hidden; box-shadow: -4px 0 12px rgba(0, 0, 0, 0.5);">'
        f'<img src="data:image/jpeg;base64,{img_base64}" style="width: 100%; height: 100%; object-fit: cover;">'
        f'</div>',
        unsafe_allow_html=True,
    )

with st.spinner("Fetching and classifying articles…"):
    articles = load_articles()

if not articles:
    st.error("No articles loaded. Check your RSS feeds or enable mock mode.")
    st.stop()

total    = len(articles)
avg_conf = sum(a.get("confidence", 0) for a in articles) / total
cats     = {a.get("category", "?") for a in articles}
n_sum    = len(st.session_state.summaries)

# Metric strip
st.markdown(
    f'<div class="metric-strip">'
    f'<div class="metric-card"><div class="metric-val">{total}</div>'
    f'<div class="metric-lbl">Articles</div></div>'
    f'<div class="metric-card"><div class="metric-val">{len(cats)}</div>'
    f'<div class="metric-lbl">Categories</div></div>'
    f'<div class="metric-card"><div class="metric-val">{avg_conf:.0%}</div>'
    f'<div class="metric-lbl">Avg Confidence</div></div>'
    f'<div class="metric-card"><div class="metric-val">{n_sum}/{total}</div>'
    f'<div class="metric-lbl">Summarised</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Category tabs
tab_labels  = [f"{CATEGORY_ICONS.get(c, '')} {c}" for c in CATEGORIES] + ["◈ All"]
tab_raw     = CATEGORIES + ["All"]
tab_objects = st.tabs(tab_labels)

for tab, label in zip(tab_objects, tab_raw):
    with tab:
        filtered = (
            articles if label == "All"
            else [a for a in articles if a.get("category") == label]
        )

        if not filtered:
            st.info(f"No articles classified as **{label}** in this batch.")
            continue

        count_word = "article" if len(filtered) == 1 else "articles"
        st.markdown(
            f'<div class="art-count">{len(filtered)} {count_word} · newest first</div>',
            unsafe_allow_html=True,
        )

        for i, article in enumerate(filtered):
            render_article(article, f"{label}_{i}")
