"""
app.py — T9.3 Tech News Tracker (Phase 1 + 2 + 3)
===================================================
Run:
    streamlit run app.py

Set GEMINI_API_KEY in .env for real Gemini summaries.
Flip USE_MOCK and FORCE_HYBRID below for offline development.
"""

import streamlit as st

# ─── Runtime switches ────────────────────────────────────────────────────────
USE_MOCK        = True    # True  → mock_feeds (offline)  |  False → live RSS
FORCE_HYBRID    = False    # True  → local classifier      |  False → BART download
MAX_ARTICLES    = 30
CACHE_TTL       = 1800    # seconds (30 min)

CATEGORIES = ["Technology", "Business", "Politics", "Sports", "Entertainment"]

CONFIDENCE_THRESHOLDS = {"high": 0.70, "medium": 0.50}

SOURCE_COLORS = {
    "TechCrunch": ("#e06c3a", "#e06c3a22"),
    "The Verge":  ("#7c5cbf", "#7c5cbf22"),
    "YourStory":  ("#1a7f5a", "#1a7f5a22"),
}

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T9.3 Tech News Tracker",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
.source-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 20px;
    margin-right: 6px;
    vertical-align: middle;
}
.conf-pill {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
}
.summary-line {
    border-left: 3px solid #4e8cff;
    padding-left: 10px;
    margin: 3px 0;
    font-size: 0.88rem;
}
.score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 3px 0;
}
.score-bar-bg {
    flex: 1;
    background: #ffffff15;
    border-radius: 4px;
    height: 7px;
}
.score-bar-fill {
    height: 7px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached resources ────────────────────────────────────────────────────────

@st.cache_resource
def _get_classifier():
    from classifier import get_classifier
    return get_classifier(force_hybrid=FORCE_HYBRID)

@st.cache_resource
def _get_gemini_client():
    from summariser import get_gemini_client
    return get_gemini_client()

@st.cache_data(ttl=CACHE_TTL)
def load_articles() -> list[dict]:
    """Phase 1 → 2 → 3. Re-runs at most once per CACHE_TTL seconds."""
    from pipeline_runner import run_pipeline
    return run_pipeline(
        use_mock=USE_MOCK,
        force_hybrid_classifier=FORCE_HYBRID,
        max_articles=MAX_ARTICLES,
        inter_call_sleep=4.0,
    )


# ─── Render helpers ───────────────────────────────────────────────────────────

def _conf_color(conf: float) -> tuple[str, str]:
    """(emoji, hex) for confidence float."""
    if conf >= CONFIDENCE_THRESHOLDS["high"]:
        return "🟢", "#2d6a4f"
    elif conf >= CONFIDENCE_THRESHOLDS["medium"]:
        return "🟡", "#856404"
    return "🔴", "#842029"


def _source_badge(source: str) -> str:
    fg, bg = SOURCE_COLORS.get(source, ("#888", "#88888822"))
    return (
        f'<span class="source-badge" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}55;">{source}</span>'
    )


def render_article(article: dict):
    title      = article.get("title", "Untitled")
    url        = article.get("url", "#")
    source     = article.get("source", "")
    published  = article.get("published")
    category   = article.get("category", "Unknown")
    confidence = article.get("confidence", 0.0)
    all_scores = article.get("all_scores", {})
    summary    = article.get("summary", "")
    sum_src    = article.get("summary_source", "fallback")

    conf_emoji, conf_color = _conf_color(confidence)
    date_str = published.strftime("%d %b %Y · %H:%M UTC") if published else ""

    with st.container(border=True):
        # Title + meta row
        c_title, c_badge = st.columns([5, 1])
        with c_title:
            st.markdown(f"**[{title}]({url})**")
            st.markdown(
                _source_badge(source)
                + f'<span style="font-size:11px;color:var(--text-color,#888);">{date_str}</span>',
                unsafe_allow_html=True,
            )
        with c_badge:
            st.markdown(
                f'<div style="text-align:right;padding-top:4px;">'
                f'<span class="conf-pill" '
                f'style="background:{conf_color}22;color:{conf_color};">'
                f'{conf_emoji} {confidence:.0%}</span><br>'
                f'<span style="font-size:10px;color:#888;display:block;margin-top:3px;">'
                f'{"★ Gemini" if sum_src == "gemini" else "○ Fallback"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 3-line summary
        if summary:
            lines = [l.strip() for l in summary.splitlines() if l.strip()][:3]
            st.markdown(
                "".join(f'<div class="summary-line">{l}</div>' for l in lines),
                unsafe_allow_html=True,
            )

        # Score breakdown
        if all_scores:
            with st.expander("Category confidence breakdown", expanded=False):
                for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    pct = int(score * 100)
                    fill_color = "#4e8cff" if label == category else "#666666"
                    bold = "600" if label == category else "400"
                    st.markdown(
                        f'<div class="score-bar-wrap">'
                        f'<span style="width:105px;font-size:12px;font-weight:{bold};">{label}</span>'
                        f'<div class="score-bar-bg">'
                        f'<div class="score-bar-fill" style="width:{pct}%;background:{fill_color};"></div>'
                        f'</div>'
                        f'<span style="font-size:12px;width:32px;text-align:right;">{score:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 T9.3 News Tracker")
    st.caption("All phases active")

    st.markdown("---")
    st.subheader("Pipeline")

    phases = {
        "Phase 1 · RSS":      ("✅", "Active"),
        "Phase 2 · Classify": ("✅", "Hybrid" if FORCE_HYBRID else "BART"),
        "Phase 3 · Summarise":("✅", "Gemini" if not FORCE_HYBRID else "Fallback"),
    }
    for name, (icon, status) in phases.items():
        st.markdown(f"{icon} **{name}** — {status}")

    st.markdown("---")
    st.subheader("Config")
    st.caption(f"Cache TTL: {CACHE_TTL // 60} min")
    st.caption(f"Max articles: {MAX_ARTICLES}")
    st.caption(f"Data: {'Mock feeds' if USE_MOCK else 'Live RSS'}")

    if st.button("🔄 Force refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main ────────────────────────────────────────────────────────────────────

st.title("Tech News Tracker")

with st.spinner("Running pipeline (Phase 1 → 2 → 3)…"):
    articles = load_articles()

if not articles:
    st.error("Pipeline returned no articles. Check your RSS feeds or enable mock mode.")
    st.stop()

# Stats row
total    = len(articles)
avg_conf = sum(a.get("confidence", 0) for a in articles) / total
gemini_n = sum(1 for a in articles if a.get("summary_source") == "gemini")
cats     = {a.get("category", "?") for a in articles}

m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles fetched",   total)
m2.metric("Categories found",   len(cats))
m3.metric("Avg confidence",     f"{avg_conf:.0%}")
m4.metric("Gemini summaries",   f"{gemini_n}/{total}")

st.divider()

# Tabs — one per category + All
tab_labels  = CATEGORIES + ["All"]
tab_objects = st.tabs(tab_labels)

for tab, label in zip(tab_objects, tab_labels):
    with tab:
        filtered = (
            articles if label == "All"
            else [a for a in articles if a.get("category") == label]
        )

        if not filtered:
            st.info(f"No articles classified as **{label}** in this batch.")
            continue

        st.caption(f"{len(filtered)} article{'s' if len(filtered) != 1 else ''} · newest first")
        for article in filtered:
            render_article(article)