"""
app.py — T9.3 Tech News Tracker (Phase 1 + 2 + on-demand Phase 3)
==================================================================
Articles load immediately (Phase 1 RSS + Phase 2 classification).
Phase 3 summarisation is on-demand: click "Summarise" next to any
article to fetch just that one summary from Gemini.

Summary state is stored in st.session_state so it survives tab
switches and doesn't reset on every Streamlit rerun.

Run:
    streamlit run app.py

Set GEMINI_API_KEY in .env for real Gemini summaries.
"""

import streamlit as st

# ─── Runtime switches ─────────────────────────────────────────────────────────
USE_MOCK     = False  # True → mock_feeds (offline) | False → live RSS
FORCE_HYBRID = True   # True → local classifier     | False → BART download
MAX_ARTICLES = 30
CACHE_TTL    = 1800   # seconds (30 min)

CATEGORIES = ["Technology", "Business", "Politics", "Sports", "Entertainment"]

SOURCE_COLORS = {
    "TechCrunch": ("#e06c3a", "#e06c3a18"),
    "The Verge":  ("#7c5cbf", "#7c5cbf18"),
    "YourStory":  ("#1a7f5a", "#1a7f5a18"),
}

# ─── Page config ──────────────────────────────────────────────────────────────
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
.summary-para {
    border-left: 3px solid #4e8cff;
    padding-left: 10px;
    margin: 6px 0;
    font-size: 0.9rem;
    line-height: 1.65;
    color: inherit;
}
.score-bar-wrap { display:flex; align-items:center; gap:8px; margin:3px 0; }
.score-bar-bg   { flex:1; background:#ffffff15; border-radius:4px; height:6px; }
.score-bar-fill { height:6px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ─────────────────────────────────────────────
# summaries: dict[article_url -> {"text": str, "source": "gemini"|"fallback"}]
# summarising: set of article_urls currently being fetched (spinner state)
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
    """
    Phase 1 (RSS) + Phase 2 (classify) only.
    Phase 3 is on-demand per article.
    """
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
    """
    Fetch full article body + call Gemini for one article.
    Stores result in session_state keyed by article URL.
    """
    url = article.get("url", article["title"])

    from summariser import summarise_article, get_gemini_client
    client = _get_gemini_client()

    # Pass the real URL so summarise_article can fetch the full body
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
    if conf >= 0.70: return "🟢", "#2d6a4f"
    if conf >= 0.50: return "🟡", "#856404"
    return "🔴", "#842029"


def _source_badge(source: str) -> str:
    fg, bg = SOURCE_COLORS.get(source, ("#888", "#88888818"))
    return (
        f'<span class="source-badge" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}55;">{source}</span>'
    )


def render_article(article: dict, idx: int):
    title      = article.get("title", "Untitled")
    url        = article.get("url", title)        # stable session_state key
    link       = article.get("url", "#")
    source     = article.get("source", "")
    published  = article.get("published")
    category   = article.get("category", "Unknown")
    confidence = article.get("confidence", 0.0)
    all_scores = article.get("all_scores", {})

    conf_emoji, conf_color = _conf_color(confidence)
    date_str = published.strftime("%d %b %Y · %H:%M UTC") if published else ""

    # Has this article already been summarised?
    cached_summary = st.session_state.summaries.get(url)
    is_summarising = url in st.session_state.summarising

    with st.container(border=True):

        # ── Row 1: title + confidence pill ───────────────────────────────
        c_title, c_conf = st.columns([5, 1])
        with c_title:
            st.markdown(f"**[{title}]({link})**")
            st.markdown(
                _source_badge(source)
                + f'<span style="font-size:11px;color:var(--text-color,#888);">'
                  f'{date_str}</span>',
                unsafe_allow_html=True,
            )
        with c_conf:
            st.markdown(
                f'<div style="text-align:right;padding-top:6px;">'
                f'<span class="conf-pill" '
                f'style="background:{conf_color}22;color:{conf_color};">'
                f'{conf_emoji} {confidence:.0%}</span>'
                f'<br><span style="font-size:10px;color:#888;">{category}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Row 2: summary area OR summarise button ───────────────────────
        if cached_summary:
            # Show the summary — split on double newline for paragraph rendering
            src_label  = "★ Gemini" if cached_summary["source"] == "gemini" else "○ Fallback"
            body_label = cached_summary.get("body_source", "")
            body_note  = {
                "full_article":  "📄 Full article text",
                "rss_description": "📰 RSS description only",
            }.get(body_label, "")

            paragraphs = [p.strip() for p in cached_summary["text"].split("\n\n") if p.strip()]
            for para in paragraphs:
                st.markdown(
                    f'<div class="summary-para">{para}</div>',
                    unsafe_allow_html=True,
                )

            col_src, col_body = st.columns([1, 2])
            col_src.caption(src_label)
            if body_note:
                col_body.caption(body_note)

        elif is_summarising:
            st.info("Fetching full article + calling Gemini…", icon="⏳")

        else:
            # The button — unique key uses index to avoid duplicate-widget errors
            if st.button(
                "✨ Summarise",
                key=f"sum_btn_{idx}",
                help="Fetch the full article text and generate a Gemini summary",
            ):
                st.session_state.summarising.add(url)
                with st.spinner("Fetching article + calling Gemini…"):
                    fetch_summary_for(article)
                st.session_state.summarising.discard(url)
                st.rerun()

        # ── Row 3: score breakdown (collapsible) ──────────────────────────
        if all_scores:
            with st.expander("Category scores", expanded=False):
                for label, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                    pct  = int(score * 100)
                    fill = "#4e8cff" if label == category else "#666666"
                    bold = "600"     if label == category else "400"
                    st.markdown(
                        f'<div class="score-bar-wrap">'
                        f'<span style="width:105px;font-size:12px;font-weight:{bold};">{label}</span>'
                        f'<div class="score-bar-bg">'
                        f'<div class="score-bar-fill" style="width:{pct}%;background:{fill};"></div>'
                        f'</div>'
                        f'<span style="font-size:12px;width:32px;text-align:right;">{score:.0%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📡 T9.3 News Tracker")

    st.markdown("---")
    st.subheader("Pipeline")
    st.markdown("✅ **Phase 1** · RSS parsed")
    st.markdown("✅ **Phase 2** · Classified")
    st.markdown("🔘 **Phase 3** · On-demand per article")

    st.markdown("---")
    st.subheader("Session")
    n_summarised = len(st.session_state.summaries)
    gemini_n     = sum(
        1 for v in st.session_state.summaries.values()
        if v["source"] == "gemini"
    )
    st.metric("Summarised", n_summarised)
    st.metric("via Gemini",  gemini_n)

    if n_summarised and st.button("🗑 Clear summaries", use_container_width=True):
        st.session_state.summaries.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Cache TTL: {CACHE_TTL // 60} min")
    st.caption(f"Source: {'Mock' if USE_MOCK else 'Live RSS'}")
    st.caption(f"Classifier: {'Hybrid' if FORCE_HYBRID else 'BART'}")

    if st.button("🔄 Refresh articles", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ─── Main ─────────────────────────────────────────────────────────────────────

st.title("Tech News Tracker")

with st.spinner("Fetching and classifying articles…"):
    articles = load_articles()

if not articles:
    st.error("No articles loaded. Check your RSS feeds or enable mock mode.")
    st.stop()

# Stats bar
total    = len(articles)
avg_conf = sum(a.get("confidence", 0) for a in articles) / total
cats     = {a.get("category", "?") for a in articles}
n_sum    = len(st.session_state.summaries)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Articles",        total)
m2.metric("Categories",      len(cats))
m3.metric("Avg confidence",  f"{avg_conf:.0%}")
m4.metric("Summarised",      f"{n_sum}/{total}")

st.divider()

# Category tabs
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

        for i, article in enumerate(filtered):
            # Use a globally unique index so button keys don't collide across tabs
            global_idx = f"{label}_{i}"
            render_article(article, global_idx)