"""
pipeline_runner.py — Full pipeline orchestrator: Phase 1 → 2 → 3
=================================================================
Chains RSS parsing, classification, and summarisation into a single
call that app.py can cache with @st.cache_data.

Run standalone to see the full pipeline output:
    python pipeline_runner.py

Set GEMINI_API_KEY in .env for real summaries (Phase 3).
Without it, Phase 3 falls back to sentence-split descriptions.
"""

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)


def run_pipeline(
    use_mock: bool = False,
    force_hybrid_classifier: bool = False,
    max_articles: int = 30,
) -> list[dict]:
    """
    Execute the full T9.3 pipeline.

    Parameters
    ----------
    use_mock                : use mock_feeds instead of real RSS (for testing)
    force_hybrid_classifier : skip BART download, use local HybridClassifier
    max_articles            : max articles to fetch and process

    Returns
    -------
    list of fully-enriched article dicts with all fields populated:
        title, description, url, source, published,
        category, confidence, all_scores,
        summary, summary_source
    """
    t_total = time.perf_counter()

    # ── Phase 1: RSS Parsing ──────────────────────────────────────────────────
    log.info("━━━ Phase 1: RSS Parsing ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if use_mock:
        from mock_feeds import fetch_articles
        log.info("Using mock feeds (offline mode)")
    else:
        from rss_parser import fetch_articles
        log.info("Fetching live RSS feeds")

    t1 = time.perf_counter()
    articles = fetch_articles(max_articles=max_articles)
    t1_elapsed = time.perf_counter() - t1
    log.info("Phase 1 done: %d articles in %.2fs", len(articles), t1_elapsed)

    if not articles:
        log.warning("No articles fetched — pipeline aborted")
        return []

    # ── Phase 2: Zero-Shot Classification ────────────────────────────────────
    log.info("━━━ Phase 2: Classification ━━━━━━━━━━━━━━━━━━━━━━━━")
    from classifier import classify_articles

    t2 = time.perf_counter()
    articles = classify_articles(
        articles,
        force_hybrid=force_hybrid_classifier,
        use_finetuned=True # make flase if we donr wanna use finetuned 
    )
    t2_elapsed = time.perf_counter() - t2
    log.info("Phase 2 done: %d articles classified in %.2fs", len(articles), t2_elapsed)

    # ── Phase 3: LLM Summarisation ───────────────────────────────────────────
    log.info("━━━ Phase 3: Summarisation ━━━━━━━━━━━━━━━━━━━━━━━━━")
    from summariser import summarise_articles, get_gemini_client

    client = get_gemini_client()
    t3 = time.perf_counter()
    articles = summarise_articles(
        articles,
        client=client,
    )
    t3_elapsed = time.perf_counter() - t3
    log.info("Phase 3 done: %d articles summarised in %.2fs", len(articles), t3_elapsed)

    # ── Pipeline stats ────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    cat_counts: dict[str, int] = {}
    gemini_count = 0
    for a in articles:
        cat_counts[a.get("category", "Unknown")] = cat_counts.get(a.get("category", "Unknown"), 0) + 1
        if a.get("summary_source") == "gemini":
            gemini_count += 1

    log.info(
        "━━━ Pipeline complete: %d articles in %.2fs "
        "(p1=%.2fs p2=%.2fs p3=%.2fs) ━━━",
        len(articles), total_elapsed, t1_elapsed, t2_elapsed, t3_elapsed,
    )
    log.info("Category breakdown: %s", cat_counts)
    log.info(
        "Summaries: %d Gemini / %d fallback",
        gemini_count, len(articles) - gemini_count,
    )

    return articles


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("T9.3 — Full Pipeline: Phase 1 → 2 → 3")
    print("=" * 65)

    articles = run_pipeline(
        use_mock=True,
        force_hybrid_classifier=True,
        max_articles=12,
    )

    print()
    for i, a in enumerate(articles, 1):
        conf_pct = f"{a.get('confidence', 0):.0%}"
        src_flag = "★" if a.get("summary_source") == "gemini" else "○"
        print(f"[{i:02d}] {a['title'][:55]}")
        print(f"     Source   : {a['source']}")
        print(f"     Category : {a.get('category','?')} ({conf_pct})")
        print(f"     Summary  : {src_flag}")
        for line in (a.get("summary") or "").splitlines():
            print(f"               {line}")
        print()

    # Category distribution table
    from collections import Counter
    cats = Counter(a.get("category", "Unknown") for a in articles)
    print("─" * 40)
    print("Category distribution:")
    for cat, count in cats.most_common():
        bar = "█" * count
        print(f"  {cat:<15} {bar} {count}")
    print("─" * 40)
    print(f"Total: {len(articles)} articles")