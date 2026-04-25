"""
rss_parser.py — Phase 1: RSS Feed Fetcher & Parser
====================================================
Fetches articles from configured RSS/Atom feeds, normalises them into
a consistent dict structure, deduplicates by URL, and returns the
latest MAX_ARTICLES articles sorted newest-first.

Run standalone:
    python rss_parser.py
"""

import feedparser
import logging
import time
from datetime import datetime, timezone
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

FEEDS: dict[str, str] = {
    "TechCrunch": "https://techcrunch.com/feed/",
    "The Verge":  "https://www.theverge.com/rss/index.xml",
    "YourStory":  "https://yourstory.com/feed",
}

# Extra feeds you can add later
OPTIONAL_FEEDS: dict[str, str] = {
    "Wired":         "https://www.wired.com/feed/rss",
    "Ars Technica":  "https://feeds.arstechnica.com/arstechnica/index",
    "MIT Tech Review": "https://www.technologyreview.com/feed/",
}

MAX_ARTICLES   = 30          # total articles kept across all feeds
DESCRIPTION_CHARS = 300      # how many chars to keep from raw description
REQUEST_TIMEOUT   = 10       # seconds before giving up on a feed
REQUEST_HEADERS   = {        # mimic a real browser to avoid 403s
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Strip the most common HTML tags from a description string."""
    import re
    text = re.sub(r"<[^>]+>", "", text)          # remove tags
    text = re.sub(r"&amp;",  "&",  text)          # unescape entities
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&nbsp;", " ",  text)
    text = re.sub(r"&#\d+;", "",   text)
    text = re.sub(r"\s+",    " ",  text).strip()  # collapse whitespace
    return text


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    """
    Extract a timezone-aware datetime from a feed entry.

    feedparser tries to normalise published/updated into a 9-tuple
    (published_parsed / updated_parsed).  If that's absent we fall
    back to the current UTC time so the article still sorts correctly.
    """
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    log.debug("No date found for entry '%s', using now()", entry.get("title", ""))
    return datetime.now(timezone.utc)


def _extract_entry(entry: feedparser.FeedParserDict, source_name: str) -> dict:
    """
    Map a raw feedparser entry → clean article dict.

    Returned keys
    -------------
    title       : str  — article headline
    description : str  — cleaned plain-text description / summary (≤300 chars)
    url         : str  — canonical link
    source      : str  — human-readable feed name
    published   : datetime — timezone-aware UTC datetime
    """
    # Title
    title = _clean_html(entry.get("title", "Untitled"))

    # Description — feedparser exposes this as "summary" or "description"
    raw_desc = entry.get("summary") or entry.get("description") or ""
    description = _clean_html(raw_desc)[:DESCRIPTION_CHARS]

    # URL — prefer "link", fall back to first alternate link
    url = entry.get("link", "")
    if not url and entry.get("links"):
        url = entry["links"][0].get("href", "")

    published = _parse_published(entry)

    return {
        "title":       title,
        "description": description,
        "url":         url,
        "source":      source_name,
        "published":   published,
        # Fields added later by classifier / summariser
        "category":    None,
        "confidence":  None,
        "summary":     None,
    }


# ─── Core function ────────────────────────────────────────────────────────────

def fetch_articles(
    feeds: Optional[dict[str, str]] = None,
    max_articles: int = MAX_ARTICLES,
) -> list[dict]:
    """
    Fetch and parse articles from all configured RSS feeds.

    Parameters
    ----------
    feeds        : dict mapping display-name → RSS URL.
                   Defaults to the global FEEDS constant.
    max_articles : Maximum number of articles to return (newest first).

    Returns
    -------
    list of article dicts (see _extract_entry for key definitions),
    sorted newest-first, deduplicated by URL, capped at max_articles.
    """
    if feeds is None:
        feeds = FEEDS

    seen_urls: set[str] = set()
    articles:  list[dict] = []

    for source_name, feed_url in feeds.items():
        log.info("Fetching %-15s  %s", source_name, feed_url)

        try:
            feed = feedparser.parse(
                feed_url,
                request_headers=REQUEST_HEADERS,
            )
        except Exception as exc:
            log.error("Failed to fetch '%s': %s", source_name, exc)
            continue

        # HTTP-level error reporting
        status = feed.get("status")
        if status and status >= 400:
            log.warning(
                "HTTP %d for '%s' — feed may block bots or require auth",
                status, source_name,
            )
            # Continue — feedparser may still have parsed a cached/redirect body
            if not feed.entries:
                continue

        if not feed.entries:
            log.warning("No entries in feed for '%s'", source_name)
            continue

        log.info("  → %d entries retrieved", len(feed.entries))

        for entry in feed.entries:
            article = _extract_entry(entry, source_name)

            # Skip articles with no URL (can't deduplicate or link them)
            if not article["url"]:
                log.debug("Skipping entry with no URL in '%s'", source_name)
                continue

            # Deduplicate by canonical URL
            if article["url"] in seen_urls:
                log.debug("Duplicate skipped: %s", article["url"][:60])
                continue

            seen_urls.add(article["url"])
            articles.append(article)

    # Sort newest-first
    articles.sort(key=lambda a: a["published"], reverse=True)

    if len(articles) > max_articles:
        log.info(
            "Trimming %d → %d articles (max_articles=%d)",
            len(articles), max_articles, max_articles,
        )
        articles = articles[:max_articles]

    log.info("Phase 1 complete: %d articles ready for Phase 2", len(articles))
    return articles


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("T9.3 — Phase 1: RSS Parser")
    print("=" * 60)

    articles = fetch_articles()

    if not articles:
        print("\n⚠  No articles fetched. Check your network / feed URLs.\n")
    else:
        for i, a in enumerate(articles, 1):
            print(f"\n[{i:02d}] {a['title']}")
            print(f"     Source : {a['source']}")
            print(f"     Date   : {a['published'].strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"     URL    : {a['url']}")
            print(f"     Desc   : {a['description'][:80]}...")

    print("\n" + "=" * 60)
    print(f"Total: {len(articles)} articles")
    print("=" * 60)
