"""
summariser.py — Phase 3: LLM Summarisation with Gemini
========================================================
Fetches the full article text from the URL, then sends it to Gemini
for a rich, substantive 3-paragraph summary.

Why full article text matters
------------------------------
RSS feed descriptions are truncated stubs (150–400 chars). Gemini can
only summarise what it receives — feeding it a stub produces a stub
summary. This module fetches the real article URL with trafilatura,
extracts the main body text (up to 8 000 chars), and sends that to
Gemini so the summary reflects the actual article content.

Quality improvements over the previous version
------------------------------------------------
1. Full article body fetched via trafilatura (best-in-class extractor)
2. Graceful fallback: RSS description used if URL fetch fails
3. Prompt rewritten: 3 substantive paragraphs, not 3 terse lines
4. max_output_tokens raised 180 → 600 (room for proper sentences)
5. temperature lowered 0.3 → 0.2 (more factual, less hallucination)
6. Debug logging prints the exact payload sent to Gemini

SDK: google-genai  (pip install google-genai)
Model: gemini-2.0-flash → fallback gemini-1.5-flash

Rate-limit handling
-------------------
Free tier: 15 RPM. We target 12 RPM (5 s gap, 20% safety margin).
TokenBucketPacer enforces the gap BEFORE every call.
Exponential back-off + jitter on 429s.

Run standalone:
    python summariser.py
"""

import logging
import os
import random
import re
import time
from typing import Optional

log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

GEMINI_MODEL   = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.5-Pro"

FREE_TIER_RPM  = 12
MIN_CALL_GAP   = 60.0 / FREE_TIER_RPM   # 5.0 s

MAX_RETRIES    = 5
BACKOFF_BASE   = 10.0
BACKOFF_MAX    = 120.0
BACKOFF_JITTER = 0.25

# Full article body: send up to 8 000 chars (~2 000 tokens of context)
ARTICLE_BODY_CHARS = 8_000

# Hard cap on what we send if full fetch fails and we fall back to RSS desc
RSS_DESC_CHARS = 1_200


# ─── Article body fetcher ─────────────────────────────────────────────────────

def fetch_article_body(url: str) -> Optional[str]:
    """
    Fetch the full article text from a URL using trafilatura.

    Returns the extracted plain-text body (up to ARTICLE_BODY_CHARS chars),
    or None if the fetch fails (network error, paywalled, 403, etc.).

    trafilatura is purpose-built for news article extraction — it strips
    ads, nav bars, comments, and boilerplate far more reliably than
    BeautifulSoup heuristics.
    """
    if not url or url == "#":
        return None
    try:
        import trafilatura

        log.debug("Fetching article body from: %s", url)

        # Build a config with a browser-like user agent so fewer sites 403
        config = trafilatura.settings.use_config()
        config.set("DEFAULT", "USER_AGENTS",
                   "Mozilla/5.0 (X11; Linux x86_64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36")
        config.set("DEFAULT", "DOWNLOAD_TIMEOUT", "8")   # 8 s max per fetch

        downloaded = trafilatura.fetch_url(url, config=config)
        if not downloaded:
            log.warning("No content downloaded from %s", url)
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,      # use fallback extractor if main fails
            favor_precision=False,  # favor recall — get more text
        )
        if not text:
            log.warning("trafilatura extracted no text from %s", url)
            return None

        body = text.strip()[:ARTICLE_BODY_CHARS]
        log.info("Fetched article body: %d chars from %s", len(body), url)
        return body

    except ImportError:
        log.warning("trafilatura not installed — falling back to RSS description. "
                    "Install with: pip install trafilatura")
        return None
    except Exception as exc:
        log.warning("Article fetch failed for %s: %s", url, exc)
        return None


# ─── Token-bucket rate pacer ──────────────────────────────────────────────────

class _TokenBucketPacer:
    def __init__(self, min_gap_seconds: float = MIN_CALL_GAP):
        self._gap  = min_gap_seconds
        self._last = 0.0

    def wait(self) -> None:
        elapsed   = time.monotonic() - self._last
        remaining = self._gap - elapsed
        if remaining > 0:
            log.debug("Rate pacer: sleeping %.2fs", remaining)
            time.sleep(remaining)
        self._last = time.monotonic()


_pacer = _TokenBucketPacer()


# ─── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior technology journalist. Your job is to read a news article
and write a clear, informative summary for a busy reader.

Write exactly 3 paragraphs. Each paragraph must be 2–3 complete sentences.

Paragraph 1 — What happened: Describe the main news event, who is involved,
and what was announced or released.

Paragraph 2 — Key details: Include the most important specific facts —
numbers, technical specifications, timelines, or named people/companies
that give the story substance.

Paragraph 3 — Why it matters: Explain the broader significance, industry
impact, competitive context, or what happens next.

Rules:
- Write in plain, confident prose. No bullet points, no headers, no markdown.
- Do not start with phrases like "This article discusses" or "The article says".
- Preserve specific numbers, names, and technical terms exactly as given.
- Each paragraph must be separated by a blank line.
- Do not add any text outside the 3 paragraphs.\
"""


def _build_prompt(title: str, body: str, source: str = "") -> str:
    """
    Build the user message sent to Gemini.
    Logs the exact payload so quality issues are debuggable.
    """
    src_line = f"Source: {source}\n" if source else ""
    prompt   = f"{src_line}Title: {title}\n\nArticle:\n{body}"

    log.info("── Gemini payload ──────────────────────────────────────")
    log.info("Title  : %s", title)
    log.info("Source : %s", source or "(unknown)")
    log.info("Body   : %d chars | first 300: %s…", len(body), body[:300].replace("\n", " "))
    log.info("────────────────────────────────────────────────────────")

    return prompt


# ─── SDK ─────────────────────────────────────────────────────────────────────

def _load_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip('"').strip("'")
    return None


def _make_client():
    api_key = _load_api_key()
    if not api_key:
        log.warning(
            "GEMINI_API_KEY not set — summaries will fall back to RSS description. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        log.info("Gemini client initialised (model: %s)", GEMINI_MODEL)
        return client
    except ImportError:
        log.error("google-genai not installed — run: pip install google-genai")
        return None
    except Exception as exc:
        log.error("Failed to create Gemini client: %s", exc)
        return None


# ─── Retry logic ─────────────────────────────────────────────────────────────

def _jittered_backoff(base: float, attempt: int) -> float:
    delay  = min(base * (2 ** (attempt - 1)), BACKOFF_MAX)
    jitter = delay * BACKOFF_JITTER * (2 * random.random() - 1)
    return max(1.0, delay + jitter)


def _call_gemini(client, title: str, body: str, source: str = "") -> Optional[str]:
    """
    Send the article to Gemini and return a cleaned 3-paragraph summary.
    Retries with exponential back-off on 429 / transient errors.
    """
    from google.genai import types as genai_types

    user_prompt  = _build_prompt(title, body, source)
    active_model = GEMINI_MODEL

    for attempt in range(1, MAX_RETRIES + 1):
        _pacer.wait()   # enforce rate limit BEFORE every call

        try:
            response = client.models.generate_content(
                model=active_model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,        # lower → more factual
                    max_output_tokens=6000,  # ~2 400 chars — room for 3 real paragraphs
                    candidate_count=1,
                ),
            )
            raw = (response.text or "").strip()
            if raw:
                log.info("Gemini response (%d chars): %s…", len(raw), raw[:120].replace("\n", " "))
                return _clean_summary(raw)
            log.warning("Attempt %d: Gemini returned empty response", attempt)

        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = any(kw in exc_str.lower() for kw in (
                "429", "quota", "rate limit", "resource_exhausted",
                "resource exhausted", "too many requests",
            ))
            if is_rate_limit:
                wait = _jittered_backoff(BACKOFF_BASE, attempt)
                log.warning("429 on attempt %d/%d — backing off %.0fs", attempt, MAX_RETRIES, wait)
                time.sleep(wait)
                _pacer._last = 0.0
                continue

            if "model not found" in exc_str.lower() or "404" in exc_str:
                if active_model != FALLBACK_MODEL:
                    log.warning("Model %s not found — switching to %s", active_model, FALLBACK_MODEL)
                    active_model = FALLBACK_MODEL
                    continue
                break

            log.error("Gemini error attempt %d/%d: %s", attempt, MAX_RETRIES, exc_str[:200])
            if attempt < MAX_RETRIES:
                time.sleep(_jittered_backoff(BACKOFF_BASE, attempt))

    return None


# ─── Post-processing ─────────────────────────────────────────────────────────

def _clean_summary(raw: str) -> str:
    """
    Clean Gemini output:
    - Strip markdown bold/italic
    - Remove numbered/bullet list prefixes
    - Normalise paragraph breaks (blank line between paras)
    - Ensure we have 1–3 paragraphs
    """
    # Strip markdown
    raw = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", raw)   # **bold** / *italic*
    raw = re.sub(r"_{1,2}([^_]+)_{1,2}",   r"\1", raw)   # __bold__ / _italic_
    raw = re.sub(r"`([^`]+)`",              r"\1", raw)   # `code`
    raw = re.sub(r"#{1,6}\s*",              "",    raw)   # ### headers

    # Remove list-item prefixes from lines
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)   # "1. " or "1) "
        line = re.sub(r"^\s*[-•*]\s+",   "", line)   # "- " or "• "
        lines.append(line)
    raw = "\n".join(lines)

    # Split into paragraphs on blank lines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]

    if not paragraphs:
        return ""

    # Join multi-line paragraphs into single-line blocks
    paragraphs = [" ".join(p.split()) for p in paragraphs]

    # Keep at most 3
    return "\n\n".join(paragraphs[:3])


# ─── Fallback summary (no Gemini) ────────────────────────────────────────────

def _fallback_summary(description: str) -> str:
    """
    Build a best-effort summary from the RSS description when Gemini is
    unavailable. Returns the full description split into up to 3 sentences.
    """
    if not description:
        return "No summary available — article description is empty."

    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if not sentences:
        return description.strip()
    if len(sentences) == 1:
        return sentences[0]

    # Return all sentences (let the UI wrap them into paragraphs)
    return "\n\n".join(sentences[:3])


# ─── Public API ──────────────────────────────────────────────────────────────

_client_singleton = None


def get_gemini_client():
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = _make_client()
    return _client_singleton


def summarise_article(article: dict, client=None) -> dict:
    """
    Generate a rich 3-paragraph summary for one article dict.

    Steps
    -----
    1. Try to fetch the full article body from article["url"] via trafilatura.
    2. If fetch fails (paywalled, 403, no network), use article["description"].
    3. Send whichever body we have to Gemini with the improved prompt.
    4. If Gemini fails, fall back to sentence-split of the description.

    Modifies article in-place and returns it.
    Adds: summary (str), summary_source (str), body_source (str)
    """
    if client is None:
        client = get_gemini_client()

    title       = article.get("title", "").strip()
    description = article.get("description", "").strip()
    url         = article.get("url", "")
    source      = article.get("source", "")

    # ── Step 1: get the richest possible body text ────────────────────────
    full_body = fetch_article_body(url)

    if full_body:
        body        = full_body
        body_source = "full_article"
        log.info("Using full article body (%d chars)", len(body))
    else:
        # Extend RSS description to its full length (no truncation)
        body        = description[:RSS_DESC_CHARS] if description else title
        body_source = "rss_description"
        log.info(
            "Full article fetch failed — using RSS description (%d chars)",
            len(body),
        )

    article["body_source"] = body_source

    # ── Step 2: call Gemini ───────────────────────────────────────────────
    if client is not None and body:
        summary = _call_gemini(client, title, body, source)
        if summary:
            article["summary"]        = summary
            article["summary_source"] = "gemini"
            return article
        log.warning("Gemini failed — using fallback for '%s'", title[:50])

    # ── Step 3: fallback ──────────────────────────────────────────────────
    article["summary"]        = _fallback_summary(description or title)
    article["summary_source"] = "fallback"
    return article


def summarise_articles(articles: list[dict], client=None) -> list[dict]:
    """Summarise all articles. Rate-pacing handled by _TokenBucketPacer."""
    if client is None:
        client = get_gemini_client()

    total = len(articles)
    log.info("Phase 3: summarising %d articles%s", total,
             f" via Gemini (~{total * MIN_CALL_GAP:.0f}s)" if client else " (fallback)")

    t0 = time.perf_counter()
    gemini_n = fallback_n = 0

    for i, article in enumerate(articles, 1):
        summarise_article(article, client=client)
        src = article.get("summary_source", "?")
        if src == "gemini": gemini_n += 1
        else: fallback_n += 1
        log.info("[%d/%d] [%-8s] [body=%-15s] %s",
                 i, total, src.upper(),
                 article.get("body_source", "?"),
                 article["title"][:50])

    log.info("Phase 3 done in %.1fs — gemini:%d fallback:%d",
             time.perf_counter() - t0, gemini_n, fallback_n)
    return articles


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO, datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("T9.3 — Phase 3: Summariser diagnostic")
    print("=" * 65)

    # Print exact payload that will be sent
    title = "OpenAI's GPT-5 Brings Real-Time Voice and Vision to ChatGPT"
    desc  = ("OpenAI has announced GPT-5, its most capable model yet, featuring "
             "real-time voice conversation and native image understanding. "
             "The release signals a major step toward multimodal AI assistants "
             "that can reason across text, audio and visual inputs simultaneously. "
             "The model is rolling out to Plus and Pro subscribers first.")
    url   = "https://techcrunch.com/2026/04/25/openai-gpt5-voice-vision/"

    print("\n[1] Attempting full article fetch…")
    body = fetch_article_body(url)
    if body:
        print(f"    ✓ Fetched {len(body)} chars of article body")
        print(f"    Preview: {body[:200]}…")
    else:
        print(f"    ✗ Fetch failed — will use RSS description ({len(desc)} chars)")
        body = desc

    print("\n[2] Prompt that will be sent to Gemini:")
    print("-" * 55)
    print(f"SYSTEM: {SYSTEM_PROMPT[:200]}…")
    print()
    prompt = _build_prompt(title, body, "TechCrunch")
    print(f"USER ({len(prompt)} chars):")
    print(prompt[:400], "…" if len(prompt) > 400 else "")
    print("-" * 55)

    print("\n[3] Running summariser…")
    article = {"title": title, "description": desc, "url": url, "source": "TechCrunch"}
    client  = get_gemini_client()
    if client is None:
        print("    ⚠  No GEMINI_API_KEY — fallback mode\n")

    summarise_article(article, client=client)

    print(f"\n    Summary source : {article['summary_source'].upper()}")
    print(f"    Body source    : {article.get('body_source','?')}")
    print(f"    Summary length : {len(article['summary'])} chars")
    print("\n    Summary:")
    for para in article["summary"].split("\n\n"):
        print(f"    ▌ {para}")
        print()
    print("=" * 65)