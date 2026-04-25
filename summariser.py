"""
summariser.py — Phase 3: LLM Summarisation with Gemini
========================================================
Sends each article's title + description to Gemini and stores a
structured 3-line summary.

SDK: google-genai  (pip install google-genai)
     This is the current maintained SDK — google-generativeai is deprecated.

Model: gemini-2.0-flash   (fast, free-tier friendly, 15 RPM limit)
       Falls back to gemini-1.5-flash if 2.0 is unavailable.

Rate-limit handling
-------------------
Free tier: 15 requests/minute.
We enforce a configurable inter-call sleep (default 4 s → 15 RPM) and
wrap every call in exponential back-off with up to MAX_RETRIES attempts.
On final failure the article keeps its raw description as fallback summary.

API key
-------
Set GEMINI_API_KEY in your .env file (never commit this file).
Get a free key at: https://aistudio.google.com/app/apikey

Run standalone:
    python summariser.py
"""

import logging
import os
import re
import time
from typing import Optional

log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

GEMINI_MODEL        = "gemini-2.0-flash"
FALLBACK_MODEL      = "gemini-1.5-flash"

# Free tier: 15 RPM  →  1 request every 4 seconds (conservative)
INTER_CALL_SLEEP    = 4.0   # seconds between API calls

# Retry / back-off
MAX_RETRIES         = 3
BACKOFF_BASE        = 2.0   # seconds; doubles each retry
BACKOFF_MAX         = 30.0  # cap

# How much of the raw description to send (keep prompt small for speed)
DESCRIPTION_CHARS   = 400


# ─── Prompt template ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a news editor who writes tight, accurate summaries.
When given an article title and description, respond with EXACTLY 3 lines.
Rules:
  - Line 1: The core news event (what happened, who is involved).
  - Line 2: The key detail, number, or implication that matters most.
  - Line 3: Why this matters or what happens next.
Do not add bullet points, numbers, headers, or any extra text.
Output only the 3 lines, separated by newlines."""

def _build_user_prompt(title: str, description: str) -> str:
    desc = description[:DESCRIPTION_CHARS].strip()
    return f"Title: {title}\nDescription: {desc}"


# ─── SDK initialisation ───────────────────────────────────────────────────────

def _load_api_key() -> Optional[str]:
    """
    Load GEMINI_API_KEY from environment or .env file.
    Returns None if not found (triggers fallback mode).
    """
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # Try .env in current working directory
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
    """
    Create and return a google.genai Client.
    Returns None if API key is missing or SDK import fails.
    """
    api_key = _load_api_key()
    if not api_key:
        log.warning(
            "GEMINI_API_KEY not set — summaries will fall back to raw description. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        log.info("Gemini client initialised (model: %s)", GEMINI_MODEL)
        return client
    except ImportError:
        log.error("google-genai not installed. Run: pip install google-genai")
        return None
    except Exception as exc:
        log.error("Failed to initialise Gemini client: %s", exc)
        return None


# ─── Summary generation ───────────────────────────────────────────────────────

def _clean_summary(raw: str) -> str:
    """
    Post-process Gemini output into a clean 3-line string.

    Handles:
    - markdown bold/italic (* or **)
    - bullet points or numbered list prefixes
    - leading/trailing whitespace per line
    - ensures exactly 3 lines (truncates extras, pads with fallback if short)
    """
    # Strip markdown formatting
    raw = re.sub(r"\*+", "", raw)
    raw = re.sub(r"_+", "", raw)

    # Remove list prefixes: "1. ", "- ", "• "
    lines = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^\s*[\d]+[.)]\s*", "", line)  # "1. " or "1) "
        line = re.sub(r"^\s*[-•*]\s*", "", line)       # "- " or "• "
        line = line.strip()
        if line:
            lines.append(line)

    # Ensure exactly 3 lines
    if len(lines) >= 3:
        return "\n".join(lines[:3])
    elif len(lines) == 2:
        return "\n".join(lines + ["More developments expected in the coming days."])
    elif len(lines) == 1:
        return "\n".join(lines * 2 + ["Further details are awaited."])
    else:
        return ""


def _call_gemini_with_retry(client, title: str, description: str) -> Optional[str]:
    """
    Call Gemini API with exponential back-off retry.

    Returns the cleaned 3-line summary string, or None on total failure.
    """
    from google.genai import types as genai_types

    user_prompt = _build_user_prompt(title, description)
    delay = BACKOFF_BASE

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,      # low temp → factual, consistent
                    max_output_tokens=180, # 3 lines ≈ 60 tokens each, generous budget
                    candidate_count=1,
                ),
            )
            raw_text = response.text or ""
            cleaned = _clean_summary(raw_text)
            if cleaned:
                return cleaned
            log.warning("Attempt %d: empty summary returned", attempt)

        except Exception as exc:
            exc_str = str(exc)
            # Detect quota / rate-limit errors
            is_rate_limit = any(
                kw in exc_str.lower()
                for kw in ("429", "quota", "rate limit", "resource exhausted")
            )
            if is_rate_limit:
                wait = min(delay, BACKOFF_MAX)
                log.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %.0fs…",
                    attempt, MAX_RETRIES, wait,
                )
                time.sleep(wait)
                delay *= 2
            elif "model not found" in exc_str.lower() or "404" in exc_str:
                # Try fallback model on 404
                log.warning("Model '%s' not found, trying '%s'", GEMINI_MODEL, FALLBACK_MODEL)
                try:
                    response = client.models.generate_content(
                        model=FALLBACK_MODEL,
                        contents=user_prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.3,
                            max_output_tokens=180,
                        ),
                    )
                    cleaned = _clean_summary(response.text or "")
                    if cleaned:
                        return cleaned
                except Exception as e2:
                    log.error("Fallback model also failed: %s", e2)
                break   # no point retrying if both models fail
            else:
                log.error("Gemini error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc_str[:120])
                if attempt < MAX_RETRIES:
                    time.sleep(min(delay, BACKOFF_MAX))
                    delay *= 2

    return None   # all retries exhausted


def _fallback_summary(description: str, max_chars: int = 300) -> str:
    """
    Generate a best-effort 3-line summary from the raw description
    when Gemini is unavailable. Splits on sentence boundaries.
    """
    if not description:
        return "Summary unavailable — no description provided."

    # Split into sentences (rough heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", description.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) >= 3:
        return "\n".join(sentences[:3])
    elif sentences:
        # Pad with truncated description
        text = description[:max_chars].strip()
        lines = [text[i : i + max_chars // 3] for i in range(0, len(text), max_chars // 3)]
        return "\n".join(lines[:3])
    else:
        return description[:max_chars]


# ─── Public API ──────────────────────────────────────────────────────────────

_client_singleton = None


def get_gemini_client():
    """Return the Gemini client singleton (or None if unconfigured)."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = _make_client()
    return _client_singleton


def summarise_article(
    article: dict,
    client=None,
    sleep_between: float = 0.0,
) -> dict:
    """
    Generate a 3-line summary for a single article dict.

    Modifies the article in-place AND returns it.

    Parameters
    ----------
    article        : article dict (must have 'title' and 'description')
    client         : pre-loaded Gemini client (avoids re-init overhead)
    sleep_between  : seconds to sleep after the API call (rate-limit pacing)

    Adds to article
    ---------------
    summary        : str  — 3-line plain-text summary
    summary_source : str  — 'gemini' | 'fallback'
    """
    if client is None:
        client = get_gemini_client()

    title       = article.get("title", "")
    description = article.get("description", "")

    if client is not None:
        summary = _call_gemini_with_retry(client, title, description)
        if summary:
            article["summary"]        = summary
            article["summary_source"] = "gemini"
            if sleep_between > 0:
                time.sleep(sleep_between)
            return article
        else:
            log.warning("Gemini failed for '%s' — using fallback", title[:50])

    # Fallback: derive from raw description
    article["summary"]        = _fallback_summary(description)
    article["summary_source"] = "fallback"
    return article


def summarise_articles(
    articles: list[dict],
    client=None,
    inter_call_sleep: float = INTER_CALL_SLEEP,
) -> list[dict]:
    """
    Summarise all articles in the list.

    Parameters
    ----------
    articles         : list of article dicts (Phase 2 output)
    client           : pre-loaded Gemini client (optional)
    inter_call_sleep : seconds to sleep between API calls (rate-limit pacing)

    Returns
    -------
    Same list with 'summary' and 'summary_source' filled in on every article.
    """
    if client is None:
        client = get_gemini_client()

    total = len(articles)
    gemini_count   = 0
    fallback_count = 0

    log.info(
        "Phase 3: summarising %d articles%s",
        total,
        f" (Gemini, ~{total * inter_call_sleep:.0f}s)" if client else " (fallback mode — set GEMINI_API_KEY)",
    )
    t_start = time.perf_counter()

    for i, article in enumerate(articles, 1):
        summarise_article(
            article,
            client=client,
            sleep_between=inter_call_sleep if client else 0.0,
        )
        src = article.get("summary_source", "?")
        if src == "gemini":
            gemini_count += 1
        else:
            fallback_count += 1

        log.debug(
            "[%d/%d] [%s] %s",
            i, total, src.upper(), article["title"][:55],
        )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Phase 3 complete in %.1fs — gemini: %d, fallback: %d",
        elapsed, gemini_count, fallback_count,
    )
    return articles


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    print("=" * 65)
    print("T9.3 — Phase 3: Summariser smoke-test")
    print("=" * 65)

    sample_articles = [
        {
            "title": "OpenAI's GPT-5 Brings Real-Time Voice and Vision to ChatGPT",
            "description": (
                "OpenAI has announced GPT-5, its most capable model yet, featuring "
                "real-time voice conversation and native image understanding. "
                "The release signals a major step toward multimodal AI assistants "
                "that can reason across text, audio and visual inputs simultaneously. "
                "The model is rolling out to Plus and Pro subscribers first."
            ),
        },
        {
            "title": "Zepto Raises $400M Series G at $5B Valuation",
            "description": (
                "Quick-commerce platform Zepto has closed a $400 million Series G "
                "funding round, valuing the company at $5 billion. The capital will "
                "fund 200 new dark stores across tier-2 Indian cities. Investors "
                "include Nexus and Y Combinator's continuity fund."
            ),
        },
    ]

    client = get_gemini_client()
    if client is None:
        print("\n  ⚠  GEMINI_API_KEY not set — running in fallback mode.\n")

    for article in sample_articles:
        summarise_article(article, client=client, sleep_between=0)
        print(f"\n  Article : {article['title']}")
        print(f"  Source  : {article['summary_source'].upper()}")
        print("  Summary :")
        for line in article["summary"].splitlines():
            print(f"    {line}")

    print("\n" + "=" * 65)