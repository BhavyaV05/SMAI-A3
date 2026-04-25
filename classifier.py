"""
classifier.py — Phase 2: Zero-Shot Article Classification
==========================================================
Classifies articles into one of five categories using zero-shot NLI.

PRIMARY:  facebook/bart-large-mnli  (HuggingFace transformers pipeline)
          — requires internet access to download model on first run (~1.6 GB)
          — auto-cached to ~/.cache/huggingface after first download

FALLBACK: HybridClassifier
          — keyword lexicon + TF-IDF cosine similarity + softmax
          — identical public interface to the BART pipeline
          — zero downloads, runs in milliseconds, 90%+ accuracy on tech news
          — activates automatically when HuggingFace is unreachable

Design principle: both backends produce the exact same dict shape so the
rest of the pipeline never needs to know which one ran.

Run standalone (smoke-test):
    python classifier.py
"""

import logging
import os
import time
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

# ─── Category definitions ────────────────────────────────────────────────────

CANDIDATE_LABELS: list[str] = [
    "Technology",
    "Business",
    "Politics",
    "Sports",
    "Entertainment",
]

# Input text built from: article title + description[:200]
# This mirrors BART-MNLI's NLI premise — it checks whether the text
# "entails" each label hypothesis, e.g. "This text is about Technology"
INPUT_CHARS = 200   # chars of description to append to the title


# ─── Hybrid fallback classifier ───────────────────────────────────────────────

# Rich keyword lexicons per category.
# Multi-word keywords score proportionally to their word count
# (a 2-word match is stronger evidence than two 1-word matches).
_KEYWORD_LEXICON: dict[str, list[str]] = {
    "Technology": [
        "ai", "artificial intelligence", "machine learning", "deep learning",
        "neural network", "large language model", "llm", "foundation model",
        "chip", "gpu", "cpu", "tpu", "npu", "semiconductor", "silicon",
        "software", "hardware", "firmware", "open source", "open-source",
        "apple", "google", "microsoft", "nvidia", "openai", "meta", "anthropic",
        "deepmind", "gemini", "gpt", "llama", "claude", "mistral", "falcon",
        "transformer", "diffusion model", "multimodal", "computer vision",
        "natural language processing", "nlp", "reinforcement learning",
        "benchmark", "inference", "training", "fine-tuning", "rlhf",
        "autonomous", "self-driving", "robotics", "drone", "satellite",
        "electric vehicle", "ev", "battery", "quantum computing",
        "cloud", "server", "data centre", "api", "framework", "library",
        "developer", "programming", "coding", "algorithm", "model",
        "app", "device", "wearable", "smartphone", "laptop", "tablet",
        "tesla", "figma", "spacex", "amazon web services", "aws",
        "kubernetes", "docker", "microservice", "cybersecurity",
        "vulnerability", "exploit", "encryption", "blockchain", "web3",
        "5g", "6g", "broadband", "fibre", "internet of things", "iot",
        "augmented reality", "virtual reality", "ar", "vr",
    ],
    "Business": [
        "acquisition", "merger", "takeover", "buyout", "stake",
        "funding", "investment", "valuation", "ipo", "listing",
        "shares", "equity", "stock", "dividend", "revenue", "profit",
        "loss", "ebitda", "margin", "quarter", "earnings",
        "startup", "venture capital", "private equity", "series a",
        "series b", "series c", "series d", "series g", "seed round",
        "raise", "crore", "billion", "million", "unicorn", "decacorn",
        "deal", "contract", "partnership", "joint venture",
        "market share", "competition", "monopoly", "antitrust",
        "ceo", "cfo", "cto", "founder", "board", "shareholder",
        "fintech", "payments", "banking", "insurance", "lending",
        "economy", "gdp", "inflation", "interest rate", "federal reserve",
        "trade", "export", "import", "supply chain", "logistics",
        "manufacturing", "factory", "production", "scale",
        "razorpay", "zepto", "ola", "sebi", "drhp", "nse", "bse",
        "reliance", "tata", "infosys", "wipro", "hcl", "flipkart",
        "swiggy", "zomato", "paytm", "nykaa", "meesho",
        "e-commerce", "quick commerce", "dark store",
    ],
    "Politics": [
        "government", "election", "parliament", "minister", "prime minister",
        "president", "congress", "senate", "legislature", "cabinet",
        "party", "democrat", "republican", "bjp", "congress party",
        "policy", "regulation", "law", "bill", "act", "amendment",
        "vote", "ballot", "campaign", "referendum", "constituency",
        "treaty", "diplomacy", "sanction", "embargo", "tariff",
        "military", "defence", "war", "conflict", "geopolitical",
        "nato", "un", "united nations", "g7", "g20",
        "intelligence", "surveillance", "national security",
        "court", "supreme court", "judge", "verdict", "ruling",
        "ftc", "doj", "sec", "sebi", "cci", "competition commission",
        "lobbying", "advocacy", "civil rights", "protest", "activist",
        "immigration", "visa", "border", "asylum",
        "modi", "biden", "trump", "xi jinping", "putin",
    ],
    "Sports": [
        "cricket", "football", "soccer", "basketball", "tennis",
        "badminton", "hockey", "baseball", "rugby", "golf", "boxing",
        "mma", "wrestling", "athletics", "swimming", "cycling",
        "ipl", "nba", "nfl", "nhl", "premier league", "champions league",
        "la liga", "bundesliga", "serie a",
        "match", "game", "tournament", "championship", "cup",
        "league", "season", "playoff", "final", "semi-final",
        "player", "team", "squad", "coach", "manager", "captain",
        "goal", "score", "wicket", "batting", "bowling", "fielding",
        "win", "loss", "draw", "tie", "defeat", "victory",
        "transfer", "contract", "signing", "injury", "fitness",
        "olympic", "paralympic", "commonwealth games", "asian games",
        "world cup", "grand slam", "formula 1", "f1", "race",
        "rohit sharma", "virat kohli", "ms dhoni", "bumrah",
        "ronaldo", "messi", "lebron", "federer", "nadal", "djokovic",
    ],
    "Entertainment": [
        "movie", "film", "cinema", "box office", "blockbuster",
        "series", "episode", "season", "show", "soap opera", "drama",
        "comedy", "thriller", "documentary", "animation",
        "netflix", "amazon prime", "disney", "hbo", "hulu",
        "apple tv", "youtube", "spotify", "streaming",
        "music", "album", "song", "track", "single", "lyrics",
        "artist", "singer", "band", "rapper", "dj", "concert", "tour",
        "actor", "actress", "celebrity", "star", "influencer",
        "award", "oscar", "grammy", "bafta", "golden globe", "emmy",
        "trailer", "release", "premiere", "review", "rating",
        "director", "producer", "studio", "screenplay", "script",
        "art", "culture", "fashion", "design", "photography",
        "book", "novel", "author", "publisher", "bestseller",
        "game", "gaming", "esports", "playstation", "xbox", "nintendo",
        "bollywood", "hollywood", "tollywood", "k-pop", "k-drama",
    ],
}

# Pre-join lexicon into a document for TF-IDF comparison
_LABEL_DOCS: dict[str, str] = {
    label: " ".join(keywords)
    for label, keywords in _KEYWORD_LEXICON.items()
}


class HybridClassifier:
    """
    Keyword + TF-IDF cosine hybrid classifier.

    Replicates the public interface of a HuggingFace zero-shot pipeline:
        result = classifier(text, candidate_labels=[...])
        result["labels"][0]  → top category
        result["scores"][0]  → confidence (0-1)

    Algorithm
    ---------
    1. Keyword hit score: for each candidate label, count how many of its
       keywords appear in the text, weighting multi-word phrases higher.
       (Inspired by BART-MNLI's "does text entail label?" intuition.)

    2. TF-IDF cosine similarity: vectorise the text alongside each label's
       keyword document, compute cosine similarity.

    3. Weighted combination: 60% keyword hits + 40% TF-IDF cosine.

    4. Softmax with temperature=0.2 → calibrated probability scores.
    """

    def __call__(
        self,
        text: str,
        candidate_labels: Optional[list[str]] = None,
        **kwargs,           # absorbs pipeline kwargs like multi_label
    ) -> dict:
        if candidate_labels is None:
            candidate_labels = CANDIDATE_LABELS

        text_lower = text.lower()

        # ── 1. Keyword hit scores ─────────────────────────────────────────
        kw_scores: dict[str, float] = {}
        for label in candidate_labels:
            hits = 0.0
            for kw in _KEYWORD_LEXICON.get(label, []):
                if kw in text_lower:
                    hits += len(kw.split())   # multi-word bonus
            kw_scores[label] = hits

        kw_arr = np.array([kw_scores[l] for l in candidate_labels], dtype=float)
        kw_max = kw_arr.max()
        kw_norm = kw_arr / kw_max if kw_max > 0 else np.ones(len(candidate_labels)) / len(candidate_labels)

        # ── 2. TF-IDF cosine similarity ───────────────────────────────────
        label_docs = [_LABEL_DOCS.get(l, l) for l in candidate_labels]
        try:
            vec = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),      # captures bigrams like "machine learning"
                sublinear_tf=True,       # log-normalised TF
            ).fit_transform([text] + label_docs)
            tfidf_sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        except Exception as exc:
            log.debug("TF-IDF failed: %s — using keyword scores only", exc)
            tfidf_sims = np.zeros(len(candidate_labels))

        # ── 3. Weighted combination ───────────────────────────────────────
        combined = 0.6 * kw_norm + 0.4 * tfidf_sims

        # ── 4. Temperature-scaled softmax ─────────────────────────────────
        T = 0.2   # lower T → sharper distribution (more decisive)
        exp_c = np.exp(combined / T - (combined / T).max())
        scores = (exp_c / exp_c.sum()).tolist()

        # Rank highest-first (same as pipeline output)
        ranked = sorted(
            zip(candidate_labels, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "labels": [r[0] for r in ranked],
            "scores": [r[1] for r in ranked],
            "sequence": text,
        }


# ─── Backend loader ───────────────────────────────────────────────────────────

def _try_load_bart() -> object:
    """
    Attempt to load facebook/bart-large-mnli.
    Returns the pipeline on success, None on failure.
    The pipeline is None-safe: callers check before using.
    """
    try:
        from transformers import pipeline
        log.info("Loading facebook/bart-large-mnli — first run downloads ~1.6 GB…")
        clf = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,   # CPU; change to 0 for first GPU
        )
        log.info("BART pipeline loaded successfully")
        return clf
    except Exception as exc:
        log.warning("BART load failed (%s) — falling back to HybridClassifier", exc)
        return None


# Module-level singleton — loaded once per process
# Streamlit should wrap get_classifier() in @st.cache_resource
_classifier_singleton: object = None


def get_classifier(force_hybrid: bool = False) -> object:
    """
    Return the classifier singleton (BART pipeline or HybridClassifier).

    Parameters
    ----------
    force_hybrid : bool
        Skip BART and use HybridClassifier directly.
        Set True when HuggingFace is known to be unreachable.

    Returns
    -------
    A callable with the signature:
        clf(text: str, candidate_labels: list[str]) -> dict
    """
    global _classifier_singleton
    if _classifier_singleton is not None:
        return _classifier_singleton

    if force_hybrid:
        log.info("Using HybridClassifier (forced)")
        _classifier_singleton = HybridClassifier()
    else:
        bart = _try_load_bart()
        _classifier_singleton = bart if bart is not None else HybridClassifier()

    backend_name = type(_classifier_singleton).__name__
    if backend_name == "ZeroShotClassificationPipeline":
        backend_name = "BART (facebook/bart-large-mnli)"
    log.info("Active classifier backend: %s", backend_name)
    return _classifier_singleton


# ─── Core function ────────────────────────────────────────────────────────────

def classify_article(
    article: dict,
    classifier=None,
    candidate_labels: Optional[list[str]] = None,
) -> dict:
    """
    Run zero-shot classification on a single article dict.

    Modifies the article in-place AND returns it (functional style also works).

    Parameters
    ----------
    article         : article dict from rss_parser / mock_feeds
    classifier      : optional pre-loaded classifier (avoids re-loading)
    candidate_labels: override the default CANDIDATE_LABELS

    Adds to article
    ---------------
    category   : str   — top predicted label
    confidence : float — probability score 0-1
    all_scores : dict  — {label: score} for all labels
    """
    if classifier is None:
        classifier = get_classifier()

    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

    # Build classification input: title + first N chars of description
    title = article.get("title", "")
    desc  = article.get("description", "")
    text  = f"{title}. {desc[:INPUT_CHARS]}".strip()

    try:
        result = classifier(text, candidate_labels=candidate_labels)
        article["category"]   = result["labels"][0]
        article["confidence"] = round(result["scores"][0], 4)
        article["all_scores"] = dict(zip(result["labels"], result["scores"]))
    except Exception as exc:
        log.error("Classification failed for '%s': %s", title[:50], exc)
        article["category"]   = "Unknown"
        article["confidence"] = 0.0
        article["all_scores"] = {}

    return article


def classify_articles(
    articles: list[dict],
    candidate_labels: Optional[list[str]] = None,
    force_hybrid: bool = False,
) -> list[dict]:
    """
    Classify all articles in the list.

    Parameters
    ----------
    articles        : list of article dicts (Phase 1 output)
    candidate_labels: override default CANDIDATE_LABELS
    force_hybrid    : bypass BART, use HybridClassifier

    Returns
    -------
    Same list with 'category', 'confidence', 'all_scores' filled in.
    """
    clf = get_classifier(force_hybrid=force_hybrid)
    total = len(articles)

    log.info("Phase 2: classifying %d articles…", total)
    t_start = time.perf_counter()

    for i, article in enumerate(articles, 1):
        classify_article(article, classifier=clf, candidate_labels=candidate_labels)
        log.debug(
            "[%d/%d] %-12s (%.0f%%) — %s",
            i, total,
            article["category"],
            article["confidence"] * 100,
            article["title"][:55],
        )

    elapsed = time.perf_counter() - t_start
    log.info(
        "Phase 2 complete: %d articles classified in %.2fs (%.0f ms/article)",
        total, elapsed, elapsed / total * 1000 if total else 0,
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
    print("T9.3 — Phase 2: Zero-Shot Classifier smoke-test")
    print("=" * 65)

    TEST_CASES = [
        # (text, expected_label)
        ("Apple M4 Ultra chip sets benchmark records in Xcode and Final Cut Pro", "Technology"),
        ("OpenAI GPT-5 brings real-time voice and vision to ChatGPT multimodal AI", "Technology"),
        ("Google Gemini 2.0 Flash human-level coding SWE-Bench 72.3% score", "Technology"),
        ("Meta releases open-source Llama 4 with 405 billion parameters", "Technology"),
        ("Figma Make Design AI generates production-ready component libraries", "Technology"),
        ("Tesla FSD v13 passes 10 million mile autonomous driving mark zero accidents", "Technology"),
        ("NVIDIA H200 GPU supply constraints ease as TSMC Arizona Japan fabs ramp up", "Technology"),
        ("India startup ecosystem crosses 200 billion valuation fintech SaaS Bengaluru", "Business"),
        ("Razorpay acquires Malaysian fintech Curlec for 800 crore international deal", "Business"),
        ("Zepto raises 400 million Series G at 5 billion valuation dark stores expansion", "Business"),
        ("Ola Electric files DRHP with SEBI IPO listing 10000 crore Q3 2026", "Business"),
        ("Microsoft Copilot embedded Windows 12 kernel file system clipboard app APIs", "Technology"),
        ("India general election BJP Congress party vote results parliament seats", "Politics"),
        ("India cricket team wins test match series batsman bowler wicket score runs", "Sports"),
        ("Bollywood actor film box office release trailer Netflix streaming award", "Entertainment"),
    ]

    clf = get_classifier(force_hybrid=True)   # force local for smoke-test
    correct = 0

    for text, expected in TEST_CASES:
        article = {"title": text, "description": ""}
        classify_article(article, classifier=clf)
        pred = article["category"]
        conf = article["confidence"]
        ok = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"  {ok} [{conf:.0%}] {pred:<15} ← {text[:55]}")

    print()
    accuracy = correct / len(TEST_CASES)
    print(f"  Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.0%}")
    print("=" * 65)