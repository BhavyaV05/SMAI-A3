"""
classifier_finetuned.py — Drop-in replacement for classifier.py using finetuned BART
=====================================================================================
This is an updated version of classifier.py that can use either:
  1. The original facebook/bart-large-mnli (zero-shot, general purpose)
  2. A finetuned model (trained on India headlines dataset)

To switch to the finetuned model, set USE_FINETUNED = True

Instructions
------------
1. First, run: python finetune_bart.py
2. Then, replace classifier.py with this file OR update classifier.py directly
3. In get_classifier(), pass use_finetuned=True to use your custom model

Or just patch the _try_load_bart() function in your existing classifier.py.
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

INPUT_CHARS = 200

# ────────────────────────────────────────────────────────────────────────────
# FINETUNED MODEL CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────
FINETUNED_MODEL_PATH = "./bart-finetuned-india-headlines"
USE_FINETUNED = True  # Set to True to use the finetuned model by default


# ─── Hybrid fallback classifier ───────────────────────────────────────────────

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

_LABEL_DOCS: dict[str, str] = {
    label: " ".join(keywords)
    for label, keywords in _KEYWORD_LEXICON.items()
}


class HybridClassifier:
    """Keyword + TF-IDF cosine hybrid classifier (identical to original)."""

    def __call__(
        self,
        text: str,
        candidate_labels: Optional[list[str]] = None,
        **kwargs,
    ) -> dict:
        if candidate_labels is None:
            candidate_labels = CANDIDATE_LABELS

        text_lower = text.lower()

        kw_scores: dict[str, float] = {}
        for label in candidate_labels:
            hits = 0.0
            for kw in _KEYWORD_LEXICON.get(label, []):
                if kw in text_lower:
                    hits += len(kw.split())
            kw_scores[label] = hits

        kw_arr = np.array([kw_scores[l] for l in candidate_labels], dtype=float)
        kw_max = kw_arr.max()
        kw_norm = kw_arr / kw_max if kw_max > 0 else np.ones(len(candidate_labels)) / len(candidate_labels)

        label_docs = [_LABEL_DOCS.get(l, l) for l in candidate_labels]
        try:
            vec = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ).fit_transform([text] + label_docs)
            tfidf_sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        except Exception as exc:
            log.debug("TF-IDF failed: %s — using keyword scores only", exc)
            tfidf_sims = np.zeros(len(candidate_labels))

        combined = 0.6 * kw_norm + 0.4 * tfidf_sims

        T = 0.2
        exp_c = np.exp(combined / T - (combined / T).max())
        scores = (exp_c / exp_c.sum()).tolist()

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

def _try_load_bart(model_path: Optional[str] = None) -> object:
    """
    Attempt to load BART model.
    
    Parameters
    ----------
    model_path : str, optional
        Path to a finetuned BART model. If None, uses the base
        facebook/bart-large-mnli from HuggingFace Hub.
    
    Returns
    -------
    The pipeline on success, None on failure.
    """
    try:
        from transformers import pipeline

        if model_path:
            log.info("Loading finetuned BART from: %s", model_path)
        else:
            log.info("Loading facebook/bart-large-mnli — first run downloads ~1.6 GB…")

        clf = pipeline(
            "zero-shot-classification",
            model=model_path or "facebook/bart-large-mnli",
            device=-1,   # CPU; change to 0 for first GPU
        )
        log.info("BART pipeline loaded successfully")
        return clf
    except Exception as exc:
        log.warning("BART load failed (%s) — falling back to HybridClassifier", exc)
        return None


_classifier_singleton: object = None


def get_classifier(
    force_hybrid: bool = False,
    use_finetuned: bool = False,
) -> object:
    """
    Return the classifier singleton.

    Parameters
    ----------
    force_hybrid : bool
        Skip BART and use HybridClassifier directly.
    use_finetuned : bool
        Load the finetuned BART model instead of the base model.
        Requires finetune_bart.py to have been run successfully.

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
        # Try to load BART (finetuned or base)
        model_path = None
        if use_finetuned:
            if os.path.exists(FINETUNED_MODEL_PATH):
                model_path = FINETUNED_MODEL_PATH
            else:
                log.warning(
                    "Finetuned model not found at %s — using base model",
                    FINETUNED_MODEL_PATH,
                )

        bart = _try_load_bart(model_path)
        _classifier_singleton = bart if bart is not None else HybridClassifier()

    backend_name = type(_classifier_singleton).__name__
    if backend_name == "ZeroShotClassificationPipeline":
        if use_finetuned:
            backend_name = "BART (finetuned on India headlines)"
        else:
            backend_name = "BART (facebook/bart-large-mnli)"

    log.info("Active classifier backend: %s", backend_name)
    return _classifier_singleton


# ─── Core functions (identical to original) ───────────────────────────────────

def classify_article(
    article: dict,
    classifier=None,
    candidate_labels: Optional[list[str]] = None,
) -> dict:
    """Run zero-shot classification on a single article dict."""
    if classifier is None:
        classifier = get_classifier()

    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

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
    use_finetuned: bool = False,
) -> list[dict]:
    """
    Classify all articles in the list.
    
    Parameters
    ----------
    articles        : list of article dicts (Phase 1 output)
    candidate_labels: override default CANDIDATE_LABELS
    force_hybrid    : bypass BART, use HybridClassifier
    use_finetuned   : use finetuned BART instead of base model

    Returns
    -------
    Same list with 'category', 'confidence', 'all_scores' filled in.
    """
    clf = get_classifier(force_hybrid=force_hybrid, use_finetuned=use_finetuned)
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
