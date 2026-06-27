# SMAI-A3 — Tech News Tracker

An end-to-end NLP pipeline that fetches tech news from RSS feeds, classifies articles, and generates rich summaries using LLMs. Built for learning, demos, and lightweight production use.

Repository: https://github.com/BhavyaV05/SMAI-A3

---

## What this project does

1. Phase 1 — RSS Parsing: fetches and normalises articles from configured RSS/Atom feeds (TechCrunch, The Verge, YourStory).
2. Phase 2 — Classification: assigns each article to one of five categories (Technology, Business, Politics, Sports, Entertainment) using zero-shot BART; falls back to a keyword+TF-IDF hybrid classifier and supports an optional finetuned BART.
3. Phase 3 — Summarisation: fetches full article text via `trafilatura` and generates a substantive multi-paragraph summary using Google Gemini (via google-generativeai SDK). Rate-limited and retried.
4. UI — Streamlit dashboard (app.py) to browse, filter and view classified articles and summaries.

---

## Repository layout

- app.py                  — Streamlit application (UI)
- pipeline_runner.py      — Orchestrates Phase 1 → 2 → 3 end-to-end
- rss_parser.py           — Phase 1: feed fetching and normalization
- classifier.py           — Phase 2: zero-shot + hybrid fallback classification
- finetune_bart.py        — Optional: finetune facebook/bart-large-mnli on India headlines
- summariser.py           — Phase 3: fetch article body and call Gemini
- mock_feeds.py           — Offline test data for development
- requirements.txt        — Python dependencies
- FINETUNE_GUIDE.md       — Detailed finetuning instructions

---

## Quick start

1. Clone the repo:

```bash
git clone https://github.com/BhavyaV05/SMAI-A3.git
cd SMAI-A3
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. (Optional) Add an API key for Google Gemini in `.env`:

```
GEMINI_API_KEY=your_key_here
```

4. Run the Streamlit UI:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running the pipeline CLI

- Run the full pipeline (mock mode):

```bash
python pipeline_runner.py
```

- Test RSS parsing only:

```bash
python rss_parser.py
```

- Test summariser (needs GEMINI_API_KEY or will use fallback):

```bash
python summariser.py
```

- Run finetuning (requires extra packages and GPU for reasonable speed):

```bash
python finetune_bart.py
```

See `FINETUNE_GUIDE.md` for details.

---

## How it works — short technical summary

Phase 1 (rss_parser.py):
- Uses `feedparser` to read feeds, cleans HTML, extracts `title`, `description`, `url`, `published`.
- Deduplicates by URL and sorts newest-first.

Phase 2 (classifier.py):
- Primary: Hugging Face `pipeline("zero-shot-classification", model="facebook/bart-large-mnli")` to get entailment-based label scores.
- Fallback: `HybridClassifier` combining keyword counts and TF-IDF cosine similarity (60% keyword, 40% TF-IDF), with softmax temperature sharpening.
- Optional: Finetuned BART trained on India headlines dataset (see finetune_bart.py / FINETUNE_GUIDE.md) for better local-domain accuracy.

Phase 3 (summariser.py):
- Fetches article body with `trafilatura` (preferred) or falls back to RSS description.
- Sends prompt + article body to Google Gemini via `google-generativeai` client.
- Enforces rate limits with a token-bucket pacer and retries with exponential backoff on 429s.
- Post-processes Gemini output (strip markdown, normalize paragraphs).

---

## Configuration & options

Edit runtime switches in `app.py` or pass flags in `pipeline_runner.py`:

- `USE_MOCK` (app.py) — use mock_feeds instead of live RSS.
- `FORCE_HYBRID` — force keyword+TF-IDF fallback (no BART download).
- `MAX_ARTICLES` — cap total articles processed.
- `GEMINI_API_KEY` — set in environment or `.env` for summaries.

Finetuned model path used by classifier: `./bart-finetuned-india-headlines` (created by finetune_bart.py when run).

---

## ML concepts used (for interviews)

- Zero-shot learning using NLI (BART + MNLI): classify without labeled training data by testing entailment for each label.
- Transfer learning and finetuning: adapt pre-trained BART to India headlines for domain performance gains.
- TF-IDF + cosine similarity: classic IR technique used in the hybrid fallback.
- Prompt engineering and LLM generation: tailored system prompt for journalist-style summaries.
- Rate limiting & exponential backoff: production-friendly API usage patterns.

Concepts intentionally not used but relevant: supervised end-to-end classifier (requires labeled dataset), dense embeddings + vector DB for semantic search, active learning loops.

---

## Production considerations

- Secrets: keep `GEMINI_API_KEY` out of source control (.env is ignored).
- Caching: Streamlit caching used for UI; consider Redis for scale.
- Rate limiting: summariser enforces a minimum gap between requests; adjust for paid tiers.
- Monitoring & logs: modules log progress and warnings; integrate with a logging/observability stack for production.

---

## Troubleshooting & tips

- If BART fails to load or memory is low: set `FORCE_HYBRID=True` to use hybrid classifier.
- If Gemini returns empty or rate-limited: check `GEMINI_API_KEY`, inspect logs, or let summariser use fallback.
- If finetuning fails due to OOM on GPU: reduce `BATCH_SIZE` in `finetune_bart.py` or train on CPU (much slower).

---

## License & attribution

This project contains third-party models and libraries (Hugging Face, Google Gemini, trafilatura). Respect their licenses and API terms.

---

If you want, I can also add a minimal CONTRIBUTING.md, CI workflow, or a Dockerfile to run this as a containerized service.

