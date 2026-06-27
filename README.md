# SMAI-A3 — Tech News Tracker

A simple, practical pipeline that fetches tech news, classifies each article into a category, and generates readable summaries. This README now focuses on how to use the project and explains what each component does and how they connect.

Repository: https://github.com/BhavyaV05/SMAI-A3

---

## What this project does (plain English)

1. Reads RSS feeds (TechCrunch, The Verge, YourStory) and extracts headlines, short descriptions and links.
2. Decides which category each article belongs to (Technology, Business, Politics, Sports, Entertainment).
3. Fetches the full article (when possible) and produces a human-style summary.
4. Shows everything in a Streamlit web UI where you can filter and read summaries.

---

## Files you need to know (what each does)

- app.py
  - Streamlit UI. Provides filters, displays article cards, and shows summaries. Adjustable switches at top (USE_MOCK, FORCE_HYBRID, MAX_ARTICLES).

- pipeline_runner.py
  - Runs the full pipeline: Phase 1 → Phase 2 → Phase 3. Useful for CLI testing and debugging (can run in mock mode).

- rss_parser.py
  - Phase 1: fetches configured RSS feeds, cleans HTML, deduplicates articles by URL, and returns a list of normalized article dicts.

- classifier.py
  - Phase 2: assigns categories to each article. By default it attempts to use Hugging Face BART zero-shot pipeline. If BART is unavailable it falls back to a fast keyword + TF-IDF hybrid classifier.

- finetune_bart.py
  - Optional: downloads an India headlines dataset and finetunes a local BART model to improve classification accuracy for Indian news.

- summariser.py
  - Phase 3: attempts to fetch the full article body with `trafilatura`, then calls Google Gemini (if GEMINI_API_KEY is set) to create a readable summary. Uses rate limiting and retries. Falls back to a short summary made from RSS description if LLM is unavailable.

- mock_feeds.py
  - Offline sample articles for development and tests without network calls.

- requirements.txt
  - Python packages to install.

---

## How to use (step-by-step)

1. Clone and install dependencies:

```bash
git clone https://github.com/BhavyaV05/SMAI-A3.git
cd SMAI-A3
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Optional: Add Gemini API key for better summaries

Create a `.env` file in the repo root with:

```
GEMINI_API_KEY=your_key_here
```

If you do not set GEMINI_API_KEY, the summariser will still work but will use a fallback that constructs summaries from RSS descriptions.

3. Quick run (single-machine)

- Start the Streamlit UI (recommended):

```bash
streamlit run app.py
```

Open http://localhost:8501. Use the sidebar to select categories, toggle mock/live feeds, and adjust article count.

- Run the full pipeline from the command line (no UI):

```bash
python pipeline_runner.py
```

This prints timings, category breakdowns, and sample summaries.

4. Development tips

- To avoid large downloads or slow model loading, set `FORCE_HYBRID=True` in `app.py` to use the fast local fallback classifier.
- For offline testing set `USE_MOCK=True` in `app.py` or run `pipeline_runner.py` with the mock option (it uses `mock_feeds.py`).
- To train a local finetuned classifier, follow `FINETUNE_GUIDE.md` and run `python finetune_bart.py` (GPU recommended).

---

## What you will see (expected behavior & outputs)

- pipeline_runner.py output:
  - Phase timings, number of articles fetched, category counts, and whether summaries were produced by Gemini or by fallback.

- app.py UI:
  - Grid of article cards. Each card shows title, source, category, confidence, and a short summary.
  - Click a card to view the full summary and link to original article.

- Files created optionally:
  - `./bart-finetuned-india-headlines/` when you run finetuning.

---

## Environment variables and config

- GEMINI_API_KEY — (optional) Google Gemini API key for high-quality LLM summaries.
- Edit top of `app.py` to change runtime options:
  - `USE_MOCK` (True/False) — use offline sample feeds
  - `FORCE_HYBRID` (True/False) — skip BART and use keyword+TF-IDF
  - `MAX_ARTICLES` (integer) — how many articles to process

---

## Troubleshooting (quick)

- If the app is slow or BART fails to load: set `FORCE_HYBRID=True` in `app.py`.
- If summaries are missing or you get rate-limited: ensure `GEMINI_API_KEY` is set and valid; summariser has built-in rate pacing and retries.
- If `trafilatura` fails to fetch article content (paywall or 403): summariser falls back to the RSS description.

---

## Next steps (optional improvements)

- Add a Dockerfile for containerized deployment.
- Add a small Redis cache for summaries to reduce LLM calls.
- Add unit tests for rss parsing and classifier fallback.

---

If you'd like, I can update the README in-place in the repository now (add this content to README.md). Would you like me to push this change?"}]}]