# Finetuning BART on India Headlines Dataset

## Quick Start

### Step 1: Install Additional Dependencies
```bash
pip install kagglehub
```

### Step 2: Get Kaggle API Credentials
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" → downloads `kaggle.json`
3. Run in Python:
   ```python
   import kagglehub
   kagglehub.login()  # Follow the prompt to upload kaggle.json
   ```

### Step 3: Run Finetuning
```bash
python finetune_bart.py
```

This will:
- Download the India headlines dataset (~500 MB)
- Parse and preprocess headlines
- Map raw categories to: Technology, Business, Politics, Sports, Entertainment
- Create 10,000+ NLI training examples (balanced positive/negative)
- Finetune facebook/bart-large-mnli for 3 epochs
- Save the finetuned model to: `./bart-finetuned-india-headlines/`

**Typical runtime:**
- GPU: 30-45 minutes
- CPU: 2-3 hours (not recommended)

### Step 4: Use Finetuned Model

#### Option A: Use `classifier_finetuned.py` (drop-in replacement)
```bash
# Rename or update in app.py
from classifier_finetuned import classify_articles

# Pass use_finetuned=True
articles = classify_articles(articles, use_finetuned=True)
```

#### Option B: Patch existing `classifier.py`
Update the `get_classifier()` function:

```python
def get_classifier(force_hybrid: bool = False, use_finetuned: bool = False) -> object:
    """Return the classifier singleton."""
    global _classifier_singleton
    if _classifier_singleton is not None:
        return _classifier_singleton

    if force_hybrid:
        log.info("Using HybridClassifier (forced)")
        _classifier_singleton = HybridClassifier()
    else:
        # NEW: Support finetuned model
        model_path = None
        if use_finetuned and os.path.exists("./bart-finetuned-india-headlines"):
            model_path = "./bart-finetuned-india-headlines"
        
        bart = _try_load_bart(model_path)  # Pass model_path to _try_load_bart
        _classifier_singleton = bart if bart is not None else HybridClassifier()

    backend_name = type(_classifier_singleton).__name__
    if backend_name == "ZeroShotClassificationPipeline":
        backend_name = "BART (facebook/bart-large-mnli)"
    log.info("Active classifier backend: %s", backend_name)
    return _classifier_singleton
```

Then update `_try_load_bart()` to accept `model_path`:

```python
def _try_load_bart(model_path: Optional[str] = None) -> object:
    """Attempt to load facebook/bart-large-mnli or finetuned variant."""
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
```

### Step 5: Integrate with App

In `app.py` or `pipeline_runner.py`:

```python
# Update the load_articles() function:
@st.cache_data(ttl=CACHE_TTL)
def load_articles() -> list[dict]:
    """Phase 1 + Phase 2 with optional finetuned model."""
    if USE_MOCK:
        from mock_feeds import fetch_articles
    else:
        from rss_parser import fetch_articles

    from classifier import classify_articles
    articles = fetch_articles(max_articles=MAX_ARTICLES)
    
    # Use finetuned model if available
    use_finetuned = os.path.exists("./bart-finetuned-india-headlines")
    articles = classify_articles(articles, force_hybrid=FORCE_HYBRID, use_finetuned=use_finetuned)
    return articles
```

## Dataset Details

**Dataset:** therohk/india-headlines-news-dataset
- **Size:** ~100,000+ headlines
- **Categories:** Auto-mapped to 5 labels (Technology, Business, Politics, Sports, Entertainment)
- **Format:** CSV with headline text and category

## Finetuning Hyperparameters

Defaults in `finetune_bart.py`:
- **Batch size:** 8
- **Learning rate:** 5e-5
- **Epochs:** 3
- **Max seq length:** 512 tokens
- **Warmup steps:** 500
- **NLI examples per category:** ~500 positive + ~500 negative = 1000 per category

Adjust `finetune_bart.py` to tune these:
```python
BATCH_SIZE = 16        # Increase for GPU, decrease for CPU
LEARNING_RATE = 5e-5   # Lower for more stable training
NUM_EPOCHS = 5         # More epochs = longer training but potentially better
```

## Expected Improvements

- **Before:** BART is general-purpose (trained on diverse Wikipedia/web text)
- **After:** BART is optimized for Indian tech news headlines
- **Expected boost:** 3-8% accuracy improvement on India headlines dataset

For example:
- Better recognition of: Flipkart, Reliance, Tata, SEBI, IPL, Bollywood
- Better handling of: Indian English, rupees (₹), local abbreviations

## Troubleshooting

### "No CSV files found"
- Ensure kagglehub downloaded the dataset correctly
- Check the path printed by `download_dataset()`
- Manually inspect the downloaded CSV structure

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in finetune_bart.py
- Or set `device=-1` to force CPU (slower)

### "Kagglehub authentication failed"
```python
import kagglehub
kagglehub.logout()
kagglehub.login()  # Re-authenticate
```

### Model still using base BART
- Ensure finetuned model saved to `./bart-finetuned-india-headlines/`
- Pass `use_finetuned=True` explicitly
- Check logs: "Active classifier backend: BART (finetuned on India headlines)"

## File Structure After Finetuning

```
SMAI-A3/
├── finetune_bart.py                      (new: finetuning script)
├── classifier_finetuned.py               (new: updated classifier)
├── bart-finetuned-india-headlines/       (created after finetuning)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ... (other HF model files)
└── ... (existing files)
```

## Next Steps

1. Run `python finetune_bart.py`
2. Wait for training to complete
3. Update app.py or pipeline_runner.py to use the finetuned model
4. Test with `streamlit run app.py`
5. Compare accuracy on India headlines vs. mock/real feeds
