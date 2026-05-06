"""
finetune_bart.py — Finetune facebook/bart-large-mnli on India Headlines Dataset
================================================================================
Loads india-news-headlines.csv, filters for tech headlines, converts them into
NLI (entailment/contradiction) pairs, and finetunes BART for zero-shot
classification.

Dataset expected columns:
    publish_date, headline_category, headline_text

Usage:
    python finetune_bart.py

Output:
    ./bart-finetuned-india-headlines/

To use the finetuned model in classifier.py:
    Set USE_FINETUNED = True  (it already points to the right output path)

Requirements:
    pip install transformers datasets torch scikit-learn pandas accelerate
"""

import logging
import os
import pandas as pd
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ─── Config ───────────────────────────────────────────────────────────────────

CSV_PATH        = "./india-news-headlines.csv"
BASE_MODEL      = "facebook/bart-large-mnli"
OUTPUT_DIR      = "./bart-finetuned-india-headlines"
CACHE_DIR       = "./hf_cache"

MAX_HEADLINES   = 5000   # tech headlines to use (keep training time reasonable)
MAX_SEQ_LENGTH  = 128    # shorter than 512 speeds things up significantly
BATCH_SIZE      = 8
NUM_EPOCHS      = 3
LEARNING_RATE   = 5e-5

# All five categories the classifier uses
CATEGORIES = ["Technology", "Business", "Politics", "Sports", "Entertainment"]

# Keywords used to identify tech headlines in the dataset
TECH_KEYWORDS = [
    "tech", "technology", "software", "hardware", "internet", "digital",
    "computer", "mobile", "startup", "innovation", "cyber", "ai",
    "artificial intelligence", "science", "gadget", "app", "it ",
]

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_gpu() -> str:
    """Return device string and log GPU info (or warn if CPU-only)."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        log.info("GPU detected: %s", name)
        return "cuda"
    log.warning("No GPU found — training on CPU will be very slow.")
    return "cpu"


def is_tech(category: str) -> bool:
    """Return True if the raw headline_category looks tech-related."""
    cat = category.lower()
    return any(kw in cat for kw in TECH_KEYWORDS)


# ─── Data pipeline ────────────────────────────────────────────────────────────

def load_tech_headlines(csv_path: str) -> list[str]:
    """
    Load the CSV and return a list of tech headline strings.

    The India headlines dataset has three columns:
        publish_date, headline_category, headline_text

    Because almost all rows are labelled 'unknown', we fall back to
    keyword-matching the headline_text itself when the category gives
    no signal.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download from Kaggle and place it in the project root."
        )

    log.info("Loading %s …", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    log.info("Loaded %d rows. Columns: %s", len(df), df.columns.tolist())

    # Normalise column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # Identify text and category columns
    text_col = next(
        (c for c in ["headline_text", "headline", "title", "text"] if c in df.columns),
        None,
    )
    cat_col = next(
        (c for c in ["headline_category", "category", "label", "section"] if c in df.columns),
        None,
    )

    if text_col is None:
        raise ValueError(f"Cannot find a headline/text column. Got: {df.columns.tolist()}")
    if cat_col is None:
        raise ValueError(f"Cannot find a category column. Got: {df.columns.tolist()}")

    log.info("Using text='%s', category='%s'", text_col, cat_col)

    df = df[[text_col, cat_col]].dropna()
    df.columns = ["text", "category"]

    # Filter tech rows:
    # 1. Category column has a tech-like label
    # 2. OR headline text itself contains a tech keyword (catches 'unknown' rows)
    cat_match  = df["category"].str.lower().apply(is_tech)
    text_match = df["text"].str.lower().apply(
        lambda t: any(kw in t for kw in TECH_KEYWORDS)
    )
    tech_df = df[cat_match | text_match].drop_duplicates(subset="text")

    log.info("Tech headlines found: %d", len(tech_df))
    if tech_df.empty:
        raise ValueError("No tech headlines found. Check TECH_KEYWORDS or the CSV.")

    headlines = tech_df["text"].head(MAX_HEADLINES).tolist()
    log.info("Using %d headlines for training.", len(headlines))
    return headlines


def build_nli_examples(headlines: list[str]) -> list[dict]:
    """
    Convert each headline into one NLI pair per category.

    BART's NLI head has exactly 3 labels (this is fixed in the pretrained weights):
        0 = contradiction
        1 = neutral
        2 = entailment

    We must use these same indices — passing num_labels=2 causes the size-mismatch
    error you saw. We skip neutral and only produce entailment / contradiction pairs:
        Technology hypothesis  → entailment   (label 2)
        Other category         → contradiction (label 0)
    """
    # BART's fixed label indices — do NOT change these
    ENTAILMENT    = 2
    CONTRADICTION = 0

    examples = []
    for headline in headlines:
        for category in CATEGORIES:
            examples.append({
                "text":     headline,
                "category": category,
                "label":    ENTAILMENT if category == "Technology" else CONTRADICTION,
            })

    n_pos = sum(1 for e in examples if e["label"] == ENTAILMENT)
    n_neg = len(examples) - n_pos
    log.info("NLI pairs: %d total  (entailment=%d, contradiction=%d)", len(examples), n_pos, n_neg)
    return examples


def make_hf_dataset(examples: list[dict]) -> DatasetDict:
    """Split examples 80/20 and return a HuggingFace DatasetDict."""
    train_ex, val_ex = train_test_split(examples, test_size=0.2, random_state=42)

    def to_ds(exs):
        return Dataset.from_dict({
            "text":     [e["text"]     for e in exs],
            "category": [e["category"] for e in exs],
            "label":    [e["label"]    for e in exs],
        })

    ds = DatasetDict({"train": to_ds(train_ex), "validation": to_ds(val_ex)})
    log.info("Train: %d  |  Validation: %d", len(ds["train"]), len(ds["validation"]))
    return ds


# ─── Tokenisation ─────────────────────────────────────────────────────────────

def tokenize(batch, tokenizer):
    """
    Tokenize premise (headline) + hypothesis ('This text is about {category}.')
    BART NLI uses this exact format during zero-shot inference, so we match it.
    """
    hypotheses = [f"This text is about {cat}." for cat in batch["category"]]
    encoded = tokenizer(
        batch["text"],
        hypotheses,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    encoded["labels"] = batch["label"]
    return encoded


# ─── Training ─────────────────────────────────────────────────────────────────

def train(ds: DatasetDict, tokenizer, model, device: str):
    """Tokenize the dataset and run the HuggingFace Trainer."""
    log.info("Tokenising dataset …")
    tokenized = ds.map(
        lambda batch: tokenize(batch, tokenizer),
        batched=True,
        remove_columns=["text", "category"],
    )

    use_fp16 = device == "cuda"

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=use_fp16,
        gradient_accumulation_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",           # disable wandb / other trackers
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    log.info("Starting training …")
    trainer.train()
    return trainer


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("BART Finetuning — India Tech Headlines")
    log.info("=" * 60)

    device = check_gpu()

    # 1. Load data
    headlines = load_tech_headlines(CSV_PATH)

    # 2. Build NLI pairs
    examples = build_nli_examples(headlines)

    # 3. HuggingFace dataset
    ds = make_hf_dataset(examples)

    # 4. Load tokenizer and model
    log.info("Loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR)
    # Do NOT pass num_labels — bart-large-mnli has a fixed 3-label head
    # (contradiction=0, neutral=1, entailment=2) baked into its pretrained weights.
    # Overriding num_labels resizes that head and causes the shape-mismatch error.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        cache_dir=CACHE_DIR,
    )
    model.to(device)

    # 5. Train
    trainer = train(ds, tokenizer, model, device)

    # 6. Save
    log.info("Saving finetuned model to: %s", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    log.info("=" * 60)
    log.info("Done! Model saved to: %s", OUTPUT_DIR)
    log.info("In classifier.py, set  USE_FINETUNED = True  to use it.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()