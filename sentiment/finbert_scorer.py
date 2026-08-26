"""
FinBERT scorer — loads ProsusAI/finbert once per process and scores
batches of text. Reads label order from the model's own config rather
than hardcoding it, since FinBERT's label ordering has been reported
inconsistent across some config versions on HuggingFace.
"""

from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

_tokenizer = None
_model = None
_id2label = None


def _load_model():
    """
    Lazy-loaded, module-level singleton — loading FinBERT takes a few seconds
    and should happen once per process, not once per article.
    """
    global _tokenizer, _model, _id2label
    if _model is None:
        logger.info(f"Loading FinBERT model: {settings.finbert_model_name} (first call only, may take a moment)")
        _tokenizer = AutoTokenizer.from_pretrained(settings.finbert_model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(settings.finbert_model_name)
        _model.eval()  # inference mode, no gradient tracking needed

        # Read label order from the model's own config rather than hardcoding it —
        # FinBERT's id2label ordering has been reported inconsistent across
        # some published config versions, so trust the loaded model, not assumptions.
        _id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
        logger.info(f"FinBERT model loaded. Label mapping: {_id2label}")

        expected = {"positive", "negative", "neutral"}
        if set(_id2label.values()) != expected:
            raise ValueError(
                f"Unexpected FinBERT label set: {_id2label.values()}. "
                f"Expected {expected}. Aborting rather than silently mislabeling sentiment."
            )
    return _tokenizer, _model, _id2label


def score_batch(texts: List[str]) -> List[Dict]:
    """
    Scores a batch of texts at once (more efficient than one-at-a-time on CPU).
    Returns a list of dicts: {label, positive, negative, neutral, compound}
    in the same order as the input texts.
    """
    if not texts:
        return []

    tokenizer, model, id2label = _load_model()

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results = []
    for prob_row in probs:
        scores = {id2label[i]: prob_row[i].item() for i in range(len(id2label))}
        label = max(scores, key=scores.get)
        compound = scores["positive"] - scores["negative"]
        results.append({
            "label": label,
            "positive": scores["positive"],
            "negative": scores["negative"],
            "neutral": scores["neutral"],
            "compound": compound,
        })
    return results