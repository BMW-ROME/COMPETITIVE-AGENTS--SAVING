"""HF inference helper.

Provides a small wrapper around huggingface_hub.InferenceClient to classify
financial and social sentiment texts, with graceful fallback when the
library or token is unavailable. Models are configurable via environment
variables and have sensible defaults.
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

try:
    from huggingface_hub import InferenceClient  # type: ignore
    _HF_OK = True
except Exception:
    InferenceClient = None  # type: ignore
    _HF_OK = False


class HFClassifier:
    """Small wrapper for Hugging Face InferenceClient.

    Reads token from HF_TOKEN or HUGGINGFACE_API_KEY and uses default models that
    can be overridden via environment variables.
    """

    def __init__(self) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.client: Optional[InferenceClient] = None
        if _HF_OK and token:
            try:
                self.client = InferenceClient(api_key=token)
            except Exception:
                self.client = None

        # Default models with env overrides
        self.fin_model = os.getenv(
            "HF_MODEL_FIN_SENTIMENT", "ProsusAI/finbert"
        )
        self.social_model = os.getenv(
            "HF_MODEL_SOCIAL_SENTIMENT", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.summarizer_model = os.getenv(
            "HF_MODEL_SUMMARIZER", "sshleifer/distilbart-cnn-12-6"
        )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _score_signed(self, preds: List[Dict[str, Any]]) -> float:
        pos = next(
            (p.get("score", 0.0) for p in preds if str(p.get("label", "")).lower().startswith("pos")),
            0.0,
        )
        neg = next(
            (p.get("score", 0.0) for p in preds if str(p.get("label", "")).lower().startswith("neg")),
            0.0,
        )
        return float(pos - neg)

    def classify_financial(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify financial sentiment for a list of texts.

        Returns list per text: {label, score, score_signed}
        score_signed in [-1,1] approximated as P(pos) - P(neg).
        """
        if not self.client:
            return [{"label": "neutral", "score": 0.0, "score_signed": 0.0} for _ in texts]
        out: List[Dict[str, Any]] = []
        for t in texts:
            try:
                preds = self.client.text_classification(t, model=self.fin_model)
                best = max(preds, key=lambda x: x.get("score", 0.0)) if preds else {"label": "neutral", "score": 0.0}
                out.append({"label": best.get("label", "neutral"), "score": best.get("score", 0.0), "score_signed": self._score_signed(preds)})
            except Exception:
                out.append({"label": "neutral", "score": 0.0, "score_signed": 0.0})
        return out

    def classify_social(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify social sentiment for a list of texts.

        Returns list per text: {label, score, score_signed}
        score_signed in [-1,1] approximated as P(pos) - P(neg).
        """
        if not self.client:
            return [{"label": "neutral", "score": 0.0, "score_signed": 0.0} for _ in texts]
        out: List[Dict[str, Any]] = []
        for t in texts:
            try:
                preds = self.client.text_classification(t, model=self.social_model)
                best = max(preds, key=lambda x: x.get("score", 0.0)) if preds else {"label": "neutral", "score": 0.0}
                out.append({"label": best.get("label", "neutral"), "score": best.get("score", 0.0), "score_signed": self._score_signed(preds)})
            except Exception:
                out.append({"label": "neutral", "score": 0.0, "score_signed": 0.0})
        return out

    def summarize(self, texts: List[str], max_words: int = 60) -> List[str]:
        """Summarize a list of texts."""
        if not self.client:
            return [""] * len(texts)
        outs: List[str] = []
        for t in texts:
            try:
                s = self.client.summarization(
                    t, model=self.summarizer_model, max_new_tokens=128, temperature=0.3
                )
                text = s if isinstance(s, str) else str(s)
                outs.append(" ".join(text.split()[:max_words]))
            except Exception:
                outs.append("")
        return outs
