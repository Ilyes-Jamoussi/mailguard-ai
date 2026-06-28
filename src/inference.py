"""Load the trained model and classify raw email text.

This is the boundary the UI imports: it owns all model/tensor logic so the
Streamlit app only has to render the returned :class:`Prediction`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from src.config import (
    CLASS_NAMES,
    METRICS_PATH,
    MODEL_CONFIG_PATH,
    VOCAB_PATH,
    WEIGHTS_PATH,
    ModelConfig,
)
from src.preprocessing import TextPreprocessor
from src.transformer_model import TransformerClassifier

_MIN_TEXT_CHARS = 10


@dataclass(frozen=True)
class Prediction:
    """Outcome of classifying one email."""

    label: str
    confidence: float
    probabilities: dict[str, float]


class SpamClassifier:
    """Trained model plus its fitted vocabulary, ready to score raw text."""

    def __init__(self, model: TransformerClassifier, preprocessor: TextPreprocessor) -> None:
        self._model = model
        self._preprocessor = preprocessor

    @classmethod
    def load(cls) -> SpamClassifier:
        """Load the exported config, weights, and vocabulary from ``models/``."""
        config = ModelConfig(**json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8")))
        model = TransformerClassifier.from_config(config)
        model.load_state_dict(load_file(str(WEIGHTS_PATH)))
        model.eval()
        return cls(model, TextPreprocessor.load(VOCAB_PATH))

    def predict(self, text: str) -> Prediction:
        """Classify ``text`` as ham or spam with class probabilities."""
        sequence = torch.tensor([self._preprocessor.text_to_sequence(text)], dtype=torch.long)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(sequence), dim=1)[0]
        index = int(probabilities.argmax())
        return Prediction(
            label=CLASS_NAMES[index],
            confidence=float(probabilities[index]),
            probabilities={name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        )


def is_analyzable(text: str) -> bool:
    """Return whether ``text`` is long enough to be worth classifying."""
    return len(text.strip()) >= _MIN_TEXT_CHARS


def load_metrics() -> dict | None:
    """Return the saved training metrics, or ``None`` if they are missing."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
