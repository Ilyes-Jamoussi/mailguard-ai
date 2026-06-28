from __future__ import annotations

import torch

from src.config import ModelConfig
from src.transformer_model import TransformerClassifier


def test_forward_returns_logits_per_class() -> None:
    config = ModelConfig(vocab_size=200, max_len=16)
    model = TransformerClassifier.from_config(config)
    tokens = torch.randint(0, config.vocab_size, (4, config.max_len))

    logits = model(tokens)

    assert logits.shape == (4, config.num_classes)


def test_padding_does_not_produce_nans() -> None:
    config = ModelConfig(vocab_size=200, max_len=16)
    model = TransformerClassifier.from_config(config)
    # A sequence with a single real token followed by padding.
    tokens = torch.zeros((1, config.max_len), dtype=torch.long)
    tokens[0, 0] = 5

    logits = model(tokens)

    assert torch.isfinite(logits).all()
