from __future__ import annotations

import torch
from torch import nn

from src.config import ModelConfig
from src.transformer_model import TransformerClassifier


def test_training_steps_reduce_loss() -> None:
    torch.manual_seed(0)
    config = ModelConfig(vocab_size=100, max_len=16)
    model = TransformerClassifier.from_config(config)
    tokens = torch.randint(1, config.vocab_size, (8, config.max_len))
    labels = torch.randint(0, config.num_classes, (8,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_before = criterion(model(tokens), labels).item()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(tokens), labels)
        loss.backward()
        optimizer.step()
    loss_after = criterion(model(tokens), labels).item()

    assert loss_after < loss_before
