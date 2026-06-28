"""Train the Transformer spam classifier and export model artifacts.

Run from the repository root with ``python -m src.train``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    CLASS_NAMES,
    METRICS_PATH,
    MODEL_CONFIG_PATH,
    MODELS_DIR,
    SEED,
    VOCAB_PATH,
    WEIGHTS_PATH,
    ModelConfig,
    TrainConfig,
)
from src.preprocessing import load_and_preprocess_data
from src.transformer_model import TransformerClassifier

logger = logging.getLogger(__name__)


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and torch RNGs so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Use CUDA when available, otherwise CPU (never hardcode the device)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmailDataset(Dataset):
    """Wrap encoded sequences and labels as tensors for a DataLoader."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    """Run one training epoch; return (mean loss, accuracy %)."""
    model.train()
    loss_sum = correct = total = 0.0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute (mean loss, accuracy %) on a loader without updating weights."""
    model.eval()
    loss_sum = correct = total = 0.0
    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        logits = model(sequences)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / len(loader), 100.0 * correct / total


@torch.no_grad()
def _predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, labels) over a loader."""
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sequences, batch_labels in loader:
        logits = model(sequences.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def _class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to offset the ham/spam imbalance."""
    counts = np.bincount(labels.astype(int))
    weights = torch.tensor(1.0 / counts, dtype=torch.float, device=device)
    return weights / weights.sum()


def train_model(train_config: TrainConfig | None = None) -> dict:
    """Train the classifier, save artifacts, and return the test metrics."""
    config = train_config or TrainConfig()
    seed_everything()
    device = resolve_device()
    logger.info("Device: %s", device)

    (x_train, y_train), (x_val, y_val), (x_test, y_test), preprocessor = load_and_preprocess_data(
        train_config=config
    )
    logger.info("Train=%d Val=%d Test=%d", len(x_train), len(x_val), len(x_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save(VOCAB_PATH)

    model_config = ModelConfig(vocab_size=len(preprocessor.word2idx))
    MODEL_CONFIG_PATH.write_text(json.dumps(asdict(model_config)), encoding="utf-8")
    model = TransformerClassifier.from_config(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Vocabulary=%d | Parameters=%d", model_config.vocab_size, total_params)

    train_loader = DataLoader(
        EmailDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(EmailDataset(x_val, y_val), batch_size=config.batch_size)
    test_loader = DataLoader(EmailDataset(x_test, y_test), batch_size=config.batch_size)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(config.epochs):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        scheduler.step()
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        logger.info(
            "Epoch %2d/%d | train %.1f%% | val %.1f%% | val_loss %.4f",
            epoch + 1,
            config.epochs,
            train_acc,
            val_acc,
            val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_file(model.state_dict(), str(WEIGHTS_PATH))
            logger.info("  saved best model (val_loss %.4f)", val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (no val_loss improvement for %d epochs)",
                    epoch + 1,
                    config.early_stopping_patience,
                )
                break

    elapsed_min = (time.time() - start) / 60.0
    logger.info("Training time: %.1f min", elapsed_min)

    model.load_state_dict(load_file(str(WEIGHTS_PATH)))
    preds, labels = _predict_all(model, test_loader, device)
    report = classification_report(
        labels, preds, target_names=list(CLASS_NAMES), output_dict=True, zero_division=0
    )
    metrics = {
        "history": history,
        "test_accuracy": float(np.mean(preds == labels)),
        "training_time_min": round(elapsed_min, 1),
        "total_params": total_params,
        "report": report,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Test accuracy: %.4f", metrics["test_accuracy"])
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model()
