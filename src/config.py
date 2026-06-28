"""Single source of truth for hyperparameters, paths, and constants.

Importing values from here (rather than re-declaring them) is what prevents
silent inconsistencies such as a model using ``max_len=512`` while the
preprocessor pads to ``256``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42

# Special token ids shared by the preprocessor (encoding) and the model (the
# padding mask is derived from ``token_id == PAD_IDX``).
PAD_IDX = 0
UNK_IDX = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

CLASS_NAMES = ("ham", "spam")

# Project paths: centralized, built with pathlib (never string concatenation).
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_PATH = PROCESSED_DATA_DIR / "spam_data.csv"

MODELS_DIR = ROOT_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "model.safetensors"
VOCAB_PATH = MODELS_DIR / "vocab.json"
MODEL_CONFIG_PATH = MODELS_DIR / "config.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

STYLES_PATH = ROOT_DIR / "assets" / "styles.css"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters for the Transformer classifier."""

    vocab_size: int = 30000
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    num_classes: int = 2
    max_len: int = 256
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop hyperparameters.

    ``epochs`` is an upper bound: training stops once the validation loss has not
    improved for ``early_stopping_patience`` epochs.
    """

    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stopping_patience: int = 4
    val_size: float = 0.15
    test_size: float = 0.15
