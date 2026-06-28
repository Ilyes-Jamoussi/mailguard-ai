"""Text cleaning, vocabulary building, and integer encoding for the classifier."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    PAD_IDX,
    PAD_TOKEN,
    SEED,
    UNK_IDX,
    UNK_TOKEN,
    ModelConfig,
    TrainConfig,
)

_URL_RE = re.compile(r"http\S+|www\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")

_DEFAULTS = ModelConfig()

# (sequences, labels) for one split.
Split = tuple[np.ndarray, np.ndarray]


def clean_text(text: str) -> str:
    """Lowercase text and replace URLs/emails, dropping non-alphabetic noise."""
    text = str(text).lower()
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" email ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


class TextPreprocessor:
    """Encode raw text as fixed-length integer sequences.

    The vocabulary must be fit on the training split only (``build_vocab``) so no
    information leaks from validation/test into the model's known tokens.
    """

    def __init__(
        self,
        vocab_size: int = _DEFAULTS.vocab_size,
        max_len: int = _DEFAULTS.max_len,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build_vocab(self, texts: Iterable[str]) -> None:
        """Populate the vocabulary with the most frequent tokens in ``texts``."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(clean_text(text).split())
        for word, _ in counter.most_common(self.vocab_size - len(self.word2idx)):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def text_to_sequence(self, text: str) -> list[int]:
        """Encode text to exactly ``max_len`` token ids (truncated or PAD-filled)."""
        ids = [self.word2idx.get(word, UNK_IDX) for word in clean_text(text).split()]
        ids = ids[: self.max_len]
        ids.extend([PAD_IDX] * (self.max_len - len(ids)))
        return ids

    def save(self, path: Path) -> None:
        """Persist the vocabulary as JSON (safe and inspectable, unlike pickle)."""
        payload = {
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "word2idx": self.word2idx,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TextPreprocessor:
        """Rebuild a preprocessor from a JSON vocabulary file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        preprocessor = cls(vocab_size=payload["vocab_size"], max_len=payload["max_len"])
        preprocessor.word2idx = {word: int(idx) for word, idx in payload["word2idx"].items()}
        preprocessor.idx2word = {idx: word for word, idx in preprocessor.word2idx.items()}
        return preprocessor


def _encode(preprocessor: TextPreprocessor, texts: np.ndarray) -> np.ndarray:
    """Encode an array of raw texts into a (n, max_len) integer matrix."""
    return np.array([preprocessor.text_to_sequence(text) for text in texts], dtype=np.int64)


def load_and_preprocess_data(
    data_path: Path = DATASET_PATH,
    train_config: TrainConfig | None = None,
) -> tuple[Split, Split, Split, TextPreprocessor]:
    """Load the dataset and return stratified train/val/test splits + a fitted preprocessor.

    The vocabulary is fit on the training texts only.
    """
    config = train_config or TrainConfig()
    df = pd.read_csv(data_path)
    texts = df["text"].to_numpy()
    labels = df["label"].to_numpy()

    holdout = config.val_size + config.test_size
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=holdout,
        random_state=SEED,
        stratify=labels,
    )
    relative_test = config.test_size / holdout
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test,
        random_state=SEED,
        stratify=y_temp,
    )

    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(x_train)

    return (
        (_encode(preprocessor, x_train), y_train),
        (_encode(preprocessor, x_val), y_val),
        (_encode(preprocessor, x_test), y_test),
        preprocessor,
    )
