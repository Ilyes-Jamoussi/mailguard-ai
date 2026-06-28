"""Encoder-only Transformer for spam classification, implemented from scratch.

Every component (multi-head self-attention, sinusoidal positional encoding,
pre-LayerNorm blocks, masked mean pooling) is built on PyTorch primitives only.
A padding mask is derived inside ``forward`` so callers just pass token ids.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
from torch import nn

from src.config import PAD_IDX, ModelConfig

_DEFAULTS = ModelConfig()


class PositionalEncoding(nn.Module):
    """Add fixed sinusoidal position signals to token embeddings."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)."""
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    """Scaled dot-product self-attention with ``nhead`` parallel heads."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model); pad_mask: (batch, seq_len), True at PAD."""
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.nhead, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = (attn @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(context)


class TransformerBlock(nn.Module):
    """Pre-LN encoder block: residual self-attention then residual feed-forward."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    """Classify a token sequence as ham/spam via a small Transformer encoder."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = _DEFAULTS.d_model,
        nhead: int = _DEFAULTS.nhead,
        num_layers: int = _DEFAULTS.num_layers,
        num_classes: int = _DEFAULTS.num_classes,
        max_len: int = _DEFAULTS.max_len,
        dropout: float = _DEFAULTS.dropout,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, nhead, d_model * 4, dropout) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> TransformerClassifier:
        """Build a model from a ``ModelConfig`` (used by training and inference)."""
        return cls(**asdict(config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) token ids. Returns logits (batch, num_classes)."""
        pad_mask = x == PAD_IDX
        h = self.embedding(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        for layer in self.layers:
            h = layer(h, pad_mask)
        h = self.norm(h)
        pooled = self._masked_mean(h, pad_mask)
        return self.classifier(pooled)

    @staticmethod
    def _masked_mean(h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Average over real (non-PAD) positions. h: (batch, seq_len, d_model)."""
        keep = (~pad_mask).unsqueeze(-1).float()
        summed = (h * keep).sum(dim=1)
        count = keep.sum(dim=1).clamp(min=1.0)
        return summed / count
