from __future__ import annotations

from src.config import PAD_IDX, UNK_IDX
from src.preprocessing import TextPreprocessor


def test_sequence_is_padded_to_max_len() -> None:
    preprocessor = TextPreprocessor(vocab_size=50, max_len=12)
    preprocessor.build_vocab(["meeting tomorrow at noon", "free prize winner now"])

    sequence = preprocessor.text_to_sequence("meeting tomorrow")

    assert len(sequence) == 12
    assert sequence[-1] == PAD_IDX


def test_sequence_is_truncated_to_max_len() -> None:
    preprocessor = TextPreprocessor(vocab_size=50, max_len=5)
    preprocessor.build_vocab(["a b c d e f g h i j"])

    sequence = preprocessor.text_to_sequence("a b c d e f g h")

    assert len(sequence) == 5


def test_unknown_words_map_to_unk() -> None:
    preprocessor = TextPreprocessor(vocab_size=50, max_len=4)
    preprocessor.build_vocab(["hello world"])

    sequence = preprocessor.text_to_sequence("hello zzqq")

    assert sequence[1] == UNK_IDX
