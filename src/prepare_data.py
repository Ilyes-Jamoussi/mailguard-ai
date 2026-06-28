"""Download the SpamAssassin public corpus and write a deduplicated CSV.

Ham and spam both come from SpamAssassin so the classifier learns the spam
signal itself rather than corpus-specific style (which would happen if ham and
spam were drawn from different datasets). No SMS or synthetic data is used.
"""

from __future__ import annotations

import io
import logging
import tarfile
import urllib.request

import pandas as pd

from src.config import DATASET_PATH, PROCESSED_DATA_DIR, SEED

logger = logging.getLogger(__name__)

_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
_ARCHIVES: tuple[tuple[str, int], ...] = (
    ("20030228_easy_ham.tar.bz2", 0),
    ("20030228_easy_ham_2.tar.bz2", 0),
    ("20030228_hard_ham.tar.bz2", 0),
    ("20030228_spam.tar.bz2", 1),
    ("20030228_spam_2.tar.bz2", 1),
)
_MIN_BODY_CHARS = 30
_MAX_BODY_CHARS = 2000


def _extract_body(raw_email: str) -> str:
    """Return the email body (text after the header block), truncated."""
    body = raw_email.split("\n\n", 1)[-1]
    return body[:_MAX_BODY_CHARS].strip()


def _download_archive(filename: str, label: int) -> list[dict[str, object]]:
    """Download one SpamAssassin archive and return its labeled email bodies."""
    url = f"{_BASE_URL}/{filename}"
    with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted constant host)
        archive_bytes = response.read()

    rows: list[dict[str, object]] = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.name.endswith("cmds"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            body = _extract_body(handle.read().decode("utf-8", errors="ignore"))
            if len(body) > _MIN_BODY_CHARS:
                rows.append({"text": body, "label": label})
    return rows


def prepare_dataset() -> pd.DataFrame:
    """Build the spam dataset and write it to ``DATASET_PATH``."""
    rows: list[dict[str, object]] = []
    for filename, label in _ARCHIVES:
        logger.info("Downloading %s ...", filename)
        rows.extend(_download_archive(filename, label))

    df = pd.DataFrame(rows)

    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    logger.info("Removed %d duplicate emails", before - len(df))

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)

    ham = int((df["label"] == 0).sum())
    spam = int((df["label"] == 1).sum())
    logger.info(
        "Saved %d emails to %s (ham=%d, spam=%d, ratio=%.1f:1)",
        len(df),
        DATASET_PATH,
        ham,
        spam,
        ham / spam if spam else 0.0,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare_dataset()
