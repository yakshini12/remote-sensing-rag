"""Caption text normalization for overlap detection.

Design decision
---------------
We keep *raw* captions in the manifest unchanged.
Normalization is used only for overlap / leakage comparisons so that
trivial punctuation and case differences do not hide duplicates.
"""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_caption(text: str) -> str:
    """Lowercase, strip accents-compatible punctuation, squeeze spaces."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
