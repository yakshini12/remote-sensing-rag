"""Canonical manifest schema for Stage 0–1.

Design decision
---------------
We store one row per *image*, with captions as a JSON list.
This matches how caption metrics (BLEU/CIDEr) expect multiple references
per image and avoids duplicating image hashes five times.

Knowledge-base rows (RSITMD) also use one-row-per-image so the leakage
audit can compare images and captions at the same granularity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "1.0.0"

REQUIRED_FIELDS = (
    "schema_version",
    "source",
    "role",
    "image_id",
    "image_path",
    "relative_path",
    "split",
    "captions",
    "num_captions",
    "class_name",
    "width",
    "height",
    "file_size_bytes",
    "sha256",
    "dhash64",
    "ok",
    "error",
)


@dataclass(slots=True)
class ImageRecord:
    """One image + its reference captions."""

    schema_version: str
    source: str  # ucm_captions | rsitmd | ...
    role: str  # caption_trainvaltest | knowledge_base
    image_id: str
    image_path: str
    relative_path: str
    split: str  # train | val | test | kb | unknown
    captions: list[str] = field(default_factory=list)
    num_captions: int = 0
    class_name: str = ""
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    sha256: str = ""
    dhash64: str = ""
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["num_captions"] = len(self.captions)
        return d


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of schema problems for one record."""
    problems: list[str] = []
    for key in REQUIRED_FIELDS:
        if key not in record:
            problems.append(f"missing field: {key}")
    if "captions" in record and not isinstance(record["captions"], list):
        problems.append("captions must be a list")
    if record.get("ok") and not record.get("sha256"):
        problems.append("ok=True but sha256 empty")
    if record.get("ok") and record.get("num_captions", 0) < 1:
        # KB and caption datasets should have text; flag empty as a problem.
        problems.append("ok=True but no captions")
    return problems
