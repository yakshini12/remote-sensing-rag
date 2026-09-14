#!/usr/bin/env python3
"""Create tiny synthetic UCM + RSITMD fixtures for Stage 0–1 smoke tests.

Injected leakage cases
----------------------
1. Hard image duplicate: identical JPEG bytes in UCM test and RSITMD KB.
2. Exact caption overlap: shared airport caption string.
3. Soft near-duplicate: beach scene with small color/geometry shift.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
UCM = ROOT / "tests" / "fixtures" / "ucm_captions"
RSITMD = ROOT / "tests" / "fixtures" / "rsitmd"


def _draw(path: Path, base: tuple[int, int, int], shapes: list) -> None:
    im = Image.new("RGB", (128, 128), base)
    d = ImageDraw.Draw(im)
    for kind, args, fill in shapes:
        if kind == "rect":
            d.rectangle(args, fill=fill)
        elif kind == "ellipse":
            d.ellipse(args, fill=fill)
        elif kind == "line":
            d.line(args, fill=fill, width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="JPEG", quality=95)


def main() -> None:
    if UCM.exists():
        shutil.rmtree(UCM)
    if RSITMD.exists():
        shutil.rmtree(RSITMD)

    ucm_spec = {
        "agricultural/ag_001.jpg": (
            (20, 90, 20),
            [("rect", (10, 10, 60, 50), (180, 220, 40)), ("rect", (70, 60, 120, 110), (40, 160, 40))],
            ["green farmland with rectangular fields", "crop fields arranged in a grid"],
        ),
        "airport/ap_001.jpg": (
            (60, 60, 60),
            [("line", [(10, 100), (120, 20)], (220, 220, 220)), ("rect", (40, 40, 55, 55), (200, 30, 30))],
            ["an airport runway with airplanes", "gray runway surrounded by buildings"],
        ),
        "beach/be_001.jpg": (
            (30, 100, 180),
            [("rect", (0, 80, 128, 128), (210, 190, 120)), ("ellipse", (20, 20, 50, 45), (255, 255, 255))],
            ["a sandy beach next to blue water", "coastal beach with waves"],
        ),
        "leak_hard/hard_001.jpg": (
            (5, 15, 80),
            [("ellipse", (30, 30, 100, 100), (0, 180, 255)), ("rect", (50, 70, 90, 95), (180, 180, 20))],
            ["shared harbor scene with ships"],
        ),
    }

    ucm_ann = {"images": []}
    for rel, (base, shapes, caps) in ucm_spec.items():
        path = UCM / rel
        _draw(path, base, shapes)
        ucm_ann["images"].append({"filename": Path(rel).name, "filepath": rel, "captions": caps})

    (UCM / "dataset.json").write_text(json.dumps(ucm_ann, indent=2) + "\n", encoding="utf-8")
    (UCM / "splits.json").write_text(
        json.dumps({"train": ["ag_001", "ap_001"], "val": ["be_001"], "test": ["hard_001"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # RSITMD KB
    hard_dst = RSITMD / "images" / "kb_hard.jpg"
    hard_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(UCM / "leak_hard" / "hard_001.jpg", hard_dst)

    rs_spec = {
        "images/kb_001.jpg": (
            (140, 40, 40),
            [("rect", (20, 20, 40, 110), (80, 20, 20)), ("rect", (60, 40, 80, 110), (100, 30, 30))],
            ["dense residential buildings with roads"],
        ),
        "images/kb_002.jpg": (
            (20, 60, 20),
            [("line", [(0, 64), (128, 40)], (30, 120, 200)), ("ellipse", (70, 70, 110, 110), (10, 100, 10))],
            ["a river flowing through forest"],
        ),
        "images/kb_beach_near.jpg": (
            (35, 105, 175),
            [("rect", (0, 82, 128, 128), (205, 185, 115)), ("ellipse", (22, 22, 52, 47), (250, 250, 250))],
            ["coastal beach with waves nearby"],
        ),
        "images/kb_airport_text.jpg": (
            (55, 55, 55),
            [("line", [(15, 110), (110, 25)], (200, 200, 200)), ("rect", (70, 70, 95, 95), (20, 20, 200))],
            ["an airport runway with airplanes"],
        ),
    }

    rs_ann = {
        "images": [
            {
                "filename": "kb_hard.jpg",
                "filepath": "images/kb_hard.jpg",
                "captions": ["shared harbor scene with ships", "boats docked at a harbor"],
            }
        ]
    }
    for rel, (base, shapes, caps) in rs_spec.items():
        _draw(RSITMD / rel, base, shapes)
        rs_ann["images"].append({"filename": Path(rel).name, "filepath": rel, "captions": caps})

    (RSITMD / "dataset_RSITMD.json").write_text(json.dumps(rs_ann, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote fixtures to:\n  {UCM}\n  {RSITMD}")


if __name__ == "__main__":
    main()
