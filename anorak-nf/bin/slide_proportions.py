#!/usr/bin/env python3
"""Per-slide growth pattern pixel counts from one post-processed Ss1 mask.

Counts, not proportions, because the paper's proportion is defined over a
tumour rather than a slide: g_j = sum_i S_ij / sum_i sum_j S_ij, where i runs
over the slides of one tumour and S_ij is the pixel count for pattern j on
slide i (Methods). Summing per-slide *proportions* would weight a small biopsy
the same as a large resection; summing counts, as here, and dividing once at
the tumour level is what the paper specifies. Per-slide proportions are still
written alongside, because they are what a slide viewer shows.

Background (black) is excluded from every denominator: it is everything the
model did not assign to a pattern, including glass.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import PATTERN_COLOURS_RGB, PATTERN_ORDER, refuse  # noqa: E402


def count_pattern_pixels(mask_path: Path) -> dict[str, int]:
    import cv2
    import numpy as np

    bgr = cv2.imread(str(mask_path))
    if bgr is None:
        refuse(f"Could not read {mask_path}.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    counts: dict[str, int] = {}
    for pattern in PATTERN_ORDER:
        red, green, blue = PATTERN_COLOURS_RGB[pattern]
        counts[pattern] = int(np.count_nonzero(
            (rgb[:, :, 0] == red) & (rgb[:, :, 1] == green) & (rgb[:, :, 2] == blue)
        ))

    # Every non-black pixel should have landed in exactly one pattern; anything
    # left over means the colours written no longer match the ones read, which
    # would show up as quietly smaller proportions rather than as an error.
    assigned = sum(counts.values())
    non_background = int(np.count_nonzero(rgb.any(axis=2)))
    if assigned != non_background:
        refuse(
            f"{mask_path.name}: {non_background - assigned} coloured pixels "
            f"match no known pattern. The palette in anorak_common is taken "
            f"from ss1_final.py; if upstream's class_colors changed, it has to "
            f"change here too."
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mask", type=Path, required=True,
                        help="post-processed <slide>_Ss1.png")
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--sample", required=True,
                        help="tumour/patient id this slide belongs to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    # Grading pools slides by this value, so a blank one is not "unknown" — it
    # is a tumour, the same one for every slide that has it. main.nf refuses a
    # list with blank samples at launch; this is the backstop where the value
    # is written.
    if not args.sample.strip():
        refuse(f"{args.slide_id}: no sample. Grading would pool this slide "
               f"with every other slide lacking one, as a single tumour.")

    counts = count_pattern_pixels(args.mask)
    total = sum(counts.values())

    row = {"slide_id": args.slide_id, "sample": args.sample,
           "pattern_pixels": total}
    for pattern in PATTERN_ORDER:
        row[f"{pattern}_px"] = counts[pattern]
    for pattern in PATTERN_ORDER:
        # Empty rather than 0.0 when the model found no pattern anywhere: a
        # slide with no growth pattern has no composition, and writing zeros
        # would average into a tumour as though it did.
        row[f"{pattern}_frac"] = (counts[pattern] / total) if total else ""

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"{args.slide_id}: {total} pattern pixels", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
