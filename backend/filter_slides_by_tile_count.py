#!/usr/bin/env python3
"""Drop slides whose tumour verdict rests on too few malignant tiles.

`select_tumour_slides.py` selects inclusively on purpose: one malignant tile
pulls a whole slide in. That is the right rule for *finding* tumour and the
wrong one for anything measured downstream, because a slide carrying three
malignant tiles out of a thousand produces a growth-pattern grade computed over
three tiles and indistinguishable, afterwards, from a grade computed over three
thousand. This is the floor that makes those slides identifiable up front.

    python filter_slides_by_tile_count.py radiogenomics_tumour_slides.csv \
        --min-malignant-tiles 10 --out radiogenomics_tumour_slides_min10.csv

Filtering only, on the columns `select_tumour_slides.py` already wrote — nothing
is recomputed here, so this can never disagree with the selection it filters.
Dry run by default: without `--out` it prints what the cut costs and writes
nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

#: Columns this script reads. A file missing them is not a slide list, and
#: `malignant_tiles` in particular is the one number the cut is made on — a
#: silently absent column would otherwise filter nothing and report success.
_REQUIRED_COLUMNS = ("slide_id", "malignant_tiles", "total_tiles")

#: Malignant-tile floors the report tabulates, so the chosen threshold is read
#: against the cohort's own distribution rather than in isolation.
_TILE_CUTS = (1, 5, 10, 25, 50, 100, 250)


def filter_slides(slides: pd.DataFrame, min_malignant_tiles: int) -> pd.DataFrame:
    """Rows with at least `min_malignant_tiles` malignant tiles, order preserved."""
    missing = [c for c in _REQUIRED_COLUMNS if c not in slides.columns]
    if missing:
        raise ValueError(
            f"slide list is missing {', '.join(missing)}; "
            f"expected the columns select_tumour_slides.py writes, got "
            f"{', '.join(slides.columns)}"
        )

    counts = _numeric_column(slides, "malignant_tiles")
    # Checked although the cut never reads it: the report sums it, and a blank
    # there is the same torn or hand-edited row as a blank malignant_tiles.
    _numeric_column(slides, "total_tiles")
    blank_ids = slides["slide_id"].astype(str).str.strip() == ""
    if blank_ids.any():
        raise ValueError(
            f"{int(blank_ids.sum())} row(s) have a blank slide_id; a row that "
            f"names no slide cannot be kept or dropped by anything here"
        )
    return slides.loc[counts >= min_malignant_tiles]


def _numeric_column(slides: pd.DataFrame, column: str) -> pd.Series:
    """`column` as numbers, refusing any value that is not one.

    Converted here, explicitly, because the file is read as text (see
    read_slide_list) — so a blank or a stray word arrives as itself rather than
    as a NaN pandas invented, and is refused rather than treated as zero.
    """
    values = pd.to_numeric(slides[column].astype(str).str.strip(), errors="coerce")
    if values.isna().any():
        bad = slides.loc[values.isna(), "slide_id"].head(5).tolist()
        raise ValueError(
            f"{int(values.isna().sum())} slides have a non-numeric {column} "
            f"(e.g. {bad}); refusing rather than treating them as zero"
        )
    return values


def read_slide_list(path: Path) -> pd.DataFrame:
    """The slide list with every cell exactly as written.

    pandas' default read guesses types and rewrites identifiers on the way in:
    a numeric `samples` column with one blank becomes floats (`1001` ->
    `1001.0`), `00123` loses its zeros, and a sample called `NA` or `None`
    becomes a blank. This script only drops rows, so whatever it reads is what
    it writes back out — and ANORAK then groups by a sample id that no longer
    matches the one it came from. Read as text; the two numeric columns are
    converted explicitly where the cut is made.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _report(slides: pd.DataFrame, kept: pd.DataFrame, min_malignant_tiles: int) -> None:
    total = len(slides)
    counts = _numeric_column(slides, "malignant_tiles")
    print(f"slides in:  {total}")
    print(f"threshold:  malignant_tiles >= {min_malignant_tiles}")
    print(f"kept:       {len(kept)} ({len(kept) / total:.1%})")
    print(f"dropped:    {total - len(kept)} ({(total - len(kept)) / total:.1%})")
    if "total_tiles" in kept.columns and len(kept):
        print(f"tiles kept: {int(_numeric_column(kept, 'total_tiles').sum()):,} "
              f"of {int(_numeric_column(slides, 'total_tiles').sum()):,}")
    print("\nmalignant tiles   slides at or above")
    for cut in _TILE_CUTS:
        n = int((counts >= cut).sum())
        mark = "  <- threshold" if cut == min_malignant_tiles else ""
        print(f"  >= {cut:<12d} {n:>6d} ({n / total:5.1%}){mark}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slides_csv", type=Path,
                        help="slide list written by select_tumour_slides.py")
    parser.add_argument("--min-malignant-tiles", type=int, default=10,
                        help="drop slides with fewer malignant tiles (default: 10)")
    parser.add_argument("--out", type=Path,
                        help="write the kept slides here; omit for a dry run")
    args = parser.parse_args(argv)

    if args.min_malignant_tiles < 1:
        parser.error("--min-malignant-tiles must be at least 1")
    if not args.slides_csv.exists():
        parser.error(f"no such file: {args.slides_csv}")

    slides = read_slide_list(args.slides_csv)
    if slides.empty:
        print(f"{args.slides_csv} has no rows", file=sys.stderr)
        return 1

    try:
        kept = filter_slides(slides, args.min_malignant_tiles)
    except ValueError as refusal:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 1
    _report(slides, kept, args.min_malignant_tiles)

    if kept.empty:
        print(f"\nno slide reaches {args.min_malignant_tiles} malignant tiles; "
              f"writing nothing", file=sys.stderr)
        return 1
    if args.out is None:
        print("\ndry run: pass --out to write the filtered list")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    kept.to_csv(tmp, index=False)
    tmp.replace(args.out)
    print(f"\nwrote {len(kept)} slides to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
