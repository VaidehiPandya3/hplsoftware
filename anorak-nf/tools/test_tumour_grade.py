#!/usr/bin/env python3
"""Tests for tumour_grade.aggregate — the multi-slide path.

A tumour's grade is computed by POOLING its slides' pixel counts and deriving
proportions from the sum, not by averaging the slides' proportions. The two
agree whenever a tumour has one slide, which is every tumour in a random
ten-slide sample, so the whole cohort's grading rests on a branch a test run
cannot reach by accident.

The case below is built so the two rules disagree about the grade, not merely
about a decimal: pooling says 2, averaging says 3. A change from one to the
other cannot pass this quietly.

Runs under pytest and standalone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from tumour_grade import aggregate  # noqa: E402

PX = ["lepidic_px", "papillary_px", "acinar_px",
      "cribriform_px", "micropapillary_px", "solid_px"]


def counts(*slides) -> pd.DataFrame:
    """slides are (slide_id, sample, {pattern: px})."""
    rows = []
    for slide_id, sample, patterns in slides:
        row = {"slide_id": slide_id, "sample": sample}
        row.update({column: 0 for column in PX})
        row.update({f"{name}_px": value for name, value in patterns.items()})
        row["pattern_pixels"] = sum(row[column] for column in PX)
        rows.append(row)
    return pd.DataFrame(rows)


#: One small slide that is pure solid, one large slide that is pure acinar.
#: Alone the first grades 3 and the second grades 2. Pooled the tumour is 1%
#: solid — grade 2. Averaged it is 50% solid — grade 3.
SMALL_SOLID_LARGE_ACINAR = counts(
    ("A", "T", {"solid": 1_000}),
    ("B", "T", {"acinar": 99_000}),
)


def test_a_tumours_slides_become_one_row():
    result = aggregate(SMALL_SOLID_LARGE_ACINAR)
    assert len(result) == 1
    assert result["slides"].iloc[0] == 2
    assert result["pattern_pixels"].iloc[0] == 100_000


def test_proportions_pool_pixels_rather_than_averaging_slides():
    row = aggregate(SMALL_SOLID_LARGE_ACINAR).iloc[0]
    assert abs(row["solid_prop"] - 0.01) < 1e-9      # pooled
    assert abs(row["acinar_prop"] - 0.99) < 1e-9
    assert abs(row["solid_prop"] - 0.50) > 0.4       # NOT the mean of 1.0 and 0.0


def test_the_grade_follows_the_pooled_fraction_not_the_slide_grades():
    row = aggregate(SMALL_SOLID_LARGE_ACINAR).iloc[0]
    assert row["predominant_pattern"] == "acinar"
    assert abs(row["high_grade_fraction"] - 0.01) < 1e-9
    assert str(row["iaslc_grade"]) == "2"

    # And each slide on its own would have graded differently, which is the
    # whole reason this matters.
    alone = aggregate(counts(("A", "T1", {"solid": 1_000}),
                             ("B", "T2", {"acinar": 99_000})))
    assert sorted(alone["iaslc_grade"].astype(str)) == ["2", "3"]


def test_a_big_high_grade_slide_still_carries_the_tumour():
    """Pooling is not a way for high grade to get averaged away."""
    row = aggregate(counts(("A", "T", {"solid": 60_000}),
                           ("B", "T", {"acinar": 40_000}))).iloc[0]
    assert abs(row["high_grade_fraction"] - 0.60) < 1e-9
    assert str(row["iaslc_grade"]) == "3"


def test_one_slide_per_tumour_is_unchanged():
    """The case the ten-slide run covered — kept so the fix cannot break it."""
    row = aggregate(counts(("A", "T", {"acinar": 700, "solid": 300}))).iloc[0]
    assert row["slides"] == 1
    assert abs(row["solid_prop"] - 0.3) < 1e-9
    assert str(row["iaslc_grade"]) == "3"


def test_a_tumour_with_no_tissue_is_ungraded_not_zero():
    row = aggregate(counts(("A", "T", {}))).iloc[0]
    assert row["pattern_pixels"] == 0
    assert str(row["iaslc_grade"]) == ""


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print(f"  ok    {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
