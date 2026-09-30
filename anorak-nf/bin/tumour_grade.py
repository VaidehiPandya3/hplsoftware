#!/usr/bin/env python3
"""Aggregate slide pixel counts to tumour proportions, predominant pattern and
IASLC grade.

Implements the paper's definitions exactly (Methods):

    g_j = sum_i S_ij / sum_i sum_j S_ij      proportion of pattern j
    P   = argmax(g_j)                        predominant pattern

    grade 1  lepidic-predominant,             high-grade < 20%
    grade 2  acinar- or papillary-predominant, high-grade < 20%
    grade 3  any tumour with high-grade >= 20%

where high-grade is solid + micropapillary + cribriform. The aggregation unit
is the tumour, not the slide, which is why this runs once over every slide's
counts rather than per slide.

Two edge cases are named rather than absorbed. A tumour whose slides carry no
pattern pixels at all has no proportions and no grade, and is written with an
empty grade and a stated reason instead of a default. And a tie for the
predominant pattern is reported, because argmax picks the first and which one
that is depends on column order, not on the tissue.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import HIGH_GRADE_PATTERNS, PATTERN_ORDER, refuse  # noqa: E402

#: The share of high-grade pattern at or above which a tumour is grade 3,
#: whatever predominates. The IASLC cutoff, not a tunable.
HIGH_GRADE_CUTOFF = 0.20


def grade_of(predominant: str, high_grade_fraction: float) -> tuple[str, str]:
    """(grade, reason) for one tumour."""
    if high_grade_fraction >= HIGH_GRADE_CUTOFF:
        return "3", f"high-grade patterns {high_grade_fraction:.1%} >= 20%"
    if predominant == "lepidic":
        return "1", f"lepidic-predominant, high-grade {high_grade_fraction:.1%}"
    if predominant in ("acinar", "papillary"):
        return "2", f"{predominant}-predominant, high-grade {high_grade_fraction:.1%}"
    # Unreachable, and deliberately not given a default. If a high-grade
    # pattern predominates then the high-grade sum is at least that pattern's
    # own share, so the three non-high-grade patterns would each have to be no
    # larger while together exceeding 80% — impossible for three patterns each
    # below 20%. A tumour arriving here means the arithmetic above it changed.
    return "", (f"{predominant}-predominant with high-grade only "
                f"{high_grade_fraction:.1%} — not covered by the IASLC rules")


def blank_samples(frame: pd.DataFrame) -> list[str]:
    """slide_ids whose sample is missing, NaN or whitespace."""
    sample = frame["sample"]
    blank = sample.isna() | (sample.astype(str).str.strip() == "")
    return frame.loc[blank, "slide_id"].astype(str).tolist()


def aggregate(counts: pd.DataFrame,
              expected_slides: dict[str, int] | None = None) -> pd.DataFrame:
    """One row per tumour.

    `expected_slides`, when given, is how many slides each tumour has in the
    slide list. A tumour with fewer counted than that is left ungraded with the
    shortfall as its reason — pooling is over *all* of a tumour's slides, and a
    grade from a subset is a different measurement that looks the same — and a
    tumour with none counted still gets a row, so the table covers every
    tumour the list named.
    """
    pixel_columns = [f"{pattern}_px" for pattern in PATTERN_ORDER]
    missing = [column for column in ["sample", *pixel_columns]
               if column not in counts.columns]
    if missing:
        refuse(f"slide counts are missing {', '.join(missing)}")

    # A blank sample is not a tumour. groupby(dropna=False) used to collect
    # every such slide into one group and grade it, so a list with an empty
    # samples column came out as a single "tumour" pooling the whole cohort,
    # well-formed and graded. main.nf refuses such a list before anything is
    # queued; this is the same rule where the pooling actually happens.
    blank = blank_samples(counts)
    if blank:
        refuse(f"{len(blank)} slides have no sample (e.g. {blank[:5]}). Grading "
               f"pools a tumour's slides by sample, so these would be graded "
               f"together as one tumour that does not exist.")

    by_tumour = counts.groupby("sample")
    summed = by_tumour[pixel_columns].sum()
    summed["slides"] = by_tumour.size()
    summed["pattern_pixels"] = summed[pixel_columns].sum(axis=1)

    rows = []
    for sample, row in summed.iterrows():
        total = row["pattern_pixels"]
        record = {"sample": sample, "slides": int(row["slides"]),
                  "pattern_pixels": int(total)}
        if expected_slides is not None:
            record["slides_expected"] = expected_slides.get(sample, 0)
            short = record["slides_expected"] - record["slides"]
            if short:
                for pattern in PATTERN_ORDER:
                    record[f"{pattern}_prop"] = ""
                record.update(predominant_pattern="", high_grade_fraction="",
                              iaslc_grade="", tie="",
                              reason=f"incomplete: {short} of "
                                     f"{record['slides_expected']} slides have "
                                     f"no counts (see anorak_missing_slides.csv)")
                rows.append(record)
                continue

        if total == 0:
            for pattern in PATTERN_ORDER:
                record[f"{pattern}_prop"] = ""
            record.update(predominant_pattern="", high_grade_fraction="",
                          iaslc_grade="", tie="",
                          reason="no growth pattern pixels on any slide")
            rows.append(record)
            continue

        proportions = {pattern: row[f"{pattern}_px"] / total
                       for pattern in PATTERN_ORDER}
        for pattern in PATTERN_ORDER:
            record[f"{pattern}_prop"] = proportions[pattern]

        high_grade = sum(proportions[pattern] for pattern in HIGH_GRADE_PATTERNS)
        best = max(proportions.values())
        winners = [p for p in PATTERN_ORDER if proportions[p] == best]
        predominant = winners[0]
        grade, reason = grade_of(predominant, high_grade)

        record.update(
            predominant_pattern=predominant,
            high_grade_fraction=high_grade,
            iaslc_grade=grade,
            tie=";".join(winners) if len(winners) > 1 else "",
            reason=reason,
        )
        rows.append(record)

    if expected_slides is not None:
        for sample, n in sorted(expected_slides.items()):
            if sample in summed.index:
                continue
            record = {"sample": sample, "slides": 0, "pattern_pixels": 0,
                      "slides_expected": n}
            for pattern in PATTERN_ORDER:
                record[f"{pattern}_prop"] = ""
            record.update(predominant_pattern="", high_grade_fraction="",
                          iaslc_grade="", tie="",
                          reason=f"incomplete: none of its {n} slides have counts "
                                 f"(see anorak_missing_slides.csv)")
            rows.append(record)

    return pd.DataFrame(rows)


def read_text_csv(path: Path) -> pd.DataFrame:
    """A CSV with every cell as the text it was written as.

    pandas' default reading guesses types, and on identifiers every guess is
    wrong: `007` becomes the integer 7 and then collides with a slide called
    `7` — a false duplicate refusal at the very last step of a cohort, which
    fails again on every resume — or silently merges with it where the two are
    samples; and `NA`, `None` and `null` become NaN, which is how a real
    identifier turns into a blank one. Numbers are converted afterwards, by
    name, where a number is meant.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def pixel_counts_as_int(slides: pd.DataFrame) -> pd.DataFrame:
    """The *_px and pattern_pixels columns as integers, or a refusal."""
    for column in [c for c in slides.columns
                   if c.endswith("_px") or c == "pattern_pixels"]:
        converted = pd.to_numeric(slides[column], errors="coerce")
        bad = slides.loc[converted.isna() | (converted % 1 != 0), "slide_id"]
        if len(bad):
            refuse(f"{column} is not a whole number for {len(bad)} slides "
                   f"(e.g. {bad.tolist()[:5]}).")
        slides[column] = converted.astype("int64")
    return slides


def reconcile(slides: pd.DataFrame, expected: pd.DataFrame,
              allow_missing: bool) -> pd.DataFrame:
    """The expected slides that have no counts, after refusing what cannot be.

    `collect()` hands this step whatever arrived, and with
    -process.errorStrategy=ignore a slide whose task failed simply does not
    arrive — nor does one that a join dropped. Before this check that produced
    a table short by however many slides, with nothing in it to say so. Now a
    missing slide is a refusal, unless the run asked for a best-effort sweep
    (`allow_missing`), in which case it is written down by name.

    Never allowed, in either mode: a slide nobody listed, or a slide whose
    sample differs from the list's. Both mean the counts are from some other
    run or some other labelling, and there is no best-effort reading of that.
    """
    for column in ("slide_id", "sample"):
        if column not in expected.columns:
            refuse(f"the expected-slides list has no {column} column")
    listed = dict(zip(expected["slide_id"], expected["sample"]))

    unexpected = [s for s in slides["slide_id"] if s not in listed]
    if unexpected:
        refuse(f"{len(unexpected)} counted slides are not in the slide list "
               f"(e.g. {unexpected[:5]}).")
    relabelled = [s for s, sample in zip(slides["slide_id"], slides["sample"])
                  if listed[s] != sample]
    if relabelled:
        refuse(f"{len(relabelled)} slides were counted under a different sample "
               f"than the slide list gives them (e.g. {relabelled[:5]}).")

    counted = set(slides["slide_id"])
    missing = expected[~expected["slide_id"].isin(counted)][["slide_id", "sample"]]
    if len(missing) and not allow_missing:
        refuse(f"{len(missing)} of {len(expected)} listed slides have no counts "
               f"(e.g. {missing['slide_id'].tolist()[:5]}), so the grading "
               f"table would be short by that many with nothing in it to say "
               f"so. For a deliberate best-effort sweep, run with "
               f"--allow_missing_slides true; the missing slides are then "
               f"listed in anorak_missing_slides.csv and their tumours left "
               f"ungraded.")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slide-counts", type=Path, nargs="+", required=True)
    parser.add_argument("--out-slides", type=Path, required=True)
    parser.add_argument("--out-tumours", type=Path, required=True)
    parser.add_argument("--expected", type=Path,
                        help="slide_id,sample for every slide the run was "
                             "given; counts are reconciled against it")
    parser.add_argument("--out-missing", type=Path,
                        help="where to list expected slides with no counts")
    parser.add_argument("--allow-missing", action="store_true",
                        help="grade what arrived and list what did not, "
                             "instead of refusing")
    args = parser.parse_args()

    frames = [read_text_csv(path) for path in args.slide_counts]
    slides = pd.concat(frames, ignore_index=True)

    duplicates = slides["slide_id"][slides["slide_id"].duplicated()].tolist()
    if duplicates:
        refuse(f"{len(duplicates)} slide_ids appear more than once "
               f"(e.g. {duplicates[:5]}); a slide counted twice inflates its "
               f"tumour's composition toward whatever it happens to contain.")
    slides = pixel_counts_as_int(slides)

    expected_slides = None
    if args.expected:
        expected = read_text_csv(args.expected)
        missing = reconcile(slides, expected, args.allow_missing)
        expected_slides = expected.groupby("sample").size().to_dict()
        if args.out_missing:
            missing.to_csv(args.out_missing, index=False)
        if len(missing):
            print(f"  MISSING: {len(missing)} of {len(expected)} listed slides "
                  f"have no counts; see {args.out_missing}", flush=True)

    slides.to_csv(args.out_slides, index=False)
    tumours = aggregate(slides, expected_slides)
    tumours.to_csv(args.out_tumours, index=False)

    graded = tumours["iaslc_grade"].astype(str)
    print(f"{len(slides)} slides -> {len(tumours)} tumours", flush=True)
    for grade in ("1", "2", "3"):
        print(f"  grade {grade}: {int((graded == grade).sum())}", flush=True)
    ungraded = int((graded == "").sum())
    if ungraded:
        print(f"  ungraded: {ungraded} (see the reason column)", flush=True)
    ties = int((tumours["tie"].astype(str) != "").sum())
    if ties:
        print(f"  ties for predominant pattern: {ties}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
