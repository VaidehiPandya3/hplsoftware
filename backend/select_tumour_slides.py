#!/usr/bin/env python3
"""Which slides in a cohort contain malignant tissue, per HPL.

ANORAK segments lung-adenocarcinoma growth patterns, so it only means anything
on slides that carry tumour. This is the gate: it turns HPL's existing per-slide
cluster composition into a slide list, and it is nearly free — the malignancy of
each of the 71 HPCs is reference data that is already in `hpc_dictionary`, and
each slide's composition is already in `hpl_profile_proportion` (3.9 MB). No new
compute, no tiles read, no model run. It also cuts the expensive half: ANORAK
tiles at 0.22 um/px against HPL's 1.8, ~67x the pixel volume, so a slide not
selected here is 67x of work not done.

Standalone and useful on its own as a cohort-composition report.

    python select_tumour_slides.py --from-kb --dataset-id LATTICeA_5x --out slides.csv
    python select_tumour_slides.py --from-csv DS_hpc_assignments.csv --out slides.csv

`--out` writes the selected slides only, because it is the file handed to
ANORAK and nothing downstream reads `is_tumour`. `--include-non-tumour` writes
every row instead, as the composition report; that file is not a slide list.

Two sources for the same number, because the KB is not always ready. Stage 6 has
not run for every cohort, and waiting on it would block this for no reason:

  --from-kb    per-slide proportions already loaded, joined to hpc_dictionary.
  --from-csv   the Stage 4 assignment CSV, joined to hpc_dictionary, for a
               cohort that is classified but not yet loaded.

Both still need the database, because `hpc_dictionary` is where malignancy
lives. `test_tumour_filter.py` asserts the two agree on a fixture: two ways of
computing one number that could silently disagree is exactly the class of bug
this codebase is written against.

The rule is *select*, not exclude, and it is deliberately inclusive: a slide
counts as tumour if it has a single malignant tile. One misassigned tile out of
18.5M can therefore pull a slide in, which is why the malignant fraction is
recorded next to every verdict rather than only the yes/no — a grade later
derived from a sliver of tissue is then identifiable instead of looking like any
other grade. The printed distribution is there so the cost of the inclusive rule
is visible even though the rule is already chosen.

Malignancy itself is read, never recomputed: `hpc_dictionary` is reference data
(CLAUDE.md), and the normalisation of its loosely-typed `malignant` column is
`backend/malignancy.py`, shared with the UI so the filter and the viewer cannot
drift on what "malignant" means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_hpc_assignments import DB_NAME, make_engine, read_assignments  # noqa: E402
from malignancy import UnrecognisedMalignancy, parse_malignant  # noqa: E402

#: Columns of the slide list, in order. `is_tumour` is the answer; everything
#: beside it is what makes the answer auditable — above all
#: `malignant_fraction`, which is how much tissue the verdict rests on.
_OUTPUT_COLUMNS = (
    "slide_id",
    "dataset_id",
    "samples",
    "malignant_fraction",
    "malignant_tiles",
    "total_tiles",
    "n_malignant_hpcs",
    "is_tumour",
)

#: Malignant-fraction floors the report tabulates. 0.00 is the rule this script
#: actually applies — any malignant tile at all — and the rows below it are what
#: a less inclusive rule would have cost, on this cohort's own distribution.
#: Deliberately dense at the bottom: the interesting slides are the ones sitting
#: on a fraction of a percent, not the ones at 40%.
_FRACTION_CUTS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25)

#: How far a slide's proportions may sum from 1 before the numbers are refused.
#: `compute_profiles()` computes them as a share of the slide's tiles, so they sum
#: to 1 by construction; a slide that sums to 0.6 has lost 40% of its proportion
#: rows somewhere, and its malignant fraction is then wrong by an unknown amount
#: in the direction that drops slides. 0.01 rather than a float epsilon because
#: these are doubles summed over up to 71 rows and may have been rounded on the
#: way in — the guard is for a missing slice, not for the last bit.
_PROPORTION_TOLERANCE = 0.01


# --- the malignancy of each cluster ---------------------------------------

def load_malignancy(engine) -> dict[str, bool]:
    """{hpc_id: is_malignant} for every row in hpc_dictionary.

    Keys are stripped strings, matching how load_hpc_assignments.py compares
    cluster IDs — `hpc_dictionary.hpc_id`'s own type is one of the four things
    kb_live_schema_2026-08-26.txt never captured, so nothing here should assume
    the driver hands back an int.

    Refuses on a `malignant` value the shared normalisation does not recognise,
    rather than treating it as non-malignant. That default is the whole reason
    this is a function and not a dict comprehension: it would silently drop
    every slide whose tumour happens to sit in that cluster, and the resulting
    cohort would be smaller with nothing to say why.
    """
    with engine.connect() as conn:
        rows = pd.read_sql(text("SELECT hpc_id, malignant FROM hpc_dictionary"), conn)
    if rows.empty:
        raise SystemExit(
            f"hpc_dictionary in {DB_NAME} is empty. It is reference data — the "
            f"71 HPCs and their malignancy — so this is a database that has not "
            f"been populated, not a cohort problem."
        )

    mapping: dict[str, bool] = {}
    unrecognised: list[tuple[str, object]] = []
    for hpc_id, value in zip(rows["hpc_id"], rows["malignant"]):
        key = str(hpc_id).strip()
        try:
            mapping[key] = parse_malignant(value)
        except UnrecognisedMalignancy:
            unrecognised.append((key, value))

    if unrecognised:
        shown = [f"hpc_id {h}: {v!r}" for h, v in unrecognised[:5]]
        raise SystemExit(
            f"{len(unrecognised)} hpc_dictionary row(s) carry a `malignant` "
            f"value that cannot be read as malignant or non-malignant, e.g. "
            f"{shown}.\n\n"
            f"Not coerced, because the safe-looking default — non-malignant — "
            f"would drop every slide whose tumour is in one of these clusters "
            f"and report a smaller cohort as a success. Fix the dictionary row, "
            f"or add the spelling to backend/malignancy.py if it is a spelling "
            f"and not a mistake."
        )

    if not any(mapping.values()):
        raise SystemExit(
            f"No hpc_dictionary row in {DB_NAME} is marked malignant "
            f"({len(mapping)} cluster(s) read). Every slide in every cohort "
            f"would be non-tumour, which is a reference-data mismatch rather "
            f"than a cohort with no cancer in it — check the `malignant` column "
            f"has not been emptied or re-spelled."
        )
    return mapping


def _check_clusters_known(assigned, malignancy: dict[str, bool], source: str) -> None:
    """Refuse a cluster ID with no hpc_dictionary row.

    Its malignancy is unknown, and the only two ways to proceed are both wrong:
    treated as non-malignant it drops slides, treated as malignant it admits
    them on evidence nobody has. This is the same reference/cohort mismatch
    load_hpc_assignments.py refuses at Stage 6, caught earlier here.
    """
    unknown = sorted({str(c).strip() for c in assigned} - set(malignancy))
    if unknown:
        raise SystemExit(
            f"{source} refers to {len(unknown)} cluster ID(s) with no "
            f"hpc_dictionary row: {unknown[:10]}"
            f"{' ...' if len(unknown) > 10 else ''}.\n\n"
            f"Their malignancy is unknown, so no slide carrying them can be "
            f"classified either way. The usual cause is an assignment made "
            f"against a different reference than the dictionary describes — "
            f"check hpc_reference against the {len(malignancy)} cluster(s) the "
            f"dictionary holds."
        )


# --- the two sources ------------------------------------------------------

def from_kb(engine, malignancy: dict[str, bool], dataset_id: str | None = None,
            tolerance: float = _PROPORTION_TOLERANCE) -> pd.DataFrame:
    """The slide list from hpl_profile_proportion + hpl_profile_summary.

    Driven from `hpl_profile_summary` with a LEFT JOIN onto the proportions, not
    the other way round: a slide whose summary row exists but whose proportion
    rows do not is a half-written Stage 6, and reading only the proportion table
    would make that slide simply absent — indistinguishable from a slide that is
    not in the cohort — instead of refused.

    The join is on (samples, slides) alone because that, and not
    (samples, slides, dataset_id), is the unique constraint the foreign key
    points at (see kb_live_schema_2026-08-26.txt's note). It therefore cannot
    fan out, and the two tables' `dataset_id` is compared afterwards rather than
    joined on, so a disagreement the schema permits is reported instead of
    quietly dropping the slide's composition.
    """
    where = "WHERE s.dataset_id = :dataset_id" if dataset_id else ""
    sql = f"""
        SELECT s.samples          AS samples,
               s.slides           AS slides,
               s.dataset_id       AS dataset_id,
               s.total_tiles      AS total_tiles,
               p.hpc_id           AS hpc_id,
               p.proportion       AS proportion,
               p.dataset_id       AS proportion_dataset_id
        FROM hpl_profile_summary s
        LEFT JOIN hpl_profile_proportion p
               ON p.samples = s.samples AND p.slides = s.slides
        {where}
    """
    params = {"dataset_id": dataset_id} if dataset_id else {}
    with engine.connect() as conn:
        rows = pd.read_sql(text(sql), conn, params=params)

    if rows.empty:
        scope = f" for dataset_id {dataset_id!r}" if dataset_id else ""
        raise SystemExit(
            f"hpl_profile_summary in {DB_NAME} holds no rows{scope}. Stage 6 "
            f"has not loaded this cohort, so its per-slide composition does not "
            f"exist yet — run the assignment CSV through --from-csv instead, "
            f"which needs only Stage 4."
        )

    orphans = sorted(rows.loc[rows["hpc_id"].isna(), "slides"].astype(str).unique())
    if orphans:
        raise SystemExit(
            f"{len(orphans)} slide(s) have an hpl_profile_summary row but no "
            f"hpl_profile_proportion rows, e.g. {orphans[:5]}.\n\n"
            f"That is a half-written Stage 6: a tile count with no composition "
            f"beside it. Every one of these slides would come out non-tumour "
            f"here for want of data, so re-run the KB load for this cohort "
            f"rather than a filter over it."
        )

    mismatched = rows[
        rows["proportion_dataset_id"].notna()
        & (rows["proportion_dataset_id"].astype(str) != rows["dataset_id"].astype(str))
    ]
    if not mismatched.empty:
        example = mismatched.iloc[0]
        raise SystemExit(
            f"{len(mismatched):,} proportion row(s) across "
            f"{mismatched['slides'].nunique()} slide(s) carry a different "
            f"dataset_id than their summary row, e.g. slide "
            f"{example['slides']!r}: summary {example['dataset_id']!r} vs "
            f"proportion {example['proportion_dataset_id']!r}.\n\n"
            f"The unique constraint on hpl_profile_summary is (samples, slides) "
            f"and does not include dataset_id, so the database permits this and "
            f"nothing else would notice — but a cohort filter cannot tell which "
            f"of the two cohorts these tiles belong to."
        )

    missing_totals = sorted(
        rows.loc[rows["total_tiles"].isna(), "slides"].astype(str).unique())
    if missing_totals:
        raise SystemExit(
            f"{len(missing_totals)} slide(s) have a NULL total_tiles in "
            f"hpl_profile_summary, e.g. {missing_totals[:5]}. The malignant "
            f"tile count is that number times the malignant proportion, so "
            f"there is nothing to report for these slides."
        )

    _refuse_null_keys(rows, ("slides", "samples", "dataset_id"),
                      "hpl_profile_summary")

    rows["hpc_id"] = rows["hpc_id"].astype(str).str.strip()
    _check_clusters_known(rows["hpc_id"], malignancy, "hpl_profile_proportion")

    # Proportions sum to 1 per slide by construction (compute_profiles()). A slide
    # that does not is missing proportion rows, and the direction of the error
    # is the dangerous one: a malignant slice lost from the sum reads as a
    # smaller malignant fraction, and at the limit as a non-tumour slide.
    sums = rows.groupby("slides")["proportion"].sum()
    off = sums[(sums - 1.0).abs() > tolerance]
    if not off.empty:
        shown = [f"{slide}: {total:.4f}" for slide, total in off.head(5).items()]
        raise SystemExit(
            f"{len(off)} slide(s) have hpl_profile_proportion rows that do not "
            f"sum to 1 (tolerance {tolerance}), e.g. {shown}.\n\n"
            f"Stage 6 computes each proportion as a share of that slide's "
            f"tiles, so a sum below 1 means rows are missing and the malignant "
            f"fraction below it is understated by an unknown amount. Re-run the "
            f"KB load for this cohort."
        )

    rows["is_malignant"] = rows["hpc_id"].map(malignancy)
    # Summed as columns rather than in a groupby-apply: the malignant share is
    # the proportion where the cluster is malignant and zero elsewhere, so a
    # plain sum gives it, and the aggregate stays vectorised over a table that
    # is 71 rows per slide across every cohort in the database.
    rows["malignant_proportion"] = rows["proportion"].where(rows["is_malignant"], 0.0)
    rows["malignant_present"] = rows["is_malignant"] & (rows["proportion"] > 0)
    # dropna=False although _refuse_null_keys() has already run: pandas' default
    # drops any group whose key holds a NULL, so a slide with a NULL dataset_id
    # or samples would leave this aggregate with no row and no error — gone
    # from the list, which is the one outcome this filter must never produce.
    # The refusal above is what makes such a key loud; this is what keeps a
    # future edit to that refusal from turning it back into a silent drop.
    grouped = rows.groupby(["slides", "dataset_id", "samples"], as_index=False,
                           dropna=False).agg(
        malignant_fraction=("malignant_proportion", "sum"),
        n_malignant_hpcs=("malignant_present", "sum"),
        # Repeated across the slide's proportion rows by the join, so "first"
        # is the value itself, not a choice among several.
        total_tiles=("total_tiles", "first"),
    )
    grouped["n_malignant_hpcs"] = grouped["n_malignant_hpcs"].astype(int)
    grouped["total_tiles"] = grouped["total_tiles"].astype(int)

    grouped = grouped.rename(columns={"slides": "slide_id"})
    # Derived rather than counted, because the KB stores the share and not the
    # count. Rounded because it is a tile count: compute_profiles() divides exact
    # integers, so one malignant tile in 20,000 comes back as exactly 1.0 here
    # rather than 0.9999.
    grouped["malignant_tiles"] = (
        grouped["malignant_fraction"] * grouped["total_tiles"]).round().astype(int)
    return _finish(grouped)


def from_csv(csv_path: Path, malignancy: dict[str, bool],
             min_margin: float = 0.0, dataset_id: str | None = None) -> pd.DataFrame:
    """The slide list from a Stage 4 assignment CSV, counting tiles directly.

    Read through `load_hpc_assignments.read_assignments()` rather than
    `pd.read_csv`, so this path inherits every guard Stage 6 already applies to
    the same file — the cluster column found by elimination, duplicate tiles
    refused, a blank vote_margin refused, short tile names repaired and counted,
    and every identifier read as the text the file holds ('007' stays '007',
    'NA' stays 'NA') rather than as whatever pandas would have guessed.
    A cohort filter that accepted a CSV Stage 6 would refuse is a filter over a
    file nobody should be trusting.

    min_margin exists only to reproduce the KB numbers: Stage 6 can be told to
    drop low-confidence tiles before computing proportions, and if it was, the
    two sources answer different questions until this matches. It defaults to 0,
    which is what Stage 6 defaults to.
    """
    frame, cluster_column = read_assignments(csv_path)
    _refuse_null_keys(frame, ("slides", "samples"), str(csv_path))
    if min_margin > 0:
        frame = frame[frame["vote_margin"] >= min_margin]
        if frame.empty:
            raise SystemExit(
                f"--min-margin {min_margin} excludes every tile in "
                f"{csv_path}. Nothing is left to attribute to a slide."
            )

    work = frame[["samples", "slides", cluster_column]].copy()
    work.columns = ["samples", "slides", "hpc_id"]
    work["hpc_id"] = work["hpc_id"].astype(str).str.strip()
    _check_clusters_known(work["hpc_id"].unique(), malignancy, str(csv_path))

    work["is_malignant"] = work["hpc_id"].map(malignancy)
    # The cluster id on malignant rows and NaN elsewhere, so one nunique() over
    # it counts distinct malignant clusters and gives 0 for a slide with none —
    # rather than counting the malignant subset separately and merging it back,
    # where anything but a left join would drop exactly the slides this filter
    # exists to label non-tumour.
    work["malignant_hpc"] = work["hpc_id"].where(work["is_malignant"])
    # dropna=False for the same reason as from_kb(): a NULL key is refused
    # above, and must never be the thing that silently drops a slide here.
    grouped = work.groupby(["slides", "samples"], as_index=False,
                           dropna=False).agg(
        total_tiles=("hpc_id", "size"),
        malignant_tiles=("is_malignant", "sum"),
        n_malignant_hpcs=("malignant_hpc", "nunique"),
    )

    grouped = grouped.rename(columns={"slides": "slide_id"})
    grouped["malignant_tiles"] = grouped["malignant_tiles"].astype(int)
    grouped["total_tiles"] = grouped["total_tiles"].astype(int)
    grouped["malignant_fraction"] = (
        grouped["malignant_tiles"] / grouped["total_tiles"])
    grouped["dataset_id"] = dataset_id
    return _finish(grouped)


def _refuse_null_keys(frame: pd.DataFrame, columns, source: str) -> None:
    """Refuse a row whose slide, sample or cohort is NULL or blank.

    These are the keys the per-slide aggregate groups on, and pandas' groupby
    drops any group with a NULL key by default — so a malignant slide with a
    NULL dataset_id simply had no row in the output, with nothing printed. The
    dataset_id mismatch guard in from_kb() cannot see it either: it skips NULL
    proportion ids, and a NULL on both sides is not a mismatch. A blank string
    is refused with NULL because that is what a NULL becomes the moment the
    list is written to CSV, and ANORAK grades by `samples`: a slide with no
    sample is a slide with no tumour to attribute a grade to.
    """
    for column in columns:
        values = frame[column]
        bad = values.isna() | (values.astype(str).str.strip() == "")
        if bad.any():
            slides = frame.loc[bad, "slides"].astype(str).unique().tolist()
            raise SystemExit(
                f"{source}: {int(bad.sum()):,} row(s) across {len(slides)} "
                f"slide(s) have a NULL or blank {column}, e.g. {slides[:5]}.\n\n"
                f"Not grouped as their own cohort and not dropped: dropping "
                f"them is what used to happen, silently, and a malignant slide "
                f"with no {column} is still a malignant slide. Fix the "
                f"{column} at its source."
            )


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    """The shared tail of both sources: the verdict, and a stable shape.

    is_tumour is taken from the fraction rather than from `malignant_tiles`,
    which matters at exactly the boundary this rule was chosen for: on the KB
    path the tile count is derived and rounded, so a slide whose malignant
    tissue rounds to zero tiles would come out non-tumour while its fraction
    says otherwise. The fraction is the number that came out of the data.
    """
    frame = frame.copy()
    frame["is_tumour"] = frame["malignant_fraction"] > 0
    for column in _OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return (frame[list(_OUTPUT_COLUMNS)]
            .sort_values("slide_id", kind="stable")
            .reset_index(drop=True))


# --- the report -----------------------------------------------------------

def fraction_distribution(frame: pd.DataFrame, cuts=_FRACTION_CUTS) -> list[dict]:
    """For each candidate malignant-fraction floor: what it keeps and drops.

    The rule is already chosen — any malignant tile — so this is not a knob
    being offered. It is the cost of that choice made visible: the 0.00 row is
    the rule in force, and the rows below say how many of those slides are
    carried by a sliver of tissue. Modelled on Stage 6's margin_tradeoff table,
    which answers the same shape of question from the cohort's own distribution
    rather than from a number someone picked.
    """
    tumour = frame[frame["is_tumour"]]
    rows = []
    for cut in cuts:
        kept = tumour[tumour["malignant_fraction"] >= cut]
        rows.append({
            "min_fraction": cut,
            "slides_kept": int(len(kept)),
            "share_kept": (len(kept) / len(tumour)) if len(tumour) else 0.0,
            "slides_dropped": int(len(tumour) - len(kept)),
            "tiles_kept": int(kept["total_tiles"].sum()),
        })
    return rows


def report(frame: pd.DataFrame) -> dict:
    """Everything the CLI prints, computed in one place so a UI can show it."""
    tumour = frame[frame["is_tumour"]]
    return {
        "slides": int(len(frame)),
        "tumour_slides": int(len(tumour)),
        "non_tumour_slides": int(len(frame) - len(tumour)),
        "total_tiles": int(frame["total_tiles"].sum()),
        "tumour_tiles": int(tumour["total_tiles"].sum()),
        # The slides the inclusive rule is *for*, and the ones a grade derived
        # from them would be least supported by.
        "single_tile_slides": int((tumour["malignant_tiles"] <= 1).sum()),
        "median_fraction": float(tumour["malignant_fraction"].median())
        if len(tumour) else 0.0,
        "min_fraction": float(tumour["malignant_fraction"].min())
        if len(tumour) else 0.0,
        "distribution": fraction_distribution(frame),
    }


def print_report(summary: dict) -> None:
    print(f"  slides       {summary['slides']:,}")
    print(f"  tumour       {summary['tumour_slides']:,} selected, "
          f"{summary['non_tumour_slides']:,} not")
    print(f"  tiles        {summary['tumour_tiles']:,} on selected slides "
          f"of {summary['total_tiles']:,} in the cohort")
    if summary["tumour_slides"]:
        print(f"  fraction     median {summary['median_fraction']:.4f}, "
              f"lowest {summary['min_fraction']:.6f}")
        print(f"  thin slides  {summary['single_tile_slides']:,} selected on one "
              f"malignant tile or fewer")

        print("\n  what a malignant-fraction floor would cost, on this cohort's "
              "own distribution")
        print(f"    {'floor':>6} {'slides kept':>13} {'%kept':>7} "
              f"{'dropped':>9} {'tiles kept':>14}")
        for row in summary["distribution"]:
            print(f"    {row['min_fraction']:>6.3f} {row['slides_kept']:>13,} "
                  f"{100 * row['share_kept']:>6.1f}% {row['slides_dropped']:>9,} "
                  f"{row['tiles_kept']:>14,}")
        print("    (the 0.000 row is the rule in force — any malignant tile "
              "selects the slide.")
        print("     The rows below it are not offered as alternatives; they say "
              "how much of")
        print("     this cohort is selected on a sliver of tissue, so a grade "
              "derived from one")
        print("     is identifiable later rather than looking like any other "
              "grade.)")


# --- CLI ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-kb", action="store_true",
                        help="Read per-slide composition from "
                             "hpl_profile_proportion (Stage 6 has run).")
    source.add_argument("--from-csv", type=Path, metavar="CSV",
                        help="Count tiles in a Stage 4 assignment CSV instead "
                             "(Stage 6 has not run).")
    parser.add_argument("--dataset-id", default=None,
                        help="Cohort to read (--from-kb), or the label written "
                             "into the output's dataset_id column "
                             "(--from-csv). Omitted with --from-kb means every "
                             "cohort in the database.")
    parser.add_argument("--out", type=Path, default=None, metavar="CSV",
                        help="Where to write the slide list: the selected "
                             "(tumour) slides only, ready to hand to ANORAK. "
                             "Omitted prints the report without writing "
                             "anything.")
    rows = parser.add_mutually_exclusive_group()
    rows.add_argument("--include-non-tumour", action="store_true",
                      help="Write every slide, is_tumour=False rows included, "
                           "as a cohort-composition report. NOT an ANORAK "
                           "input: nothing downstream filters on is_tumour.")
    # Kept so an existing command line still parses; it is now the default.
    rows.add_argument("--tumour-only", action="store_true",
                      help="The default; accepted for existing command lines.")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="--from-csv only: drop tiles below this "
                             "vote_margin before counting, to match a Stage 6 "
                             "load that was given the same floor. Default 0, "
                             "which is Stage 6's default.")
    parser.add_argument("--proportion-tolerance", type=float,
                        default=_PROPORTION_TOLERANCE,
                        help="--from-kb only: how far a slide's proportions may "
                             f"sum from 1 before refusing (default "
                             f"{_PROPORTION_TOLERANCE}).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_margin and args.from_kb:
        raise SystemExit(
            "--min-margin applies to --from-csv only. The KB's proportions were "
            "already computed under whatever floor Stage 6 was given, and this "
            "cannot re-apply one to a share.")

    engine = make_engine()
    malignancy = load_malignancy(engine)
    n_malignant = sum(1 for v in malignancy.values() if v)
    print(f"Dictionary   : {len(malignancy)} clusters, {n_malignant} malignant")

    if args.from_kb:
        scope = args.dataset_id or "every cohort"
        print(f"Source       : hpl_profile_proportion ({scope})")
        frame = from_kb(engine, malignancy, dataset_id=args.dataset_id,
                        tolerance=args.proportion_tolerance)
    else:
        print(f"Source       : {args.from_csv}")
        frame = from_csv(args.from_csv, malignancy,
                         min_margin=args.min_margin,
                         dataset_id=args.dataset_id)

    print_report(report(frame))

    if args.out:
        # Tumour-only unless asked otherwise. This file is what the UI tells
        # people to give ANORAK, and no consumer of it reads is_tumour: the
        # full table written by default meant every normal slide in a cohort
        # was tiled and graded, at 67x HPL's pixel volume, with a growth
        # pattern "grade" that looked like any other. The composition report
        # is still one flag away, and named so it cannot be mistaken for a
        # slide list.
        out = frame if args.include_non_tumour else frame[frame["is_tumour"]]
        if out.empty:
            raise SystemExit(
                f"No slide in this cohort is tumour, so there is no slide list "
                f"to write to {args.out}. Pass --include-non-tumour to write "
                f"the composition report instead.")
        # .tmp-then-rename, so an interrupted write leaves no half-file for the
        # next stage to size its Slurm array from.
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(args.out)
        left_out = len(frame) - len(out)
        print(f"\nWrote        : {len(out):,} row(s) to {args.out}"
              + (f" ({left_out:,} non-tumour slide(s) left out; "
                 f"--include-non-tumour keeps them)" if left_out else ""))
    else:
        print("\n(no --out — nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
