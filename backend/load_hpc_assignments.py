#!/usr/bin/env python3
"""Load a cluster-assignment CSV into the Knowledge Bank (tile_registry).

Stage 6, and the step that makes Stage 4's output visible: until a tile's
hpc_id is in tile_registry, the slide viewer's join
(tile_coordinates -> tile_registry -> hpc_dictionary) returns nothing and the
CSV is just a file on disk.

    tile_coordinates.slide_tile ──┐
                                  ├── tile_registry.hpc_id ──> hpc_dictionary
    assignment CSV (slides,tiles)─┘                            (pattern, malignant,
                                                                inflammation, ...)

Everything here is written around one fact: this is the only script in the
pipeline that mutates the shared KB, and its failure mode is not a crash. A
naming mismatch between the CSV and tile_registry silently updates nothing; a
partial match silently updates some tiles and leaves others carrying stale
cluster IDs from an earlier reference. Both look like success. So it refuses to
commit unless the match rate clears a threshold, it reports exactly what it
would change before changing it, and --dry-run is the default posture for
anything unfamiliar.

Writes five columns, all added by migrate_tile_registry_confidence.sql:
    hpc_id, hpc_vote_margin, hpc_neighbor_distance, hpc_assigned_at, hpc_reference

Usage:
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --dry-run
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --commit
    python load_hpc_assignments.py --csv DS_hpc_assignments.csv --commit --min-margin 0.25
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_url import database_url  # noqa: E402
import kb_stage  # noqa: E402
from run_record import record_run  # noqa: E402
from slide_naming import (  # noqa: E402
    make_slide_tile_series,
    normalize_tile_names,
    tile_name_verdict,
)

# Same defaults and env names as tile_server_v2_.py, so a shell configured for
# the server needs no extra setup here. Duplicated rather than imported: that
# module opens HDF5 handles and builds a FastAPI app at import time, which is a
# lot to drag in for a connection string.
DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

# Columns assign_hpc_clusters.py writes besides the cluster ID itself.
_META = ("samples", "slides", "tiles")
_CONFIDENCE = ("vote_margin", "neighbor_distance")

#: Columns *this module* derives and may find already present, because a CSV
#: that has been round-tripped through a frame read_assignments() touched
#: carries them back. They are recomputed either way and must never be
#: candidates for the cluster column.
#:
#: This matters more than it looks. The cluster column is found by elimination,
#: so an unexpected column has two failure modes and only one of them is loud:
#: alongside a real cluster column it makes two candidates and refuses, but on a
#: CSV whose cluster column is absent it would be the only candidate left and
#: get loaded as hpc_id — a registry full of tile names where cluster IDs belong,
#: every row well-formed.
_DERIVED = ("slide_tile",)

# Below this share of CSV rows matching tile_registry, refuse to commit. The
# number this guards against is not 0% — a total mismatch is obvious. It is the
# 3% that means the CSV and the registry disagree about slide naming for all but
# a handful of tiles, which reads as "it worked" in any summary line.
_MIN_MATCH_RATE = 0.95

# Tiles per lookup query. Only reached by the no-staging fallback path — see
# _lookup_present_chunked — which exists so a database where the scratch table
# cannot be created still previews rather than failing.
_LOOKUP_CHUNK = 10_000

#: Connections used to COPY the assignments into the scratch table. Staging is
#: the one part of this stage that parallelises safely, because the scratch
#: table is not Knowledge Bank state (see kb_stage's docstring); the write
#: itself is one statement in one transaction and always will be.
#: Four rather than a core count: this is bound by how fast one Postgres backend
#: can ingest COPY, and past a handful of writers they contend on the same
#: relation extension lock instead of going faster.
_STAGE_WORKERS = 4

#: Workers Postgres may use for the preview's join. The preview is a read, so
#: this is free parallelism with no correctness surface at all.
_QUERY_WORKERS = 4


def make_engine():
    return create_engine(
        database_url(DB_NAME, user=DB_USER, password=DB_PASS,
                     host=DB_HOST, port=DB_PORT),
        pool_pre_ping=True,
    )


def _read_csv(csv_path: Path) -> pd.DataFrame:
    """The CSV as text, read on as many cores as the installed pandas will use.

    A 7.8M-row assignment CSV is minutes of single-threaded parsing on pandas'
    default C engine; the pyarrow engine threads the parse. The fallback is not
    politeness — pyarrow is not among the container extras this pipeline
    installs, so this script has to keep working where it is absent, and a
    reader that raised on a machine without it would take Stage 6 down for a
    speedup.

    Every column comes back as the literal text in the file, with no NA
    guessing (`dtype=str, keep_default_na=False`), and both readers agree on
    that exactly. Letting pandas infer types is a rewrite of the identifier
    columns rather than a reading of them: an all-digit slide id loses its
    zeros ('007' -> 7), one blank or 'NA' anywhere in the column turns every
    other id in it into a float ('1001' -> 1001.0), and a slide or sample
    actually called 'NA', 'None' or 'null' becomes NaN. slide_tile is built
    from those values, so the key comes out as '1001.0_18_15.JPEG' — which
    joins nothing, or joins a different slide that really is called '7'. The
    numeric columns are converted by name in read_assignments(), where a blank
    is still a blank and can be refused as one.
    """
    try:
        return _read_csv_threaded(csv_path)
    except Exception as e:  # noqa: BLE001 - any reader failure falls back
        print(f"  (threaded CSV reader unavailable: {type(e).__name__}: {e}; "
              f"using the default single-threaded reader)", file=sys.stderr)
        return _read_csv_default(csv_path)


def _read_csv_default(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def _read_csv_threaded(csv_path: Path) -> pd.DataFrame:
    """pyarrow's reader, told every column is a string before it parses.

    Not `pd.read_csv(engine="pyarrow", dtype=str)`, which looks equivalent and
    is not: that engine lets pyarrow infer each column first and casts the
    result afterwards, so '007' is parsed as the integer 7 and handed back as
    the string '7'. Only pyarrow's own column_types reaches the parser.
    """
    import csv
    import pyarrow as pa
    import pyarrow.csv as pa_csv

    with open(csv_path, newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise ValueError("no header row")
    if len(set(header)) != len(header):
        # pandas renames a repeated column ('x', 'x.1'); pyarrow keeps both
        # under one name. Left to the default reader so the two cannot differ.
        raise ValueError(f"repeated column name(s) in {header}")
    table = pa_csv.read_csv(csv_path, convert_options=pa_csv.ConvertOptions(
        column_types={name: pa.string() for name in header},
        null_values=[], strings_can_be_null=False,
        quoted_strings_can_be_null=False))
    return table.to_pandas()


def _blank(values: pd.Series) -> pd.Series:
    """Empty after stripping — what an absent field is once read as text."""
    return values.isna() | (values.astype(str).str.strip() == "")


def read_assignments(csv_path: Path) -> tuple[pd.DataFrame, str]:
    """The CSV plus the name of its cluster column.

    The cluster column is named for the reference's groupby ('leiden_2.5'), so
    it is found by elimination rather than by name — the same rule the server's
    validator uses, and for the same reason: hardcoding it breaks the moment the
    reference changes resolution.
    """
    frame = _read_csv(csv_path)
    known = set(_META) | set(_CONFIDENCE) | set(_DERIVED) | {"hpc_reference"}
    missing = [c for c in (*_META, *_CONFIDENCE) if c not in frame.columns]
    if missing:
        raise SystemExit(
            f"{csv_path} is missing {missing}. Expected the output of "
            f"assign_hpc_clusters.py; found columns {list(frame.columns)}."
        )
    cluster_columns = [c for c in frame.columns if c not in known]
    if len(cluster_columns) != 1:
        raise SystemExit(
            f"Could not identify the cluster column in {csv_path}: "
            f"{cluster_columns or 'none'} left after the known ones. "
            f"Columns: {list(frame.columns)}."
        )
    if frame.empty:
        raise SystemExit(f"{csv_path} holds no assignments.")

    # A CSV produced from a .h5 packaged before make_hpl_hdf5.py started storing
    # the ".jpeg" suffix carries the short form, which joins neither
    # tile_registry nor tile_coordinates and merges zero rows against Kai's
    # reference CSV. Nothing about it needs recomputing — the cluster IDs and
    # margins are correct, only the label is short — so the suffix is appended
    # here and the count reported, rather than the load being refused.
    #
    # Mixed still refuses: some names suffixed and some not is what a resume
    # straddling the fix leaves behind, the two sides are indistinguishable by
    # name, and guessing would attach correct cluster IDs to the wrong tiles.
    # A confidence column that is not numeric. One malformed line in a
    # 7.8M-row CSV — a header repeated by a concatenation, a torn write — turns
    # the whole column to object dtype, and every comparison against it
    # afterwards either raises deep in pandas or silently misbehaves. Refused
    # with the offending values rather than coerced: a file with a stray row in
    # it is a file whose row count nobody should trust, and dropping the row
    # quietly would leave a cohort short by an unknown amount.
    #
    # The columns arrive as text (see _read_csv), so a blank field is "" here,
    # not NaN. It is excluded from this check and becomes NaN, so an empty
    # vote_margin still reaches its own refusal below rather than being called
    # "not a number" — or, worse, coerced to 0. 'NA' or 'nan' written into a
    # margin is not blank, and is refused here as the non-number it is.
    for column in _CONFIDENCE:
        if column not in frame.columns:
            continue
        empty = _blank(frame[column])
        coerced = pd.to_numeric(frame[column].where(~empty), errors="coerce")
        bad = coerced.isna() & ~empty
        if bad.any():
            examples = frame.loc[bad, column].astype(str).head(5).tolist()
            raise SystemExit(
                f"{csv_path} has {int(bad.sum()):,} row(s) whose {column} is not "
                f"a number, e.g. {examples}. That is usually a header repeated "
                f"mid-file by a concatenation, or a torn write from an "
                f"interrupted job.\n\n"
                f"Check the file's row count against the projections .h5 before "
                f"loading it — if shards were merged by hand, re-merge them with "
                f"merge_assignment_shards.py, which checks the total against the "
                f"source rather than believing the parts."
            )
        frame[column] = coerced

    # An identity field that is empty. Read as text, a blank slide or tile is
    # "" rather than NaN, and would build the key '_18_15.JPEG' — a row that
    # joins nothing and hides inside the 5% the match-rate guard allows, while
    # its cluster still counts toward a slide called ''. A blank cluster id is
    # the same torn row from the other side: it used to become NaN, which a
    # text hpc_id column receives as the string 'nan'. Refused by name; a blank
    # *sample* is not, because it is not part of any key this stage joins on,
    # and select_tumour_slides.py refuses it itself with its own reasons.
    for column in ("slides", "tiles", cluster_columns[0]):
        empty = _blank(frame[column])
        if empty.any():
            rows = (frame.index[empty][:5] + 2).tolist()   # 1-based, after header
            raise SystemExit(
                f"{csv_path} has {int(empty.sum()):,} row(s) with an empty "
                f"{column}, e.g. at line(s) {rows}. assign_hpc_clusters.py "
                f"writes every one of these for every row, so this is a torn "
                f"or hand-edited file rather than a tile without one. Check "
                f"the row count against the projections .h5 before loading."
            )

    verdict = tile_name_verdict(frame["tiles"])
    if verdict == "mixed":
        raise SystemExit(
            f"{csv_path} has SOME tile names with a file extension and some "
            f"without (e.g. {frame['tiles'].iloc[0]!r}). That is what an "
            f"assignment built from a half-migrated .h5 looks like, and the two "
            f"forms cannot be told apart by name. Repackage and re-assign the "
            f"dataset rather than loading this."
        )
    frame["tiles"], renamed = normalize_tile_names(frame["tiles"])

    # tile_coordinates.slide_tile is "<slides>_<tiles>" upper-cased, e.g.
    # TCGA-55-7574-01Z-00-DX1_18_15.JPEG. Built by the shared helper so this and
    # the dataset-registration step cannot drift apart on the key they join on.
    # Rebuilt from slides + tiles even when the CSV already had a slide_tile
    # column, and deliberately not trusted: one written before tile names were
    # normalised holds the short form (`..._18_15`) and would join nothing,
    # which is the failure this key exists to prevent. The rebuild is from the
    # same two columns the normalisation above just corrected.
    rebuilt = make_slide_tile_series(frame["slides"], frame["tiles"])
    disagreed = 0
    example = None
    if "slide_tile" in frame.columns:
        # Compared rather than silently overwritten. A disagreement is worth a
        # number: all of them differing is the ordinary case (a column written
        # before the tile names were normalised, so `..._18_15` against
        # `..._18_15.JPEG`), and that is benign because the rebuild is what
        # joins. A *handful* differing is not benign — it means the CSV's
        # slides or tiles columns are not what its slide_tile was built from,
        # and the match rate below will be the only other sign of it.
        supplied = frame["slide_tile"].astype(str).str.strip().str.upper()
        differs = supplied != rebuilt
        disagreed = int(differs.sum())
        if disagreed:
            first = differs.idxmax()
            example = (str(supplied.loc[first]), str(rebuilt.loc[first]))
            print(f"  note: {csv_path.name} carried a slide_tile column and "
                  f"{disagreed:,} of {len(frame):,} value(s) disagree with the "
                  f"key rebuilt from slides + tiles, e.g. {example[0]!r} -> "
                  f"{example[1]!r}. The rebuild is what joins tile_registry.",
                  file=sys.stderr)
        else:
            print(f"  note: {csv_path.name} carried a slide_tile column and "
                  f"every value matches the key rebuilt from slides + tiles.",
                  file=sys.stderr)
    frame["slide_tile"] = rebuilt

    # Two rows claiming the same tile.
    #
    # The old writer sent one UPDATE per row, so duplicates applied in file
    # order and the last one silently won. The staged join cannot reproduce
    # that and should not try: `UPDATE ... FROM` picks an arbitrary matching
    # staging row, and which one is a property of the query plan rather than of
    # the data. Refused instead — and refusing is the right answer for the file
    # regardless of how it is written, because a duplicated tile means the CSV's
    # row count is not the cohort's tile count, so every per-slide proportion
    # computed from it is already wrong by an unknown amount.
    #
    # This is what an overlapping shard merge looks like.
    # merge_assignment_shards.py checks the total against the source precisely
    # so this cannot arrive, which makes reaching it a sign the parts were
    # concatenated by hand.
    duplicated = frame["slide_tile"].duplicated(keep=False)
    if duplicated.any():
        counts = frame.loc[duplicated, "slide_tile"].value_counts()
        raise SystemExit(
            f"{csv_path} has {int(duplicated.sum()):,} row(s) across "
            f"{len(counts):,} tile(s) that appear more than once, e.g. "
            f"{[f'{k} x{v}' for k, v in counts.head(3).items()]}.\n\n"
            f"Two assignments for one tile cannot both be loaded, and which one "
            f"would win is not something this can decide for you. The usual "
            f"cause is shard outputs concatenated by hand where their ranges "
            f"overlapped — re-merge them with merge_assignment_shards.py, which "
            f"checks the row total against the projections .h5 rather than "
            f"believing the parts."
        )

    # A vote_margin that is empty rather than wrong.
    #
    # The non-numeric guard above deliberately only catches values that were
    # *something* before coercion, so a blank cell arrives here as NaN having
    # passed every check. It used to be written to hpc_vote_margin as
    # float('nan'); COPY's CSV format reads an empty field as SQL NULL, so the
    # per-row and staged writers would disagree on exactly these rows. Refused
    # rather than reconciled: the margin is the number this stage's entire
    # confidence story rests on — it is what the --min-margin trade-off table is
    # computed from — and a blank one is a torn write, not a tile with no
    # confidence.
    blank = int(frame["vote_margin"].isna().sum())
    if blank:
        raise SystemExit(
            f"{csv_path} has {blank:,} row(s) with an empty vote_margin. That is "
            f"an interrupted or truncated write rather than a tile without a "
            f"confidence — assign_hpc_clusters.py writes a margin for every row "
            f"it emits. Check the row count against the projections .h5 before "
            f"loading."
        )

    # On the frame rather than in the return tuple, which every caller and test
    # already unpacks as exactly (frame, cluster_column).
    frame.attrs["tile_names_normalized"] = renamed
    frame.attrs["slide_tile_supplied"] = "slide_tile" in frame.columns
    frame.attrs["slide_tile_disagreed"] = disagreed
    frame.attrs["slide_tile_example"] = example
    return frame, cluster_columns[0]


#: Leave-one-out accuracy per vote_margin band, measured on the production
#: reference (CLASSIFIER_TUNING_2026-08-13.md §7: 20,000 tiles, k=10,
#: distance-weighted). (upper edge, share correct) — the band is [previous
#: edge, this edge).
#:
#: These are the only numbers that turn a margin into an accuracy, and they were
#: measured *within LATTICeA*, the cohort the 71 HPCs were defined on. Applying
#: them to another cohort assumes a margin of 0.3 means there what it means
#: here, which is exactly the cohort-shift question that cannot be settled
#: without labels. So anything derived from them is an estimate, and says so.
_MARGIN_ACCURACY_BANDS = (
    (0.10, 0.617),
    (0.25, 0.812),
    (0.50, 0.955),
    (0.75, 0.995),
    (1.01, 1.000),
)

_MARGIN_CUTS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)


def margin_tradeoff(frame: pd.DataFrame, cuts=_MARGIN_CUTS) -> list[dict]:
    """For each candidate min_margin: tiles kept, and their expected accuracy.

    The question this answers is the one actually being decided at Stage 6 — how
    many tiles a confidence floor costs, and what it buys — and it can only be
    answered from the cohort's own margin distribution. Everything else about
    picking a threshold is guesswork against a distribution nobody has looked
    at.

    Expected accuracy is each kept tile's band accuracy, averaged. It is an
    estimate under LATTICeA calibration (see _MARGIN_ACCURACY_BANDS), not a
    measurement of this cohort — which has no labels to measure against.
    """
    margins = pd.to_numeric(frame["vote_margin"], errors="coerce").to_numpy(dtype=float)
    edges = np.array([0.0, *(edge for edge, _ in _MARGIN_ACCURACY_BANDS)])
    accuracies = np.array([acc for _, acc in _MARGIN_ACCURACY_BANDS])
    per_tile = accuracies[np.clip(np.digitize(margins, edges) - 1,
                                  0, len(accuracies) - 1)]

    total = len(margins)
    rows = []
    for cut in cuts:
        kept = margins >= cut
        n = int(kept.sum())
        rows.append({
            "min_margin": float(cut),
            "tiles_kept": n,
            "share_kept": (n / total) if total else 0.0,
            "expected_accuracy": float(per_tile[kept].mean()) if n else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# The two statements that replaced 20 million round trips.
#
# Both join tile_registry against a scratch table holding this CSV's keys
# (kb_stage) instead of carrying the keys in the statement. The join predicate
# is UPPER(tile_registry.slide_tile) = stage.slide_tile in both, matching what
# the per-row writer did — make_slide_tile_series already upper-cases the CSV
# side, and migrate_indexes.sql's idx_tr_slide_tile_upper is an index on
# exactly this expression.
# ---------------------------------------------------------------------------

_PRESENT_SQL = """
    SELECT UPPER(t.slide_tile) AS slide_tile, t.hpc_id, t.hpc_reference
    FROM tile_registry t
    JOIN {stage} s ON s.slide_tile = UPPER(t.slide_tile)
"""

_UPDATE_FROM_SQL = """
    UPDATE tile_registry SET
        hpc_id = s.cluster_id,
        hpc_vote_margin = s.margin,
        hpc_neighbor_distance = s.distance,
        hpc_reference = s.reference,
        hpc_assigned_at = :assigned_at
    FROM {stage} AS s
    WHERE UPPER(tile_registry.slide_tile) = s.slide_tile
"""

#: The same update for a database without UPDATE ... FROM. One correlated
#: subquery per column, which is why kb_stage indexes the scratch key: without
#: that index this is five scans of the scratch table per registry row.
#: Only the tests take this path in practice, and the equivalence test runs both
#: against the same fixture so they cannot drift.
_UPDATE_CORRELATED_SQL = """
    UPDATE tile_registry SET
        hpc_id = (SELECT s.cluster_id FROM {stage} s
                  WHERE s.slide_tile = UPPER(tile_registry.slide_tile)),
        hpc_vote_margin = (SELECT s.margin FROM {stage} s
                  WHERE s.slide_tile = UPPER(tile_registry.slide_tile)),
        hpc_neighbor_distance = (SELECT s.distance FROM {stage} s
                  WHERE s.slide_tile = UPPER(tile_registry.slide_tile)),
        hpc_reference = (SELECT s.reference FROM {stage} s
                  WHERE s.slide_tile = UPPER(tile_registry.slide_tile)),
        hpc_assigned_at = :assigned_at
    WHERE UPPER(tile_registry.slide_tile) IN (SELECT slide_tile FROM {stage})
"""


def supports_update_from(bind) -> bool:
    """Whether this database can do UPDATE ... FROM.

    PostgreSQL always could; SQLite gained it in 3.33 (2020), and the tests run
    on whichever SQLite the interpreter was built against. Feature-detected
    rather than assumed so the fast statement is used wherever it exists and the
    portable one is a fallback rather than the test suite's private path.
    """
    name = bind.dialect.name
    if name == "postgresql":
        return True
    if name == "sqlite":
        return tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2]) >= (3, 33)
    return False


def _update_sql(bind, stage: str) -> str:
    template = (_UPDATE_FROM_SQL if supports_update_from(bind)
                else _UPDATE_CORRELATED_SQL)
    return template.format(stage=stage)


def _enable_parallel_query(conn, workers: int = _QUERY_WORKERS) -> None:
    """Ask Postgres to parallelise the preview's join.

    The preview is a pure read, so this has no correctness surface: parallel
    workers return the same rows, and the only thing that changes is how many
    backends scan tile_registry. Note it cannot help the write — PostgreSQL does
    not parallelise the workers of a DML statement, whatever it does with the
    plan underneath — which is why the write's speedup had to come from issuing
    one statement instead of millions rather than from more cores.

    Best-effort: a database that refuses the SET (a restricted role, a pooler
    that disallows it) still previews correctly, just serially.
    """
    if not kb_stage.is_postgres(conn):
        return
    try:
        conn.execute(text(f"SET max_parallel_workers_per_gather = {int(workers)}"))
    except Exception as e:  # noqa: BLE001 - a speed setting, never a requirement
        print(f"  (could not enable parallel query: {type(e).__name__}: {e})",
              file=sys.stderr)


def _lookup_present_chunked(conn, tiles: list[str]) -> pd.DataFrame:
    """The pre-staging preview lookup, kept as the fallback.

    1,850 expanding-IN queries for an 18.5M-row CSV, each re-planned, which is
    why staging exists. It stays because it needs nothing but SELECT: a role
    that cannot create the scratch table can still preview, and a preview that
    refused would block the one step in this stage that changes nothing.
    """
    lookup = text("""
        SELECT UPPER(slide_tile) AS slide_tile, hpc_id, hpc_reference
        FROM tile_registry
        WHERE UPPER(slide_tile) IN :tiles
    """).bindparams(bindparam("tiles", expanding=True))
    pieces = []
    for start in range(0, len(tiles), _LOOKUP_CHUNK):
        pieces.append(pd.read_sql(
            lookup, conn, params={"tiles": tiles[start:start + _LOOKUP_CHUNK]}))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["slide_tile", "hpc_id", "hpc_reference"])


def inspect(engine, frame: pd.DataFrame, cluster_column: str,
            min_margin: float = 0.0, *, stage: str | None = None,
            stage_workers: int = _STAGE_WORKERS,
            timing: dict | None = None) -> dict:
    """What loading this CSV would do, computed without changing anything.

    Deliberately a separate pass rather than a count taken during the write:
    the point is to be able to look before committing, and a preview derived
    from the write path would only exist after the write.

    min_margin previews compute_profiles()'s own exclusion: leave-one-out
    validation against the reference put tiles below 0.1 vote_margin at 57%
    correct and 0.1-0.25 at 76%, against 92%+ once margin clears 0.25 — so a
    tile below threshold is disproportionately likely to be wrong, and
    "excluded_from_aggregates" is how many of those exist in this CSV before
    anything is decided.
    """
    tiles = frame["slide_tile"]

    # Stage the keys and join once, rather than send them back in 10,000-key
    # IN lists. `stage` lets the caller hand over a scratch table it has
    # already filled — main() stages the full payload once and previews against
    # it, so a dry-run-then-commit does not pay for staging twice.
    owned = None
    if stage is None:
        owned = kb_stage.stage_table_name("kb_preview")
        try:
            seconds = kb_stage.stage_frame(engine, owned, frame[["slide_tile"]],
                                           workers=stage_workers)
            if timing is not None:
                timing["stage"] = timing.get("stage", 0.0) + seconds
            stage = owned
        except Exception as e:  # noqa: BLE001 - see _lookup_present_chunked
            print(f"  (could not create a staging table "
                  f"({type(e).__name__}: {e}); previewing with the chunked "
                  f"lookup instead, which is slower but needs only SELECT)",
                  file=sys.stderr)
            with engine.begin() as conn:
                kb_stage.drop_stage(conn, owned)
            owned = None
            stage = None

    started = time.perf_counter()
    try:
        with engine.connect() as conn:
            _enable_parallel_query(conn)
            if stage is not None:
                present = pd.read_sql(text(_PRESENT_SQL.format(stage=stage)), conn)
            else:
                present = _lookup_present_chunked(conn, tiles.tolist())
            clusters = pd.read_sql(
                text("SELECT hpc_id FROM hpc_dictionary"), conn
            )["hpc_id"].astype(str).str.strip().tolist()
    finally:
        if owned is not None:
            with engine.begin() as conn:
                kb_stage.drop_stage(conn, owned)
    if timing is not None:
        timing["preview"] = timing.get("preview", 0.0) + (
            time.perf_counter() - started)

    matched = set(present["slide_tile"])
    assigned = frame[cluster_column].astype(str).str.strip()

    # What tile_registry.hpc_id actually is, and whether this CSV's cluster
    # column can be written into it. Reported here because the preview stages
    # only the join key: without it, the dry run is clean and the commit is
    # what discovers the type mismatch, 18.5M COPYed rows later.
    cluster_type = kb_stage.cluster_stage_type(engine)
    unwritable, unwritable_examples = kb_stage.unwritable_cluster_ids(
        assigned, cluster_type)

    # Render the IDs the way the integer column will hold them before comparing
    # them to hpc_dictionary. A cluster column with a single missing value reads
    # as float64, so astype(str) gives "45.0" while the dictionary's integer
    # column gives "45" — and every ID in the cohort is then reported as having
    # no dictionary row. That refusal is believable and names the wrong cause,
    # which is the failure this file is written against.
    if kb_stage.is_integer_type(cluster_type):
        numbers = pd.to_numeric(assigned, errors="coerce")
        whole = numbers.notna() & (numbers == numbers.round())
        if whole.any():
            assigned = assigned.mask(
                whole, numbers.where(whole).astype("Int64").astype(str))

    already = present[present["hpc_id"].notna()]
    other_reference = already[
        already["hpc_reference"].notna()
        & (already["hpc_reference"] != frame["hpc_reference"].iloc[0])
    ] if "hpc_reference" in present.columns else already.iloc[0:0]

    return {
        "rows": len(frame),
        "matched": len(matched),
        "unmatched": len(frame) - len(matched),
        # islice over a generator rather than a list comprehension then a
        # slice: the comprehension built the whole miss list first, which for a
        # cohort that matches nothing is an 18.5-million-element list to show
        # five names from.
        "unmatched_examples": list(islice(
            (t for t in tiles if t not in matched), 5)),
        # A cluster ID with no hpc_dictionary row joins to NULL in the viewer:
        # the tile gets a cluster but no pattern, malignancy or inflammation.
        "unknown_clusters": sorted(set(assigned) - set(clusters)),
        "known_clusters": len(clusters),
        # The column's real type, and how many cluster IDs it cannot hold. A
        # non-zero count here is a refusal at commit, not a warning.
        "cluster_column_type": cluster_type,
        "unwritable_cluster_ids": unwritable,
        "unwritable_cluster_examples": unwritable_examples,
        "overwriting": int(len(already)),
        "overwriting_other_reference": int(len(other_reference)),
        "distribution": assigned.value_counts().head(5).to_dict(),
        "low_margin": int((frame["vote_margin"] < 0.1).sum()),
        "reference": str(frame["hpc_reference"].iloc[0]),
        "min_margin": min_margin,
        "excluded_from_aggregates": int((frame["vote_margin"] < min_margin).sum()) if min_margin > 0 else 0,
        # How many tile names had ".jpeg" appended on the way in. Zero for a CSV
        # written after the packaging fix; non-zero says the match rate below
        # was only reachable because of that correction.
        "tile_names_normalized": int(frame.attrs.get("tile_names_normalized", 0)),
        # Whether the CSV brought its own slide_tile, and how much of it
        # disagreed with the key that actually joins. All of it differing is
        # the ordinary pre-normalisation case; a few values differing means the
        # CSV's own columns disagree with each other.
        "slide_tile_supplied": bool(frame.attrs.get("slide_tile_supplied", False)),
        "slide_tile_disagreed": int(frame.attrs.get("slide_tile_disagreed", 0)),
        "slide_tile_example": frame.attrs.get("slide_tile_example"),
        # What each candidate threshold would cost and buy, from this cohort's
        # own margins. Computed here so the UI and the CLI show the same table.
        "margin_tradeoff": margin_tradeoff(frame),
    }


def load(engine, frame: pd.DataFrame, cluster_column: str, *,
         profiles: tuple[pd.DataFrame, pd.DataFrame] | None = None,
         stage: str | None = None, stage_workers: int = _STAGE_WORKERS,
         vacuum: bool = True, rebuild_hpc_index: bool = False,
         timing: dict | None = None) -> int:
    """Write the assignments. Returns registry rows changed.

    One transaction for everything that lands in the Knowledge Bank: a
    half-loaded registry, with some tiles on the new reference and some on the
    old, is not a state anything downstream can interpret.

    The write is a single statement joined against a scratch table, not one
    UPDATE per tile. What that replaced, at 18.5M tiles, was 18.5M statement
    executions preceded by a Python list of 18.5M six-key dicts — several GB of
    interpreter objects built before the first row was sent. Staging happens
    *outside* the transaction, on several connections, which is only sound
    because the scratch table is not Knowledge Bank state: a half-staged table
    is discarded and remade, and `kb_stage.stage_frame` verifies its row count
    against the frame before this function will join against it.

    `stage` accepts a scratch table the caller already filled, so main() does
    not stage the same 18.5M keys for the preview and again for the write.
    """
    timing = timing if timing is not None else {}
    now = datetime.now(timezone.utc)

    owned = None
    if stage is None:
        owned = kb_stage.stage_table_name("kb_load")
        # Read off tile_registry.hpc_id rather than assumed: see
        # kb_stage.cluster_stage_type. Both the DDL and the staged values depend
        # on it, so it is resolved once, here, before anything is written.
        cluster_type = kb_stage.cluster_stage_type(engine)
        timing["stage"] = timing.get("stage", 0.0) + kb_stage.stage_frame(
            engine, owned,
            kb_stage.build_stage_frame(frame, cluster_column, cluster_type),
            workers=stage_workers, cluster_type=cluster_type)
        stage = owned

    updated = 0
    try:
        # One transaction covering the tiles and both aggregates. Splitting them
        # would allow a registry whose per-tile clusters and per-slide
        # proportions came from different runs, which is worse than either being
        # stale: nothing downstream can tell that has happened.
        with engine.begin() as conn:
            if rebuild_hpc_index:
                _drop_hpc_index(conn)
            started = time.perf_counter()
            result = conn.execute(text(_update_sql(conn, stage)),
                                  {"assigned_at": now})
            updated = result.rowcount if result.rowcount is not None else 0
            timing["tiles"] = timing.get("tiles", 0.0) + (
                time.perf_counter() - started)
            print(f"  tiles {updated:,} registry rows updated", flush=True)

            if profiles is not None:
                started = time.perf_counter()
                proportions, summary = profiles
                written = replace_profiles(conn, proportions, summary)
                for table, count in written.items():
                    print(f"  {table}: {count:,} rows", flush=True)
                timing["aggregates"] = timing.get("aggregates", 0.0) + (
                    time.perf_counter() - started)

            if rebuild_hpc_index:
                started = time.perf_counter()
                _create_hpc_index(conn)
                timing["hpc_index"] = timing.get("hpc_index", 0.0) + (
                    time.perf_counter() - started)
    finally:
        # Dropped whether or not the write committed: it is scratch either way,
        # and leaving it behind on a failure is how a database accumulates
        # tables nobody can date.
        if owned is not None:
            with engine.begin() as conn:
                kb_stage.drop_stage(conn, owned)

    if vacuum:
        timing["vacuum"] = timing.get("vacuum", 0.0) + _vacuum_analyze(engine)
    return updated


#: The index on the column this stage rewrites on every row.
_HPC_INDEX = "idx_tr_hpc_id"


def _drop_hpc_index(conn) -> None:
    """Drop idx_tr_hpc_id so the update does not maintain it row by row.

    Because hpc_id is indexed, none of these updates can be a HOT update: every
    one of the 18.5M rows churns this index as well as the heap. Dropping it for
    the duration and rebuilding it once is less total work.

    Off by default, and it should stay that way unless someone has decided the
    cost is worth paying. DROP INDEX takes an ACCESS EXCLUSIVE lock on
    tile_registry, and because this runs inside the write's transaction that
    lock is held for the whole load — so the slide viewer and the chatbot block
    on every query against tile_registry until the write commits, which for a
    full cohort is not a short time. Inside the transaction is nonetheless the
    only safe place for it: a failure rolls back the DROP along with everything
    else, so there is no outcome where the load fails and the index is simply
    gone.
    """
    if not kb_stage.is_postgres(conn):
        return
    print(f"  dropping {_HPC_INDEX} for the duration of the write "
          f"(tile_registry is locked until this commits)", flush=True)
    conn.execute(text(f"DROP INDEX IF EXISTS {_HPC_INDEX}"))


def _create_hpc_index(conn) -> None:
    if not kb_stage.is_postgres(conn):
        return
    print(f"  rebuilding {_HPC_INDEX}", flush=True)
    conn.execute(text(
        f"CREATE INDEX IF NOT EXISTS {_HPC_INDEX} ON tile_registry (hpc_id)"))


def _vacuum_analyze(engine) -> float:
    """VACUUM ANALYZE tile_registry after the write. Returns seconds taken.

    Not a tuning nicety. Under MVCC an UPDATE writes a new row version and
    leaves the old one dead, so updating every row of tile_registry roughly
    doubles the table on disk — no write strategy avoids that, staged or per-row.
    Without a vacuum that space is only reclaimed whenever autovacuum next gets
    to a table this size, and until it does every sequential scan the viewer
    does reads both versions of every row. ANALYZE matters for a second reason:
    hpc_id went from mostly NULL to fully populated, and the planner's old
    statistics say otherwise.

    Runs outside the write's transaction because VACUUM cannot run inside one,
    and after it, so a failed load does not spend the time. Never fatal — the
    rows are committed by the time this is called, and reporting a completed
    write as failed because the housekeeping did not run is the more misleading
    of the two errors.
    """
    if not kb_stage.is_postgres(engine):
        return 0.0
    started = time.perf_counter()
    print("  vacuum analyze tile_registry (reclaiming the old row versions "
          "this update leaves behind) ...", flush=True)
    try:
        pooled = engine.raw_connection()
        try:
            # The DBAPI connection itself rather than the pool's proxy: VACUUM
            # cannot run inside a transaction, and AUTOCOMMIT has to be set on
            # the psycopg2 connection, not on something wrapping it.
            raw = pooled.dbapi_connection
            raw.set_isolation_level(0)  # ISOLATION_LEVEL_AUTOCOMMIT
            try:
                with raw.cursor() as cursor:
                    cursor.execute("VACUUM ANALYZE tile_registry")
            finally:
                raw.set_isolation_level(1)  # as the pool handed it over
        finally:
            pooled.close()
    except Exception as e:  # noqa: BLE001 - housekeeping, never the verdict
        print(f"  WARNING: VACUUM ANALYZE tile_registry failed "
              f"({type(e).__name__}: {e}). The load itself committed. Run it by "
              f"hand, or leave it to autovacuum — until then tile_registry holds "
              f"a dead row version per updated tile and the planner's statistics "
              f"for hpc_id are stale.", file=sys.stderr)
    return time.perf_counter() - started


def report_timing(elapsed: float, timing: dict) -> None:
    """Where the time went, in Stage 4's format, because Stage 6 had no such
    line and every claim about its cost was therefore arithmetic."""
    if not timing:
        return
    accounted = sum(timing.values())
    print("Time spent: " + ", ".join(
        f"{name} {seconds:,.0f}s ({seconds / max(elapsed, 1e-9) * 100:.0f}%)"
        for name, seconds in sorted(timing.items(), key=lambda kv: -kv[1])))
    if elapsed - accounted > 0.05 * elapsed:
        print(f"            unaccounted {elapsed - accounted:,.0f}s "
              f"({(elapsed - accounted) / max(elapsed, 1e-9) * 100:.0f}%)")


def compute_profiles(frame: pd.DataFrame, cluster_column: str,
                     cancer_type: str | None, min_margin: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two per-slide aggregates, from the same CSV the tiles came from.

    Definitions taken from the scripts that first populated these tables
    (Filling_out_kb.ipynb and csv_files/For_KB/filling_out_kb_hs.py) so the rows
    this writes are the same shape as the rows already there:

      hpl_profile_proportion  per (samples, slides, hpc_id): that cluster's share
                              of the slide's tiles, summing to 1 per slide.
      hpl_profile_summary     per (samples, slides): tile count and modal cluster.

    They matter because they are what the chatbot and the HPC panels read — not
    tile_registry. Loading per-tile assignments without refreshing these leaves
    the UI showing new clusters per tile and old proportions per slide, with
    nothing to indicate the two disagree.

    min_margin drops tiles below that vote_margin before either aggregate is
    computed. tile_registry is untouched either way — every tile keeps its own
    hpc_id and margin regardless — this only decides what counts toward the
    per-slide numbers people actually read. total_tiles shrinks along with it
    rather than staying at the slide's full tile count: the alternative, a
    total_tiles that counts tiles the proportions and dominant_hpc never saw,
    would report a number next to a composition it does not match.
    """
    work = frame[["samples", "slides", cluster_column, "vote_margin"]].copy()
    work.columns = ["samples", "slides", "hpc_id", "vote_margin"]
    if min_margin > 0:
        work = work[work["vote_margin"] >= min_margin]
    work["hpc_id"] = work["hpc_id"].astype(str).str.strip()

    proportions = work.groupby(["samples", "slides", "hpc_id"], as_index=False).size()
    proportions["proportion"] = proportions.groupby(["samples", "slides"])["size"].transform(
        lambda x: x / x.sum()
    )
    proportions = proportions.drop(columns=["size"])

    summary = work.groupby(["samples", "slides"], as_index=False).agg(
        total_tiles=("hpc_id", "count"),
        dominant_hpc=("hpc_id", lambda x: x.value_counts().idxmax()),
    )
    if cancer_type is not None:
        summary["cancer_type"] = cancer_type
    return proportions, summary


def _existing_columns(conn, table: str) -> set[str]:
    """Columns the live table actually has.

    Inspected rather than assumed: these tables predate this script and were
    filled by hand from notebooks, so an insert naming a column that is not
    there fails the whole transaction — including the tile_registry update that
    had nothing to do with it.
    """
    return {c["name"] for c in sqlalchemy_inspect(conn).get_columns(table)}


def _catch_up_id_sequence(conn, table: str) -> int | None:
    """Advance `table`'s serial id sequence past its largest id, if it lags.

    Both aggregate tables take `id` from a sequence, and the INSERT below names
    no id. A sequence behind MAX(id) — what a data-only restore or a COPY with
    explicit ids leaves, since neither calls setval — hands out an id a row
    already holds: on 2026-09-27 the RADIOGENOMICS load of 18.5M tiles failed
    at its very last statement on `hpl_profile_summary_pkey (id)=(1)`, and the
    single transaction took every tile update back with it. Nothing references
    `id` (both foreign keys into hpl_profile_summary are on (samples, slides)),
    so moving the sequence forward can put no row in the wrong place; it is only
    ever moved forward, and the move is reported.

    Postgres only — SQLite's INTEGER PRIMARY KEY picks max+1 itself. setval is
    not transactional, so a later rollback keeps the advance, which is harmless.
    Returns the new value, or None when nothing was changed.
    """
    if conn.dialect.name != "postgresql":
        return None
    sequence = conn.execute(
        text("SELECT pg_get_serial_sequence(:t, 'id')"), {"t": table}).scalar()
    if not sequence:
        return None
    largest = conn.execute(text(f"SELECT MAX(id) FROM {table}")).scalar()
    if largest is None:
        return None
    last = conn.execute(
        text("SELECT pg_sequence_last_value(CAST(:s AS regclass))"), {"s": sequence}).scalar()
    if last is not None and last >= largest:
        return None
    conn.execute(text("SELECT setval(CAST(:s AS regclass), :v)"),
                 {"s": sequence, "v": int(largest)})
    print(f"  {table}: id sequence {sequence} was at {last}, behind MAX(id) "
          f"{largest}; advanced it to {largest} so the insert cannot reuse an id",
          file=sys.stderr)
    return int(largest)


def replace_profiles(conn, proportions: pd.DataFrame, summary: pd.DataFrame) -> dict:
    """Swap in the aggregates for just the slides being loaded.

    Scoped by slide, not a whole-table rebuild: loading ten slides must not
    delete the proportions for every other slide in the KB. Delete-then-insert
    rather than upsert because a slide's cluster set changes between references —
    a cluster that no longer appears must lose its row, and an upsert would leave
    it behind at its old proportion.

    Order matters, and not in the obvious way. In Postgres
    hpl_profile_proportion carries
    FOREIGN KEY (samples, slides) REFERENCES hpl_profile_summary ON DELETE CASCADE,
    so this cannot be a per-table delete-then-insert loop:

      * inserting a proportion row for a slide with no summary row yet — every
        slide on a cohort's first load — violates the FK and rolls back the whole
        transaction, taking the tile_registry update with it;
      * inserting proportions first for a slide that *does* exist, then deleting
        its summary row, cascades away the proportions just written, leaving a
        summary row with no proportions.

    So: both deletes first (child, then parent), then both inserts (parent, then
    child). The SQLite tests cannot see this — their fixtures create the two
    tables without the foreign key — so it is asserted against a real FK in
    test_kb_load.py rather than left to the schema.
    """
    # UPPER as well as TRIM: migrate_indexes.sql normalised the live columns to
    # UPPER(TRIM(...)), so matching on TRIM alone finds nothing for any slide
    # whose CSV casing differs. That failure is silent in the worst way — the
    # DELETE removes zero rows, the INSERT still runs, and the slide ends up with
    # two sets of aggregates that both look plausible.
    slides = sorted({s.strip().upper() for s in summary["slides"].astype(str)})
    bind = {"slides": slides}

    # dataset_id is NOT NULL on both aggregate tables in the live database, and
    # compute_profiles() cannot know it — the assignment CSV does not carry a
    # cohort. Resolved here from tile_registry, which registration filled, so
    # the aggregates are scoped to exactly the cohort their tiles belong to
    # rather than to whatever the caller believed.
    #
    # This was a latent break, not a new requirement: before it, the INSERT
    # below raised NotNullViolation on any real Postgres, and because load() is
    # one transaction the rollback took the tile_registry update with it. It had
    # never surfaced because no cohort had ever got past the 95% match gate to
    # reach this line, and the SQLite test fixtures declare dataset_id nullable.
    dataset_by_slide = {}
    if "dataset_id" in _existing_columns(conn, "hpl_profile_summary"):
        rows = conn.execute(
            text("SELECT UPPER(TRIM(slides)) AS s, dataset_id FROM tile_registry "
                 "WHERE UPPER(TRIM(slides)) IN :slides AND dataset_id IS NOT NULL "
                 "GROUP BY 1, 2").bindparams(bindparam("slides", expanding=True)),
            bind,
        ).fetchall()
        for slide, dataset_id in rows:
            # A slide already claimed by two cohorts is refused at registration,
            # so this keeps the first and does not invent a resolution.
            dataset_by_slide.setdefault(slide, dataset_id)

        missing = [s for s in slides if s not in dataset_by_slide]
        if missing:
            raise SystemExit(
                f"{len(missing)} slide(s) have no dataset_id in tile_registry "
                f"(e.g. {missing[:3]}). The aggregates cannot be scoped to a "
                f"cohort without one. Register the dataset first — "
                f"register_dataset.py, or the 'Register in the Knowledge Bank' "
                f"step in the UI."
            )

        for frame in (summary, proportions):
            frame["dataset_id"] = (frame["slides"].astype(str).str.strip().str.upper()
                                   .map(dataset_by_slide))

    # Refuse to delete another cohort's aggregates.
    #
    # These deletes are scoped by slide NAME, not by cohort, because that is
    # what the rewrite has to match. But a slide name is not unique across
    # cohorts, so on a shared database this could silently remove the rows a
    # different dataset_id owns — turning "load a new cohort" into "quietly
    # replace an old one", which is the exact failure this codebase is written
    # against.
    #
    # It cannot simply leave them, either: hpl_profile_summary is UNIQUE on
    # (samples, slides) with no dataset_id, so two cohorts cannot both hold a
    # row for the same slide even in principle. Adding would hit the
    # constraint. So the only honest options are delete-and-replace or refuse,
    # and refusing is the one that never destroys data nobody asked to touch.
    if dataset_by_slide:
        ours = set(dataset_by_slide.values())
        conflicting = conn.execute(
            text("SELECT DISTINCT UPPER(TRIM(slides)), dataset_id "
                 "FROM hpl_profile_summary "
                 "WHERE UPPER(TRIM(slides)) IN :slides "
                 "  AND dataset_id IS NOT NULL AND dataset_id NOT IN :ours"
                 ).bindparams(bindparam("slides", expanding=True),
                              bindparam("ours", expanding=True)),
            {**bind, "ours": sorted(ours)},
        ).fetchall()
        if conflicting:
            listed = ", ".join(f"{s} (owned by {d})" for s, d in conflicting[:5])
            raise SystemExit(
                f"{len(conflicting)} slide(s) already have aggregates belonging to a "
                f"different cohort: {listed}. Loading would delete them, and "
                f"hpl_profile_summary's UNIQUE (samples, slides) has no dataset_id, "
                f"so both cannot coexist. Resolve which cohort owns these slides "
                f"before loading — this will not overwrite another dataset's rows."
            )

    def _delete(table: str) -> None:
        conn.execute(
            text(f"DELETE FROM {table} WHERE UPPER(TRIM(slides)) IN :slides").bindparams(
                bindparam("slides", expanding=True)
            ),
            bind,
        )

    def _insert(table: str, frame: pd.DataFrame) -> int:
        columns = _existing_columns(conn, table)
        usable = [c for c in frame.columns if c in columns]
        skipped = [c for c in frame.columns if c not in columns]
        if skipped:
            print(f"  {table}: no column(s) {skipped}; not writing them", file=sys.stderr)
        placeholders = ", ".join(f":{c}" for c in usable)
        conn.execute(
            text(f"INSERT INTO {table} ({', '.join(usable)}) VALUES ({placeholders})"),
            frame[usable].to_dict("records"),
        )
        return len(frame)

    _delete("hpl_profile_proportion")
    _delete("hpl_profile_summary")
    for table in ("hpl_profile_summary", "hpl_profile_proportion"):
        _catch_up_id_sequence(conn, table)
    written = {
        "hpl_profile_summary": _insert("hpl_profile_summary", summary),
        "hpl_profile_proportion": _insert("hpl_profile_proportion", proportions),
    }
    membership = replace_slide_membership(conn, proportions)
    if membership is not None:
        written["slide_hpc_membership"] = membership
    return written


def replace_slide_membership(conn, proportions: pd.DataFrame):
    """Refresh slide_hpc_membership for the slides being loaded.

    Which HPCs appear on which slide — derived from the same assignment as the
    proportions, and refreshed with them, because it is read alongside them.
    The reader is not obvious: app/hpc_chat_handlers_v23.py:334 enumerates
    every table in the database with `insp.get_table_names()`, keeps any that
    has an hpc_id or dominant_hpc column, skipping only hpc_dictionary and
    h_latent_vectors, and renders up to five matching rows straight to the
    user. So this table is answered out of the chatbot without ever being named
    in a query — which is why a grep for it finds nothing and why it had been
    left stale, showing 19,493 rows from cohorts nobody was asking about.

    Derived from `proportions` rather than from the raw assignment so that
    min_margin applies here too. A cluster excluded from a slide's proportions
    but still listed as present would let the chatbot report a slide as
    containing an HPC the aggregate table has no row for, and the two are shown
    side by side.

    Returns None when the table is absent — it has no CREATE TABLE in git older
    than migrate_kb_base_tables.sql, so a database predating that has no such
    table and this must not fail the load.

    Scoped by slide_id and not by dataset_id, because the table has no
    dataset_id column: two cohorts holding the same slide id share these rows.
    Recorded as a property of the schema rather than worked around here.
    """
    if not sqlalchemy_inspect(conn).has_table("slide_hpc_membership"):
        print("  slide_hpc_membership: table absent; not writing it", file=sys.stderr)
        return None

    pairs = (proportions[["slides", "hpc_id"]]
             .dropna()
             .drop_duplicates())
    slides = sorted({str(s).strip().upper() for s in pairs["slides"]})
    if not slides:
        return 0

    # compute_profiles() carries hpc_id as a string, because the cluster column
    # is named for the reference's groupby and its values arrive as whatever the
    # CSV held — "3" from an int column, "3.0" from a float one. This table's
    # column is integer, so int("3.0") would raise and take the tile_registry
    # update down with it.
    records = []
    for slide, hpc in zip(pairs["slides"], pairs["hpc_id"]):
        try:
            cluster = int(float(str(hpc).strip()))
        except (TypeError, ValueError):
            # All or nothing: a membership missing the clusters that happen not
            # to be numeric is a table that looks complete and under-reports,
            # which is worse than one that was not refreshed and says so.
            print(f"  slide_hpc_membership: cluster id {hpc!r} is not an integer; "
                  f"not writing this table", file=sys.stderr)
            return None
        records.append({"slide_id": str(slide).strip().upper(), "hpc_id": cluster})

    conn.execute(
        text("DELETE FROM slide_hpc_membership WHERE UPPER(TRIM(slide_id)) IN :slides")
        .bindparams(bindparam("slides", expanding=True)),
        {"slides": slides},
    )
    conn.execute(
        text("INSERT INTO slide_hpc_membership (slide_id, hpc_id) "
             "VALUES (:slide_id, :hpc_id)"),
        records,
    )
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="Output of assign_hpc_clusters.py.")
    parser.add_argument("--commit", action="store_true",
                        help="Actually write. Without it this only reports.")
    parser.add_argument("--min-match-rate", type=float, default=_MIN_MATCH_RATE,
                        help="Refuse to commit below this share of CSV rows matching "
                             "tile_registry.")
    parser.add_argument("--cancer-type", type=str, default=None,
                        help="Value for hpl_profile_summary.cancer_type (e.g. LUAD). "
                             "Omitted leaves it unset rather than guessing.")
    parser.add_argument("--skip-profiles", action="store_true",
                        help="Only update tile_registry. The per-slide aggregates the "
                             "chatbot and HPC panels read will then disagree with it.")
    parser.add_argument("--allow-unknown-clusters", action="store_true",
                        help="Load cluster IDs that have no hpc_dictionary row. They "
                             "will show in the viewer with no annotations.")
    # See the same flags on register_dataset.py: when this runs as the Slurm
    # job the server submits, the job is what knows whether the write committed.
    parser.add_argument("--record-run", default=None, metavar="SUBMISSION_ID",
                        help="Record the outcome against this run in "
                             "slurm_dataset_runs.")
    parser.add_argument("--record-run-db", default=None, metavar="DBNAME",
                        help="Database holding slurm_dataset_runs. Run tracking "
                             "stays in production whichever Knowledge Bank the "
                             "rows go to, so this is separate from DB_NAME.")
    parser.add_argument("--record-kb-target", default=None,
                        help="Recorded as kb_load_kb_target, so the run says "
                             "which Knowledge Bank it filled.")
    parser.add_argument("--stage-workers", type=int, default=_STAGE_WORKERS,
                        metavar="N",
                        help="Connections used to COPY the assignments into the "
                             "scratch table the write joins against. Staging is "
                             "the only part of this stage that parallelises "
                             "safely; the write is one statement in one "
                             "transaction. 1 takes the parallel path out of the "
                             "picture entirely.")
    parser.add_argument("--no-vacuum", action="store_true",
                        help="Skip the VACUUM ANALYZE after the write. Updating "
                             "every row leaves a dead version of each behind, so "
                             "skipping this leaves tile_registry roughly twice "
                             "its size and the planner's hpc_id statistics stale "
                             "until autovacuum reaches it.")
    parser.add_argument("--rebuild-hpc-index", action="store_true",
                        help="Drop idx_tr_hpc_id for the duration of the write "
                             "and rebuild it after. Less total work, because "
                             "every row's update otherwise maintains that index "
                             "too — but it holds an ACCESS EXCLUSIVE lock on "
                             "tile_registry for the whole load, so the viewer "
                             "and chatbot block until it commits.")
    parser.add_argument("--min-margin", type=float, default=0.0,
                        help="Exclude tiles below this vote_margin from "
                             "hpl_profile_proportion/summary. tile_registry keeps every "
                             "tile's hpc_id and margin regardless of this flag — it only "
                             "changes what counts toward the per-slide aggregates. "
                             "0 (default) excludes nothing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    timing: dict[str, float] = {}

    _t = time.perf_counter()
    frame, cluster_column = read_assignments(args.csv)
    timing["read csv"] = time.perf_counter() - _t

    engine = make_engine()

    # Scratch tables left by a job killed between CREATE and DROP. Swept here
    # rather than left to accumulate, and only ones older than a day, so this
    # can never remove the table under a load running on another node right now.
    stale = kb_stage.sweep_stale(engine)
    if stale:
        print(f"  swept {len(stale)} stale staging table(s) from earlier runs: "
              f"{stale[:3]}")

    # Staged once and used by both the preview and the write. A dry run only
    # ever reads the keys, so it does not pay to ship the payload columns it
    # will not look at.
    cluster_type = kb_stage.cluster_stage_type(engine)
    stage_frame = (kb_stage.build_stage_frame(frame, cluster_column, cluster_type)
                   if args.commit else frame[["slide_tile"]])
    stage = kb_stage.stage_table_name("kb_load")
    try:
        timing["stage"] = kb_stage.stage_frame(
            engine, stage, stage_frame, workers=max(1, args.stage_workers),
            cluster_type=cluster_type)
        _run(args, engine, frame, cluster_column, stage, timing, started)
    finally:
        with engine.begin() as conn:
            kb_stage.drop_stage(conn, stage)


def _run(args, engine, frame, cluster_column, stage, timing, started) -> None:
    report = inspect(engine, frame, cluster_column, min_margin=args.min_margin,
                     stage=stage, timing=timing)

    print(f"CSV            : {args.csv}")
    print(f"  rows         {report['rows']:,}   cluster column '{cluster_column}'")
    print(f"  reference    {report['reference']}")
    if report.get("tile_names_normalized"):
        print(f"  tile names   .jpeg appended to "
              f"{report['tile_names_normalized']:,} name(s) so they match the "
              f"Knowledge Bank's '18_15.jpeg' form; the CSV on disk still holds "
              f"the short form")
    if report.get("slide_tile_supplied"):
        if report["slide_tile_disagreed"]:
            supplied, key = report["slide_tile_example"]
            print(f"  slide_tile   the CSV's own column disagrees with the join "
                  f"key on {report['slide_tile_disagreed']:,} of "
                  f"{report['rows']:,} row(s) ({supplied} -> {key}); the "
                  f"rebuilt key is what joins")
        else:
            print(f"  slide_tile   the CSV's own column matches the rebuilt "
                  f"join key on every row")
    print(f"  matched      {report['matched']:,} of {report['rows']:,} "
          f"({report['matched'] / report['rows'] * 100:.1f}%) in tile_registry")
    if report["unmatched"]:
        print(f"  unmatched    {report['unmatched']:,}, e.g. {report['unmatched_examples']}")
    print(f"  overwriting  {report['overwriting']:,} tiles that already have a cluster"
          + (f" ({report['overwriting_other_reference']:,} from a different reference)"
             if report["overwriting_other_reference"] else ""))
    print(f"  clusters     {report['known_clusters']} in hpc_dictionary; "
          f"largest here: " + ", ".join(f"{k}={v:,}" for k, v in report["distribution"].items()))
    print(f"  low margin   {report['low_margin']:,} tiles below 0.1")

    if report.get("margin_tradeoff"):
        print("\n  what a --min-margin would cost and buy, on this cohort's own "
              "margins:")
        print(f"    {'cut':>5} {'tiles kept':>14} {'%kept':>7} "
              f"{'expected accuracy':>19}")
        for row in report["margin_tradeoff"]:
            print(f"    {row['min_margin']:>5.2f} {row['tiles_kept']:>14,} "
                  f"{row['share_kept'] * 100:>6.1f}% "
                  f"{row['expected_accuracy'] * 100:>18.2f}%")
        print("    (expected accuracy is this cohort's margin distribution "
              "weighted by leave-one-out accuracy measured on the reference — an "
              "estimate, not a measurement of this cohort)")
    if args.min_margin > 0:
        print(f"  min margin   {args.min_margin} — excludes "
              f"{report['excluded_from_aggregates']:,} tile(s) from the aggregates below")

    profiles = None
    if not args.skip_profiles:
        profiles = compute_profiles(frame, cluster_column, args.cancer_type,
                                    min_margin=args.min_margin)
        proportions, summary = profiles
        print(f"  aggregates   {len(proportions):,} proportion rows and "
              f"{len(summary):,} summary rows across "
              f"{summary['slides'].nunique()} slide(s)")
        if args.cancer_type is None:
            print("               (cancer_type not set — pass --cancer-type to fill it)")

    match_rate = report["matched"] / report["rows"]
    problems = []
    if match_rate < args.min_match_rate:
        problems.append(
            f"only {match_rate * 100:.1f}% of rows match tile_registry (need "
            f"{args.min_match_rate * 100:.0f}%). The usual cause is a slide-naming "
            f"difference between the .h5 and the registry, not missing tiles — "
            f"check the unmatched examples above against "
            f"`SELECT slide_tile FROM tile_registry LIMIT 5`."
        )
    if report.get("unwritable_cluster_ids"):
        problems.append(
            f"{report['unwritable_cluster_ids']:,} cluster ID(s) are not whole "
            f"numbers, and tile_registry.hpc_id is "
            f"{report['cluster_column_type']}: "
            f"{report['unwritable_cluster_examples']}. No flag makes these "
            f"writable — the assignment CSV's cluster column is wrong. This is "
            f"the check that turns job 1243334's failed UPDATE into a refusal "
            f"before any row is staged."
        )
    if report["unknown_clusters"] and not args.allow_unknown_clusters:
        problems.append(
            f"{len(report['unknown_clusters'])} cluster ID(s) have no hpc_dictionary "
            f"row: {report['unknown_clusters'][:10]}. Those tiles would show a cluster "
            f"with no pattern or malignancy annotation. Pass "
            f"--allow-unknown-clusters if that is intended."
        )

    if problems:
        print("\nRefusing to load:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    if not args.commit:
        print("\nDry run — nothing written. Re-run with --commit to load.")
        report_timing(time.perf_counter() - started, timing)
        return

    print(f"\nLoading into {DB_NAME}.tile_registry ...")
    try:
        updated = load(engine, frame, cluster_column, profiles=profiles,
                       stage=stage, vacuum=not args.no_vacuum,
                       rebuild_hpc_index=args.rebuild_hpc_index,
                       timing=timing)
    except BaseException as e:
        if args.record_run and args.record_run_db:
            record_run(args.record_run_db, args.record_run,
                       kb_load_error=f"{type(e).__name__}: {e}"[:2000])
        raise
    print(f"Updated {updated:,} rows.")
    if args.record_run and args.record_run_db:
        record_run(
            args.record_run_db, args.record_run,
            kb_load_done=True,
            kb_load_at=datetime.now(timezone.utc),
            kb_load_rows=int(updated),
            kb_load_reference=str(frame["hpc_reference"].iloc[0]),
            kb_load_error=None,
            **({"kb_load_kb_target": args.record_kb_target}
               if args.record_kb_target else {}),
        )
    if updated != report["matched"]:
        print(
            f"WARNING: updated {updated:,} but expected {report['matched']:,}. The "
            f"registry changed between the preview and the write.",
            file=sys.stderr,
        )
    elapsed = time.perf_counter() - started
    print(f"Loaded    : {updated:,} tiles in {elapsed:.1f}s "
          f"({updated / max(elapsed, 1e-9):,.0f} tiles/s)")
    report_timing(elapsed, timing)


if __name__ == "__main__":
    main()
