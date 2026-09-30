"""Loading cluster assignments into the Knowledge Bank.

This is the only script in the pipeline that writes to the shared KB, and its
failure mode is silence rather than a crash. If the CSV and tile_registry
disagree about slide naming, the UPDATE matches nothing and reports success. If
they agree for only some slides, it updates those and leaves the rest carrying
cluster IDs from an older reference — a registry in two states at once, which
nothing downstream can detect because every row still looks valid.

So the tests here are mostly about the refusals, and about the join key, since a
wrong `slide_tile` is exactly how the silent-no-op happens.

Runs against SQLite standing in for Postgres. That is enough because the loader
deliberately uses portable SQL — an expanding IN rather than PostgreSQL's
ANY(array) — and what is under test is the matching logic and the guards, not
the driver.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, text  # noqa: E402

import load_hpc_assignments as loader  # noqa: E402

SLIDE = "TCGA-55-7574-01Z-00-DX1"
REFERENCE = "hpc_reference_leiden_2p5_fold2"


def _make_kb(tmp_path: Path, registry_tiles, clusters=("0", "1", "2"), existing=None):
    """A KB with just the two tables the loader touches."""
    engine = create_engine(f"sqlite:///{tmp_path / 'kb.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE tile_registry (
                slide_tile TEXT, image_index INTEGER, hpc_id INTEGER,
                hpc_vote_margin REAL, hpc_neighbor_distance REAL,
                hpc_reference TEXT, hpc_assigned_at TIMESTAMP
            )"""))
        # hpc_id INTEGER, matching kb_live_schema_2026-08-26.txt. It said TEXT
        # here until 2026-09-10, following schema.sql — the stale dump CLAUDE.md
        # warns about — and because SQLite types values dynamically, every test
        # passed while the staged TEXT column could not be assigned into the
        # real integer column at all. See kb_stage.cluster_stage_type.
        conn.execute(text("CREATE TABLE hpc_dictionary (hpc_id INTEGER, malignant TEXT)"))
        for i, tile in enumerate(registry_tiles):
            conn.execute(
                text("INSERT INTO tile_registry (slide_tile, image_index, hpc_id, "
                     "hpc_reference) VALUES (:t, :i, :h, :r)"),
                {"t": tile, "i": i,
                 "h": (existing or {}).get(tile),
                 "r": REFERENCE if (existing or {}).get(tile) else None},
            )
        for cluster in clusters:
            conn.execute(text("INSERT INTO hpc_dictionary VALUES (:h, 'True')"),
                         {"h": cluster})
    return engine


def _make_csv(tmp_path: Path, n=20, slide=SLIDE, clusters=("0", "1", "2"), name="a.csv"):
    rows = [{
        "samples": "S1", "slides": slide, "tiles": f"{i}_{i}.jpeg",
        "leiden_2.5": clusters[i % len(clusters)],
        "vote_margin": 0.05 if i % 10 == 0 else 0.8,
        "neighbor_distance": 1.2,
        "hpc_reference": REFERENCE,
    } for i in range(n)]
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _table_names(engine):
    """Every table, to prove a refused load left no scratch table behind."""
    from sqlalchemy import inspect as sqlalchemy_inspect
    return sqlalchemy_inspect(engine).get_table_names()


def _registry_tiles(n=20, slide=SLIDE):
    """slide_tile as tile_coordinates stores it: "<slides>_<tiles>"."""
    return [f"{slide}_{i}_{i}.jpeg" for i in range(n)]


def test_join_key_matches_the_registrys_format(tmp_path):
    """The whole loader hinges on this. tile_coordinates stores
    '<slides>_<tiles>' — TCGA-55-7574-01Z-00-DX1_18_15.jpeg — and getting it
    wrong makes every UPDATE match nothing while reporting success."""
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    assert column == "leiden_2.5"
    assert frame["slide_tile"].iloc[0] == f"{SLIDE}_0_0.JPEG"
    # Upper-cased, because the server joins with UPPER() on both sides.
    assert frame["slide_tile"].str.isupper().all()


def test_load_populates_every_column_the_viewer_reads(tmp_path):
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 20 and report["unmatched"] == 0
    assert report["unknown_clusters"] == []
    assert report["low_margin"] == 2  # i % 10 == 0

    assert loader.load(engine, frame, column) == 20
    with engine.connect() as conn:
        got = pd.read_sql(text("SELECT * FROM tile_registry ORDER BY image_index"), conn)
    assert got["hpc_id"].notna().all()
    assert got["hpc_vote_margin"].notna().all()
    assert got["hpc_neighbor_distance"].notna().all()
    assert (got["hpc_reference"] == REFERENCE).all()
    assert got["hpc_assigned_at"].notna().all()
    # Row-for-row: tile i must carry the cluster the CSV gave tile i. Compared
    # as integers because that is what tile_registry.hpc_id is — the reason
    # this assertion used to read .astype(str) is the same stale schema.sql
    # that made the staged column TEXT.
    expected = pd.read_csv(csv)["leiden_2.5"].astype(int).tolist()
    assert got["hpc_id"].astype(int).tolist() == expected


def test_a_naming_mismatch_is_refused_not_reported_as_success(tmp_path):
    """The failure this script exists to prevent: the registry knows the slide
    by another name, so the UPDATE matches nothing at all."""
    csv = _make_csv(tmp_path, slide="TCGA-55-7574")          # registry has the full ID
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 0
    assert report["matched"] / report["rows"] < loader._MIN_MATCH_RATE
    assert report["unmatched_examples"], "must name the tiles that did not match"


def test_a_partial_match_is_refused(tmp_path):
    """More dangerous than a total mismatch, because a summary line still reads
    as plausible. Half the tiles updated leaves the registry in two states."""
    csv = _make_csv(tmp_path, n=20)
    engine = _make_kb(tmp_path, _registry_tiles(n=10))  # only half are registered
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 10
    assert report["matched"] / report["rows"] < loader._MIN_MATCH_RATE


def test_clusters_absent_from_the_dictionary_are_flagged(tmp_path):
    """A cluster with no hpc_dictionary row joins to NULL in the viewer: the
    tile shows a cluster with no pattern, malignancy or inflammation."""
    csv = _make_csv(tmp_path, clusters=("0", "1", "99"))
    engine = _make_kb(tmp_path, _registry_tiles(), clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["unknown_clusters"] == ["99"]


def test_overwriting_an_earlier_reference_is_reported(tmp_path):
    """Reassigning against a different reference is legitimate, but it changes
    the meaning of every ID in the registry, so it cannot happen silently."""
    tiles = _registry_tiles()
    engine = _make_kb(tmp_path, tiles, existing={t: "7" for t in tiles})
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)
    assert report["overwriting"] == 20


def test_inspect_changes_nothing(tmp_path):
    """--dry-run has to be trustworthy, or nobody will use it."""
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    with engine.connect() as conn:
        before = pd.read_sql(text("SELECT * FROM tile_registry"), conn).to_json()
    loader.inspect(engine, frame, column)
    with engine.connect() as conn:
        after = pd.read_sql(text("SELECT * FROM tile_registry"), conn).to_json()
    assert before == after


def test_loading_twice_is_idempotent(tmp_path):
    """Re-running the same CSV must converge, not accumulate — the obvious
    reaction to a job that looked like it half-finished is to run it again."""
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles())
    frame, column = loader.read_assignments(csv)

    loader.load(engine, frame, column)
    with engine.connect() as conn:
        first = pd.read_sql(
            text("SELECT slide_tile, hpc_id, hpc_vote_margin FROM tile_registry "
                 "ORDER BY image_index"), conn).to_json()
    loader.load(engine, frame, column)
    with engine.connect() as conn:
        second = pd.read_sql(
            text("SELECT slide_tile, hpc_id, hpc_vote_margin FROM tile_registry "
                 "ORDER BY image_index"), conn).to_json()
    assert first == second


def test_lookup_chunking_does_not_lose_tiles(tmp_path):
    """The chunked lookup must not drop rows at the seams.

    inspect() joins a staging table now, so this reaches the chunked path only
    by calling it directly — which is the point: the path still exists as the
    fallback for a role that cannot create the scratch table, and a fallback
    nothing exercises is a fallback nobody knows is broken.
    """
    original = loader._LOOKUP_CHUNK
    try:
        loader._LOOKUP_CHUNK = 3  # forces ragged chunks against 20 rows
        csv = _make_csv(tmp_path, n=20)
        engine = _make_kb(tmp_path, _registry_tiles(n=20))
        frame, column = loader.read_assignments(csv)
        with engine.connect() as conn:
            present = loader._lookup_present_chunked(
                conn, frame["slide_tile"].tolist())
        assert len(set(present["slide_tile"])) == 20
    finally:
        loader._LOOKUP_CHUNK = original


def test_the_staged_preview_and_the_chunked_one_agree(tmp_path):
    """Two ways of asking the same question. The staged join is the one that
    runs; the chunked lookup is what it replaced. They must return the same
    registry rows, or the speedup changed the report."""
    csv = _make_csv(tmp_path, n=20)
    engine = _make_kb(tmp_path, _registry_tiles(n=20),
                      existing={_registry_tiles(n=20)[0].upper(): "1"})
    frame, column = loader.read_assignments(csv)

    staged = loader.inspect(engine, frame, column)
    with engine.connect() as conn:
        chunked = loader._lookup_present_chunked(
            conn, frame["slide_tile"].tolist())

    assert staged["matched"] == len(set(chunked["slide_tile"]))
    assert staged["overwriting"] == int(chunked["hpc_id"].notna().sum())


# --- the staged write ----------------------------------------------------
# The write used to be one UPDATE per tile. It is one statement joined against
# a scratch table now, and the only thing that matters about that change is
# that the Knowledge Bank ends up holding exactly what it held before.

def _write_row_by_row(engine, frame, cluster_column, now):
    """The pre-staging writer, kept here as the thing to compare against.

    Deliberately a copy rather than the real function: what is under test is
    that the new statement reproduces the old behaviour, and a test that
    imported the old code would stop testing that the day the old code was
    deleted.
    """
    records = [
        {
            "slide_tile": slide_tile,
            "hpc_id": str(cluster).strip(),
            "margin": float(margin),
            "distance": None if pd.isna(distance) else float(distance),
            "reference": str(reference),
            "assigned_at": now,
        }
        for slide_tile, cluster, margin, distance, reference in zip(
            frame["slide_tile"], frame[cluster_column], frame["vote_margin"],
            frame["neighbor_distance"], frame["hpc_reference"])
    ]
    statement = text("""
        UPDATE tile_registry
        SET hpc_id = :hpc_id, hpc_vote_margin = :margin,
            hpc_neighbor_distance = :distance, hpc_reference = :reference,
            hpc_assigned_at = :assigned_at
        WHERE UPPER(slide_tile) = :slide_tile
    """)
    updated = 0
    with engine.begin() as conn:
        for start in range(0, len(records), 5000):
            result = conn.execute(statement, records[start:start + 5000])
            updated += result.rowcount if result.rowcount is not None else 0
    return updated


def _registry_rows(engine):
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT slide_tile, hpc_id, hpc_vote_margin, hpc_neighbor_distance, "
            "hpc_reference FROM tile_registry ORDER BY slide_tile")).fetchall()


def test_the_staged_write_matches_the_row_by_row_one_exactly(tmp_path):
    """The whole justification for the rewrite. Same CSV, same registry, two
    writers: every column of every row must come out identical, not merely the
    hpc_id, and the reported row count must match too."""
    csv = _make_csv(tmp_path, n=40)
    tiles = _registry_tiles(n=40)
    frame, column = loader.read_assignments(csv)

    for name in ("old", "new"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    old_engine = _make_kb(tmp_path / "old", tiles)
    new_engine = _make_kb(tmp_path / "new", tiles)

    old_count = _write_row_by_row(old_engine, frame, column,
                                  loader.datetime.now(loader.timezone.utc))
    new_count = loader.load(new_engine, frame, column, vacuum=False)

    assert old_count == new_count == 40
    assert _registry_rows(old_engine) == _registry_rows(new_engine)


def test_both_update_statements_produce_the_same_registry(tmp_path):
    """UPDATE ... FROM and the portable correlated form are two spellings of
    one write. Forced against the same fixture so the fallback cannot quietly
    diverge from the statement that actually runs in production."""
    csv = _make_csv(tmp_path, n=25)
    tiles = _registry_tiles(n=25)
    frame, column = loader.read_assignments(csv)

    results = {}
    for label, supports in (("update_from", True), ("correlated", False)):
        (tmp_path / label).mkdir(parents=True, exist_ok=True)
        engine = _make_kb(tmp_path / label, tiles)
        original = loader.supports_update_from
        try:
            loader.supports_update_from = lambda bind, _s=supports: _s
            assert loader.load(engine, frame, column, vacuum=False) == 25
        finally:
            loader.supports_update_from = original
        results[label] = _registry_rows(engine)

    assert results["update_from"] == results["correlated"]


def test_the_scratch_table_does_not_outlive_the_write(tmp_path):
    """A staging table left behind is a table nobody can date, on the one
    database this pipeline shares. Dropped on the way out of both paths."""
    csv = _make_csv(tmp_path, n=10)
    engine = _make_kb(tmp_path, _registry_tiles(n=10))
    frame, column = loader.read_assignments(csv)

    def _stage_tables():
        with engine.connect() as conn:
            return [r[0] for r in conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'hpl_stage_%'"))]

    loader.inspect(engine, frame, column)
    assert _stage_tables() == []
    loader.load(engine, frame, column, vacuum=False)
    assert _stage_tables() == []


def test_a_short_staging_table_refuses_rather_than_loading_a_subset(tmp_path):
    """The check that turns a staging bug into a refusal.

    A scratch table short by a slice would make the join update a subset of the
    cohort and report success — this pipeline's characteristic failure. Forced
    by having the copy step drop its last chunk.
    """
    import kb_stage

    csv = _make_csv(tmp_path, n=10)
    engine = _make_kb(tmp_path, _registry_tiles(n=10))
    frame, column = loader.read_assignments(csv)

    original = kb_stage._insert_chunk
    try:
        kb_stage._insert_chunk = lambda conn, table, columns, chunk: original(
            conn, table, columns, chunk.iloc[:-1])
        try:
            loader.load(engine, frame, column, vacuum=False)
        except SystemExit as e:
            assert "holds 9" in str(e) and "has 10" in str(e), str(e)
        else:
            raise AssertionError("a short staging table was accepted")
    finally:
        kb_stage._insert_chunk = original

    # And nothing was written.
    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT COUNT(*) FROM tile_registry WHERE hpc_id IS NOT NULL"
        )).scalar_one() == 0


def test_staging_slices_are_contiguous_and_cover_everything(tmp_path):
    """A split that overlapped or gapped would stage the wrong rows. Checked
    arithmetically rather than eyed, at sizes that do not divide evenly."""
    import kb_stage

    for total in (0, 1, 7, 10, 999, 18_485_499):
        for parts in (1, 3, 4, 16):
            bounds = kb_stage._slice_bounds(total, parts)
            assert sum(b - a for a, b in bounds) == total, (total, parts)
            assert all(a < b for a, b in bounds), (total, parts)
            for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
                assert prev_end == next_start, (total, parts)
            if bounds:
                assert bounds[0][0] == 0 and bounds[-1][1] == total


def test_parallel_staging_stages_every_row(tmp_path):
    """More than one worker must not lose or duplicate a slice. SQLite has no
    COPY so this exercises the serial fallback's arithmetic, and the row-count
    check above is what guards the parallel path itself."""
    import kb_stage

    csv = _make_csv(tmp_path, n=37)
    engine = _make_kb(tmp_path, _registry_tiles(n=37))
    frame, column = loader.read_assignments(csv)
    original = kb_stage.CHUNK_ROWS
    try:
        kb_stage.CHUNK_ROWS = 5  # ragged chunks against 37 rows
        assert loader.load(engine, frame, column, stage_workers=4,
                           vacuum=False) == 37
    finally:
        kb_stage.CHUNK_ROWS = original


def test_a_stale_scratch_table_name_carries_a_readable_timestamp(tmp_path):
    """sweep_stale dates a leftover table by the timestamp in its name. If the
    name and the parser disagree, a sweep either drops nothing or drops a table
    out from under a running load."""
    import kb_stage
    import time as _time

    name = kb_stage.stage_table_name("kb load")
    assert name.startswith("hpl_stage_kb_load_")
    stamp = int(name.rsplit("_", 2)[2])
    assert abs(stamp - _time.time()) < 60


# --- the refusals the staged write made necessary ------------------------

def test_two_rows_for_one_tile_are_refused(tmp_path):
    """The old writer applied duplicates in file order and let the last win.
    A join cannot: `UPDATE ... FROM` picks an arbitrary match. Refused, which
    is also right on the merits — a duplicated tile means the CSV's row count
    is not the cohort's, so its proportions are already wrong."""
    frame = pd.read_csv(_make_csv(tmp_path, n=6))
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    path = tmp_path / "dup.csv"
    frame.to_csv(path, index=False)

    try:
        loader.read_assignments(path)
    except SystemExit as e:
        assert "more than once" in str(e)
        assert "merge_assignment_shards.py" in str(e)
    else:
        raise AssertionError("a CSV with a duplicated tile was accepted")


def test_an_empty_vote_margin_is_refused(tmp_path):
    """A blank margin passed the non-numeric guard, which only catches values
    that were something before coercion. It has to be caught now because the
    two writers disagreed on it — NaN one way, NULL the other — and because a
    blank margin is a torn write, not a tile without confidence."""
    frame = pd.read_csv(_make_csv(tmp_path, n=6))
    frame.loc[3, "vote_margin"] = None
    path = tmp_path / "blank.csv"
    frame.to_csv(path, index=False)

    try:
        loader.read_assignments(path)
    except SystemExit as e:
        assert "empty vote_margin" in str(e)
    else:
        raise AssertionError("a CSV with a blank vote_margin was accepted")


def test_a_supplied_slide_tile_that_agrees_is_reported_as_agreeing(tmp_path):
    """Keeping slide_tile in the CSV is useful and supported. When it matches
    the rebuilt key, say so — otherwise the note reads like a warning about a
    file that is entirely fine."""
    frame = pd.read_csv(_make_csv(tmp_path, n=8))
    frame["slide_tile"] = (frame["slides"] + "_" + frame["tiles"]).str.upper()
    path = tmp_path / "agrees.csv"
    frame.to_csv(path, index=False)

    loaded, column = loader.read_assignments(path)
    assert column == "leiden_2.5"
    assert loaded.attrs["slide_tile_supplied"] is True
    assert loaded.attrs["slide_tile_disagreed"] == 0


def test_a_supplied_slide_tile_that_disagrees_is_counted_with_an_example(tmp_path):
    """The case worth catching. A slide_tile built from different columns than
    the CSV's own slides/tiles is invisible except as a match rate — so the
    disagreement is counted and one example shown, rather than the column
    being silently overwritten."""
    frame = pd.read_csv(_make_csv(tmp_path, n=8))
    frame["slide_tile"] = (frame["slides"] + "_" + frame["tiles"]).str.upper()
    frame.loc[3, "slide_tile"] = "SOMETHING-ELSE_9_9.JPEG"
    path = tmp_path / "disagrees.csv"
    frame.to_csv(path, index=False)

    loaded, _ = loader.read_assignments(path)
    assert loaded.attrs["slide_tile_disagreed"] == 1
    supplied, rebuilt = loaded.attrs["slide_tile_example"]
    assert supplied == "SOMETHING-ELSE_9_9.JPEG"
    assert rebuilt == loaded["slide_tile"].iloc[3]


def test_a_stale_supplied_slide_tile_still_joins(tmp_path):
    """The ordinary pre-normalisation case: every value differs, because the
    column holds the short form. It must be reported and then ignored — the
    rebuild is taken from the tile names the normalisation just corrected, so
    the load still reaches a full match rate."""
    frame = pd.read_csv(_make_csv(tmp_path, n=12))
    frame["tiles"] = frame["tiles"].str.replace(".jpeg", "", regex=False)
    frame["slide_tile"] = (frame["slides"] + "_" + frame["tiles"]).str.upper()
    path = tmp_path / "stale.csv"
    frame.to_csv(path, index=False)

    loaded, column = loader.read_assignments(path)
    assert loaded.attrs["slide_tile_disagreed"] == 12
    engine = _make_kb(tmp_path, _registry_tiles(n=12))
    assert loader.inspect(engine, loaded, column)["matched"] == 12


def test_duplicates_are_caught_on_the_rebuilt_key_not_the_supplied_one(tmp_path):
    """Why the rebuild has to happen before the duplicate check.

    The join key is upper-cased, so two rows whose slide names differ only in
    case are distinct in the CSV's own slide_tile column and become the same
    key once rebuilt. The collision exists only after the rebuild, so
    deduplicating on the supplied value would miss it and let an arbitrary one
    of the two win — the exact ambiguity the duplicate guard exists to refuse.

    Not hypothetical: this cohort's dataset_id is RADIOGENOMICS and its tile
    folder is Radiogenomics, and case has already cost a day here.
    """
    frame = pd.read_csv(_make_csv(tmp_path, n=6))
    extra = frame.iloc[[5]].copy()
    extra["slides"] = extra["slides"].str.lower()
    frame = pd.concat([frame, extra], ignore_index=True)
    frame["slide_tile"] = frame["slides"] + "_" + frame["tiles"]
    # Distinct in the file's own column — this is what a dedup on the supplied
    # value would see, and why it would not fire.
    assert frame["slide_tile"].is_unique
    path = tmp_path / "collide.csv"
    frame.to_csv(path, index=False)

    try:
        loader.read_assignments(path)
    except SystemExit as e:
        assert "more than once" in str(e), str(e)
    else:
        raise AssertionError("a post-normalisation key collision was accepted")


def test_a_clean_csv_is_not_caught_by_either_new_refusal(tmp_path):
    """The guards above must fail on bad input, not on ordinary input — the
    pair of them sits in front of every load this pipeline does."""
    frame, column = loader.read_assignments(_make_csv(tmp_path, n=50))
    assert len(frame) == 50
    assert frame["slide_tile"].is_unique


def test_a_csv_carrying_slide_tile_still_identifies_its_cluster_column(tmp_path):
    """A CSV round-tripped through a frame this module touched comes back with a
    slide_tile column. The cluster column is found by elimination, so that
    extra column made two candidates and the load refused — on a file whose
    cluster IDs were perfectly fine."""
    frame = pd.read_csv(_make_csv(tmp_path, n=8))
    frame["slide_tile"] = (frame["slides"].str.upper() + "_"
                           + frame["tiles"].str.upper())
    path = tmp_path / "round_tripped.csv"
    frame.to_csv(path, index=False)

    loaded, column = loader.read_assignments(path)
    assert column == "leiden_2.5"
    assert loaded["slide_tile"].iloc[0] == f"{SLIDE.upper()}_0_0.JPEG"


def test_a_stale_slide_tile_column_is_rebuilt_not_trusted(tmp_path):
    """The dangerous version of the same file: a slide_tile written before the
    tile names were normalised holds the short form and joins nothing. Taking
    the column at face value would turn a repairable CSV into a 0% match rate
    reported as a naming mismatch."""
    frame = pd.read_csv(_make_csv(tmp_path, n=8))
    frame["slide_tile"] = frame["slides"].str.upper() + "_0_0"  # short and wrong
    path = tmp_path / "stale_key.csv"
    frame.to_csv(path, index=False)

    loaded, _ = loader.read_assignments(path)
    assert loaded["slide_tile"].tolist() == [
        f"{SLIDE.upper()}_{i}_{i}.JPEG" for i in range(8)]

    engine = _make_kb(tmp_path, _registry_tiles(n=8))
    assert loader.inspect(engine, loaded, "leiden_2.5")["matched"] == 8


def test_a_csv_with_no_cluster_column_does_not_load_slide_tile_as_one(tmp_path):
    """The loud failure had a silent twin. With slide_tile outside the known
    set and no cluster column present, it was the only candidate left — so tile
    names would have been written into tile_registry.hpc_id, every row
    well-formed and every cluster wrong."""
    frame = pd.read_csv(_make_csv(tmp_path, n=8)).drop(columns=["leiden_2.5"])
    frame["slide_tile"] = frame["slides"].str.upper() + "_0_0.JPEG"
    path = tmp_path / "no_cluster.csv"
    frame.to_csv(path, index=False)

    try:
        loader.read_assignments(path)
    except SystemExit as e:
        assert "Could not identify the cluster column" in str(e)
        assert "none left after the known ones" in str(e), str(e)
    else:
        raise AssertionError("slide_tile was accepted as the cluster column")


def test_both_csv_readers_produce_the_same_assignments(tmp_path):
    """read_assignments reads on several threads where pyarrow is installed and
    falls back where it is not. The two readers must not disagree about values,
    since which one runs is a property of the machine."""
    path = _make_csv(tmp_path, n=30)
    threaded = loader._read_csv(path)
    default = pd.read_csv(path)
    assert list(threaded.columns) == list(default.columns)
    for column in default.columns:
        assert (threaded[column].astype(str).tolist()
                == default[column].astype(str).tolist()), column


def test_both_csv_readers_keep_identifiers_as_text(tmp_path):
    """And they must agree on the text, exactly, for the ids inference breaks.

    The threaded reader is pyarrow's own, told the column types up front:
    `pd.read_csv(engine="pyarrow", dtype=str)` infers first and casts after,
    so an all-digit '007' column comes back as '7' from it while the default
    reader returns '007' — the two readers disagreeing on exactly the value
    the join key is built from, depending on what is installed."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("  (pyarrow not installed; the threaded reader is not exercised)")
        return
    rows = (_rows("007", 3) + _rows("1001", 3, start=3, sample="NA")
            + _rows("None", 3, start=6, sample=""))
    path = _text_csv(tmp_path, rows)

    threaded = loader._read_csv_threaded(path)
    default = loader._read_csv_default(path)

    assert threaded.to_dict("list") == default.to_dict("list"), (
        threaded.to_dict("list"), default.to_dict("list"))
    assert threaded["slides"].tolist() == ["007"] * 3 + ["1001"] * 3 + ["None"] * 3
    assert threaded["samples"].tolist() == ["S1"] * 3 + ["NA"] * 3 + [""] * 3


# --- the per-slide aggregates --------------------------------------------
# hpl_profile_proportion and hpl_profile_summary are what the chatbot and the
# HPC panels actually read — not tile_registry. Loading tiles without refreshing
# these leaves the UI showing new clusters per tile and old proportions per
# slide, with nothing to indicate the two came from different runs.

def _add_profile_tables(engine, rows=()):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL)"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, cancer_type TEXT,
                total_tiles INTEGER, dominant_hpc TEXT)"""))
        for slide, hpc, prop in rows:
            conn.execute(
                text("INSERT INTO hpl_profile_proportion (samples, slides, hpc_id, "
                     "proportion) VALUES ('S1', :s, :h, :p)"),
                {"s": slide, "h": hpc, "p": prop},
            )


def _add_profile_tables_with_fk(engine, rows=()):
    """The aggregate tables *with* the foreign key Postgres actually has.

    schema.sql:486-487 gives hpl_profile_proportion
    FOREIGN KEY (samples, slides) REFERENCES hpl_profile_summary ON DELETE CASCADE.
    _add_profile_tables above omits it, which is why the ordering bug in
    replace_profiles survived every existing test: without the constraint, both
    the child-before-parent insert and the cascade-away-what-was-just-written
    case are invisible.

    SQLite enforces foreign keys only when asked, per connection.
    """
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, cancer_type TEXT,
                total_tiles INTEGER, dominant_hpc TEXT,
                UNIQUE (samples, slides))"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL,
                FOREIGN KEY (samples, slides)
                    REFERENCES hpl_profile_summary (samples, slides)
                    ON DELETE CASCADE)"""))
        for samples, slides, hpc, prop in rows:
            conn.execute(
                text("INSERT INTO hpl_profile_summary (samples, slides, total_tiles, "
                     "dominant_hpc) VALUES (:sa, :sl, 1, :h)"),
                {"sa": samples, "sl": slides, "h": hpc},
            )
            conn.execute(
                text("INSERT INTO hpl_profile_proportion (samples, slides, hpc_id, "
                     "proportion) VALUES (:sa, :sl, :h, :p)"),
                {"sa": samples, "sl": slides, "h": hpc, "p": prop},
            )


def test_first_load_of_a_new_cohort_does_not_violate_the_foreign_key(tmp_path):
    """The case that fires on every slide of a cohort's first load.

    hpl_profile_proportion references hpl_profile_summary, so inserting a
    proportion row for a slide that has no summary row yet is rejected — and
    because the loader writes everything in one transaction, that rollback would
    take the tile_registry update with it. Loading Radiogenomics for the first
    time is exactly this case for all ten slides.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'fk.sqlite'}")
    _add_profile_tables_with_fk(engine)  # empty: nothing to be a parent yet

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)


def test_reload_does_not_cascade_away_the_rows_it_just_wrote(tmp_path):
    """The second FK failure, and the quieter one.

    Deleting a slide's summary row cascades to its proportions. Insert the
    proportions before that delete and they are silently removed again, leaving
    a summary row with no proportions — a slide whose HPC panel is simply empty,
    with no error anywhere.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'fk2.sqlite'}")
    # Pre-existing aggregates for the same sample/slide, as a reload would meet.
    _add_profile_tables_with_fk(
        engine, rows=[(summary["samples"].iloc[0], summary["slides"].iloc[0], "9", 1.0)]
    )

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)
        # The stale cluster 9 row is gone, not left beside the new ones.
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion WHERE hpc_id = '9'")
        ).scalar() == 0


def test_reload_matches_rows_stored_in_a_different_case(tmp_path):
    """migrate_indexes.sql normalised the live columns to UPPER(TRIM(...)).
    Matching on TRIM alone deletes nothing, and the insert still runs — so the
    slide silently ends up with two full sets of aggregates.
    """
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, None)

    engine = create_engine(f"sqlite:///{tmp_path / 'case.sqlite'}")
    _add_profile_tables_with_fk(
        engine,
        rows=[(summary["samples"].iloc[0].upper(),
               summary["slides"].iloc[0].upper(), "9", 1.0)],
    )

    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)

    with engine.connect() as conn:
        # One summary row, not two. The upper-cased one must have been replaced.
        assert conn.execute(text("SELECT COUNT(*) FROM hpl_profile_summary")).scalar() == 1
        assert conn.execute(
            text("SELECT COUNT(*) FROM hpl_profile_proportion")).scalar() == len(proportions)


def test_proportions_sum_to_one_per_slide(tmp_path):
    csv = _make_csv(tmp_path, n=30, clusters=("0", "1", "2"))
    frame, column = loader.read_assignments(csv)
    proportions, summary = loader.compute_profiles(frame, column, "LUAD")

    totals = proportions.groupby(["samples", "slides"])["proportion"].sum()
    assert all(abs(t - 1.0) < 1e-9 for t in totals), totals.to_dict()
    # 30 tiles cycling through three clusters -> a third each.
    assert sorted(proportions["proportion"].round(6)) == [round(1 / 3, 6)] * 3


def test_summary_counts_and_dominant_cluster(tmp_path):
    # 10 tiles: cluster "0" six times, "1" four times -> dominant is "0".
    rows = [{
        "samples": "S1", "slides": SLIDE, "tiles": f"{i}_{i}.jpeg",
        "leiden_2.5": "0" if i < 6 else "1",
        "vote_margin": 0.9, "neighbor_distance": 1.0, "hpc_reference": REFERENCE,
    } for i in range(10)]
    csv = tmp_path / "dom.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    frame, column = loader.read_assignments(csv)
    _, summary = loader.compute_profiles(frame, column, "LUAD")
    assert len(summary) == 1
    assert summary["total_tiles"].iloc[0] == 10
    assert summary["dominant_hpc"].iloc[0] == "0"
    assert summary["cancer_type"].iloc[0] == "LUAD"


def test_cancer_type_is_not_invented(tmp_path):
    """The original script hardcoded 'LUAD'. Guessing a cancer type into a KB
    that may hold several would be a quiet data error."""
    csv = _make_csv(tmp_path)
    frame, column = loader.read_assignments(csv)
    _, summary = loader.compute_profiles(frame, column, None)
    assert "cancer_type" not in summary.columns


def test_profiles_replace_only_the_loaded_slides(tmp_path):
    """Loading ten slides must not delete the proportions of every other slide."""
    csv = _make_csv(tmp_path, n=9, clusters=("0", "1", "2"))
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    _add_profile_tables(engine, rows=[
        (SLIDE, "7", 1.0),              # stale row for the slide being loaded
        ("SOME-OTHER-SLIDE", "3", 1.0),  # must survive
    ])

    frame, column = loader.read_assignments(csv)
    loader.load(engine, frame, column,
                profiles=loader.compute_profiles(frame, column, "LUAD"))

    with engine.connect() as conn:
        got = pd.read_sql(text("SELECT * FROM hpl_profile_proportion"), conn)
    assert "SOME-OTHER-SLIDE" in set(got["slides"]), "unrelated slide was deleted"
    mine = got[got["slides"] == SLIDE]
    # The stale cluster-7 row is gone, replaced by the three real clusters.
    assert set(mine["hpc_id"]) == {"0", "1", "2"}, set(mine["hpc_id"])
    assert abs(mine["proportion"].sum() - 1.0) < 1e-9


def test_ids_come_from_the_sequence_not_a_range(tmp_path):
    """The notebooks assigned id = range(1, n+1), which is right exactly once and
    collides with every existing row afterwards."""
    csv = _make_csv(tmp_path, n=9)
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    _add_profile_tables(engine, rows=[("OTHER", "3", 1.0)])

    frame, column = loader.read_assignments(csv)
    loader.load(engine, frame, column,
                profiles=loader.compute_profiles(frame, column, "LUAD"))
    with engine.connect() as conn:
        ids = pd.read_sql(text("SELECT id FROM hpl_profile_proportion"), conn)["id"]
    assert ids.is_unique, "ids collided with the pre-existing row"


def test_missing_aggregate_column_does_not_abort_the_load(tmp_path):
    """These tables were filled by hand from notebooks, so a column may not be
    there. Inserting a name that does not exist would fail the whole
    transaction, taking the tile_registry update with it."""
    csv = _make_csv(tmp_path, n=9)
    engine = _make_kb(tmp_path, _registry_tiles(n=9))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL)"""))
        # No cancer_type column here.
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, total_tiles INTEGER, dominant_hpc TEXT)"""))

    frame, column = loader.read_assignments(csv)
    assert loader.load(engine, frame, column,
                       profiles=loader.compute_profiles(frame, column, "LUAD")) == 9
    with engine.connect() as conn:
        assert len(pd.read_sql(text("SELECT * FROM hpl_profile_summary"), conn)) == 1


# --- slide_hpc_membership -------------------------------------------------
#
# The reader is not obvious and that is the point: app/hpc_chat_handlers_v23.py
# :334 enumerates every table in the database, keeps any with an hpc_id or
# dominant_hpc column, skipping only hpc_dictionary and h_latent_vectors, and
# renders matching rows straight to the user. So this table is answered out of
# the chatbot without ever being named in a query — a grep for it finds nothing,
# which is exactly how it came to hold 19,493 rows for cohorts nobody was asking
# about while looking unreferenced.


def _add_membership_table(engine, rows=()):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE slide_hpc_membership (
                slide_id TEXT NOT NULL, hpc_id INTEGER NOT NULL,
                PRIMARY KEY (slide_id, hpc_id))"""))
        for slide, hpc in rows:
            conn.execute(
                text("INSERT INTO slide_hpc_membership (slide_id, hpc_id) "
                     "VALUES (:s, :h)"), {"s": slide, "h": hpc})


def test_membership_is_refreshed_with_the_aggregates(tmp_path):
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine)
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": 3, "proportion": 0.6},
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": 7, "proportion": 0.4},
    ])
    with engine.begin() as conn:
        n = loader.replace_slide_membership(conn, proportions)
    assert n == 2
    with engine.connect() as conn:
        got = set(conn.execute(text(
            "SELECT slide_id, hpc_id FROM slide_hpc_membership")).fetchall())
    assert got == {("SLIDE-A", 3), ("SLIDE-A", 7)}


def test_a_cluster_that_no_longer_appears_loses_its_row(tmp_path):
    """Delete-then-insert, not upsert: a slide's cluster set changes between
    references, and a cluster left behind would be reported by the chatbot as
    present on a slide whose proportions have no row for it."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine, rows=[("SLIDE-A", 3), ("SLIDE-A", 99)])
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": 3, "proportion": 1.0},
    ])
    with engine.begin() as conn:
        loader.replace_slide_membership(conn, proportions)
    with engine.connect() as conn:
        got = set(conn.execute(text(
            "SELECT slide_id, hpc_id FROM slide_hpc_membership")).fetchall())
    assert got == {("SLIDE-A", 3)}, "cluster 99 survived a load that dropped it"


def test_another_slides_membership_is_untouched(tmp_path):
    """Scoped by slide, like the aggregates. Loading ten slides must not empty
    the membership of every other slide in the Knowledge Bank."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine, rows=[("OTHER-SLIDE", 42)])
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": 3, "proportion": 1.0},
    ])
    with engine.begin() as conn:
        loader.replace_slide_membership(conn, proportions)
    with engine.connect() as conn:
        got = set(conn.execute(text(
            "SELECT slide_id, hpc_id FROM slide_hpc_membership")).fetchall())
    assert ("OTHER-SLIDE", 42) in got


def test_slide_id_is_upper_cased_to_match_the_normalised_columns(tmp_path):
    """migrate_indexes.sql normalised the live identity columns to
    UPPER(TRIM(...)). A row written in the CSV's own casing is a row the
    DELETE will not find on the next load, so the slide accumulates two
    memberships that both look plausible."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine)
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "  slide-a  ", "hpc_id": 3, "proportion": 1.0},
    ])
    with engine.begin() as conn:
        loader.replace_slide_membership(conn, proportions)
    with engine.connect() as conn:
        got = conn.execute(text("SELECT slide_id FROM slide_hpc_membership")).scalar()
    assert got == "SLIDE-A"


def test_a_missing_membership_table_does_not_fail_the_load(tmp_path):
    """It has no CREATE TABLE in git older than migrate_kb_base_tables.sql, so a
    database that predates that migration has no such table — and losing the
    whole tile_registry update over an optional aggregate would be far worse
    than not writing it."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    # deliberately no _add_membership_table
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": 3, "proportion": 1.0},
    ])
    with engine.begin() as conn:
        assert loader.replace_slide_membership(conn, proportions) is None


def test_string_cluster_ids_from_the_csv_are_coerced(tmp_path):
    """compute_profiles carries hpc_id as a string, and a float-typed CSV column
    yields "3.0" — int("3.0") raises, and this runs inside the transaction that
    carries the tile_registry update."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine)
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": "3.0", "proportion": 0.5},
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": "7", "proportion": 0.5},
    ])
    with engine.begin() as conn:
        assert loader.replace_slide_membership(conn, proportions) == 2
    with engine.connect() as conn:
        got = set(conn.execute(text(
            "SELECT slide_id, hpc_id FROM slide_hpc_membership")).fetchall())
    assert got == {("SLIDE-A", 3), ("SLIDE-A", 7)}


def test_a_non_numeric_cluster_id_skips_the_whole_table(tmp_path):
    """All or nothing. A membership missing whichever clusters are not numeric
    is a table that looks complete and under-reports — worse than one that was
    not refreshed and said so. And it must not take the load down with it."""
    engine = _make_kb(tmp_path, [])
    _add_profile_tables(engine)
    _add_membership_table(engine, rows=[("SLIDE-A", 1)])
    proportions = pd.DataFrame([
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": "3", "proportion": 0.5},
        {"samples": "S1", "slides": "SLIDE-A", "hpc_id": "stroma", "proportion": 0.5},
    ])
    with engine.begin() as conn:
        assert loader.replace_slide_membership(conn, proportions) is None
    with engine.connect() as conn:
        got = set(conn.execute(text(
            "SELECT slide_id, hpc_id FROM slide_hpc_membership")).fetchall())
    assert got == {("SLIDE-A", 1)}, "the pre-existing rows must be left alone"


# --- dataset_id on the aggregates ----------------------------------------
#
# Found by running the load against a real PostgreSQL for the first time on
# 2026-08-26. dataset_id is NOT NULL on both aggregate tables in the live
# database and compute_profiles() cannot know it — the assignment CSV carries no
# cohort — so the INSERT raised NotNullViolation, and because load() is one
# transaction the rollback took the tile_registry update with it.
#
# It had never surfaced for two reasons worth keeping: no cohort had ever got
# past the 95% match gate to reach that line, and every fixture above declares
# dataset_id nullable, which is exactly the way a test fixture can be more
# forgiving than production and hide a certainty.


def _add_profile_tables_with_dataset_id(engine):
    """The aggregate tables as the live database actually has them: dataset_id
    present and NOT NULL."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, cancer_type TEXT,
                total_tiles INTEGER, dominant_hpc TEXT,
                dataset_id TEXT NOT NULL)"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                samples TEXT, slides TEXT, hpc_id TEXT, proportion REAL,
                dataset_id TEXT NOT NULL)"""))


def test_aggregates_take_their_dataset_id_from_the_registry(tmp_path):
    engine = _make_kb(tmp_path, ["SLIDE-A_1_1.JPEG"])
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN dataset_id TEXT"))
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN slides TEXT"))
        conn.execute(text("UPDATE tile_registry SET slides='SLIDE-A', "
                          "dataset_id='RADIOGENOMICS'"))
    _add_profile_tables_with_dataset_id(engine)

    summary = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                             "cancer_type": "LUAD", "total_tiles": 1,
                             "dominant_hpc": "3"}])
    proportions = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                                 "hpc_id": "3", "proportion": 1.0}])
    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)
    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT dataset_id FROM hpl_profile_summary")).scalar() == "RADIOGENOMICS"
        assert conn.execute(text(
            "SELECT dataset_id FROM hpl_profile_proportion")).scalar() == "RADIOGENOMICS"


def test_an_unregistered_slide_is_refused_by_name(tmp_path):
    """The companion. Without a dataset_id the aggregates cannot be scoped, and
    the message has to name the missing step rather than let Postgres raise a
    not-null violation from inside a rollback."""
    engine = _make_kb(tmp_path, ["SLIDE-A_1_1.JPEG"])
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN dataset_id TEXT"))
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN slides TEXT"))
        conn.execute(text("UPDATE tile_registry SET slides='SLIDE-A'"))  # no dataset_id
    _add_profile_tables_with_dataset_id(engine)

    summary = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                             "cancer_type": None, "total_tiles": 1,
                             "dominant_hpc": "3"}])
    proportions = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                                 "hpc_id": "3", "proportion": 1.0}])
    try:
        with engine.begin() as conn:
            loader.replace_profiles(conn, proportions, summary)
    except SystemExit as e:
        assert "register_dataset.py" in str(e), str(e)
    else:
        raise AssertionError("a slide with no dataset_id must be refused by name")


def test_another_cohorts_aggregates_are_never_deleted(tmp_path):
    """The deletes in replace_profiles are scoped by slide NAME, not by cohort.
    A slide name is not unique across cohorts, so without this guard loading a
    new dataset would silently remove the rows an older one owns."""
    engine = _make_kb(tmp_path, ["SLIDE-A_1_1.JPEG"])
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN dataset_id TEXT"))
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN slides TEXT"))
        conn.execute(text("UPDATE tile_registry SET slides='SLIDE-A', dataset_id='RADIOGENOMICS'"))
    _add_profile_tables_with_dataset_id(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO hpl_profile_summary (samples,slides,dataset_id,total_tiles) "
                          "VALUES ('OLD','SLIDE-A','TCGA_LUAD_5X',999)"))

    summary = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A", "cancer_type": None,
                             "total_tiles": 1, "dominant_hpc": "3"}])
    proportions = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                                 "hpc_id": "3", "proportion": 1.0}])
    try:
        with engine.begin() as conn:
            loader.replace_profiles(conn, proportions, summary)
    except SystemExit as e:
        assert "different cohort" in str(e), str(e)
    else:
        raise AssertionError("loading over another cohort's slide must be refused")

    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT total_tiles FROM hpl_profile_summary WHERE dataset_id='TCGA_LUAD_5X'"
        )).scalar() == 999, "the other cohort's row was modified"


def test_the_cohort_guard_does_not_block_a_reload_of_the_same_cohort(tmp_path):
    """The companion. Re-loading a cohort over its own rows is the normal case
    and must still work, or the guard is just a lockout."""
    engine = _make_kb(tmp_path, ["SLIDE-A_1_1.JPEG"])
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN dataset_id TEXT"))
        conn.execute(text("ALTER TABLE tile_registry ADD COLUMN slides TEXT"))
        conn.execute(text("UPDATE tile_registry SET slides='SLIDE-A', dataset_id='RADIOGENOMICS'"))
    _add_profile_tables_with_dataset_id(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO hpl_profile_summary (samples,slides,dataset_id,total_tiles) "
                          "VALUES ('S1','SLIDE-A','RADIOGENOMICS',1)"))

    summary = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A", "cancer_type": None,
                             "total_tiles": 7, "dominant_hpc": "3"}])
    proportions = pd.DataFrame([{"samples": "S1", "slides": "SLIDE-A",
                                 "hpc_id": "3", "proportion": 1.0}])
    with engine.begin() as conn:
        loader.replace_profiles(conn, proportions, summary)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT total_tiles FROM hpl_profile_summary")).scalar() == 7
        assert conn.execute(text("SELECT count(*) FROM hpl_profile_summary")).scalar() == 1


# --- the threshold decision, priced in tiles -----------------------------
#
# Choosing a min_margin was previously a number typed against a distribution
# nobody had looked at. The trade-off table answers the question actually being
# decided — how many tiles a confidence floor costs and what it buys — from the
# cohort's own margins, so the person running the load can make the call.


def test_the_tradeoff_prices_every_cut_in_tiles(tmp_path):
    import numpy as np

    frame = pd.DataFrame({"vote_margin": np.concatenate([
        np.full(700, 1.0),          # unanimous
        np.full(200, 0.05),         # near-ties
        np.full(100, 0.40),
    ])})

    rows = loader.margin_tradeoff(frame)
    by_cut = {r["min_margin"]: r for r in rows}

    assert by_cut[0.0]["tiles_kept"] == 1000
    assert by_cut[0.0]["share_kept"] == 1.0
    # The 200 near-ties are exactly what a 0.10 floor removes.
    assert by_cut[0.10]["tiles_kept"] == 800
    assert by_cut[0.50]["tiles_kept"] == 700


def test_a_higher_cut_never_lowers_the_expected_accuracy(tmp_path):
    """The whole point of the table: the columns move in opposite directions, and
    the reader is choosing where to stop."""
    import numpy as np

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"vote_margin": rng.uniform(0, 1, 5000)})

    rows = loader.margin_tradeoff(frame)
    kept = [r["tiles_kept"] for r in rows]
    accuracy = [r["expected_accuracy"] for r in rows]

    assert kept == sorted(kept, reverse=True), "tiles kept must fall with the cut"
    assert accuracy == sorted(accuracy), "expected accuracy must rise with the cut"
    assert 0.6 < accuracy[0] < 1.0


def test_the_tradeoff_reaches_the_preview_report(tmp_path):
    """It has to be on the report, or neither the UI nor the CLI can show it."""
    csv = _make_csv(tmp_path, n=20)
    engine = _make_kb(tmp_path, _registry_tiles(20))
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)

    assert report["margin_tradeoff"], "no trade-off table on the report"
    assert {"min_margin", "tiles_kept", "share_kept", "expected_accuracy"} <= \
        set(report["margin_tradeoff"][0])


# --- a confidence column that is not a number ----------------------------


def test_a_non_numeric_margin_is_refused_with_the_offending_value(tmp_path):
    """One malformed line in a 7.8M-row CSV — a header repeated by a
    concatenation, or a torn write — turns the column to object dtype, and every
    comparison against it afterwards raises deep in pandas. Refused with the
    value, because a file with a stray row is a file whose row count nobody
    should trust."""
    csv = _make_csv(tmp_path, n=20)
    # Written as text, the way the real file got it: a line from a
    # concatenation, not a value assigned into a float column.
    lines = csv.read_text().splitlines()
    lines.insert(8, lines[0])                        # the header, repeated
    csv.write_text("\n".join(lines) + "\n")

    try:
        loader.read_assignments(csv)
    except SystemExit as e:
        assert "not a number" in str(e), str(e)
        assert "vote_margin" in str(e)
        assert "merge_assignment_shards.py" in str(e)
    else:
        raise AssertionError("a non-numeric vote_margin was accepted")


def test_a_clean_csv_still_loads(tmp_path):
    """The guard has to be able to pass, or it is just a broken loader."""
    csv = _make_csv(tmp_path, n=20)

    frame, _column = loader.read_assignments(csv)

    assert len(frame) == 20
    assert frame["vote_margin"].dtype.kind == "f"


# --- standalone runner ---------------------------------------------------

# --- the type the cluster IDs land in --------------------------------------

def test_the_preview_reports_cluster_ids_the_column_cannot_hold(tmp_path):
    """The preview stages only the join key, so it has to check this separately.

    Job 1243334 is why: the dry run was clean, the operator committed, 18.5M
    rows were COPYed, and the UPDATE then refused text for an integer column.
    A number on the preview is the difference between finding that out now and
    finding it out at the end.
    """
    csv = _make_csv(tmp_path, clusters=("0", "unknown", "2"))
    engine = _make_kb(tmp_path, _registry_tiles(20), clusters=("0", "2"))
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)

    assert report["cluster_column_type"] == "BIGINT", report["cluster_column_type"]
    assert report["unwritable_cluster_ids"] == 7, report["unwritable_cluster_ids"]
    assert report["unwritable_cluster_examples"] == ["unknown"]


def test_a_clean_cohort_reports_nothing_unwritable(tmp_path):
    """The check has to be silent when there is nothing wrong with the CSV,
    or it is noise on every load."""
    csv = _make_csv(tmp_path)
    engine = _make_kb(tmp_path, _registry_tiles(20))
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)

    assert report["unwritable_cluster_ids"] == 0
    assert report["unwritable_cluster_examples"] == []
    # And it still loads, into the integer column, row for row.
    assert loader.load(engine, frame, column) == 20


def test_the_load_refuses_before_staging_a_cohort_it_cannot_write(tmp_path):
    """Not a partial load and not a coerced one: nothing staged, nothing
    written, and the KB exactly as it was."""
    csv = _make_csv(tmp_path, clusters=("0", "unknown", "2"))
    engine = _make_kb(tmp_path, _registry_tiles(20))
    frame, column = loader.read_assignments(csv)

    try:
        loader.load(engine, frame, column)
    except ValueError as e:
        assert "whole numbers" in str(e), str(e)
    else:
        raise AssertionError("loaded a cohort whose cluster IDs cannot be written")

    with engine.connect() as conn:
        written = conn.execute(
            text("SELECT COUNT(*) FROM tile_registry WHERE hpc_id IS NOT NULL")
        ).scalar()
    assert written == 0, "the KB was touched by a load that refused"

    leftovers = _table_names(engine)
    assert not [t for t in leftovers if t.startswith("hpl_stage_")], leftovers


def test_a_float_cluster_column_is_not_reported_as_unknown_clusters(tmp_path):
    """A single missing value makes pandas read the column as float64.

    Then astype(str) renders "45.0" where hpc_dictionary's integer column
    renders "45", every ID in the cohort looks unknown, and the load refuses
    with a message about missing dictionary rows — believable, and about the
    wrong thing entirely. The comparison has to use the rendering the write
    will use.
    """
    import numpy as np

    rows = pd.read_csv(_make_csv(tmp_path, n=20))
    rows["leiden_2.5"] = rows["leiden_2.5"].astype(float)
    rows.loc[19, "leiden_2.5"] = np.nan          # what makes the dtype float
    rows = rows.iloc[:19]                        # and drop the unusable row
    csv = tmp_path / "float_clusters.csv"
    rows.to_csv(csv, index=False)

    engine = _make_kb(tmp_path, _registry_tiles(20))
    frame, column = loader.read_assignments(csv)

    report = loader.inspect(engine, frame, column)

    assert report["unknown_clusters"] == [], report["unknown_clusters"]
    assert all("." not in str(k) for k in report["distribution"]), \
        report["distribution"]
    # And it is writable, because 45.0 is a whole number.
    assert report["unwritable_cluster_ids"] == 0


# --- identifiers are read as text, not guessed at ---------------------------
# pandas' type inference rewrites identifiers rather than reading them: an
# all-digit slide id loses its zeros ('007' -> 7), one blank or 'NA' elsewhere
# in the column turns every number in it into a float ('1001' -> 1001.0), and
# 'NA' / 'None' / 'null' become NaN. The key built from those is
# '7_0_0.JPEG' or '1001.0_0_0.JPEG', which joins nothing — or joins a
# different slide that really is called '7'.

_HEADER = ("samples", "slides", "tiles", "leiden_2.5", "vote_margin",
           "neighbor_distance", "hpc_reference")


def _text_csv(tmp_path: Path, rows, name="ids.csv"):
    """rows of (samples, slides, tiles, cluster, vote_margin), written as the
    literal text given — not through a DataFrame, whose own dtypes would decide
    what the file says before the reader under test ever sees it."""
    import csv as csv_module
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(_HEADER)
        for sample, slide, tile, cluster, margin in rows:
            writer.writerow([sample, slide, tile, cluster, margin, "1.2", REFERENCE])
    return path


def _rows(slide, n, sample="S1", start=0, clusters=("0", "1", "2")):
    return [(sample, slide, f"{i}_{i}.jpeg", clusters[i % len(clusters)], "0.8")
            for i in range(start, start + n)]


def test_a_zero_padded_slide_id_round_trips(tmp_path):
    """'007' is a slide id, not the number seven."""
    csv = _text_csv(tmp_path, _rows("007", 20))
    engine = _make_kb(tmp_path, _registry_tiles(20, slide="007"))

    frame, column = loader.read_assignments(csv)

    assert frame["slides"].iloc[0] == "007", frame["slides"].iloc[0]
    assert frame["slide_tile"].iloc[0] == "007_0_0.JPEG", frame["slide_tile"].iloc[0]
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 20, report["unmatched_examples"]


def test_the_key_built_from_a_numeric_slide_id_is_the_registrys_key(tmp_path):
    """The registry side builds its key with make_slide_tile from the same text,
    so the two must agree character for character — the same assertion as
    test_join_key_matches_the_registrys_format, on the ids inference breaks."""
    from slide_naming import make_slide_tile
    csv = _text_csv(tmp_path, _rows("007", 3) + _rows("1001", 3, start=3)
                    + _rows("NA", 3, start=6))

    frame, _column = loader.read_assignments(csv)

    expected = ([make_slide_tile("007", f"{i}_{i}.jpeg") for i in range(3)]
                + [make_slide_tile("1001", f"{i}_{i}.jpeg") for i in range(3, 6)]
                + [make_slide_tile("NA", f"{i}_{i}.jpeg") for i in range(6, 9)])
    assert frame["slide_tile"].tolist() == expected, frame["slide_tile"].tolist()


def test_a_numeric_slide_id_beside_an_na_slide_does_not_become_a_float(tmp_path):
    """One 'NA' in the column is enough: pandas reads it as NaN, the column
    becomes float64, and every other slide id in it gains a '.0'."""
    csv = _text_csv(tmp_path, _rows("1001", 10) + _rows("NA", 10, start=10))
    engine = _make_kb(tmp_path, _registry_tiles(10, slide="1001")
                      + [f"NA_{i}_{i}.jpeg" for i in range(10, 20)])

    frame, column = loader.read_assignments(csv)

    assert set(frame["slides"]) == {"1001", "NA"}, set(frame["slides"])
    assert not frame["slide_tile"].str.contains(r"\.0_").any(), \
        frame["slide_tile"].tolist()
    report = loader.inspect(engine, frame, column)
    assert report["matched"] == 20, report["unmatched_examples"]


def test_sample_ids_come_out_as_written(tmp_path):
    """'NA' is a sample id, '007' keeps its zeros, and a blank sample elsewhere
    in the column does not turn '1001' into '1001.0' — in the frame and in the
    aggregates written from it."""
    csv = _text_csv(tmp_path, _rows("SLIDE_A", 4, sample="NA")
                    + _rows("SLIDE_B", 4, sample="007", start=4)
                    + _rows("SLIDE_C", 4, sample="1001", start=8)
                    + _rows("SLIDE_D", 4, sample="", start=12))

    frame, column = loader.read_assignments(csv)

    by_slide = frame.groupby("slides")["samples"].first().to_dict()
    assert by_slide == {"SLIDE_A": "NA", "SLIDE_B": "007", "SLIDE_C": "1001",
                        "SLIDE_D": ""}, by_slide
    _proportions, summary = loader.compute_profiles(frame, column, None)
    # Every slide still has its summary row: groupby drops a NaN key, which is
    # what a blank or 'NA' sample used to become.
    assert sorted(summary["samples"]) == ["", "007", "1001", "NA"], \
        summary["samples"].tolist()


def test_two_slides_that_only_differ_by_zero_padding_are_not_duplicates(tmp_path):
    """'007' and '7' are different slides. Read as numbers they are the same
    one, and the duplicate-tile guard refuses a clean file for the wrong
    reason."""
    csv = _text_csv(tmp_path, _rows("007", 5) + _rows("7", 5))

    frame, _column = loader.read_assignments(csv)

    assert frame["slide_tile"].is_unique
    assert set(frame["slides"]) == {"007", "7"}


def test_cluster_ids_are_read_as_written_and_compared_as_the_column_holds_them(tmp_path):
    """The cluster column stays text in the frame, and the preview renders it
    the way tile_registry.hpc_id (integer) will hold it before comparing it
    with hpc_dictionary — so neither '1' vs 1 nor a stray blank elsewhere can
    make known clusters look unknown."""
    csv = _text_csv(tmp_path, _rows("S", 20))
    engine = _make_kb(tmp_path, _registry_tiles(20, slide="S"))

    frame, column = loader.read_assignments(csv)

    assert frame[column].map(type).eq(str).all(), frame[column].map(type).unique()
    report = loader.inspect(engine, frame, column)
    assert report["unknown_clusters"] == [], report["unknown_clusters"]
    assert loader.load(engine, frame, column) == 20
    with engine.connect() as conn:
        got = pd.read_sql(text("SELECT hpc_id FROM tile_registry "
                               "ORDER BY image_index"), conn)["hpc_id"]
    assert got.astype(int).tolist() == [i % 3 for i in range(20)]


def test_an_unknown_cluster_id_is_still_flagged_when_read_as_text(tmp_path):
    csv = _text_csv(tmp_path, _rows("S", 20, clusters=("0", "1", "99")))
    engine = _make_kb(tmp_path, _registry_tiles(20, slide="S"))

    frame, column = loader.read_assignments(csv)

    assert loader.inspect(engine, frame, column)["unknown_clusters"] == ["99"]


def test_an_empty_cluster_id_is_refused(tmp_path):
    """A blank cluster id is a torn row. It used to become NaN, which is
    written into a text hpc_id as the string 'nan' and survives every check."""
    rows = _rows("S", 6)
    rows[2] = (*rows[2][:3], "", rows[2][4])
    csv = _text_csv(tmp_path, rows)

    try:
        loader.read_assignments(csv)
    except SystemExit as e:
        assert "empty leiden_2.5" in str(e), str(e)
    else:
        raise AssertionError("a CSV with a blank cluster id was accepted")


def test_an_empty_slide_or_tile_is_refused(tmp_path):
    """A row with no slide builds the key '_0_0.JPEG' (or, inferred, 'NAN_...'),
    joins nothing, and hides inside the 5% the match-rate guard allows."""
    for index in (1, 2):
        rows = _rows("S", 40)
        rows[7] = tuple("" if i == index else v for i, v in enumerate(rows[7]))
        csv = _text_csv(tmp_path, rows, name=f"blank_{index}.csv")
        column = _HEADER[index]
        try:
            loader.read_assignments(csv)
        except SystemExit as e:
            assert f"empty {column}" in str(e), str(e)
        else:
            raise AssertionError(f"a CSV with a blank {column} was accepted")


def test_an_empty_vote_margin_written_as_text_is_still_refused_as_empty(tmp_path):
    """Read as text, a blank margin is '' rather than NaN. It must still reach
    the empty-margin refusal and not be mistaken for a non-numeric one or
    coerced to 0."""
    rows = _rows("S", 6)
    rows[3] = (*rows[3][:4], "")
    csv = _text_csv(tmp_path, rows)

    try:
        loader.read_assignments(csv)
    except SystemExit as e:
        assert "empty vote_margin" in str(e), str(e)
    else:
        raise AssertionError("a CSV with a blank vote_margin was accepted")


def test_an_na_vote_margin_is_refused_rather_than_read_as_missing(tmp_path):
    rows = _rows("S", 6)
    rows[3] = (*rows[3][:4], "NA")
    csv = _text_csv(tmp_path, rows)

    try:
        loader.read_assignments(csv)
    except SystemExit as e:
        assert "vote_margin" in str(e), str(e)
    else:
        raise AssertionError("a CSV with vote_margin 'NA' was accepted")


def test_the_match_rate_guard_still_fires_on_text_ids(tmp_path):
    """Half of '007' registered: the refusal has to see 50%, exactly as it
    would for a TCGA barcode."""
    csv = _text_csv(tmp_path, _rows("007", 20))
    engine = _make_kb(tmp_path, _registry_tiles(10, slide="007"))

    frame, column = loader.read_assignments(csv)
    report = loader.inspect(engine, frame, column)

    assert report["matched"] == 10
    assert report["matched"] / report["rows"] < loader._MIN_MATCH_RATE


def test_a_duplicated_zero_padded_tile_is_still_refused(tmp_path):
    csv = _text_csv(tmp_path, _rows("007", 5) + _rows("007", 1, start=2))

    try:
        loader.read_assignments(csv)
    except SystemExit as e:
        assert "more than once" in str(e), str(e)
        assert "007_2_2.JPEG" in str(e), str(e)
    else:
        raise AssertionError("a duplicated tile was accepted")


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_kb_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
