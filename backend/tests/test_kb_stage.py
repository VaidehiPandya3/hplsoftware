"""The PostgreSQL-only half of Stage 6's write.

Stage 6 stages the assignment CSV into a scratch table and joins against it,
instead of sending one UPDATE per tile. Most of that is proven in
test_kb_load.py, against SQLite, which is enough for the part that matters:
that the join produces exactly the registry the per-row loop produced.

What SQLite cannot reach is the code that only exists on PostgreSQL — the COPY
payload, the UNLOGGED scratch DDL, the choice of UPDATE ... FROM, and the
VACUUM. Those run only on the cluster, and "it will be fine there" is how this
pipeline's silent failures get written. So they are tested here against stand-in
binds and a stand-in psycopg2 cursor: not a substitute for running it on a real
database, but enough that a wrong column order, a NaN written as 0, or the
portable statement shipping to production is a failing test rather than a
surprise in an 18-million-row write.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import kb_stage  # noqa: E402
import load_hpc_assignments as loader  # noqa: E402


class _Dialect:
    def __init__(self, name):
        self.name = name


class _Bind:
    """Enough of a SQLAlchemy bind for the dialect checks."""
    def __init__(self, name):
        self.dialect = _Dialect(name)


class _Cursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def copy_expert(self, statement, buffer):
        self.log.append((statement, buffer.read()))


class _RawConnection:
    """Stands in for psycopg2's connection, recording what COPY was handed."""
    def __init__(self):
        self.copies = []

    def cursor(self):
        return _Cursor(self.copies)


_STAGE_COLUMNS = tuple(kb_stage._STAGE_DDL_TYPES)


def _frame():
    return pd.DataFrame({
        "samples": ["S1"] * 3,
        "slides": ["SLIDE-A"] * 3,
        "tiles": ["1_1.jpeg", "2_2.jpeg", "3_3.jpeg"],
        "slide_tile": ["SLIDE-A_1_1.JPEG", "SLIDE-A_2_2.JPEG", "SLIDE-A_3_3.JPEG"],
        "leiden_2.5": ["0", "12", "3"],
        "vote_margin": [0.8, 0.25, 0.5],
        "neighbor_distance": [1.2, float("nan"), 0.9],
        "hpc_reference": ["ref"] * 3,
    })


# --- which statement production actually gets -----------------------------

def test_postgres_gets_update_from_not_the_portable_fallback(_=None):
    """The correlated-subquery statement is five scans of the scratch table per
    registry row. It exists for databases without UPDATE ... FROM; shipping it
    to PostgreSQL by accident would undo the entire speedup while every test
    still passed."""
    sql = loader._update_sql(_Bind("postgresql"), "hpl_stage_x")
    assert "FROM hpl_stage_x AS s" in sql
    assert "SELECT s.cluster_id" not in sql
    assert loader.supports_update_from(_Bind("postgresql"))


def test_an_unknown_database_gets_the_portable_statement(_=None):
    """A dialect nobody has checked must fall back, not be assumed capable."""
    sql = loader._update_sql(_Bind("mysql"), "hpl_stage_x")
    assert "SELECT s.cluster_id FROM hpl_stage_x s" in sql
    assert not loader.supports_update_from(_Bind("mysql"))


def test_both_statements_write_the_same_five_columns(_=None):
    """The two spellings must set the same columns. A column present in one and
    missing from the other leaves whichever database takes that path with a
    stale value beside four fresh ones — valid-looking, and wrong."""
    columns = ("hpc_id", "hpc_vote_margin", "hpc_neighbor_distance",
               "hpc_reference", "hpc_assigned_at")
    for template in (loader._UPDATE_FROM_SQL, loader._UPDATE_CORRELATED_SQL):
        for column in columns:
            assert f"{column} =" in template, (column, template[:40])


def test_the_scratch_table_is_unlogged_on_postgres_only(_=None):
    """UNLOGGED is what stops this scratch table paying WAL for rows that are
    worthless the moment the write commits. It is also PostgreSQL-only syntax,
    so it must not reach SQLite."""
    statements = []

    class _Conn(_Bind):
        def execute(self, statement):
            statements.append(str(statement))

    kb_stage.create_stage(_Conn("postgresql"), "st", ["slide_tile", "cluster_id"])
    assert "CREATE UNLOGGED TABLE st" in statements[0]
    assert "DOUBLE PRECISION" not in statements[0]  # neither column is numeric

    statements.clear()
    kb_stage.create_stage(_Conn("sqlite"), "st", ["slide_tile", "margin"])
    assert "UNLOGGED" not in statements[0]
    assert "margin REAL" in statements[0]


# --- what COPY is actually handed ----------------------------------------

def test_the_copy_payload_is_headerless_csv_in_the_declared_column_order(_=None):
    """COPY matches fields to columns by position. A frame whose column order
    drifted from the COPY statement's column list would load every value into
    the wrong column — margins into distances, cluster ids into references —
    and COPY would report success."""
    raw = _RawConnection()
    stage = kb_stage.build_stage_frame(_frame(), "leiden_2.5")
    kb_stage._copy_chunk(raw, "st", list(stage.columns), stage)

    (statement, payload), = raw.copies
    assert statement.startswith(
        "COPY st (slide_tile, cluster_id, margin, distance, reference) FROM STDIN")
    assert "WITH (FORMAT csv)" in statement
    lines = payload.strip("\n").split("\n")
    assert len(lines) == 3, payload
    assert lines[0] == "SLIDE-A_1_1.JPEG,0,0.8,1.2,ref"
    # No header row: COPY would read it as data and fail on the numeric columns
    # only by luck.
    assert "slide_tile" not in payload


def test_a_missing_neighbor_distance_copies_as_null_not_as_zero(_=None):
    """The old writer sent `None if pd.isna(distance) else float(distance)`, so
    a tile with no neighbour distance held SQL NULL. In CSV format COPY reads an
    unquoted empty field as NULL, which reproduces that — whereas a 0 would be a
    distance, and a distance of zero means an exact match."""
    raw = _RawConnection()
    stage = kb_stage.build_stage_frame(_frame(), "leiden_2.5")
    kb_stage._copy_chunk(raw, "st", list(stage.columns), stage)

    (_, payload), = raw.copies
    second = payload.strip("\n").split("\n")[1]
    assert second == "SLIDE-A_2_2.JPEG,12,0.25,,ref", second
    assert "nan" not in payload.lower()


def test_the_staged_frame_does_not_carry_a_timestamp_column(_=None):
    """assigned_at is one value for the whole load, passed to the UPDATE as a
    bind parameter. Staging it would be 18.5 million copies of one timestamp
    pushed over a socket to be read back unchanged."""
    stage = kb_stage.build_stage_frame(_frame(), "leiden_2.5")
    assert list(stage.columns) == [
        "slide_tile", "cluster_id", "margin", "distance", "reference"]
    assert ":assigned_at" in loader._UPDATE_FROM_SQL


def test_the_copy_payload_is_chunked_rather_than_one_buffer(_=None):
    """One StringIO for 18.5M rows is gigabytes of CSV text in this process,
    on top of the frame it was built from. Chunked, so peak memory is one
    chunk's worth however large the cohort."""
    raw = _RawConnection()
    stage = kb_stage.build_stage_frame(_frame(), "leiden_2.5")
    original = kb_stage.CHUNK_ROWS
    try:
        kb_stage.CHUNK_ROWS = 2
        for offset in range(0, len(stage), kb_stage.CHUNK_ROWS):
            kb_stage._copy_chunk(raw, "st", list(stage.columns),
                                 stage.iloc[offset:offset + kb_stage.CHUNK_ROWS])
    finally:
        kb_stage.CHUNK_ROWS = original
    assert len(raw.copies) == 2
    assert len(raw.copies[0][1].strip("\n").split("\n")) == 2
    assert len(raw.copies[1][1].strip("\n").split("\n")) == 1


# --- the sweep ------------------------------------------------------------

def test_no_staged_column_is_named_so_the_chatbot_would_render_it(_=None):
    """app/hpc_chat_handlers_v23.py enumerates every table in the database,
    keeps any carrying an hpc_id or dominant_hpc column, and renders up to five
    matching rows straight to the user — skipping only two tables by name. A
    scratch table with an hpc_id column would therefore be answered out of the
    chatbot for as long as a load is running, under a name nobody recognises.

    Pinned as a test rather than a comment because the tempting name for this
    column is exactly the wrong one: it is written into tile_registry.hpc_id,
    so calling it hpc_id here reads as consistency.
    """
    assert "hpc_id" not in _STAGE_COLUMNS
    assert "dominant_hpc" not in _STAGE_COLUMNS


def test_the_scratch_name_prefix_is_what_the_chatbot_skips(_=None):
    """The second line of defence. Renaming the column is what actually keeps
    the chatbot out; the prefix skip is what keeps a future column named less
    carefully from undoing that. Both have to agree on the prefix."""
    handlers = (BACKEND.parent / "app" / "hpc_chat_handlers_v23.py").read_text()
    assert f'startswith("{kb_stage._NAME_PREFIX}")' in handlers
    assert kb_stage.stage_table_name("x").startswith(kb_stage._NAME_PREFIX)


def test_the_sweep_is_a_no_op_off_postgres(_=None):
    """It reads pg_tables. On anything else it must return nothing rather than
    raise, because inspect() calls it on whatever engine it was handed."""
    assert kb_stage.sweep_stale(_Bind("sqlite")) == []


# --- the staged column has to be assignable into the real one --------------
#
# Job 1243334 staged cluster_id as TEXT, COPYed 18,485,499 rows, and then died
# on `column "hpc_id" is of type integer but expression is of type text`. Every
# test passed first: SQLite types values dynamically, so its TEXT column held
# integers happily, and test_kb_load.py's fixture declared hpc_id TEXT anyway,
# following schema.sql — the stale dump CLAUDE.md warns against. No amount of
# loading proves the types agree on SQLite, so the agreement is asserted here.

def _registry(tmp_path, hpc_id_type):
    from sqlalchemy import create_engine, text
    engine = create_engine(
        f"sqlite:///{tmp_path}/kb_{hpc_id_type.split('(')[0].lower()}.db")
    with engine.begin() as conn:
        conn.execute(text(
            f"CREATE TABLE tile_registry (slide_tile TEXT, "
            f"hpc_id {hpc_id_type}, hpc_reference TEXT)"))
    return engine


def test_the_staged_cluster_column_follows_the_real_column(tmp_path):
    """Read off tile_registry, not off a constant or a comment."""
    assert kb_stage.cluster_stage_type(_registry(tmp_path, "INTEGER")) == "BIGINT"
    assert kb_stage.cluster_stage_type(_registry(tmp_path, "BIGINT")) == "BIGINT"
    assert kb_stage.cluster_stage_type(_registry(tmp_path, "TEXT")) == "TEXT"
    assert kb_stage.cluster_stage_type(_registry(tmp_path, "VARCHAR(100)")) == "TEXT"


def test_an_unreadable_registry_falls_back_rather_than_raising(tmp_path):
    """A database with no tile_registry is the caller's problem: Stage 6 refuses
    on the match rate a moment later, with a message about the actual cause."""
    from sqlalchemy import create_engine
    assert kb_stage.cluster_stage_type(
        create_engine(f"sqlite:///{tmp_path}/empty.db")) == "TEXT"


def test_the_scratch_ddl_uses_the_type_it_was_handed(_=None):
    statements = []

    class _Conn(_Bind):
        def execute(self, statement):
            statements.append(str(statement))

    kb_stage.create_stage(_Conn("postgresql"), "st",
                          ["slide_tile", "cluster_id"], "BIGINT")
    assert "cluster_id BIGINT" in statements[0], statements[0]
    assert "cluster_id TEXT" not in statements[0]


def test_integer_cluster_ids_are_staged_as_integers_not_floats(_=None):
    """'45.0' is what astype(str) makes of a float column, and COPY rejects it
    for a BIGINT column one row into an 18.5M-row load."""
    frame = _frame()
    frame["leiden_2.5"] = frame["leiden_2.5"].astype(float)
    staged = kb_stage.build_stage_frame(frame, "leiden_2.5", "BIGINT")
    assert staged["cluster_id"].dtype.kind == "i", staged["cluster_id"].dtype
    assert all("." not in v for v in staged["cluster_id"].astype(str))


def test_a_text_column_still_gets_text(_=None):
    """The other KB shape has to keep working: the type is read, not assumed."""
    staged = kb_stage.build_stage_frame(_frame(), "leiden_2.5", "TEXT")
    assert staged["cluster_id"].dtype == object
    assert staged["cluster_id"].tolist() == [
        str(v).strip() for v in _frame()["leiden_2.5"]]


def test_a_cluster_id_that_is_not_a_whole_number_is_refused_before_staging(_=None):
    """Refused rather than coerced, and before the COPY.

    Rounding or dropping 'unknown' would attach some other cluster's ID to
    those tiles, which is the failure this stage is written against — and the
    refusal has to name the values, because each kind means a different defect
    in the assignment CSV.
    """
    frame = _frame()
    frame.loc[frame.index[0], "leiden_2.5"] = "unknown"
    try:
        kb_stage.build_stage_frame(frame, "leiden_2.5", "BIGINT")
    except ValueError as e:
        assert "unknown" in str(e) and "whole numbers" in str(e), str(e)
    else:
        raise AssertionError("a non-integer cluster ID was staged for an "
                             "integer column")


def test_the_unwritable_count_is_available_without_raising(_=None):
    """What the preview reports. The preview stages only the join key, so it
    cannot discover this by staging — without the count, a clean dry run is
    followed by a commit that fails."""
    frame = _frame()
    frame.loc[frame.index[0], "leiden_2.5"] = "45.5"
    values = frame["leiden_2.5"].astype(str)
    assert kb_stage.unwritable_cluster_ids(values, "BIGINT") == (1, ["45.5"])
    assert kb_stage.unwritable_cluster_ids(values, "TEXT") == (0, [])


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="hpl_stage_test_")))
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
