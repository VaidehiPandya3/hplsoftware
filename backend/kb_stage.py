#!/usr/bin/env python3
"""Bulk-load a frame into a scratch table, so the Knowledge Bank write can be
one statement instead of eighteen million.

Why this exists
---------------
Stage 6 used to write `tile_registry` with one `UPDATE ... WHERE
UPPER(slide_tile) = :slide_tile` per tile, sent through `executemany` in
batches. At 18.5M tiles that is 18.5M statement executions — each one a plan
execution, an index probe, a heap update and its own WAL record — plus a Python
list of 18.5M six-key dicts built before any of it starts, which is several GB
of interpreter objects on its own. The preview had the mirror-image problem:
1,850 queries, each an expanding `IN` carrying 10,000 literal keys, re-planned
every time.

Both collapse to a single statement once the keys are on the server: stage the
frame, then join. What this module does is get the frame there quickly.

  * On PostgreSQL, `COPY ... FROM STDIN` — the only bulk path psycopg2 has that
    does not go through the statement planner at all.
  * Into an UNLOGGED table, because this is scratch: its contents are worthless
    the moment the transaction that reads them commits, so paying WAL for them
    buys nothing. (Losing an UNLOGGED table's rows to a crash is exactly the
    case where the load is being re-run anyway.)
  * In parallel, on several connections, when asked. This is safe *because* the
    staging table is not Knowledge Bank state: a half-staged table is discarded
    and re-staged, not repaired, so it does not need the single-transaction
    guarantee the actual write does. Which is the whole reason staging is
    separated from writing here rather than done inside the write's
    transaction, where parallelism would be impossible.

On anything that is not PostgreSQL — which is to say the tests, which run
against SQLite — there is no COPY, and staging falls back to chunked
`executemany` INSERTs. That is slower and does not matter: what the tests are
for is proving the *join* produces the same rows as the old per-row loop, and
the join is the same SQL either way.

The scratch table is a real table, not a TEMPORARY one, for two reasons: a
temporary table is scoped to one session, which rules out staging it from
several connections at once, and a temporary table on a pooled SQLAlchemy
connection outlives the code that created it in a way that is very hard to see.
A real table with a name carrying the pid and a timestamp is visible, greppable,
and dropped in a `finally`. `sweep_stale()` exists for the case where even that
did not happen — a job killed between CREATE and DROP.
"""

from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import pandas as pd
from sqlalchemy import text

#: Rows per COPY / executemany. Bounded so a stage of any size holds one
#: chunk's worth of CSV text in memory rather than the whole frame's — at 18.5M
#: rows the single-buffer version is gigabytes of string.
CHUNK_ROWS = 500_000

#: Column types the scratch table is created with. Loose where it can be —
#: this table is joined and read once, never constrained — but NOT for
#: cluster_id, which is assigned straight into tile_registry.hpc_id and so has
#: to be a type PostgreSQL will accept there. TEXT is the fallback only;
#: `cluster_stage_type` reads the real column. See its docstring for the load
#: this got wrong.
_STAGE_DDL_TYPES = {
    "slide_tile": "TEXT",
    # Named "cluster_id" and not "hpc_id" on purpose. app/hpc_chat_handlers_v23.py
    # enumerates every table in the database, keeps any carrying an hpc_id or
    # dominant_hpc column, and renders up to five matching rows straight to the
    # user — skipping only two tables by name. A scratch table with an hpc_id
    # column is therefore answered out of the chatbot for as long as a load is
    # running, without ever being named in a query. CLAUDE.md warns that a grep
    # will not find every reader of this schema; this is that warning arriving.
    "cluster_id": "TEXT",
    "margin": "DOUBLE PRECISION",
    "distance": "DOUBLE PRECISION",
    "reference": "TEXT",
}
_SQLITE_TYPES = dict(_STAGE_DDL_TYPES, margin="REAL", distance="REAL")

_NAME_PREFIX = "hpl_stage_"


def is_postgres(bind) -> bool:
    return bind.dialect.name == "postgresql"


def stage_table_name(purpose: str) -> str:
    """A name that says which process made it and when.

    Both parts earn their place: the pid is what tells you whether the job that
    owns a leftover table is still running, and the timestamp is what orders two
    leftovers from the same recycled pid.
    """
    safe = "".join(c if c.isalnum() else "_" for c in purpose).strip("_").lower()
    return f"{_NAME_PREFIX}{safe}_{os.getpid()}_{int(time.time())}"


#: PostgreSQL type names that mean "a whole number". A staged TEXT column
#: cannot be assigned into any of them: there is no implicit cast from text to
#: integer, unlike between the numeric types, which is why margin and distance
#: never had this problem.
_INTEGER_TYPES = ("integer", "bigint", "smallint", "int", "int2", "int4",
                  "int8", "serial", "bigserial")


def is_integer_type(sql_type: str) -> bool:
    """Whether a column of this type holds whole numbers.

    Public because the preview needs the same answer the staging does: how a
    cluster ID should be *rendered* for comparison depends on it, not only how
    it is stored.
    """
    return str(sql_type).strip().lower().split("(")[0] in _INTEGER_TYPES


def cluster_stage_type(bind, table: str = "tile_registry",
                       column: str = "hpc_id") -> str:
    """The type to stage cluster IDs as: whatever the column they land in is.

    Not a constant, because this repository has three disagreeing records of
    that column and only one of them is the database. `schema.sql` says
    varchar(100) and is a stale 2025-10-23 dump; the tests' fixture said TEXT;
    kb_live_schema_2026-08-26.txt, transcribed from the live
    database's own column listing, says integer — and the live database is the one the UPDATE runs
    against. Staging TEXT sent job 1243334 into

        column "hpc_id" is of type integer but expression is of type text

    after COPYing 18.5M rows, because SQLite types columns dynamically and
    accepted every test.

    Read rather than assumed, so a KB where the column really is varchar keeps
    working and neither has to be guessed at.
    """
    from sqlalchemy import inspect as sqlalchemy_inspect
    try:
        columns = sqlalchemy_inspect(bind).get_columns(table)
    except Exception:  # noqa: BLE001 - a missing table is the caller's problem
        return "TEXT"
    for info in columns:
        if info["name"] == column:
            if is_integer_type(str(info["type"])):
                # BIGINT rather than INTEGER: it is assignment-compatible with
                # any narrower integer column, and a cluster ID that does not
                # fit int4 is a refusal at the guard, not a silent overflow here.
                return "BIGINT"
            return "TEXT"
    return "TEXT"


def unwritable_cluster_ids(values: pd.Series, cluster_type: str) -> tuple:
    """(count, examples) of cluster IDs the target column cannot hold.

    Non-raising, so the preview can report the number *before* a commit
    discovers it. The preview stages only the join key — it never touches
    cluster_id — so without this a dry run passes and the write is the thing
    that finds out, which is the order this pipeline exists to avoid.
    """
    if not is_integer_type(cluster_type):
        return 0, []
    text_values = values.astype(str).str.strip()
    numbers = pd.to_numeric(text_values, errors="coerce")
    unusable = numbers.isna() | (numbers != numbers.round())
    if not unusable.any():
        return 0, []
    return (int(unusable.sum()),
            text_values[unusable].drop_duplicates().head(5).tolist())


def _integer_cluster_ids(values: pd.Series) -> pd.Series:
    """Cluster IDs as whole numbers, refusing anything that is not one.

    The refusal belongs here, before 18.5M rows are COPYed, and it has to name
    the offending values: a cluster column holding '45.0', 'unknown' or an
    empty string is a different defect in the assignment CSV each time, and the
    examples in the message are what say which.
    """
    count, examples = unwritable_cluster_ids(values, "BIGINT")
    if count:
        raise ValueError(
            f"{count:,} of {len(values):,} cluster ID(s) are not whole numbers, "
            f"and tile_registry.hpc_id is an integer column, so they cannot be "
            f"written: {examples}. Fix the assignment CSV's cluster column — "
            f"this is refused before staging rather than after, because the "
            f"alternative is a failed UPDATE at the end of a COPY of every row.")
    return pd.to_numeric(values.astype(str).str.strip()).astype("int64")


def build_stage_frame(frame: pd.DataFrame, cluster_column: str,
                     cluster_type: str = "TEXT") -> pd.DataFrame:
    """The five columns the write needs, named as the scratch table names them.

    `cluster_type` is what `cluster_stage_type` read off the target column. It
    changes the *values*, not just the DDL: staged as an integer column, the
    ids have to be written as `45` and not `45.0`, which is what
    `frame[col].astype(str)` produces from a float dtype and what COPY then
    rejects one row into the load.

    `hpc_assigned_at` is *not* here. The old loop put the same
    `datetime.now(timezone.utc)` on all 18.5M dicts; staging it would be 18.5M
    copies of one value pushed over a socket to be read back unchanged, so it is
    passed to the UPDATE as a single bind parameter instead. Same value on every
    row, which is what it always was.
    """
    cluster = frame[cluster_column]
    cluster = (_integer_cluster_ids(cluster) if is_integer_type(cluster_type)
               # str().strip() exactly as the old record-building loop did, for
               # a KB whose hpc_id really is a text column.
               else cluster.astype(str).str.strip())
    return pd.DataFrame({
        "slide_tile": frame["slide_tile"],
        # See _STAGE_DDL_TYPES for why this is not called hpc_id here.
        "cluster_id": cluster,
        "margin": pd.to_numeric(frame["vote_margin"], errors="coerce"),
        "distance": pd.to_numeric(frame["neighbor_distance"], errors="coerce"),
        "reference": frame["hpc_reference"].astype(str),
    })


def create_stage(conn, table: str, columns: Sequence[str],
                 cluster_type: str | None = None) -> None:
    types = dict(_STAGE_DDL_TYPES if is_postgres(conn) else _SQLITE_TYPES)
    if cluster_type:
        types["cluster_id"] = cluster_type
    body = ", ".join(f"{c} {types[c]}" for c in columns)
    unlogged = "UNLOGGED " if is_postgres(conn) else ""
    conn.execute(text(f"CREATE {unlogged}TABLE {table} ({body})"))


def index_stage(conn, table: str) -> None:
    """Index the join key.

    On PostgreSQL this is close to free and lets the planner choose a merge or
    nested-loop join where a hash of 18.5M keys would not fit `work_mem`; on
    SQLite, where the portable UPDATE is a correlated subquery per column, it is
    the difference between one scan and one scan per row.
    """
    conn.execute(text(f"CREATE INDEX {table}_key ON {table} (slide_tile)"))


def drop_stage(conn, table: str) -> None:
    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def _copy_chunk(raw_connection, table: str, columns: Sequence[str],
                chunk: pd.DataFrame) -> None:
    """One COPY of one slice, via psycopg2's copy_expert.

    FORMAT csv rather than the default text format because pandas already knows
    how to write CSV, quoting and all, and because the CSV format's default NULL
    representation is the unquoted empty field — which is exactly what
    `to_csv` writes for a NaN. So a NaN neighbor_distance becomes SQL NULL
    without a special case, matching the old loop's explicit
    `None if pd.isna(distance) else float(distance)`.
    """
    buffer = io.StringIO()
    chunk.to_csv(buffer, index=False, header=False, columns=list(columns))
    buffer.seek(0)
    with raw_connection.cursor() as cursor:
        cursor.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv)",
            buffer,
        )


def _insert_chunk(conn, table: str, columns: Sequence[str],
                  chunk: pd.DataFrame) -> None:
    placeholders = ", ".join(f":{c}" for c in columns)
    conn.execute(
        text(f"INSERT INTO {table} ({', '.join(columns)}) "
             f"VALUES ({placeholders})"),
        chunk[list(columns)].to_dict("records"),
    )


def _slice_bounds(total: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping, covering — checked by test rather than eyed.

    A staging split that dropped or duplicated a slice would produce a scratch
    table of the wrong size, and the size check in `stage_frame` is what turns
    that into a refusal instead of a partial load.
    """
    if parts < 1:
        parts = 1
    step = -(-total // parts) if total else 0
    return [(start, min(start + step, total))
            for start in range(0, total, step)] if step else []


def stage_frame(engine, table: str, frame: pd.DataFrame, *,
                workers: int = 1, index: bool = True,
                cluster_type: str | None = None) -> float:
    """Create `table`, fill it from `frame`, index it. Returns seconds taken.

    Runs outside any Knowledge Bank transaction, on its own connections, and
    verifies the row count it ends up with against the frame it was handed
    before returning. That check is the point of doing it here: a scratch table
    that is short by a slice would make the subsequent join update a subset of
    the tiles and report success, which is this pipeline's characteristic
    failure rather than an unlikely one.
    """
    started = time.perf_counter()
    columns = list(frame.columns)

    with engine.begin() as conn:
        create_stage(conn, table, columns, cluster_type)

    bounds = _slice_bounds(len(frame), max(1, workers)) if workers > 1 else None

    if is_postgres(engine) and workers > 1 and bounds:
        def _stage_slice(bound):
            start, stop = bound
            # One connection, one transaction per worker. They cannot share the
            # write's transaction and do not need to: see the module docstring.
            with engine.begin() as conn:
                raw = conn.connection.dbapi_connection
                part = frame.iloc[start:stop]
                for offset in range(0, len(part), CHUNK_ROWS):
                    _copy_chunk(raw, table, columns,
                                part.iloc[offset:offset + CHUNK_ROWS])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() rather than leaving the map lazy: an exception in a worker
            # is only raised when its result is consumed, and an unconsumed
            # failure here would leave a short staging table looking complete.
            list(pool.map(_stage_slice, bounds))
    else:
        with engine.begin() as conn:
            raw = conn.connection.dbapi_connection if is_postgres(conn) else None
            for offset in range(0, len(frame), CHUNK_ROWS):
                chunk = frame.iloc[offset:offset + CHUNK_ROWS]
                if raw is not None:
                    _copy_chunk(raw, table, columns, chunk)
                else:
                    _insert_chunk(conn, table, columns, chunk)

    with engine.begin() as conn:
        staged = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        if index:
            index_stage(conn, table)

    if staged != len(frame):
        raise SystemExit(
            f"Staging table {table} holds {staged:,} rows but the assignment "
            f"frame has {len(frame):,}. Nothing has been written to the "
            f"Knowledge Bank. This is a bug in the staging step, not in the "
            f"data — re-running is safe, but if it recurs, re-run with "
            f"--stage-workers 1 to take the parallel path out of the picture."
        )
    return time.perf_counter() - started


def sweep_stale(engine, older_than_seconds: int = 24 * 3600) -> list[str]:
    """Drop scratch tables left by jobs that died between CREATE and DROP.

    Only tables whose embedded timestamp is older than `older_than_seconds`, so
    a sweep can never take the table out from under a load running right now —
    including one started by a different job on a different node, which is
    precisely the case a pid check cannot settle.
    """
    if not is_postgres(engine):
        return []
    cutoff = time.time() - older_than_seconds
    dropped = []
    with engine.begin() as conn:
        names = [row[0] for row in conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() "
            "AND tablename LIKE :pattern"), {"pattern": f"{_NAME_PREFIX}%"})]
        for name in names:
            parts = name.rsplit("_", 2)
            if len(parts) != 3 or not parts[2].isdigit():
                continue
            if int(parts[2]) < cutoff:
                drop_stage(conn, name)
                dropped.append(name)
    return dropped
