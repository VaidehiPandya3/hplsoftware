#!/usr/bin/env python3
"""Why is the slide viewer fast against one Knowledge Bank and timing out on the other?

`/slide/{id}/tiles_meta` and `/slide/{id}/adjacency` are the same SQL whichever
target they are pointed at, so a 30-second client timeout on `test` while
`production` answers instantly is not a difference in the code. It is a
difference in the *databases*, and there are only a few candidates:

  1. migrate_indexes.sql has not been run against that database, or was last run
     before §8 added the expression indexes (2026-09-08). Both queries filter on
     UPPER(TRIM(tc.slides)) and join on UPPER(tr.slide_tile); PostgreSQL matches
     index expressions rather than values, so without idx_tc_slides_upper and
     idx_tr_slide_tile_upper the only plan available is a sequential scan of
     tile_coordinates and a scan-plus-hash of tile_registry.

     HANDOFF_2026-08-26.md's one-off setup runs migrate_all.sql once, at
     createdb time. Nothing re-runs it, so every index added to the migration
     after a database was created is missing from that database and present in
     whichever one someone re-ran it against by hand.

  2. The tables were never ANALYZEd. A cohort registered into an empty database
     leaves the planner with no statistics until autovacuum catches up, and it
     will choose a scan over an index that exists.

  3. The cohort is simply much bigger there. A test KB is where an 18.5M-tile
     cohort gets registered *first*, so "test" is routinely the larger database,
     and a missing index is only slow in proportion to the rows it isn't
     skipping — which is why the same gap is invisible on a small cohort.

Read-only, apart from `SET statement_timeout` on its own connection. It opens
connections, reads catalogs and runs EXPLAIN (ANALYZE) on the viewer's own
query; nothing here writes to any table.

    python backend/diagnose_kb_indexes.py
    python backend/diagnose_kb_indexes.py --databases hpl_kb_test --slide <SLIDE_ID>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_url import database_url  # noqa: E402

DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")

MIGRATION = Path(__file__).resolve().parent / "migrate_indexes.sql"

#: The viewer's own query, character for character with tile_server_v2_.py's
#: slide_tiles_meta. Timing anything else would be timing a different plan —
#: test_index_coverage.py pins the two together.
VIEWER_SQL = """
    SELECT
        tc.slide_tile, tc.slides, tc.tiles,
        tc.col, tc.row,
        tc.x_5x, tc.y_5x, tc.x_native, tc.y_native,
        tr.hpc_id, tr.image_index AS h5_index,
        hd.inflammation, hd.necrosis, hd.malignant
    FROM tile_coordinates tc
    LEFT JOIN tile_registry tr
      ON UPPER(tr.slide_tile) = UPPER(tc.slide_tile)
    LEFT JOIN hpc_dictionary hd
      ON hd.hpc_id = tr.hpc_id
    WHERE UPPER(TRIM(tc.slides)) = :slide_id
"""

#: CREATE INDEX statements with a literal name. The two built by `EXECUTE
#: format(...)` in §6/7 are deliberately not matched — their names are %I
#: placeholders, they cover reference tables of a few dozen rows, and neither is
#: on any path this script is about.
_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([A-Za-z_]\w*)\s+ON\s+([A-Za-z_]\w*)\s*\(([^;]*?)\)\s*;",
    re.IGNORECASE | re.S)

#: The ones the viewer's plan depends on, called out separately from the rest:
#: a database missing any of these cannot answer tiles_meta without a scan.
VIEWER_INDEXES = ("idx_tc_slides_upper", "idx_tr_slide_tile_upper",
                  "idx_tc_slide_tile_upper")


def declared_indexes() -> dict:
    """{index_name: (table, expression)} from migrate_indexes.sql."""
    sql = MIGRATION.read_text()
    return {m.group(1).lower(): (m.group(2).lower(), " ".join(m.group(3).split()))
            for m in _CREATE_INDEX.finditer(sql)}


def engine_for(db_name: str):
    return create_engine(
        database_url(db_name, user=DB_USER, password=DB_PASS,
                     host=DB_HOST, port=DB_PORT),
        pool_pre_ping=True)


def _table_facts(conn, table: str) -> dict:
    """Row estimate and analyse history. reltuples is -1 on a table that has
    never been analysed at all, which is itself the answer to a slow plan."""
    row = conn.execute(text(
        "SELECT c.reltuples::bigint, s.last_analyze, s.last_autoanalyze "
        "FROM pg_class c LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid "
        "WHERE c.oid = to_regclass(:t)"), {"t": table}).fetchone()
    if row is None:
        return {"exists": False}
    estimate, last_analyze, last_autoanalyze = row
    return {"exists": True, "rows_estimate": estimate,
            "analyzed": last_analyze or last_autoanalyze}


def _pick_slide(conn) -> str | None:
    """Any slide with tiles. Unordered and LIMIT 1, so this costs one page
    rather than a scan even on the database that has not got the index."""
    row = conn.execute(text(
        "SELECT UPPER(TRIM(slides)) FROM tile_coordinates "
        "WHERE slides IS NOT NULL LIMIT 1")).fetchone()
    return row[0] if row else None


def _time_viewer_query(conn, slide: str, timeout_s: int) -> dict:
    """EXPLAIN (ANALYZE) the viewer's query once — the plan and the wall clock
    come out of the same run, so what is reported is what was measured."""
    conn.execute(text(f"SET statement_timeout = {int(timeout_s * 1000)}"))
    started = time.time()
    try:
        plan = "\n".join(r[0] for r in conn.execute(
            text("EXPLAIN (ANALYZE, BUFFERS) " + VIEWER_SQL),
            {"slide_id": slide}))
    except DBAPIError as e:
        if "canceling statement due to statement timeout" in str(e).lower():
            return {"slide": slide, "timed_out": True, "seconds": timeout_s}
        return {"slide": slide, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
    elapsed = time.time() - started
    match = re.search(r"Execution Time: ([\d.]+) ms", plan)
    scans = sorted({m.group(1) for m in re.finditer(
        r"Seq Scan on (\w+)", plan)} & {"tile_coordinates", "tile_registry"})
    return {"slide": slide, "timed_out": False,
            "seconds": float(match.group(1)) / 1000 if match else elapsed,
            "seq_scans": scans, "plan": plan}


def inspect_db(db_name: str, declared: dict, slide: str | None,
               timeout_s: int) -> dict:
    out = {"database": db_name, "missing": [], "absent_tables": []}
    try:
        with engine_for(db_name).connect() as conn:
            present = {r[0].lower() for r in conn.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname NOT IN ('pg_catalog', 'information_schema')"))}
            tables = {t for t, _ in declared.values()}
            live = {t for t in tables
                    if conn.execute(text("SELECT to_regclass(:t)"),
                                    {"t": t}).scalar() is not None}
            out["absent_tables"] = sorted(tables - live)
            for name, (table, expression) in sorted(declared.items()):
                if table in live and name not in present:
                    out["missing"].append((name, table, expression))

            out["tables"] = {t: _table_facts(conn, t)
                             for t in ("tile_coordinates", "tile_registry")}
            if out["tables"]["tile_coordinates"]["exists"]:
                target = slide or _pick_slide(conn)
                out["timing"] = (_time_viewer_query(conn, target, timeout_s)
                                 if target else {"error": "no rows in tile_coordinates"})
    except Exception as e:  # noqa: BLE001 - a database that will not open is a result
        out["error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    return out


def report(result: dict, show_plan: bool) -> None:
    print(f"{result['database']}:")
    if result.get("error"):
        print(f"  unavailable — {result['error']}\n")
        return

    for table, facts in result.get("tables", {}).items():
        if not facts["exists"]:
            print(f"  {table:<18} absent")
            continue
        rows = facts["rows_estimate"]
        estimate = "never analysed" if rows < 0 else f"~{rows:,} rows"
        when = facts["analyzed"].strftime("%Y-%m-%d %H:%M") if facts["analyzed"] else "never"
        print(f"  {table:<18} {estimate} (last analysed {when})")

    if result["absent_tables"]:
        print(f"  tables absent      {', '.join(result['absent_tables'])} "
              f"(their indexes were not checked)")

    if not result["missing"]:
        print("  indexes            all of migrate_indexes.sql's are present")
    else:
        print(f"  indexes MISSING    {len(result['missing'])}")
        for name, table, expression in result["missing"]:
            flag = "  <- the viewer needs this" if name in VIEWER_INDEXES else ""
            print(f"    {name} on {table} ({expression}){flag}")

    timing = result.get("timing", {})
    if timing.get("error"):
        print(f"  viewer query       {timing['error']}")
    elif timing.get("timed_out"):
        print(f"  viewer query       slide {timing['slide']}: still running after "
              f"{timing['seconds']}s — this is the viewer's timeout, reproduced")
    elif timing:
        scans = (", ".join(timing["seq_scans"]) + " scanned in full"
                 if timing["seq_scans"] else "no sequential scan")
        print(f"  viewer query       slide {timing['slide']}: "
              f"{timing['seconds']:.2f}s, {scans}")
        if show_plan:
            print("\n" + "\n".join("    " + l for l in timing["plan"].splitlines()))
    print()


def verdict(results: list) -> None:
    print("verdict:")
    readable = [r for r in results if not r.get("error")]
    if not readable:
        print(f"  No database could be read. Fix the connection first — DB_HOST "
              f"is currently {DB_HOST!r}.")
        return

    for r in readable:
        viewer_missing = [m[0] for m in r["missing"] if m[0] in VIEWER_INDEXES]
        timing = r.get("timing", {})
        slow = timing.get("timed_out") or timing.get("seconds", 0) > 5

        if viewer_missing:
            print(f"  {r['database']}: missing {', '.join(viewer_missing)}. The "
                  f"viewer's filter is an expression, so these are the only "
                  f"indexes that can serve it.")
            print(f"    psql -d {r['database']} -f backend/migrate_indexes.sql")
            print(f"    Re-running it is safe and idempotent: every CREATE is IF "
                  f"NOT EXISTS and the normalising UPDATEs skip rows that already "
                  f"match. Building an index does take a lock that blocks writes "
                  f"to that table until it finishes, so do it while nothing is "
                  f"loading.")
        elif r["missing"]:
            print(f"  {r['database']}: {len(r['missing'])} index(es) from the "
                  f"migration are missing, none of them on the viewer's path. "
                  f"Same fix, less urgency.")
        elif slow:
            stale = [t for t, f in r.get("tables", {}).items()
                     if f.get("exists") and not f.get("analyzed")]
            print(f"  {r['database']}: every index is present and the query is "
                  f"still slow.")
            if stale:
                print(f"    {', '.join(stale)} has never been analysed, so the "
                      f"planner is choosing without statistics: "
                      f"ANALYZE {' '.join(stale)};")
            else:
                print(f"    Re-run with --plan and read where the time goes.")
        else:
            print(f"  {r['database']}: indexes complete, viewer query fast. "
                  f"Whatever is slow is not this.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--databases", nargs="+", default=["hpl_kb", "hpl_kb_test"],
                        help="Knowledge Banks to check.")
    parser.add_argument("--slide", default=None,
                        help="Slide to time the viewer's query on. Defaults to "
                             "any slide that database has tiles for.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Seconds to let the query run before reporting it "
                             "as the timeout the viewer sees. Default 60.")
    parser.add_argument("--plan", action="store_true",
                        help="Print the full EXPLAIN output.")
    args = parser.parse_args()

    declared = declared_indexes()
    print(f"migrate_indexes.sql declares {len(declared)} named index(es)")
    print(f"connecting as {DB_USER}@{DB_HOST}:{DB_PORT}\n")

    slide = args.slide.strip().upper() if args.slide else None
    results = [inspect_db(db, declared, slide, args.timeout)
               for db in args.databases]
    for r in results:
        report(r, args.plan)
    verdict(results)


if __name__ == "__main__":
    main()
