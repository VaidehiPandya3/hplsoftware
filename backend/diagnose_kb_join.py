#!/usr/bin/env python3
"""Why does Stage 6 report a 0% match rate?

Stage 6 refuses below a 95% match rate, and the refusal names the match rate
rather than the reason — deliberately, because a low rate is almost always a
naming difference and the message says so. But 0% is different from 3%: it means
*nothing* joined, and that has only a handful of possible causes, all
environmental rather than in the code. `register_dataset.py:166` and
`load_hpc_assignments.py` build the join key with the same
`slide_naming.make_slide_tile_series`, so given the same inputs and the same
database they cannot disagree.

This checks each candidate against every database it is pointed at and prints a
verdict:

  1. The tiles were registered into a different Knowledge Bank than the load is
     reading. DB_NAME governs which one, and it defaults to hpl_kb — so
     registering through the UI with "Test" selected and then loading from the
     CLI with DB_NAME unset produces exactly this.
  2. tile_registry.slide_tile is NULL for the registered rows.
     register_dataset.py's _insert reflects the live table and drops columns it
     lacks, printing a note to stderr — so a registration against a table
     without that column writes every other column and leaves the join key
     empty. migrate_indexes.sql is what adds it.
  3. The keys are genuinely spelled differently, in which case the samples
     printed side by side show where.

Read-only. It opens connections and runs SELECTs; nothing here writes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_url import database_url  # noqa: E402
from slide_naming import (  # noqa: E402
    make_slide_tile_series,
    normalize_tile_names,
    tile_name_verdict,
)

DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")


def engine_for(db_name: str):
    return create_engine(
        database_url(db_name, user=DB_USER, password=DB_PASS,
                     host=DB_HOST, port=DB_PORT),
        pool_pre_ping=True,
    )


def sample_keys(csv_path: Path, rows: int) -> tuple[list[str], dict]:
    """The first `rows` join keys, built exactly as the loader builds them.

    Same normalisation and the same helper, so a key that does not match here
    would not match in the load either — the point is to reproduce the loader's
    view of the CSV, not to second-guess it.
    """
    frame = pd.read_csv(csv_path, nrows=rows,
                        usecols=lambda c: c in ("slides", "tiles"))
    verdict = tile_name_verdict(frame["tiles"])
    frame["tiles"], renamed = normalize_tile_names(frame["tiles"])
    keys = make_slide_tile_series(frame["slides"], frame["tiles"]).tolist()
    return keys, {
        "rows_read": len(frame),
        "tile_name_verdict": verdict,
        "names_normalized": renamed,
    }


def inspect_db(db_name: str, keys: list[str]) -> dict:
    out = {"database": db_name}
    try:
        engine = engine_for(db_name)
        with engine.connect() as conn:
            inspector = sqlalchemy_inspect(conn)
            if not inspector.has_table("tile_registry"):
                out["error"] = "no tile_registry table"
                return out
            columns = {c["name"] for c in inspector.get_columns("tile_registry")}
            out["has_slide_tile"] = "slide_tile" in columns
            out["rows"] = conn.execute(
                text("SELECT COUNT(*) FROM tile_registry")).scalar_one()
            if not out["has_slide_tile"]:
                # The join would raise rather than return 0%, so this is only
                # reachable on a database the load has not been pointed at yet.
                out["error"] = ("tile_registry has no slide_tile column — run "
                                "migrate_indexes.sql")
                return out
            out["null_slide_tile"] = conn.execute(text(
                "SELECT COUNT(*) FROM tile_registry WHERE slide_tile IS NULL"
            )).scalar_one()
            out["samples"] = [r[0] for r in conn.execute(text(
                "SELECT slide_tile FROM tile_registry "
                "WHERE slide_tile IS NOT NULL LIMIT 3"))]
            # The loader's own predicate, against a temp list of keys. Chunked
            # so a large sample does not become one enormous parameter list.
            matched = 0
            for start in range(0, len(keys), 10_000):
                chunk = keys[start:start + 10_000]
                placeholders = ", ".join(f":k{i}" for i in range(len(chunk)))
                matched += conn.execute(
                    text(f"SELECT COUNT(DISTINCT UPPER(slide_tile)) "
                         f"FROM tile_registry "
                         f"WHERE UPPER(slide_tile) IN ({placeholders})"),
                    {f"k{i}": k for i, k in enumerate(chunk)},
                ).scalar_one()
            out["matched"] = matched
    except Exception as e:  # noqa: BLE001 - a database that will not open is a result
        out["error"] = f"{type(e).__name__}: {e}".split("\n")[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True,
                        help="The assignment CSV Stage 6 would load.")
    parser.add_argument("--databases", nargs="+",
                        default=["hpl_kb", "hpl_kb_test"],
                        help="Knowledge Banks to check.")
    parser.add_argument("--sample", type=int, default=20_000,
                        help="CSV rows to build keys from. The match rate on a "
                             "leading slice is enough to identify the database; "
                             "it is not a substitute for the real preview.")
    args = parser.parse_args()

    keys, csv_info = sample_keys(args.csv, args.sample)
    print(f"CSV        : {args.csv}")
    print(f"  sampled  {csv_info['rows_read']:,} row(s); tile names "
          f"{csv_info['tile_name_verdict']}, .jpeg appended to "
          f"{csv_info['names_normalized']:,}")
    print(f"  first key {keys[0]!r}")
    print(f"  connecting as {DB_USER}@{DB_HOST}:{DB_PORT}\n")

    results = [inspect_db(db, keys) for db in args.databases]
    for r in results:
        print(f"{r['database']}:")
        if r.get("error"):
            print(f"  unavailable — {r['error']}")
            print()
            continue
        rate = r["matched"] / max(len(keys), 1) * 100
        print(f"  tile_registry rows   {r['rows']:,}")
        print(f"  slide_tile NULL      {r['null_slide_tile']:,}")
        print(f"  matched this sample  {r['matched']:,} of {len(keys):,} "
              f"({rate:.1f}%)")
        for s in r["samples"]:
            print(f"  sample key           {s!r}")
        print()

    best = max((r for r in results if not r.get("error")),
               key=lambda r: r.get("matched", 0), default=None)

    print("verdict:")
    if best is None:
        print("  No database could be read. Fix the connection first — DB_HOST "
              "is currently " + repr(DB_HOST) + ".")
        return
    if best.get("matched", 0) == 0:
        print("  Nothing matched in any database checked.")
        nulls = [r for r in results if r.get("null_slide_tile")]
        if nulls:
            print(f"  {nulls[0]['database']} has "
                  f"{nulls[0]['null_slide_tile']:,} row(s) with a NULL "
                  f"slide_tile — that column is the join key, and "
                  f"register_dataset.py drops it when the live table lacks it. "
                  f"Run migrate_indexes.sql, then re-register.")
        elif best.get("rows"):
            print("  The registry has rows and a populated slide_tile, so the "
                  "two sides spell the key differently. Compare 'first key' "
                  "above against the sample keys — that difference is the bug.")
        else:
            print("  tile_registry is empty here, so this cohort was never "
                  "registered into any database checked. Run Stage 5 first.")
        return
    rate = best["matched"] / max(len(keys), 1) * 100
    print(f"  {best['database']} matches {rate:.1f}% of the sample.")
    if rate >= 95:
        print(f"  Load from it with:\n\n"
              f"    DB_NAME={best['database']} python load_hpc_assignments.py "
              f"--csv {args.csv} --commit --min-margin 0.25 "
              f"--cancer-type LUAD\n\n"
              f"  --record-run-db stays hpl_kb whichever Knowledge Bank the "
              f"tiles go into: run tracking lives in production.")
    else:
        print("  Below the 95% the load requires. A partial match is a naming "
              "difference on some slides rather than a wrong database — check "
              "the unmatched examples the preview prints.")


if __name__ == "__main__":
    main()
