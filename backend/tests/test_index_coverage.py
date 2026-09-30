"""Every UPPER() a query filters on must be an expression some index declares.

PostgreSQL matches index *expressions*, not values. migrate_indexes.sql §8
indexes UPPER(TRIM(slides)) and normalises the column so UPPER(TRIM(x)) = x for
every row, which makes the index redundant in content and load-bearing in
planning. So a query that says UPPER(slides) — same result, one function short
— cannot use it and falls back to a sequential scan.

That is not theoretical. /slide/{id}/tiles_meta and /slide/{id}/adjacency
filtered on UPPER(tc.slides), and once Radiogenomics registered 18.5M rows into
tile_coordinates the viewer stopped opening at all: a 30s client timeout on a
query that had been fast for as long as the table was small. Nothing about the
SQL looks wrong, the index exists, and the plan is where the difference lives —
which is why this is a test and not a code comment.

Expression *shapes* are compared, not (table, expression) pairs: resolving an
alias to a table from raw SQL text is more machinery than the check is worth,
and a shape no index anywhere declares is already the bug.
"""

import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

MIGRATION = BACKEND / "migrate_indexes.sql"

#: Modules whose SQL runs against the Knowledge Bank.
QUERY_SOURCES = ("tile_server_v2_.py", "load_hpc_assignments.py",
                 "register_dataset.py", "kb_stage.py", "diagnose_kb_join.py",
                 "dataset_rollup.py", "precompute_tiles.py",
                 "diagnose_kb_indexes.py")

#: UPPER(TRIM(col)) or UPPER(col), with an optional alias — and nothing else, so
#: prose inside a comment ("UPPER(...)") is not mistaken for an expression.
_COLUMN = r"[A-Za-z_][A-Za-z0-9_]*"
_EXPR = re.compile(
    rf"UPPER\(\s*(?:TRIM\(\s*)?(?:{_COLUMN}\.)?({_COLUMN})\s*\)?\s*\)",
    re.IGNORECASE)
_WRAPPED_IN_TRIM = re.compile(r"UPPER\(\s*TRIM\(", re.IGNORECASE)


def _shapes(text: str) -> set:
    """Normalised expression shapes: {"upper(trim(slides))", "upper(slide_tile)"}."""
    found = set()
    for match in _EXPR.finditer(text):
        column = match.group(1).lower()
        trimmed = bool(_WRAPPED_IN_TRIM.match(match.group(0)))
        found.add(f"upper(trim({column}))" if trimmed else f"upper({column})")
    return found


def indexed_shapes() -> set:
    """What migrate_indexes.sql declares an index on."""
    sql = MIGRATION.read_text()
    shapes = set()
    for statement in re.findall(r"CREATE INDEX[^;]*;", sql, re.IGNORECASE | re.S):
        on = statement.split(" ON ", 1)[-1] if " ON " in statement else ""
        shapes |= _shapes(on)
    return shapes


def queried_shapes() -> dict:
    """{shape: [where it appears]} across the modules that query the KB."""
    used = {}
    for name in QUERY_SOURCES:
        path = BACKEND / name
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for shape in _shapes(line):
                used.setdefault(shape, []).append(f"{name}:{lineno}")
    return used


# --- the check -------------------------------------------------------------

def test_every_queried_expression_is_indexed(_tmp=None):
    indexed = indexed_shapes()
    problems = []
    for shape, places in sorted(queried_shapes().items()):
        if shape not in indexed:
            problems.append(f"{shape} — no index declares it; used at "
                            f"{', '.join(places[:4])}")
    assert not problems, (
        "these filters cannot use an index and will scan the whole table:\n  "
        + "\n  ".join(problems)
        + f"\n\nindexed shapes: {sorted(indexed)}")


def test_the_viewers_two_queries_use_the_indexed_form(_tmp=None):
    """The specific regression, pinned.

    Both endpoints feed the slide viewer, both filter one slide out of
    tile_coordinates, and both were UPPER(tc.slides) until 2026-09-15.
    """
    server = (BACKEND / "tile_server_v2_.py").read_text()
    assert "UPPER(TRIM(tc.slides)) = :slide_id" in server
    assert "UPPER(tc.slides) = :slide_id" not in server, (
        "a viewer query is back to the unindexable form")


def test_the_migration_still_declares_what_those_queries_need(_tmp=None):
    """The other side of the same agreement: the index has to exist.

    Dropping idx_tc_slides_upper would make the fix above meaningless, and the
    symptom would be a slow viewer rather than an error.
    """
    indexed = indexed_shapes()
    for shape in ("upper(trim(slides))", "upper(slide_tile)"):
        assert shape in indexed, (shape, sorted(indexed))


# --- the runtime half: the index has to exist in the database too ----------
#
# Everything above is static — it proves the query and the migration agree. It
# cannot prove a given database has had the migration run against it, and that
# is a separate failure with the same symptom: production answers instantly and
# hpl_kb_test times out, from identical code. diagnose_kb_indexes.py is what
# checks a live database; these pin it to the server it is diagnosing.

def _diagnostic():
    import diagnose_kb_indexes
    return diagnose_kb_indexes


def test_the_diagnostics_query_is_the_viewers_query(_tmp=None):
    """Timing a query that differs from the server's would time a different plan.

    Shapes, not text: the diagnostic has no `tc.`/`tr.` aliasing obligation and
    the comments differ. What must not differ is which expressions the planner
    is handed.
    """
    server = (BACKEND / "tile_server_v2_.py").read_text()
    tiles_meta = server.split("def slide_tiles_meta", 1)[1].split("@app.get", 1)[0]
    viewer = _shapes(tiles_meta)
    assert _shapes(_diagnostic().VIEWER_SQL) == viewer, (
        "diagnose_kb_indexes.VIEWER_SQL no longer filters and joins on what "
        "slide_tiles_meta does, so its timings describe a different query")


def test_the_diagnostic_knows_which_indexes_the_viewer_needs(_tmp=None):
    """Its VIEWER_INDEXES have to be real names from the migration, or the
    verdict recommends nothing for the one case it exists to catch."""
    diagnostic = _diagnostic()
    declared = diagnostic.declared_indexes()
    for name in diagnostic.VIEWER_INDEXES:
        assert name in declared, (name, sorted(declared))
    # And they have to be the ones covering the shapes the query filters on.
    covered = set()
    for name in diagnostic.VIEWER_INDEXES:
        _table, expression = declared[name]
        covered |= _shapes(expression)
    assert {"upper(trim(slides))", "upper(slide_tile)"} <= covered, covered


def test_an_index_inside_a_do_block_is_still_parsed(_tmp=None):
    """§8's newest index is created inside a DO block, guarded on the table
    existing. A parser that only read top-level statements would report it as
    present on every database and never mention it."""
    assert "idx_shm_slide_id_upper" in _diagnostic().declared_indexes()


def test_a_dynamically_named_index_is_not_read_as_a_name(_tmp=None):
    """§6/7 build their names with format('... %I ...'). Capturing "%I" as an
    index name would make it missing from every database, forever."""
    declared = _diagnostic().declared_indexes()
    assert not [n for n in declared if "%" in n], sorted(declared)


# --- and that it can fail --------------------------------------------------

def test_a_filter_one_function_short_of_the_index_is_reported(_tmp=None):
    """The bug reduced: same result, different expression, no index."""
    indexed = _shapes("tile_coordinates (UPPER(TRIM(slides)))")
    queried = _shapes("WHERE UPPER(tc.slides) = :slide_id")
    assert indexed == {"upper(trim(slides))"}
    assert queried == {"upper(slides)"}
    assert not (queried <= indexed), "the checker cannot tell the two apart"


def test_the_matching_form_is_not_reported(_tmp=None):
    indexed = _shapes("tile_coordinates (UPPER(TRIM(slides)))")
    assert _shapes("WHERE UPPER(TRIM(tc.slides)) = :slide_id") <= indexed


def test_prose_in_a_comment_is_not_read_as_an_expression(_tmp=None):
    """The fix's own comment says UPPER(...) and UPPER(TRIM(...)); neither is a
    column, and a checker that flagged them would fail on its own explanation."""
    assert _shapes("-- UPPER(TRIM(...)), not UPPER(...): see migrate_indexes") == set()


def test_a_database_missing_the_viewers_index_is_reported(_tmp=None):
    """The reduced form of the live bug: same code, same query, one database
    that has had migrate_indexes.sql run against it and one that has not."""
    diagnostic = _diagnostic()
    declared = diagnostic.declared_indexes()
    present_everything = set(declared)
    stale_database = present_everything - {"idx_tc_slides_upper"}

    def missing(present):
        return [n for n, (_t, _e) in sorted(declared.items()) if n not in present]

    assert missing(present_everything) == []
    assert missing(stale_database) == ["idx_tc_slides_upper"]
    assert "idx_tc_slides_upper" in diagnostic.VIEWER_INDEXES, (
        "the one index whose absence times the viewer out is not on the list "
        "the verdict escalates")


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="hpl_index_test_")))
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
