"""Can a fresh checkout build the schema the code needs?

migrate_all.sql's header says running it against an empty database builds the
whole schema. That was false for eight of the live database's seventeen tables
and for dataset_id, the column that scopes every cohort — none of which had a
CREATE TABLE or ADD COLUMN anywhere in git. The failure was not visible as a
missing table: migrate_indexes.sql's `UPDATE tile_coordinates ...` blew up
under ON_ERROR_STOP, so the error named an index migration.

These tests are written the way the rest of this suite is: each one has a
companion that proves the check can come out bad, because a coverage test that
cannot fail is the same as no test.

They are static checks over the .sql files. There is no PostgreSQL on the
machines this suite runs on, so nothing here proves the DDL executes — only
that every relation the live database holds is named by some CREATE TABLE, and
that the order in migrate_all.sql puts the base tables before the migrations
that alter them.
"""

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BACKEND = REPO / "backend"

BASE_TABLES_SQL = BACKEND / "migrate_kb_base_tables.sql"
MIGRATE_ALL_SQL = BACKEND / "migrate_all.sql"
SCHEMA_SQL = REPO / "schema.sql"
SCHEMA_SNAPSHOT = BACKEND / "kb_live_schema_2026-08-26.txt"

# The live database, from `\dt` on 2026-08-26. Recorded in
# backend/kb_live_schema_2026-08-26.txt; kept here too so this test states its
# own expectation rather than parsing a text file for it.
LIVE_TABLES = frozenset({
    "dataset_config",
    "h_latent_vectors",
    "hpc_dictionary",
    "hpc_malignant_details",
    "hpc_non_malignant_details",
    "hpc_survival_analysis",
    "hpl_profile_proportion",
    "hpl_profile_summary",
    "slide_hpc_membership",
    "slurm_dataset_run_jobs",
    "slurm_dataset_runs",
    "tile_coordinates",
    "tile_hpc_heatmap",
    "tile_hpc_heatmap_old",
    "tile_registry",
    "wsi_metadata",
    "wsi_registry",
})

# Deliberately still uncovered: `\d` was never captured for these four, so
# their real column list is unknown. schema.sql defines them, but its
# tile_registry is provably out of date against the live database, so its
# hpc_dictionary cannot be trusted either. Listing them here rather than
# excluding them silently — this set should shrink to empty, and the test
# below fails if it ever grows.
KNOWN_UNCOVERED = frozenset({
    "hpc_dictionary",
    "hpc_malignant_details",
    "hpc_non_malignant_details",
    "hpc_survival_analysis",
})

_CREATE_TABLE = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?(?:public\.)?[\"']?([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)


def _sql_files():
    return sorted(BACKEND.glob("*.sql")) + [SCHEMA_SQL]


def _statements_only(text):
    """The file with its `--` comments removed.

    Needed because these files carry more prose than SQL, and a check for a
    column name matches the paragraph explaining which migration owns it just
    as happily as it matches a real ALTER TABLE.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )


def created_tables():
    """Every table any .sql file in this repository can create."""
    found = set()
    for path in _sql_files():
        found.update(m.lower() for m in _CREATE_TABLE.findall(path.read_text()))
    return found


# --- the coverage claim --------------------------------------------------

def test_every_live_table_can_be_created_from_git(_tmp=None):
    created = created_tables()
    missing = (LIVE_TABLES - created) - KNOWN_UNCOVERED
    assert not missing, (
        f"{len(missing)} live table(s) have no CREATE TABLE anywhere in git: "
        f"{sorted(missing)}. A fresh database cannot be built, and the failure "
        f"will surface as an error in whichever migration touches them first, "
        f"not as 'table missing'."
    )


def test_the_uncovered_set_is_not_quietly_growing(_tmp=None):
    """KNOWN_UNCOVERED is an admission, not a licence. If a table is added to
    the live database and nobody writes DDL for it, the test above must not be
    silenced by appending it here — so check the admissions are still exactly
    the four cluster reference tables and that they really are uncovered."""
    created = created_tables()
    assert KNOWN_UNCOVERED <= LIVE_TABLES, "an uncovered name that is not a live table"
    still_uncovered = {t for t in KNOWN_UNCOVERED if t not in created}
    # schema.sql does define all four, so they ARE matched by created_tables().
    # The point of the set is that those definitions are not trustworthy, which
    # a regex cannot see. Assert the situation is unchanged rather than pretend.
    assert still_uncovered == set(), (
        f"{sorted(still_uncovered)} are now absent from every .sql file too — "
        f"that is a second, worse problem than being defined out of date."
    )


def test_the_coverage_check_can_fail(_tmp=None):
    """Proves test_every_live_table_can_be_created_from_git is load-bearing:
    a table nothing creates must be reported."""
    created = created_tables()
    invented = "a_table_that_does_not_exist_anywhere"
    assert invented not in created
    missing = ({invented} | LIVE_TABLES) - created - KNOWN_UNCOVERED
    assert missing == {invented}, (
        "the coverage computation did not flag a table that provably has no "
        "CREATE TABLE — it is not actually checking anything"
    )


# --- the base-tables migration itself ------------------------------------

def test_base_tables_migration_creates_the_eight_that_nothing_else_did(_tmp=None):
    text = BASE_TABLES_SQL.read_text()
    created_here = {m.lower() for m in _CREATE_TABLE.findall(text)}
    expected = {
        "dataset_config", "h_latent_vectors", "slide_hpc_membership",
        "tile_coordinates", "tile_hpc_heatmap", "tile_hpc_heatmap_old",
        "wsi_metadata", "wsi_registry",
    }
    assert expected <= created_here, f"not created: {sorted(expected - created_here)}"


def test_it_does_not_recreate_tables_another_migration_owns(_tmp=None):
    """slurm_dataset_runs' primary key is replaced by
    migrate_dataset_runs_async.sql, and wsi_registry's processing_* columns are
    owned by migrate_processing_status.sql. Two files creating the same table
    is how the two definitions drift apart."""
    stripped = _statements_only(BASE_TABLES_SQL.read_text())
    created_here = {m.lower() for m in _CREATE_TABLE.findall(stripped)}
    for owned in ("slurm_dataset_runs", "slurm_dataset_run_jobs"):
        assert owned not in created_here, f"{owned} is created twice"
    assert "processing_status" not in stripped, (
        "processing_status belongs to migrate_processing_status.sql"
    )


def test_every_statement_is_idempotent(_tmp=None):
    """Every other migration can be re-run against the live database as a
    no-op, and migrate_all.sql's design depends on that being true of all of
    them — it is why there is no version table."""
    stripped = _statements_only(BASE_TABLES_SQL.read_text())
    bare_creates = re.findall(
        r"create\s+table\s+(?!if\s+not\s+exists)", stripped, re.IGNORECASE
    )
    assert not bare_creates, f"{len(bare_creates)} CREATE TABLE without IF NOT EXISTS"

    bare_indexes = re.findall(
        r"create\s+(?:unique\s+)?index\s+(?!if\s+not\s+exists)", stripped, re.IGNORECASE
    )
    assert not bare_indexes, f"{len(bare_indexes)} CREATE INDEX without IF NOT EXISTS"

    bare_add_columns = re.findall(
        r"add\s+column\s+(?!if\s+not\s+exists)", stripped, re.IGNORECASE
    )
    assert not bare_add_columns, f"{len(bare_add_columns)} ADD COLUMN without IF NOT EXISTS"


def test_the_idempotency_check_can_fail(_tmp=None):
    """The regex above must actually reject a bare CREATE TABLE."""
    bad = "CREATE TABLE some_table (id integer);"
    assert re.findall(r"create\s+table\s+(?!if\s+not\s+exists)", bad, re.IGNORECASE)
    good = "CREATE TABLE IF NOT EXISTS some_table (id integer);"
    assert not re.findall(r"create\s+table\s+(?!if\s+not\s+exists)", good, re.IGNORECASE)


def test_the_heatmap_gets_all_seventy_one_probability_columns(_tmp=None):
    """71 clusters, p_hpc_0 .. p_hpc_70. The loop bound is the only place the
    count appears, so an off-by-one here is 70 columns of probabilities and one
    silently absent — exactly the shape of failure this codebase is written
    against."""
    text = BASE_TABLES_SQL.read_text()
    assert re.search(r"FOR\s+i\s+IN\s+0\.\.70\s+LOOP", text), (
        "the p_hpc_* loop does not run 0..70"
    )
    assert "'p_hpc_' || i" in text


def test_row_is_quoted_because_it_is_a_keyword(_tmp=None):
    text = BASE_TABLES_SQL.read_text()
    assert '"row"' in text, 'the tile_coordinates "row" column must be quoted'


# --- ordering inside migrate_all.sql -------------------------------------

def _include_order():
    text = MIGRATE_ALL_SQL.read_text()
    return [m for m in re.findall(r"^\s*\\ir\s+(\S+)", text, re.MULTILINE)]


def test_migrate_all_runs_the_base_tables(_tmp=None):
    assert "migrate_kb_base_tables.sql" in _include_order()


def test_base_tables_run_before_anything_that_alters_them(_tmp=None):
    """migrate_indexes.sql UPDATEs tile_coordinates and wsi_registry;
    migrate_processing_status.sql ALTERs wsi_registry. Both fail on an empty
    database if the base tables have not been created yet — which is the exact
    bug this file fixes, so the ordering is the fix."""
    order = _include_order()
    base = order.index("migrate_kb_base_tables.sql")
    for dependent in ("migrate_indexes.sql", "migrate_processing_status.sql",
                      "migrate_tile_registry_confidence.sql"):
        assert order.index(dependent) > base, (
            f"{dependent} runs before the tables it alters are created"
        )


def test_every_included_migration_exists_on_disk(_tmp=None):
    for name in _include_order():
        assert (BACKEND / name).is_file(), f"migrate_all.sql includes a missing {name}"


# --- the record the DDL was transcribed from -----------------------------

def test_the_schema_snapshot_is_present_and_names_every_live_table(_tmp=None):
    """migrate_kb_base_tables.sql was transcribed by hand from a psql capture.
    If that capture is not in the repo, the DDL has no provenance and the next
    person cannot check it."""
    assert SCHEMA_SNAPSHOT.is_file(), f"{SCHEMA_SNAPSHOT.name} is missing"
    text = SCHEMA_SNAPSHOT.read_text()
    absent = sorted(t for t in LIVE_TABLES if t not in text)
    assert not absent, f"the snapshot does not mention {absent}"


def test_schema_sql_says_it_is_stale(_tmp=None):
    """schema.sql is a 2025-10-23 pg_dump whose tile_registry disagrees with
    the live table on the primary key, on hpc_id's type, and on six columns.
    It is kept for the cluster reference tables, so it has to say so itself."""
    head = SCHEMA_SQL.read_text()[:4000]
    assert "STALE" in head, "schema.sql no longer warns that it is out of date"
    assert "migrate_kb_base_tables.sql" in head, (
        "schema.sql does not point at what to use instead"
    )


# --- the indexes the previews depend on ----------------------------------
#
# Every lookup in the two Knowledge Bank stages filters on a *function* of the
# key — WHERE UPPER(slide_tile) IN :tiles — and Postgres matches an expression
# index by expression, not by value. So a plain index on slide_tile does not
# apply, however normalised the stored values are, and those queries were
# sequential scans: Stage 6's preview chunks 18.5M keys 10,000 at a time, which
# is ~1,850 full scans of an 18.5M-row table to dry-run a load that writes
# nothing.
#
# Nothing about that fails, which is why it needs a test: the only symptom is
# that a preview takes longer than the write it is previewing. If someone edits
# the query to UPPER(TRIM(slide_tile)) — the form used elsewhere in the same
# file — the index silently stops applying and the slowness comes back.

INDEXES_SQL = BACKEND / "migrate_indexes.sql"

_EXPRESSION_LOOKUPS = (
    # (file, expression as the query writes it, table it filters)
    ("load_hpc_assignments.py", "UPPER(slide_tile)", "tile_registry"),
    ("load_hpc_assignments.py", "UPPER(TRIM(slides))", "tile_registry"),
    # register_dataset.py's collision check builds the column name into the
    # expression (f"UPPER({key})"), one query over four tables, so the literal
    # to look for is the template rather than any one column.
    ("register_dataset.py", "UPPER({key})", None),
)

#: The tables register_dataset._foreign_scope reaches through that template, and
#: the column each is keyed by (_KEY_COLUMN). Both slide_tile tables and both
#: slide_id tables therefore need the expression indexed.
_TEMPLATED_TABLES = (
    ("tile_registry", "UPPER(slide_tile)"),
    ("tile_coordinates", "UPPER(slide_tile)"),
    ("wsi_registry", "UPPER(slide_id)"),
    ("wsi_metadata", "UPPER(slide_id)"),
)


def _normalise(sql: str) -> str:
    """Whitespace- and case-insensitive, with SQL comments removed — a comment
    mentioning CREATE INDEX is not a CREATE INDEX, which is exactly what the
    first version of this test tripped over."""
    without_comments = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"\s+", "", without_comments).upper()


def test_every_expression_filtered_on_is_indexed_as_that_expression(_tmp=None):
    indexes = _normalise(INDEXES_SQL.read_text())

    missing = []
    for filename, expression, table in _EXPRESSION_LOOKUPS:
        source = (BACKEND / filename).read_text()
        # The source is Python, so no SQL comment stripping — just whitespace.
        if re.sub(r"\s+", "", expression).upper() not in \
                re.sub(r"\s+", "", source).upper():
            missing.append(f"{filename} no longer filters on {expression}")
            continue
        targets = ([(table, expression)] if table
                   else list(_TEMPLATED_TABLES))
        for target_table, target_expression in targets:
            wanted = _normalise(f"ON {target_table} ({target_expression})")
            if wanted not in indexes:
                missing.append(
                    f"{target_table} is filtered by {target_expression} in "
                    f"{filename}, and migrate_indexes.sql has no index on that "
                    f"expression")
    assert not missing, "; ".join(missing)


def test_the_index_check_can_fail(_tmp=None):
    """An expression nothing indexes must be reported, or the test above is
    just reading a file."""
    indexes = _normalise(INDEXES_SQL.read_text())

    assert _normalise("ON tile_registry (LOWER(slide_tile))") not in indexes


def test_the_new_indexes_are_idempotent_like_the_rest(_tmp=None):
    """This file is run repeatedly against a live database."""
    # Comments stripped first: this file explains itself at length, and a
    # sentence about CREATE INDEX is not one.
    text = re.sub(r"--[^\n]*", "", INDEXES_SQL.read_text())
    creates = re.findall(r"CREATE (?:UNIQUE )?INDEX(?: IF NOT EXISTS)?", text)

    assert creates, "no CREATE INDEX found at all"
    assert all("IF NOT EXISTS" in c for c in creates), \
        "an index is created unguarded; re-running the migration would fail"


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_schema_test_"))
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
