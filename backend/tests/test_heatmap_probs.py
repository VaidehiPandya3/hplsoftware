"""The heatmap overlay must cost one slide, not the whole table.

/slide/{id}/tiles_meta merges tile_hpc_heatmap's 71 p_hpc_* columns into its
response. It used to get them from `SELECT * FROM tile_hpc_heatmap` — all
149 MB, cached per target for the life of the process, read lazily on the first
tiles_meta request. Two consequences, one of which stopped the viewer working:

  * server startup read 149 MB it might never use;
  * run against the database through an SSH tunnel — which is how the UI is
    used from a laptop — that read cannot finish inside the client's 30-second
    timeout, and because it is lazy, the *first* slide anyone opens is the one
    that fails. The traceback looks like a slow query on tile_coordinates,
    which was also true and also fixed, and is not this.

The table's primary key is slide_tile and no request has ever needed another
slide's rows, so the fix is to fetch by key. These tests pin the shape of that:
filtered, cached-when-absent, and never the whole table.
"""

import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, text  # noqa: E402

import tile_server_v2_ as srv  # noqa: E402

TILES = ["SLIDE-A_0_0.JPEG", "SLIDE-A_1_1.JPEG"]


def _heatmap_db(tmp_path, rows=TILES, table="tile_hpc_heatmap"):
    """A stand-in heatmap with three of the 71 probability columns."""
    engine = create_engine(f"sqlite:///{tmp_path}/heatmap.db")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE {table} (slide_tile TEXT PRIMARY KEY, "
                          f"p_hpc_0 REAL, p_hpc_1 REAL, p_hpc_2 REAL)"))
        for i, tile in enumerate(rows):
            conn.execute(text(f"INSERT INTO {table} VALUES (:t, :a, :b, :c)"),
                         {"t": tile, "a": 0.1 * i, "b": 0.2, "c": 0.7})
    return engine


def _with_engine(engine):
    """Point the server's engine factory at a stand-in, and clear its caches."""
    original = srv._get_engine
    srv._get_engine = lambda kb_target=srv.KB_PRODUCTION: engine
    srv._heatmap_columns_cache.clear()
    return original


def _restore(original):
    srv._get_engine = original
    srv._heatmap_columns_cache.clear()


def test_only_the_slides_own_rows_are_fetched(tmp_path):
    other = ["SLIDE-B_9_9.JPEG"]
    engine = _heatmap_db(tmp_path, rows=TILES + other)
    original = _with_engine(engine)
    try:
        probs = srv._heatmap_probs_for_tiles(srv.KB_PRODUCTION, TILES)
    finally:
        _restore(original)

    assert probs is not None
    assert sorted(probs["slide_tile"]) == sorted(TILES), "another slide came back"
    assert [c for c in probs.columns if c.startswith("p_hpc_")] == \
        ["p_hpc_0", "p_hpc_1", "p_hpc_2"]


def test_a_missing_table_is_cached_as_absent(tmp_path):
    """Nothing writes this table, so on the test KB it is absent — and a probe
    per viewer open would be a failed query on every request."""
    engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
    original = _with_engine(engine)
    try:
        assert srv._heatmap_columns(srv.KB_PRODUCTION) is None
        assert srv._heatmap_probs_for_tiles(srv.KB_PRODUCTION, TILES) is None
        target = srv._resolve_kb_target(srv.KB_PRODUCTION)
        assert target in srv._heatmap_columns_cache, "absence was not cached"
        assert srv._heatmap_columns_cache[target] is None
    finally:
        _restore(original)


def test_a_table_with_no_probability_columns_is_treated_as_absent(tmp_path):
    """An empty shell of a table must cost the overlay, not the metadata."""
    engine = create_engine(f"sqlite:///{tmp_path}/shell.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE tile_hpc_heatmap (slide_tile TEXT)"))
    original = _with_engine(engine)
    try:
        assert srv._heatmap_columns(srv.KB_PRODUCTION) is None
        assert srv._heatmap_probs_for_tiles(srv.KB_PRODUCTION, TILES) is None
    finally:
        _restore(original)


def test_no_tiles_means_no_query(tmp_path):
    """A slide with no tile_coordinates rows must not ask for every key."""
    engine = _heatmap_db(tmp_path)
    original = _with_engine(engine)
    try:
        assert srv._heatmap_probs_for_tiles(srv.KB_PRODUCTION, []) is None
    finally:
        _restore(original)


def test_tiles_with_no_heatmap_rows_return_none_rather_than_empty(tmp_path):
    """Every cohort loaded since the notebook is in this state, so it is the
    normal path, not an edge case: the merge is skipped entirely."""
    engine = _heatmap_db(tmp_path)
    original = _with_engine(engine)
    try:
        assert srv._heatmap_probs_for_tiles(
            srv.KB_PRODUCTION, ["RADIOGENOMICS-X_3_3.JPEG"]) is None
    finally:
        _restore(original)


# --- what must not come back ----------------------------------------------

def test_the_whole_table_is_never_read(tmp_path):
    """The regression itself: every query must be bounded.

    Asserted on the SQL actually issued rather than on the source text. The two
    are not the same check — the merge produces identical output either way, so
    the cost is invisible in the result, and a docstring that quotes the old
    query is not the old query. Recording what reaches the database is the only
    version of this test that cannot be fooled by prose.
    """
    engine = _heatmap_db(tmp_path)
    original_engine = _with_engine(engine)
    original_read = srv.pd.read_sql
    issued = []

    def recording_read_sql(sql, con, **kwargs):
        issued.append(str(sql))
        return original_read(sql, con, **kwargs)

    srv.pd.read_sql = recording_read_sql
    try:
        srv._heatmap_columns(srv.KB_PRODUCTION)
        srv._heatmap_probs_for_tiles(srv.KB_PRODUCTION, TILES)
    finally:
        srv.pd.read_sql = original_read
        _restore(original_engine)

    heatmap_queries = [q for q in issued if "tile_hpc_heatmap" in q]
    assert heatmap_queries, "nothing queried the heatmap at all"
    for query in heatmap_queries:
        bounded = "LIMIT 0" in query or "WHERE slide_tile IN" in query
        assert bounded, f"unbounded read of the heatmap: {query[:120]}"


def test_startup_does_not_warm_the_probabilities(_tmp=None):
    """Startup may learn the column names; it must not read the rows."""
    source = (BACKEND / "tile_server_v2_.py").read_text()
    lifespan = source[source.index("async def lifespan"):
                      source.index("async def lifespan") + 1600]
    assert "_heatmap_columns(KB_PRODUCTION)" in lifespan
    assert "_load_heatmap_probs" not in source, (
        "the whole-table loader is back")


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="hpl_heatmap_test_")))
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
