"""The box the viewer draws must be the size of the tile underneath it.

Tiles are tessellated at 1.8 µm/px, so their stride in native pixels is
`round(224 * 1.8 / native_mpp)` — derived per slide, because native mpp is a
property of the scan. The tile server asserted 0.252 µm/px for every slide and
published `tile_size_native: 1600` from a module constant, which every overlay
in both UIs uses to size its rectangles and the tile inspector uses to crop its
region.

Only 520 of the 1,598 slides in wsi_metadata are 0.252 µm/px. On the 468 finer
ones the grid was drawn undersized and opened a visible gutter between boxes
(10.1% of the pitch at 0.2265); on the 610 coarser ones — the 20x scans at
~0.50 — the boxes came out at roughly twice the tile and overlapped their
neighbours. Nothing failed: the overlay rendered, and every box sat on a real
tile at the right origin.

These tests pin the size, where the number is allowed to come from, and the
grid indices the adjacency map is built on. Several assert the *old* behaviour
was wrong, because a test that only shows the fixed code agreeing with itself
would have passed before the fix on the one slide anyone checked.
"""

import re
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import pandas as pd  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

import tile_server_v2_ as srv  # noqa: E402
from auto_tile_from_mask import (DEFAULT_NATIVE_MPP, TARGET_MPP,  # noqa: E402
                                 TARGET_TILE_PX, native_tile_px,
                                 parse_native_mpp)

SLIDE = "TCGA-05-4244-01Z-00-DX1"

# Real values out of wsi_metadata.csv, one per failure mode.
FINER_MPP = 0.2325      # 1,734 px pitch — the gapped screenshot
ASSUMED_MPP = 0.2520    # 1,600 px pitch — the only case the constant fitted
COARSER_MPP = 0.5015    # 804 px pitch — 20x, boxes twice the tile
SCANNER_MPPS = [0.2265, FINER_MPP, 0.2465, ASSUMED_MPP, 0.4939, COARSER_MPP]


class FakeSlide:
    """Just the OpenSlide surface the pitch and info paths touch."""

    def __init__(self, mpp):
        self.properties = {} if mpp is None else {
            "openslide.mpp-x": str(mpp), "openslide.mpp-y": str(mpp)}
        self.level_count = 1
        self.level_dimensions = [(60000, 40000)]


def _coords_db(tmp_path, rows, name="coords"):
    """A stand-in tile_coordinates: (slides, col, row, x_native, y_native)."""
    engine = create_engine(f"sqlite:///{tmp_path}/{name}.db")
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE tile_coordinates ('
                          'slide_tile TEXT PRIMARY KEY, slides TEXT, '
                          '"col" INTEGER, "row" INTEGER, '
                          'x_native INTEGER, y_native INTEGER)'))
        for slides, col, row, x, y in rows:
            conn.execute(
                text('INSERT INTO tile_coordinates VALUES (:t, :s, :c, :r, :x, :y)'),
                {"t": f"{slides}_{col}_{row}.JPEG", "s": slides,
                 "c": col, "r": row, "x": x, "y": y})
    return engine


def _empty_db(tmp_path):
    """A KB with no tile_coordinates at all — a slide registered but not tiled."""
    return create_engine(f"sqlite:///{tmp_path}/empty.db")


def _with(engine, slide):
    """Point the server's engine factory and slide opener at stand-ins."""
    originals = (srv._get_engine, srv._open_slide)
    srv._get_engine = lambda kb_target=srv.KB_PRODUCTION: engine
    srv._open_slide = lambda slide_id, kb_target=srv.KB_PRODUCTION: slide
    srv._tile_pitch_cache.clear()
    return originals


def _restore(originals):
    srv._get_engine, srv._open_slide = originals
    srv._tile_pitch_cache.clear()


def _tiles_along_a_row(pitch, n, slides=SLIDE):
    """What auto_tile_from_mask writes for one row of a tessellation."""
    return [(slides, col, 0, col * pitch, 0) for col in range(n)]


# --- the size itself -----------------------------------------------------

def test_the_grid_closes_at_every_scanner_resolution(tmp_path):
    """Each box's right edge must be the next box's left edge — no gutter, no
    overlap — whatever the scan's mpp."""
    for mpp in SCANNER_MPPS:
        pitch = native_tile_px(mpp)
        originals = _with(_empty_db(tmp_path), FakeSlide(mpp))
        try:
            size, _ = srv._tile_size_native(SLIDE)
        finally:
            _restore(originals)

        edges = [(col * pitch, col * pitch + size) for col in range(8)]
        for (_, right), (left, _) in zip(edges, edges[1:]):
            assert right == left, (
                f"mpp {mpp}: box ends at {right}, next tile starts at {left} "
                f"({left - right:+d} px)")


def test_the_old_constant_gapped_and_overlapped_real_slides(_tmp=None):
    """The failure this fixes, stated as an assertion: a fixed 1600 px box
    does not fit a slide that was not scanned at 0.252 µm/px."""
    assert native_tile_px(ASSUMED_MPP) == srv.TILE_SIZE_NATIVE == 1600

    gap = native_tile_px(FINER_MPP) - srv.TILE_SIZE_NATIVE
    assert gap == 134, f"expected a 134 px gutter at {FINER_MPP}, got {gap}"

    overlap = srv.TILE_SIZE_NATIVE - native_tile_px(COARSER_MPP)
    assert overlap == 796, f"expected 796 px of overlap at {COARSER_MPP}"


def test_the_pitch_is_the_tilers_own_arithmetic(_tmp=None):
    """One definition, imported rather than restated. A second copy would be
    correct on the day it was written and silent when either number moved."""
    tiler = (BACKEND / "auto_tile_from_mask.py").read_text()
    assert "tile_px_native = native_tile_px(" in tiler, (
        "the tiler no longer derives its stride from the shared function")

    server = (BACKEND / "tile_server_v2_.py").read_text()
    imports = server[server.index("from auto_tile_from_mask import"):]
    assert "native_tile_px" in imports[:imports.index("\n\n")], \
        "the server rebuilt its own copy of the arithmetic"
    assert native_tile_px(FINER_MPP) == round(TARGET_TILE_PX * TARGET_MPP / FINER_MPP)


# --- where the number is allowed to come from ----------------------------

def test_the_tiles_own_coordinates_beat_the_slides_metadata(tmp_path):
    """x_native = col * pitch, so the rows carry the stride Stage 1 actually
    used. That beats re-deriving it: a cohort tiled at a non-default
    target_mpp, or before a default moved, still gets a grid that closes."""
    pitch = native_tile_px(FINER_MPP)
    engine = _coords_db(tmp_path, _tiles_along_a_row(pitch, 12))
    # The slide claims the assumed mpp; the tiles say otherwise.
    originals = _with(engine, FakeSlide(ASSUMED_MPP))
    try:
        size, source = srv._tile_size_native(SLIDE)
    finally:
        _restore(originals)

    assert size == pitch, f"read {size}, tiles were cut at {pitch}"
    assert source == "tile_coordinates"


def test_two_pitches_in_one_slide_are_refused(tmp_path):
    """A slide tiled twice at different strides has no single right box size.
    Picking the commoner one draws a grid that fits part of the slide, which
    is the same class of bug with a smaller blast radius."""
    rows = _tiles_along_a_row(1600, 6) + [
        (SLIDE, col, 9, col * 1734, 9 * 1734) for col in range(1, 6)]
    originals = _with(_coords_db(tmp_path, rows), FakeSlide(FINER_MPP))
    try:
        assert srv._pitch_from_coordinates(SLIDE, srv.KB_PRODUCTION) is None
        size, source = srv._tile_size_native(SLIDE)
    finally:
        _restore(originals)

    assert source == "slide mpp", "a contradictory table was read anyway"
    assert size == native_tile_px(FINER_MPP)


def test_an_untiled_slide_falls_back_to_its_own_mpp(tmp_path):
    """Registered, not yet tiled: there are no coordinates to read, and the
    slide's mpp is still a better answer than a constant."""
    originals = _with(_empty_db(tmp_path), FakeSlide(FINER_MPP))
    try:
        size, source = srv._tile_size_native(SLIDE)
    finally:
        _restore(originals)

    assert (size, source) == (native_tile_px(FINER_MPP), "slide mpp")


def test_a_slide_with_no_usable_mpp_says_it_is_guessing(tmp_path):
    """The documented fallback, reported as a fallback rather than as fact."""
    for claimed in (None, "", "0", "not a number"):
        originals = _with(_empty_db(tmp_path), FakeSlide(claimed))
        try:
            size, source = srv._tile_size_native(SLIDE)
        finally:
            _restore(originals)

        assert size == srv.TILE_SIZE_NATIVE
        assert source == f"default mpp {DEFAULT_NATIVE_MPP}", \
            f"mpp {claimed!r} was presented as the slide's own"
        assert parse_native_mpp(claimed) is None


def test_a_zero_mpp_does_not_divide_by_zero(_tmp=None):
    """openslide.mpp-x is whatever the scanner wrote. This used to reach the
    division as float('0')."""
    assert parse_native_mpp("0") is None
    assert parse_native_mpp("-1") is None
    assert parse_native_mpp("nan") is None


def test_the_coordinate_pitch_is_cached_but_the_guess_is_not(tmp_path):
    """A slide gets a guess today and the real answer once Stage 5 runs. The
    guess must not outlive the registration that replaced it."""
    originals = _with(_empty_db(tmp_path), FakeSlide(ASSUMED_MPP))
    try:
        assert srv._tile_size_native(SLIDE)[1] == "slide mpp"
    finally:
        srv._get_engine, srv._open_slide = originals

    pitch = native_tile_px(FINER_MPP)
    engine = _coords_db(tmp_path, _tiles_along_a_row(pitch, 8), name="later")
    srv._get_engine = lambda kb_target=srv.KB_PRODUCTION: engine
    srv._open_slide = lambda slide_id, kb_target=srv.KB_PRODUCTION: FakeSlide(ASSUMED_MPP)
    try:
        assert srv._tile_size_native(SLIDE) == (pitch, "tile_coordinates")
    finally:
        _restore(originals)


# --- what the UIs are handed ---------------------------------------------

def test_slide_info_publishes_the_slides_own_size(tmp_path):
    """Both viewers size every rectangle from this payload."""
    pitch = native_tile_px(FINER_MPP)
    engine = _coords_db(tmp_path, _tiles_along_a_row(pitch, 10))
    originals = _with(engine, FakeSlide(FINER_MPP))
    try:
        info = srv.slide_info(SLIDE, kb_target=srv.KB_PRODUCTION)
    finally:
        _restore(originals)

    assert info["tile_size_native"] == pitch
    assert info["tile_size_native_source"] == "tile_coordinates"
    # The 5x tile is still 224 px; what changed is how much native slide it
    # covers, so the scale has to move with it or the two disagree.
    assert info["tile_size_5x"] == TARGET_TILE_PX
    assert info["scale_5x_to_native"] == pitch / TARGET_TILE_PX


# --- the grid the adjacency map is built on ------------------------------

def _row_of_alternating_tiles(pitch, n):
    df = pd.DataFrame(_tiles_along_a_row(pitch, n),
                      columns=["slides", "col", "row", "x_native", "y_native"])
    df["slide_tile"] = [f"{SLIDE}_{c}_0.JPEG" for c in df["col"]]
    df["hpc_id"] = [1 + (c % 2) for c in df["col"]]
    return df


def test_adjacency_counts_every_touching_pair_on_a_finer_scan(_tmp=None):
    """n alternating tiles in a row touch n-1 times."""
    pitch = native_tile_px(FINER_MPP)
    df = _row_of_alternating_tiles(pitch, 24)
    counts, _ = srv._compute_adjacency(df, pitch)
    assert counts.get("1_2") == 23, counts


def test_dividing_by_the_wrong_pitch_loses_neighbours(_tmp=None):
    """Why col/row is read rather than rederived. x_native // 1600 on a 1734
    pitch drifts a cell every ~12 columns, and the two tiles either side of
    the skipped cell stop being neighbours — an adjacency map that is quietly
    short, on a grid that still looks like a grid."""
    pitch = native_tile_px(FINER_MPP)
    df = _row_of_alternating_tiles(pitch, 24).drop(columns=["col", "row"])
    counts, _ = srv._compute_adjacency(df, srv.TILE_SIZE_NATIVE)

    assert counts.get("1_2", 0) < 23, (
        "the wrong pitch cost nothing here, so this test proves nothing")
    # And the stored grid, which the endpoint now selects, is unaffected by it.
    stored, _ = srv._compute_adjacency(_row_of_alternating_tiles(pitch, 24),
                                       srv.TILE_SIZE_NATIVE)
    assert stored.get("1_2") == 23


def test_the_adjacency_query_selects_the_stored_grid(_tmp=None):
    """col/row have to reach _compute_adjacency for it to prefer them."""
    source = (BACKEND / "tile_server_v2_.py").read_text()
    endpoint = source[source.index("def slide_adjacency"):]
    endpoint = endpoint[:endpoint.index("@app.get", 1)]
    assert 'tc."col", tc."row"' in endpoint, "the grid is not selected"
    assert "_tile_size_native(slide_id, kb_target)" in endpoint, (
        "the fallback pitch is still the module constant")


# --- what the pyramid viewer hovers against ------------------------------
#
# The same pitch, now answering "which tile is under the pointer" rather than
# "how big is the box". The drawing side is above; this is the index the
# viewer looks a pointer up in, and it fails the same quiet way — a tile named
# for a point that is actually in its neighbour reads exactly like a correct
# answer. app_v28.py imports streamlit at module level, so the builder is
# pulled out by source and exec'd against stubs, the same way
# test_pipeline_steps.py takes _pipeline_steps.


def _build_osd_tile_index():
    source = (BACKEND.parent / "app" / "app_v28.py").read_text()
    start = source.index("def build_osd_tile_index")
    end = source.index("\ndef ", start)
    namespace = {"pd": pd}
    exec(compile(source[start:end], "probe", "exec"), namespace)
    return namespace["build_osd_tile_index"]


def _frame(pitch, rows):
    return pd.DataFrame([
        {"col": c, "row": r, "x_native": c * pitch, "y_native": r * pitch,
         "slide_tile": f"{SLIDE}_{c}_{r}.JPEG", "hpc_id": hpc}
        for c, r, hpc in rows
    ])


def test_the_hover_index_carries_the_grid_and_the_label(_tmp=None):
    """One compact row per tile: [col, row, slide_tile, hpc_id]. The viewer
    keys on col/row and prints the other two, so a column dropped here is a
    readout that names nothing."""
    index = _build_osd_tile_index()(_frame(1734, [(0, 0, 12), (1, 0, 7)]), 1734)
    assert index == [
        [0, 0, f"{SLIDE}_0_0.JPEG", 12],
        [1, 0, f"{SLIDE}_1_0.JPEG", 7],
    ], index


def test_an_unlabelled_tile_stays_in_the_index(_tmp=None):
    """A tile registered but not yet loaded into the KB has hpc_id NULL. It is
    still a tile, and hovering it must say so rather than report empty space —
    that state is every tile of a cohort between Stage 5 and Stage 6."""
    frame = _frame(1734, [(0, 0, 12)])
    frame.loc[0, "hpc_id"] = float("nan")
    index = _build_osd_tile_index()(frame, 1734)
    assert index == [[0, 0, f"{SLIDE}_0_0.JPEG", None]], index


def test_the_index_is_not_capped_like_the_overlay(_tmp=None):
    """build_osd_overlay_records stops at 6,000 because drawing that many SVG
    rects is slow. A Map lookup is not, and a pointer over tile 8,000 has to be
    answered — silently dropping those is a viewer that goes blank in one
    corner of large slides."""
    rows = [(c, r, 1) for c in range(90) for r in range(90)]  # 8,100 tiles
    index = _build_osd_tile_index()(_frame(1734, rows), 1734)
    assert len(index) == len(rows), len(index)


def test_a_frame_without_the_grid_falls_back_to_its_coordinates(_tmp=None):
    """tiles_meta selects col/row, but a caller passing a frame that lacks them
    must still land in the lattice — the same pitch the viewer hit-tests with,
    so both sides agree about which cell a tile is."""
    frame = pd.DataFrame([{"x_native": 2 * 1734, "y_native": 1734,
                           "slide_tile": f"{SLIDE}_2_1.JPEG", "hpc_id": 3}])
    index = _build_osd_tile_index()(frame, 1734)
    assert index == [[2, 1, f"{SLIDE}_2_1.JPEG", 3]], index


def test_the_fallback_at_the_wrong_pitch_files_the_tile_in_the_wrong_cell(_tmp=None):
    """Why that fallback divides by the resolved pitch and not a constant.

    Column 12 is where 1,734 and the old constant 1,600 first disagree —
    20,808 px is column 12 of the grid the tiler actually cut and column 13 to
    anything reading it at 1,600. The same off-by-one, one column later each
    twelfth column, is what _compute_adjacency's wrong pitch did to
    neighbours; here it means the pointer over a tile is told about the tile
    beside it.
    """
    frame = pd.DataFrame([{"x_native": 12 * 1734, "y_native": 1734,
                           "slide_tile": f"{SLIDE}_12_1.JPEG", "hpc_id": 3}])
    assert _build_osd_tile_index()(frame, 1734)[0][:2] == [12, 1]
    assert _build_osd_tile_index()(frame, 1600)[0][:2] == [13, 1]


# --- how heavily the grid is outlined -------------------------------------
#
# Two viewers draw the same tile boxes, and the numbers deciding how heavy the
# outline is live in two files: the Streamlit iframe's inline JS and
# overlayBuilders.js. Nothing imports one from the other — the React app is a
# port, not a shared library — so a value changed in one and not the other is a
# grid that reads differently depending on which UI you opened, with no error
# anywhere. This pins them together.

_STROKE_CONSTANTS = ("GRID_STROKE_FRACTION", "GRID_STROKE_MIN", "GRID_STROKE_MAX",
                     "GRID_HALO_COLOR", "GRID_HALO_EXTRA")


def _declared_constants(source: str) -> dict:
    """<NAME> = <value>; for the constants above, whichever file they are in."""
    found = {}
    for name in _STROKE_CONSTANTS:
        match = re.search(rf"(?:export\s+)?const {name} = ([^;\n]+)", source)
        if match:
            found[name] = match.group(1).strip()
    return found


def test_both_viewers_outline_the_grid_identically(_tmp=None):
    """A weight changed in one UI and not the other is a grid that looks
    different depending on which one you opened."""
    streamlit = _declared_constants((BACKEND.parent / "app" / "app_v28.py").read_text())
    react = _declared_constants(
        (BACKEND.parent / "frontend" / "src" / "components" / "viewer"
         / "overlayBuilders.js").read_text())

    assert set(streamlit) == set(_STROKE_CONSTANTS), f"Streamlit is missing {set(_STROKE_CONSTANTS) - set(streamlit)}"
    assert set(react) == set(_STROKE_CONSTANTS), f"React is missing {set(_STROKE_CONSTANTS) - set(react)}"
    assert streamlit == react, f"the two viewers disagree: {streamlit} vs {react}"


def test_the_outline_is_a_fraction_of_the_tile_not_a_pixel_count(_tmp=None):
    """The rule this replaced was min(4, w/35): right up to a ~140 px tile and
    a hairline past it, because the cap held while the cell kept growing. The
    click inspector draws ~4.2% of a tile at any scale, and matching that is
    the whole point."""
    react = _declared_constants(
        (BACKEND.parent / "frontend" / "src" / "components" / "viewer"
         / "overlayBuilders.js").read_text())
    fraction = float(react["GRID_STROKE_FRACTION"])
    ceiling = float(react["GRID_STROKE_MAX"])

    assert 0.03 <= fraction <= 0.06, fraction
    # The ceiling has to sit far enough out that an ordinary zoom never reaches
    # it — otherwise this is the old rule with a bigger number.
    assert ceiling / fraction > 140, (
        "the cap bites before a tile is 140 px across, which is where the old "
        "rule started thinning out")


def test_the_viewer_is_handed_the_index_and_the_pitch(_tmp=None):
    """A lookup nothing passes to the viewer answers no pointer at all."""
    source = (BACKEND.parent / "app" / "app_v28.py").read_text()
    call = source[source.index("render_openseadragon_viewer(\n"):]
    call = call[:call.index(")\n")]
    assert "tile_index=build_osd_tile_index(df, tile_size_native)" in call, call
    assert "tile_size_native=tile_size_native" in call, call


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="hpl_pitch_test_")))
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
