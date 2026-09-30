"""The slide_tile join key, and the tile-name convention it depends on.

One convention has to hold across four places that were written at different
times: Kai's reference CSVs ("18_15.jpeg"), the KB's tile_coordinates and
tile_registry ("..._18_15.JPEG"), the packaged .h5, and the assignments CSV
derived from it. make_hpl_hdf5.py stored "18_15" instead, which broke the last
two against the first two — the Radiogenomics KB load matched 0.0% of 38,892
tiles, and assign_hpc_clusters.py --validate-against would have merged zero rows
against Kai's labels.

These tests pin the convention to Kai's CSV, which is the authority. The short
form is no longer refused — the loaders append the suffix, because the mapping
is a bijection — so what has to be proved here is that the repair reaches the
same key the migrated file would, and that the one unrepairable case (mixed)
still refuses.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

from slide_naming import (  # noqa: E402
    make_slide_tile,
    make_slide_tile_series,
    normalize_tile_names,
    tile_name_verdict,
    tiles_missing_suffix,
)

# A real Radiogenomics slide: spaces, hyphens, and a timestamp whose dots must
# not be mistaken for a file extension.
RADIO_SLIDE = "BB232560 A3-1 - 2023-10-11 16.41.02"


def test_key_matches_the_form_stored_in_the_kb(tmp_path):
    assert (make_slide_tile("TCGA-55-7574-01Z-00-DX1", "18_15.jpeg")
            == "TCGA-55-7574-01Z-00-DX1_18_15.JPEG")


def test_slide_names_containing_dots_keep_their_tile_suffix(tmp_path):
    """The timestamp in a Radiogenomics slide name ends in '.02'. Anything that
    inspects the concatenated key for an extension reads that as one."""
    key = make_slide_tile(RADIO_SLIDE, "24_10.jpeg")
    assert key.endswith("_24_10.JPEG"), key
    assert key == f"{RADIO_SLIDE.upper()}_24_10.JPEG"


def test_key_does_not_invent_a_missing_suffix(tmp_path):
    """The repair belongs in make_hpl_hdf5.py, not here. A key silently fixed
    up would join, while the same file's (slides, tiles) columns still failed
    --validate-against — one symptom cured, the cause left in place."""
    assert make_slide_tile(RADIO_SLIDE, "24_10") == f"{RADIO_SLIDE.upper()}_24_10"


def test_vectorised_matches_the_scalar_definition(tmp_path):
    """The Series path runs over millions of rows; the scalar one is the
    definition. They must not drift."""
    slides = pd.Series([RADIO_SLIDE, "TCGA-55-7574-01Z-00-DX1", " padded ", "x.y.z"])
    tiles = pd.Series(["24_10.jpeg", "18_15.JPEG", " 1_2.jpeg ", "0_0.jpeg"])
    got = make_slide_tile_series(slides, tiles).tolist()
    want = [make_slide_tile(s, t) for s, t in zip(slides, tiles)]
    assert got == want


def test_suffix_guard_can_fire_and_can_pass(tmp_path):
    """The guard has to come out bad on the broken form, or it proves nothing."""
    assert tiles_missing_suffix(pd.Series(["24_10", "25_10", "26_10"]))
    assert not tiles_missing_suffix(pd.Series(["24_10.jpeg", "25_10.jpeg"]))
    # Empty or blank input is not evidence of the bug either way.
    assert not tiles_missing_suffix(pd.Series([], dtype=str))
    assert not tiles_missing_suffix(pd.Series(["", "  "]))


def test_suffix_guard_handles_hdf5_bytes(tmp_path):
    """HDF5 hands back bytes, CSVs hand back str. str(b"18_15.jpeg") is
    "b'18_15.jpeg'" — last character a quote — so a guard that does not decode
    reports a perfectly good .h5 as missing its suffix, and packaging refuses
    every file it should accept."""
    assert not tiles_missing_suffix([b"18_15.jpeg", b"14_6.jpeg"])
    assert tiles_missing_suffix([b"18_15", b"14_6"])
    # Mixed str/bytes must not change the verdict either way.
    assert not tiles_missing_suffix([b"18_15.jpeg", "14_6.jpeg"])


def test_packaged_tile_names_match_kais_reference_csv(tmp_path):
    """The convention is not ours to choose — Kai's CSV is the reference the
    clusters came from, and the acceptance test merges against it on
    (slides, tiles). Read the real file rather than restating its format.
    """
    reference_csv = REPO / "TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv"
    if not reference_csv.is_file():
        return  # not present in every checkout; the other tests still bind this
    head = pd.read_csv(reference_csv, nrows=50)
    assert not tiles_missing_suffix(head["tiles"]), (
        f"Kai's reference CSV tiles look like {head['tiles'].iloc[0]!r}; the "
        f"convention this repo packages to must match it"
    )

    # And what make_hpl_hdf5.py builds must be that same form. Asserted against
    # the function rather than the file's text: a source-string check breaks on
    # any refactor while proving nothing about the value actually stored, and
    # what is stored is separately covered end-to-end by
    # test_packaging_guards.test_packaging_actually_stores_the_suffix.
    from make_hpl_hdf5 import _tile_name

    class _Row:
        col, row = 24, 10

    assert _tile_name(_Row()) == "24_10.jpeg"
    assert not tiles_missing_suffix([_tile_name(_Row())])


def test_loader_repairs_a_csv_from_a_stale_h5(tmp_path):
    """End to end through read_assignments: the suffix-less form must come out
    matching the Knowledge Bank's key, not surface as an unexplained 0% match
    rate and not be turned away at the door."""
    from load_hpc_assignments import read_assignments

    csv_path = tmp_path / "assignments.csv"
    pd.DataFrame({
        "samples": ["BB232560"] * 3,
        "slides": [RADIO_SLIDE] * 3,
        "tiles": ["24_10", "25_10", "26_10"],
        "leiden_2.5": [25, 50, 28],
        "vote_margin": [0.9, 0.4, 0.2],
        "neighbor_distance": [1.0, 2.0, 3.0],
        "hpc_reference": ["hpc_reference_leiden_2p5_fold2"] * 3,
    }).to_csv(csv_path, index=False)

    frame, _cluster_column = read_assignments(csv_path)

    # Loaded rather than refused: nothing in this CSV needs recomputing — the
    # cluster IDs and margins are correct and only the label was short — so the
    # suffix is appended and counted on the way in.
    assert frame["slide_tile"].tolist() == [
        f"{RADIO_SLIDE.upper()}_24_10.JPEG",
        f"{RADIO_SLIDE.upper()}_25_10.JPEG",
        f"{RADIO_SLIDE.upper()}_26_10.JPEG",
    ]
    assert frame.attrs["tile_names_normalized"] == 3

    # The same CSV with the convention applied loads, and builds the KB's key.
    good = pd.read_csv(csv_path)
    good["tiles"] = good["tiles"] + ".jpeg"
    good.to_csv(csv_path, index=False)
    frame, cluster_column = read_assignments(csv_path)
    assert cluster_column == "leiden_2.5"
    assert frame["slide_tile"].iloc[0] == f"{RADIO_SLIDE.upper()}_24_10.JPEG"


# --- the repair, and the case it must refuse ------------------------------


def test_the_verdict_separates_the_three_states(tmp_path):
    assert tile_name_verdict(["24_10", "25_10"]) == "short"
    assert tile_name_verdict([b"24_10.jpeg", "25_10.jpeg"]) == "done"
    assert tile_name_verdict(["24_10", "25_10.jpeg"]) == "mixed"
    # Nothing to append to is not the same as "needs migrating".
    assert tile_name_verdict([]) == "done"
    assert tile_name_verdict(["", "  "]) == "done"


def test_the_verdict_sees_a_mixed_file_the_old_guard_missed(tmp_path):
    """tiles_missing_suffix() reads the first 100 names and answers True only
    if none carry an extension, so one suffixed name anywhere in that window
    makes a half-migrated file look fine. That blind spot is why the verdict
    reads every name."""
    tiles = ["24_10"] * 500 + ["25_10.jpeg"]

    assert tiles_missing_suffix(tiles) is True   # says "just short", wrongly
    assert tile_name_verdict(tiles) == "mixed"   # sees the straddle


def test_the_repair_only_ever_appends(tmp_path):
    names, changed = normalize_tile_names([b"24_10", "25_10.jpeg", "26_10.png"])

    assert names == ["24_10.jpeg", "25_10.jpeg", "26_10.png"]
    assert changed == 1, "a name that already has an extension is not touched"


def test_the_repaired_key_is_the_key_the_kb_stores(tmp_path):
    """The whole justification: repairing on read must land on exactly the key
    a correctly-packaged .h5 would have produced."""
    repaired, _ = normalize_tile_names(["24_10"])

    assert make_slide_tile(RADIO_SLIDE, repaired[0]) == \
        make_slide_tile(RADIO_SLIDE, "24_10.jpeg")


def test_the_loader_refuses_a_half_migrated_csv(tmp_path):
    """The counterpart to test_loader_repairs_a_csv_from_a_stale_h5: repairing
    one side of a straddled resume would attach correct cluster IDs to the
    wrong tiles."""
    from load_hpc_assignments import read_assignments

    csv_path = tmp_path / "half.csv"
    pd.DataFrame({
        "samples": ["BB232560"] * 3,
        "slides": [RADIO_SLIDE] * 3,
        "tiles": ["24_10", "25_10.jpeg", "26_10"],
        "leiden_2.5": [25, 50, 28],
        "vote_margin": [0.9, 0.4, 0.2],
        "neighbor_distance": [1.0, 2.0, 3.0],
        "hpc_reference": ["hpc_reference_leiden_2p5_fold2"] * 3,
    }).to_csv(csv_path, index=False)

    try:
        read_assignments(csv_path)
    except SystemExit as e:
        assert "Repackage" in str(e), str(e)
    else:
        raise AssertionError("a half-migrated CSV must be refused")


def test_the_loader_and_the_migrator_agree_on_every_verdict(tmp_path):
    """Two implementations of one rule, so they are pinned to each other. If
    they drift, a file the loader repairs is one the migrator calls mixed, or
    the other way round."""
    from migrate_tile_names import classify

    for tiles in (["24_10", "25_10"],
                  ["24_10.jpeg", "25_10.jpeg"],
                  ["24_10", "25_10.jpeg"],
                  [b"24_10", b"25_10.jpeg"]):
        assert tile_name_verdict(tiles) == classify(tiles)[0], tiles

    # One deliberate difference, pinned so it stays deliberate: classify() asks
    # whether the name ends in ".jpeg" (it is deciding what to write), while the
    # verdict asks whether it has any extension at all (it is deciding whether
    # to append). A .png is "already suffixed, leave it" to the loader and
    # "not a .jpeg" to the migrator. Nothing in this pipeline writes one —
    # auto_tile_from_mask.py saves JPEG — so the case is theoretical, but the
    # safe direction is the loader's: never append to a name that has one.
    assert tile_name_verdict(["24_10.png"]) == "done"
    assert classify(["24_10.png"])[0] == "short"


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_key_test_"))
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
