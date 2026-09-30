"""Migrating tile names in artifacts packaged before the suffix fix.

This is a data migration on files the pipeline is judged by, so the tests are
mostly about what it must REFUSE to do, and about the one thing that made the
original bug invisible: `tiles` is a fixed-length byte column, so a rewrite that
does not widen the dtype truncates the suffix straight back off and reports
success.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from migrate_tile_names import classify, migrate_csv, migrate_h5  # noqa: E402
from slide_naming import tiles_missing_suffix  # noqa: E402

SCRIPT = BACKEND / "migrate_tile_names.py"
SLIDE = "BB232560 A3-1 - 2023-10-11 16.41.02"


def _write_csv(path: Path, tiles, rows=None):
    n = rows or len(tiles)
    pd.DataFrame({
        "samples": ["BB232560"] * n,
        "slides": [SLIDE] * n,
        "tiles": tiles,
        "leiden_2.5": list(range(n)),
        "vote_margin": [0.5] * n,
        "neighbor_distance": [1.0] * n,
        "hpc_reference": ["hpc_reference_leiden_2p5_fold2"] * n,
    }).to_csv(path, index=False)


def _write_h5(path: Path, tiles):
    width = max(len(t) for t in tiles)
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (len(tiles), 2, 2, 3), dtype="uint8")
        f.create_dataset("samples", data=np.array([b"BB232560"] * len(tiles)))
        f.create_dataset("slides", data=np.array([SLIDE.encode()] * len(tiles)))
        f.create_dataset(
            "tiles",
            data=np.array([t.encode() for t in tiles], dtype=f"S{width}"),
        )


def test_classify_tells_the_three_states_apart(tmp_path):
    assert classify(["24_10", "25_10"])[0] == "short"
    assert classify(["24_10.jpeg", "25_10.jpeg"])[0] == "done"
    assert classify(["24_10", "25_10.jpeg"])[0] == "mixed"


def test_csv_migration_preserves_every_other_column(tmp_path):
    """The cluster IDs, margins and distances are correct already — a migration
    that altered any of them would be a re-run wearing a disguise."""
    csv = tmp_path / "a.csv"
    _write_csv(csv, ["24_10", "25_10", "7_3"])
    before = pd.read_csv(csv)

    result = migrate_csv(csv, commit=True)
    after = pd.read_csv(result["output"])

    assert result["written"] and result["verdict"] == "short"
    assert after["tiles"].tolist() == ["24_10.jpeg", "25_10.jpeg", "7_3.jpeg"]
    for column in ("samples", "slides", "leiden_2.5", "vote_margin",
                   "neighbor_distance", "hpc_reference"):
        assert after[column].tolist() == before[column].tolist(), column
    # And the input is left alone.
    assert pd.read_csv(csv)["tiles"].tolist() == ["24_10", "25_10", "7_3"]


def test_csv_migration_is_a_no_op_when_already_correct(tmp_path):
    csv = tmp_path / "b.csv"
    _write_csv(csv, ["24_10.jpeg", "25_10.jpeg"])
    result = migrate_csv(csv, commit=True)
    assert result["verdict"] == "done" and not result["written"]
    assert not (tmp_path / "b_tilenames.csv").exists()


def test_h5_migration_widens_the_fixed_length_column(tmp_path):
    """The bug that made the original fix inert, in migration form. The stored
    dtype was sized for "24_10"; assigning "24_10.jpeg" into it truncates back
    to "24_10" and every check downstream still passes."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, ["24_10", "25_10", "7_3"])

    with h5py.File(h5_path, "r") as f:
        assert f["tiles"].dtype.itemsize == 5, "fixture must start narrow"

    result = migrate_h5(h5_path, commit=True)
    assert result["written"]

    with h5py.File(h5_path, "r") as f:
        stored = [t.decode() for t in f["tiles"][:]]
        assert f["tiles"].dtype.itemsize >= 10, f["tiles"].dtype
    assert stored == ["24_10.jpeg", "25_10.jpeg", "7_3.jpeg"], stored
    assert not tiles_missing_suffix([s.encode() for s in stored])


def test_h5_migration_keeps_row_count_and_siblings_aligned(tmp_path):
    """image_index is the row position in this file, so a migration that
    reordered or dropped a row would silently repoint every tile in the KB."""
    h5_path = tmp_path / "packaged.h5"
    names = [f"{c}_{r}" for c in range(4) for r in range(5)]
    _write_h5(h5_path, names)

    migrate_h5(h5_path, commit=True)

    with h5py.File(h5_path, "r") as f:
        assert [t.decode() for t in f["tiles"][:]] == [n + ".jpeg" for n in names]
        assert f["samples"].shape[0] == len(names)
        assert f["slides"].shape[0] == len(names)
        assert f["img"].shape[0] == len(names)


def test_mixed_names_are_refused_not_half_migrated(tmp_path):
    """A .h5 written by a resume straddling the fix holds both forms, and the
    two halves cannot be told apart from their names. Patching it would leave a
    file that looks migrated and is not."""
    h5_path = tmp_path / "mixed.h5"
    _write_h5(h5_path, ["24_10", "25_10.jpeg", "7_3"])
    assert migrate_h5(h5_path, commit=True)["verdict"] == "mixed"
    with h5py.File(h5_path, "r") as f:  # untouched
        assert [t.decode() for t in f["tiles"][:]] == ["24_10", "25_10.jpeg", "7_3"]

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--h5", str(h5_path), "--commit"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert out.returncode == 2, out.stdout
    assert "MIXED" in out.stdout and "Repackage" in out.stdout


def test_dry_run_writes_nothing(tmp_path):
    csv = tmp_path / "c.csv"
    _write_csv(csv, ["24_10", "25_10"])
    h5_path = tmp_path / "c.h5"
    _write_h5(h5_path, ["24_10", "25_10"])

    assert not migrate_csv(csv, commit=False)["written"]
    assert not (tmp_path / "c_tilenames.csv").exists()

    assert not migrate_h5(h5_path, commit=False)["written"]
    with h5py.File(h5_path, "r") as f:
        assert [t.decode() for t in f["tiles"][:]] == ["24_10", "25_10"]

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--csv", str(csv)],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert out.returncode == 0, out.stderr
    assert "Dry run" in out.stdout


def _write_headerless(path: Path, tiles, reference="hpc_reference_leiden_2p5_fold2"):
    """An assignments CSV with no header row, as found in the wild."""
    import csv as csv_module
    with path.open("w", newline="") as f:
        csv_module.writer(f).writerows(
            [["BB232073", SLIDE, t, "28", "0.736", "7.2785106", reference]
             for t in tiles]
        )


def test_headerless_csv_is_recovered_without_losing_a_row(tmp_path):
    """The reason this is not cosmetic: pandas reads row 1 as the header, so a
    headerless CSV silently loses its first tile AND names every column after
    that row's values."""
    from migrate_tile_names import detect_headerless

    csv = tmp_path / "headerless.csv"
    tiles = [f"{c}_21" for c in range(21, 26)]
    _write_headerless(csv, tiles)

    # What the default read does to it — one row short.
    assert len(pd.read_csv(csv)) == len(tiles) - 1

    names = detect_headerless(csv)
    assert names == ["samples", "slides", "tiles", "leiden_2.5", "vote_margin",
                     "neighbor_distance", "hpc_reference"], names

    result = migrate_csv(csv, commit=True)
    after = pd.read_csv(result["output"])
    assert len(after) == len(tiles), "every row must survive"
    assert after["tiles"].tolist() == [t + ".jpeg" for t in tiles]


def test_a_real_header_is_not_mistaken_for_data(tmp_path):
    """detect_headerless guesses column names by position, so it must only fire
    where the first row cannot possibly be a header."""
    from migrate_tile_names import detect_headerless

    csv = tmp_path / "proper.csv"
    _write_csv(csv, ["24_10", "25_10"])
    assert detect_headerless(csv) is None

    # Nor on something that merely has seven columns.
    other = tmp_path / "other.csv"
    pd.DataFrame({c: ["x"] for c in "abcdefg"}).to_csv(other, index=False)
    assert detect_headerless(other) is None


def test_cluster_column_is_derived_from_the_reference_or_asked_for(tmp_path):
    """The cluster column is named for the reference's groupby, which a
    headerless file does not record. Deriving it from the reference name is an
    inference, so it must be reported and overridable — and must refuse rather
    than invent when the name does not say."""
    from migrate_tile_names import cluster_column_from_reference, detect_headerless

    assert cluster_column_from_reference("hpc_reference_leiden_2p5_fold2") == "leiden_2.5"
    assert cluster_column_from_reference("hpc_reference_leiden_5p0_fold0") == "leiden_5.0"
    assert cluster_column_from_reference("some_other_reference") is None

    silent = tmp_path / "silent.csv"
    _write_headerless(silent, ["21_21"], reference="a_custom_reference")
    try:
        detect_headerless(silent)
    except SystemExit as e:
        assert "--cluster-column" in str(e), str(e)
    else:
        raise AssertionError("must refuse rather than guess the resolution")

    assert detect_headerless(silent, cluster_column="leiden_2.5")[3] == "leiden_2.5"


def test_headerless_with_correct_tile_names_still_gets_its_header(tmp_path):
    """A missing header is reason enough to rewrite on its own."""
    csv = tmp_path / "hdr_only.csv"
    _write_headerless(csv, ["21_21.jpeg", "22_21.jpeg"])
    result = migrate_csv(csv, commit=True)
    assert result["verdict"] == "done" and result["written"]
    after = pd.read_csv(result["output"])
    assert len(after) == 2 and after["tiles"].tolist() == ["21_21.jpeg", "22_21.jpeg"]


def test_migrated_csv_joins_what_the_loader_builds(tmp_path):
    """The point of the whole exercise: the migrated CSV must produce the same
    slide_tile the Knowledge Bank stores. Checked through read_assignments, so
    this fails if the loader's key or its refusal ever drift."""
    from load_hpc_assignments import read_assignments

    csv = tmp_path / "d.csv"
    _write_csv(csv, ["24_10", "25_10"])

    # The loader now repairs the short form itself, so the unmigrated CSV loads
    # — and has to produce exactly the key the migrated one does. Migrating the
    # file on disk is still worth doing (every other reader of that CSV, and
    # --validate-against, see the short names), but it is no longer the only
    # way through.
    unmigrated, _ = read_assignments(csv)
    assert unmigrated.attrs["tile_names_normalized"] == 2

    migrated = Path(migrate_csv(csv, commit=True)["output"])
    frame, cluster_column = read_assignments(migrated)
    assert cluster_column == "leiden_2.5"
    assert frame["slide_tile"].tolist() == [
        f"{SLIDE.upper()}_24_10.JPEG", f"{SLIDE.upper()}_25_10.JPEG"
    ]
    assert frame["slide_tile"].tolist() == unmigrated["slide_tile"].tolist()
    assert frame.attrs["tile_names_normalized"] == 0


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_migrate_test_"))
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
