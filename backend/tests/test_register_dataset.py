"""Registering a new dataset's identity rows into the Knowledge Bank.

This is the step that did not exist: Stage 5 (load_hpc_assignments.py) only
ever UPDATEs tile_registry.hpc_id, so a dataset that has never touched the KB
refuses at 0% match rate by construction — there is nothing there to update.
These tests are mostly about the two things that make a registration wrong in
a way that is hard to notice afterwards: image_index not actually matching the
.h5's row order, and a --replace or collision touching a dataset_id it
shouldn't.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, event, text  # noqa: E402

import register_dataset as rd  # noqa: E402

SLIDE = "BB232560 A3-1 - 2023-10-11 16.41.02"


def _write_h5(path: Path, rows):
    """rows: list of (sample, slide, tile) in .h5 row order."""
    samples = [r[0].encode() for r in rows]
    slides = [r[1].encode() for r in rows]
    tiles = [r[2].encode() for r in rows]
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (len(rows), 2, 2, 3), dtype="uint8")
        f.create_dataset("samples", data=np.array(samples, dtype=f"S{max(len(s) for s in samples)}"))
        f.create_dataset("slides", data=np.array(slides, dtype=f"S{max(len(s) for s in slides)}"))
        f.create_dataset("tiles", data=np.array(tiles, dtype=f"S{max(len(t) for t in tiles)}"))


def _write_metadata(tile_dir: Path, tile_dataset_name: str, slide_id: str, tile_rows):
    """tile_rows: list of (col, row) -> writes a real Stage-1-shaped CSV."""
    slide_dir = tile_dir / tile_dataset_name / slide_id
    slide_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for col, row in tile_rows:
        name = f"{col}_{row}.jpeg"
        records.append({
            "slides": slide_id, "tiles": name, "slide_tile": f"{slide_id}_{name}",
            "col": col, "row": row, "x_5x": col * 224, "y_5x": row * 224,
            "x_native": col * 1600, "y_native": row * 1600, "tissue_percent": 80.0,
        })
    pd.DataFrame(records).to_csv(slide_dir / f"{slide_id}_tile_metadata.csv", index=False)


def _kb_engine(tmp_path: Path, with_dataset_id_column=True):
    engine = create_engine(f"sqlite:///{tmp_path / 'kb.sqlite'}")

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    dsid = "dataset_id TEXT" if with_dataset_id_column else ""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE tile_registry (
                slide_tile TEXT PRIMARY KEY, samples TEXT, slides TEXT, tiles TEXT,
                image_index INTEGER, h5_source_path TEXT, hpc_id TEXT,
                hpc_vote_margin REAL, hpc_neighbor_distance REAL,
                hpc_reference TEXT, hpc_assigned_at TIMESTAMP, {dsid})"""))
        conn.execute(text(f"""
            CREATE TABLE tile_coordinates (
                slide_tile TEXT PRIMARY KEY, slides TEXT, tiles TEXT,
                col INTEGER, row INTEGER, x_5x INTEGER, y_5x INTEGER,
                x_native INTEGER, y_native INTEGER, h5_index INTEGER, {dsid})"""))
        conn.execute(text(f"""
            CREATE TABLE wsi_registry (
                slide_id TEXT PRIMARY KEY, sample_id TEXT, file_uuid TEXT,
                filename TEXT, hpc_path TEXT, file_size_bytes BIGINT,
                mtime_utc TIMESTAMP, added_at TIMESTAMP, {dsid})"""))
        conn.execute(text(f"""
            CREATE TABLE wsi_metadata (
                slide_id TEXT PRIMARY KEY, sample_id TEXT, level_count INTEGER,
                level_dimensions_json TEXT, level_downsamples_json TEXT,
                mpp_x REAL, mpp_y REAL, objective_power REAL, vendor TEXT,
                scanner_model TEXT, scanner_date TIMESTAMP, tile_width INTEGER,
                tile_height INTEGER, quickhash TEXT, {dsid})"""))
        conn.execute(text(f"""
            CREATE TABLE dataset_config (
                dataset_id TEXT PRIMARY KEY, target_mpp REAL,
                tile_size_5x_px INTEGER)"""))
    return engine


def _write_raw_slides(raw_dir: Path, names):
    """Files OpenSlide would accept, with real bytes so st_size is meaningful."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (raw_dir / name).write_bytes(b"not really a slide, but a real file")
    return raw_dir


def test_image_index_is_the_h5_row_position_not_csv_order(tmp_path):
    """The central claim: image_index must match the ACTUAL array position in
    the .h5, which need not be the order Stage 1's metadata lists tiles in."""
    h5_path = tmp_path / "packaged.h5"
    # .h5 row order deliberately shuffled relative to how metadata will list them.
    _write_h5(h5_path, [
        ("S1", SLIDE, "25_10.jpeg"),
        ("S1", SLIDE, "24_10.jpeg"),
        ("S1", SLIDE, "7_3.jpeg"),
    ])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10), (7, 3)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    by_tile = plan["registry"].set_index("tiles")["image_index"].to_dict()
    assert by_tile["25_10.jpeg"] == 0
    assert by_tile["24_10.jpeg"] == 1
    assert by_tile["7_3.jpeg"] == 2

    coords_by_tile = plan["coordinates"].set_index("tiles")["h5_index"].to_dict()
    assert coords_by_tile["25_10.jpeg"] == 0
    assert coords_by_tile["24_10.jpeg"] == 1


def test_slide_with_no_metadata_is_reported_not_dropped_silently(tmp_path):
    """A slide missing its Stage 1 CSV still has real tiles in the .h5 — they
    go into tile_registry with no coordinates, and the gap is reported, not
    silently absorbed."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "1_1.jpeg"), ("S1", "OTHER SLIDE", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(1, 1)])
    # "OTHER SLIDE" has no metadata CSV at all.

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    assert len(plan["registry"]) == 2, "both tiles still register"
    assert len(plan["coordinates"]) == 1, "only the one with metadata gets coordinates"
    assert any("OTHER SLIDE" in m for m in plan["missing_slides"])


def test_legacy_h5_has_the_suffix_appended_rather_than_being_refused(tmp_path):
    """A .h5 packaged before the tile-name fix used to be refused here and sent
    to migrate_tile_names.py. The mapping "24_10" -> "24_10.jpeg" is total and
    lossless (auto_tile_from_mask.py writes nothing else), so it is applied on
    the way in — and counted, because a silent correction to identity is the
    thing this codebase is written against."""
    h5_path = tmp_path / "legacy.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10")])  # no suffix

    frame = rd.read_h5_identity(h5_path)

    assert frame["tiles"].tolist() == ["24_10.jpeg"]
    assert frame["slide_tile"].tolist() == [f"{SLIDE.upper()}_24_10.JPEG"]
    assert frame.attrs["tile_names_normalized"] == 1


def test_a_half_migrated_h5_is_still_refused(tmp_path):
    """Mixed is the one state that cannot be repaired: the rows on either side
    of a resume that straddled the fix are indistinguishable by name, so
    appending would attach real cluster IDs to the wrong tiles. Note the old
    guard could not even see this case — tiles_missing_suffix() samples and
    requires ALL names to be short — so this is a refusal that did not exist."""
    h5_path = tmp_path / "half.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10"), ("S1", SLIDE, "25_10.jpeg")])
    try:
        rd.read_h5_identity(h5_path)
    except SystemExit as e:
        assert "Repackage" in str(e), str(e)
    else:
        raise AssertionError("a half-migrated .h5 must be refused")


def test_short_stage1_metadata_still_joins_the_normalised_h5(tmp_path):
    """Both sides have to be normalised or neither. Fixing only the .h5 turns
    the old loud refusal into tiles_with_coordinates: 0 — the same bug, now
    silent and shaped exactly like a successful registration."""
    h5_path = tmp_path / "legacy.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    # Rewrite Stage 1's CSV into the pre-fix short form.
    csv = tile_dir / "Radiogenomics" / SLIDE / f"{SLIDE}_tile_metadata.csv"
    frame = pd.read_csv(csv)
    frame["tiles"] = frame["tiles"].str.replace(".jpeg", "", regex=False)
    frame["slide_tile"] = frame["slide_tile"].str.replace(".jpeg", "", regex=False)
    frame.to_csv(csv, index=False)

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                                 "RADIOGENOMICS")

    assert len(plan["coordinates"]) == 1, "the short-named coordinates must still join"
    assert plan["tile_names_normalized"] == {"h5": 1, "coordinates": 1}


def test_first_registration_writes_both_tables_in_one_transaction(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg"), ("S1", SLIDE, "25_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    engine = _kb_engine(tmp_path)
    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert written == {"tile_registry": 2, "tile_coordinates": 2}

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT slide_tile, hpc_id, dataset_id FROM tile_registry"
        )).mappings().all()
        assert len(rows) == 2
        assert all(r["hpc_id"] is None for r in rows), "hpc_id is Stage 5's job, not this one's"
        assert all(r["dataset_id"] == "RADIOGENOMICS" for r in rows)


def test_second_registration_without_replace_is_refused(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = _kb_engine(tmp_path)
    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)

    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "--replace" in str(e), str(e)
    else:
        raise AssertionError("re-registering the same dataset_id must be refused")

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tile_registry")).scalar() == 1


def test_replace_only_touches_its_own_dataset_id(tmp_path):
    """The one thing a --replace must never do: delete or overwrite another
    cohort's rows, even ones that happen to share table space."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg"), ("S1", SLIDE, "25_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10), (25, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tile_registry (slide_tile, samples, slides, tiles, "
            "image_index, dataset_id) VALUES ('TCGA-1_1_1.JPEG', 'TCGA-1', "
            "'TCGA-1', '1_1.jpeg', 0, 'TCGA')"
        ))
        conn.execute(text(
            "INSERT INTO tile_coordinates (slide_tile, slides, tiles, col, row, "
            "dataset_id) VALUES ('TCGA-1_1_1.JPEG', 'TCGA-1', '1_1.jpeg', 1, 1, 'TCGA')"
        ))

    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    # Replace RADIOGENOMICS with itself — must not touch TCGA's row.
    rd.commit(engine, plan, "RADIOGENOMICS", replace=True)

    with engine.connect() as conn:
        tcga_rows = conn.execute(text(
            "SELECT COUNT(*) FROM tile_registry WHERE dataset_id = 'TCGA'"
        )).scalar()
        radio_rows = conn.execute(text(
            "SELECT COUNT(*) FROM tile_registry WHERE dataset_id = 'RADIOGENOMICS'"
        )).scalar()
    assert tcga_rows == 1, "TCGA's row must survive a RADIOGENOMICS --replace"
    assert radio_rows == 2


def test_slide_tile_collision_across_datasets_is_refused_not_reassigned(tmp_path):
    """Two different dataset_ids producing the same slide_tile key is a real
    problem — a naming collision or a misidentified cohort — and must be
    surfaced as an error, never silently resolved by whichever ran last."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    colliding_key = plan["registry"]["slide_tile"].iloc[0]

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO tile_registry (slide_tile, samples, slides, tiles, "
            "image_index, dataset_id) VALUES (:k, 'X', 'X', 'x.jpeg', 0, 'OTHER_COHORT')"
        ), {"k": colliding_key})

    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "different dataset_id" in str(e).lower() or "DIFFERENT dataset_id" in str(e)
    else:
        raise AssertionError("a cross-cohort slide_tile collision must be refused")


def test_dry_run_writes_nothing(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    engine = _kb_engine(tmp_path)

    result = rd.preview(engine, plan, "RADIOGENOMICS")
    assert result["tiles_in_h5"] == 1
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM tile_registry")).scalar() == 0


def test_missing_column_in_a_live_table_does_not_abort_the_whole_write(tmp_path):
    """These tables predate this script. A table missing a column this script
    assumes must drop that column from the insert and say so, not fail the
    whole transaction — same reasoning as load_hpc_assignments.py."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    engine = create_engine(f"sqlite:///{tmp_path / 'narrow.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE tile_registry (
                slide_tile TEXT PRIMARY KEY, samples TEXT, slides TEXT,
                tiles TEXT, image_index INTEGER, dataset_id TEXT)"""))
        # No h5_source_path column.
        conn.execute(text("""
            CREATE TABLE tile_coordinates (
                slide_tile TEXT PRIMARY KEY, slides TEXT, tiles TEXT,
                col INTEGER, row INTEGER, x_5x INTEGER, y_5x INTEGER,
                x_native INTEGER, y_native INTEGER, h5_index INTEGER, dataset_id TEXT)"""))

    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert written == {"tile_registry": 1, "tile_coordinates": 1}


def test_registration_takes_stage5_from_zero_percent_to_full_match(tmp_path):
    """The whole point, end to end.

    Before registering, Stage 5 matches 0% — not because of a naming bug but
    because tile_registry has no rows for this cohort at all, and Stage 5 only
    ever UPDATEs. After registering, the same CSV matches every tile. This is
    the exact failure the Radiogenomics load hit, so it is asserted rather than
    described.
    """
    import load_hpc_assignments as loader

    h5_path = tmp_path / "packaged.h5"
    tile_rows = [(24, 10), (25, 10), (7, 3)]
    _write_h5(h5_path, [("S1", SLIDE, f"{c}_{r}.jpeg") for c, r in tile_rows])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, tile_rows)

    # Stage 4's CSV, in the form migrate_tile_names.py produces.
    csv_path = tmp_path / "assignments.csv"
    pd.DataFrame({
        "samples": ["S1"] * 3,
        "slides": [SLIDE] * 3,
        "tiles": [f"{c}_{r}.jpeg" for c, r in tile_rows],
        "leiden_2.5": [28, 50, 19],
        "vote_margin": [0.9, 0.4, 0.2],
        "neighbor_distance": [1.0, 2.0, 3.0],
        "hpc_reference": ["hpc_reference_leiden_2p5_fold2"] * 3,
    }).to_csv(csv_path, index=False)

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE hpc_dictionary (hpc_id TEXT, malignant TEXT)"))
        for cluster in ("28", "50", "19"):
            conn.execute(text("INSERT INTO hpc_dictionary VALUES (:h, 'True')"),
                         {"h": cluster})

    frame, cluster_column = loader.read_assignments(csv_path)

    before = loader.inspect(engine, frame, cluster_column)
    assert before["matched"] == 0, "nothing to match before registration"
    assert before["unmatched"] == 3

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)

    after = loader.inspect(engine, frame, cluster_column)
    assert after["matched"] == 3, after
    assert after["unmatched"] == 0
    assert after["matched"] / after["rows"] >= loader._MIN_MATCH_RATE


# --- slide identity: wsi_registry, wsi_metadata, dataset_config ----------
#
# These exist because registering tiles without registering slides passes every
# check the pipeline has and still leaves the cohort invisible: _open_slide()
# in tile_server_v2_.py resolves paths from wsi_registry alone. Before this,
# the only INSERT into wsi_registry was the interactive single-slide upload
# path, so no bulk Slurm run had ever registered a slide.


def test_slides_are_matched_to_raw_files_by_the_same_rule_stage1_used(tmp_path):
    """slide_id_from_raw_path() is what tile_mask.py derived Stage 1's names
    with, so a GDC file "{barcode}.{uuid}.svs" must be found for the barcode
    the .h5 carries — not only for a file named exactly after it."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "TCGA-55-7574-01Z-00-DX1", "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "TCGA-55-7574-01Z-00-DX1", [(24, 10)])
    raw = _write_raw_slides(tmp_path / "raw", [
        "TCGA-55-7574-01Z-00-DX1.0f1c7e5a-9b2d-4c3e-8a1f-2b3c4d5e6f70.svs",
    ])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    assert len(plan["wsi_registry"]) == 1
    row = plan["wsi_registry"].iloc[0]
    assert row["slide_id"] == "TCGA-55-7574-01Z-00-DX1"
    assert row["file_uuid"] == "0f1c7e5a-9b2d-4c3e-8a1f-2b3c4d5e6f70"
    assert row["file_size_bytes"] > 0
    assert not plan["slides_without_files"]


def test_slide_id_is_upper_cased_because_every_reader_looks_it_up_that_way(tmp_path):
    """_load_wsi_map() and _open_slide() both upper-case before looking up. A
    row written in the .h5's own casing exists and is never found."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "lower-case-slide", "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "lower-case-slide", [(24, 10)])
    raw = _write_raw_slides(tmp_path / "raw", ["lower-case-slide.svs"])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    assert plan["wsi_registry"].iloc[0]["slide_id"] == "LOWER-CASE-SLIDE"


def test_a_slide_with_two_candidate_files_is_refused_not_picked(tmp_path):
    """Two files claiming one slide_id means the viewer would serve whichever
    happened to sort first. Being wrong about which physical slide a cohort's
    tiles came from is not a tie to break silently."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "AMBIG", "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "AMBIG", [(24, 10)])
    raw = tmp_path / "raw"
    _write_raw_slides(raw / "batch_a", ["AMBIG.svs"])
    _write_raw_slides(raw / "batch_b", ["AMBIG.ndpi"])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    assert len(plan["ambiguous_slides"]) == 1
    assert "AMBIG" in plan["ambiguous_slides"][0]
    assert plan["wsi_registry"].empty, "an ambiguous slide must not be registered"


def test_a_slide_with_no_raw_file_is_reported_and_its_tiles_still_register(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "HAS-FILE", "1_1.jpeg"), ("S1", "NO-FILE", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "HAS-FILE", [(1, 1)])
    _write_metadata(tile_dir, "Radiogenomics", "NO-FILE", [(2, 2)])
    raw = _write_raw_slides(tmp_path / "raw", ["HAS-FILE.svs"])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    assert plan["slides_without_files"] == ["NO-FILE"]
    assert len(plan["wsi_registry"]) == 1
    assert len(plan["registry"]) == 2, "both slides' tiles still register"


def test_non_slide_files_in_the_raw_directory_are_not_registered(tmp_path):
    """A partially-transferred .svs.part or a stray manifest must not become a
    wsi_registry row whose path 404s the first time somebody clicks it."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    raw = _write_raw_slides(tmp_path / "raw", ["SLIDE-A.svs.part", "SLIDE-A.csv"])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    assert plan["wsi_registry"].empty
    assert plan["slides_without_files"] == ["SLIDE-A"]


def test_without_raw_dir_nothing_slide_level_is_written(tmp_path):
    """The pre-existing tiles-only behaviour has to stay reachable unchanged,
    since re-registering tiles for a cohort whose slides are already in
    wsi_registry is a real thing to want."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    assert plan["wsi_registry"].empty
    assert plan["wsi_metadata"].empty
    assert plan["dataset_config"].empty

    engine = _kb_engine(tmp_path)
    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert set(written) == {"tile_registry", "tile_coordinates"}
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM wsi_registry")).scalar() == 0


def test_dataset_config_is_written_only_when_both_numbers_are_given(tmp_path):
    """Its two columns are NOT NULL, and a row asserting the wrong tiling
    geometry is worse than no row — every coordinate conversion downstream
    would trust it."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "24_10.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", SLIDE, [(24, 10)])

    without = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                    str(h5_path), "RADIOGENOMICS", target_mpp=1.8)
    assert without["dataset_config"].empty, "one number alone must write nothing"

    with_both = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                      str(h5_path), "RADIOGENOMICS",
                                      target_mpp=1.8, tile_size_5x_px=224)
    assert len(with_both["dataset_config"]) == 1
    row = with_both["dataset_config"].iloc[0]
    assert row["target_mpp"] == 1.8 and row["tile_size_5x_px"] == 224


def test_slide_id_collision_across_datasets_is_refused(tmp_path):
    """Sharper than the tile-level collision: slide_id is wsi_registry's whole
    primary key, so overwriting one repoints the viewer at another cohort's
    file while every tile row still says otherwise."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SHARED-SLIDE", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SHARED-SLIDE", [(1, 1)])
    raw = _write_raw_slides(tmp_path / "raw", ["SHARED-SLIDE.svs"])

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO wsi_registry (slide_id, hpc_path, dataset_id) "
            "VALUES ('SHARED-SLIDE', '/somewhere/else.svs', 'TCGA_LUAD_5X')"))

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "DIFFERENT dataset_id" in str(e), str(e)
    else:
        raise AssertionError("a slide claimed by another cohort must be refused")

    with engine.connect() as conn:
        path = conn.execute(text(
            "SELECT hpc_path FROM wsi_registry WHERE slide_id='SHARED-SLIDE'")).scalar()
        assert path == "/somewhere/else.svs", "the other cohort's row was modified"
        assert conn.execute(text("SELECT COUNT(*) FROM tile_registry")).scalar() == 0, \
            "the refusal must roll the tile writes back too"


def test_slide_and_tile_tables_are_written_in_one_transaction(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg"), ("S1", "SLIDE-A", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1), (2, 2)])
    raw = _write_raw_slides(tmp_path / "raw", ["SLIDE-A.svs"])

    engine = _kb_engine(tmp_path)
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw,
                                 target_mpp=1.8, tile_size_5x_px=224)
    written = rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    assert written == {"wsi_registry": 1, "dataset_config": 1,
                       "tile_registry": 2, "tile_coordinates": 2}

    with engine.connect() as conn:
        for table, expected in (("wsi_registry", 1), ("dataset_config", 1),
                                ("tile_registry", 2), ("tile_coordinates", 2)):
            n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            assert n == expected, f"{table} has {n}, expected {expected}"


def test_replace_clears_the_slide_tables_too(tmp_path):
    """A --replace that dropped tiles but left wsi_registry would leave a
    superseded run's slide paths pointing at files the new run may not use."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    raw = _write_raw_slides(tmp_path / "raw", ["SLIDE-A.svs"])
    engine = _kb_engine(tmp_path)

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    rd.commit(engine, plan, "RADIOGENOMICS", replace=False)

    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "wsi_registry" in str(e), str(e)
    else:
        raise AssertionError("a second registration must be refused")

    rd.commit(engine, plan, "RADIOGENOMICS", replace=True)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM wsi_registry")).scalar() == 1, \
            "--replace must leave exactly one row, not two"


def test_a_slide_with_two_sample_ids_is_reported(tmp_path):
    """samples is per-tile in the .h5 but a slide-level fact. A slide mapped to
    two sample ids would put half its tiles under the wrong patient."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("SAMPLE-A", "SLIDE-A", "1_1.jpeg"),
                        ("SAMPLE-B", "SLIDE-A", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1), (2, 2)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    assert plan["conflicting_samples"] == ["SLIDE-A"]


def test_the_conflicting_sample_check_can_fail(tmp_path):
    """The companion: a slide with one consistent sample must NOT be flagged,
    or the check above is just always true."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("SAMPLE-A", "SLIDE-A", "1_1.jpeg"),
                        ("SAMPLE-A", "SLIDE-A", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1), (2, 2)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")
    assert plan["conflicting_samples"] == []


def test_numeric_slide_ids_in_stage1_metadata_still_join_the_h5(tmp_path):
    """Stage 1's CSV spells the slide the way the .h5 does, but pandas reads an
    all-digit slides column as numbers: '007' comes back as 7, and 'NA' as NaN.
    The coordinates' key is then '7_1_1.JPEG' against the .h5's '007_1_1.JPEG',
    and registration reports tiles_with_coordinates: 0 for slides that were
    tiled perfectly well."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "007", "1_1.jpeg"), ("S2", "NA", "2_2.jpeg"),
                        ("S3", "1001", "3_3.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "007", [(1, 1)])
    _write_metadata(tile_dir, "Radiogenomics", "NA", [(2, 2)])
    _write_metadata(tile_dir, "Radiogenomics", "1001", [(3, 3)])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS")

    assert sorted(plan["coordinates"]["slide_tile"]) == [
        "007_1_1.JPEG", "1001_3_3.JPEG", "NA_2_2.JPEG"], \
        plan["coordinates"]["slide_tile"].tolist()
    assert sorted(plan["coordinates"]["slides"]) == ["007", "1001", "NA"]


def test_committing_to_a_database_without_the_base_tables_is_refused(tmp_path):
    """A database built from schema.sql has no wsi_registry at all. Refuse with
    the migration to run, rather than raising an OperationalError naming a
    table nobody knew was missing."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    raw = _write_raw_slides(tmp_path / "raw", ["SLIDE-A.svs"])

    engine = _kb_engine(tmp_path)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE wsi_registry"))

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics",
                                 str(h5_path), "RADIOGENOMICS", raw_dir=raw)
    try:
        rd.commit(engine, plan, "RADIOGENOMICS", replace=False)
    except SystemExit as e:
        assert "migrate_kb_base_tables.sql" in str(e), str(e)
    else:
        raise AssertionError("a missing base table must be refused by name")


def test_missing_values_reach_the_database_as_null_not_as_nan(tmp_path):
    """Building a DataFrame promotes any column with a gap to float64, so an
    absent mpp arrives as float('nan'). sqlite refuses it outright; psycopg2
    coerces it into a numeric column, which is worse because it succeeds — a
    slide whose mpp is unknown would read back as a number.
    """
    assert rd._native(float("nan")) is None
    assert rd._native(pd.NaT) is None
    assert rd._native(None) is None
    assert isinstance(rd._native(pd.Timestamp("2026-08-26 12:00")), datetime)
    assert not isinstance(rd._native(pd.Timestamp("2026-08-26 12:00")), pd.Timestamp)
    # and it must leave real values alone
    assert rd._native(1.8) == 1.8
    assert rd._native("SLIDE-A") == "SLIDE-A"


def test_a_frame_with_gaps_writes_nulls(tmp_path):
    """The end-to-end version of the check above, through the real insert."""
    engine = _kb_engine(tmp_path)
    frame = pd.DataFrame([
        {"slide_id": "A", "mpp_x": 0.252, "objective_power": 40.0, "dataset_id": "D"},
        {"slide_id": "B", "mpp_x": None, "objective_power": None, "dataset_id": "D"},
    ])
    with engine.begin() as conn:
        rd._insert(conn, "wsi_metadata", frame,
                   ("slide_id", "mpp_x", "objective_power", "dataset_id"))
    with engine.connect() as conn:
        got = conn.execute(text(
            "SELECT mpp_x FROM wsi_metadata WHERE slide_id='B'")).scalar()
    assert got is None, f"an absent mpp came back as {got!r}, not NULL"


def test_file_uuid_is_none_for_a_slide_that_has_no_uuid(tmp_path):
    """None because there is no uuid to report, not because one is missing —
    the distinction matters when auditing which slides came from the GDC."""
    from slide_naming import file_uuid_from_raw_path
    assert file_uuid_from_raw_path("BB232560 A3-1 - 2023-10-11 16.41.02.svs") is None
    assert file_uuid_from_raw_path(
        "TCGA-55-7574-01Z-00-DX1.0f1c7e5a-9b2d-4c3e-8a1f-2b3c4d5e6f70.svs"
    ) == "0f1c7e5a-9b2d-4c3e-8a1f-2b3c4d5e6f70"


# --- subset registration --------------------------------------------------
#
# Registering a handful of slides before committing 14,044 of them. The risk is
# not that it registers too few — it is that it registers too few and looks like
# it worked, so every refusal below is about narrowing silently.


def test_subset_registers_only_the_requested_slides(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg"), ("S1", "SLIDE-B", "2_2.jpeg"),
                        ("S1", "SLIDE-C", "3_3.jpeg")])
    tile_dir = tmp_path / "tiles"
    for slide, cr in (("SLIDE-A", (1, 1)), ("SLIDE-B", (2, 2)), ("SLIDE-C", (3, 3))):
        _write_metadata(tile_dir, "Radiogenomics", slide, [cr])

    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                                 "RADIOGENOMICS", scope="subset",
                                 slide_names=["SLIDE-B"])
    assert plan["slides"] == ["SLIDE-B"]
    assert len(plan["registry"]) == 1
    assert plan["registry"]["slides"].tolist() == ["SLIDE-B"]
    assert plan["scope"] == "subset"


def test_a_subset_slide_not_in_the_h5_is_refused_by_name(tmp_path):
    """Registering three of four requested slides would look like success and
    leave the fourth missing with nothing to say so."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    try:
        rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                              "RADIOGENOMICS", scope="subset",
                              slide_names=["SLIDE-A", "NOT-PACKAGED"])
    except SystemExit as e:
        assert "NOT-PACKAGED" in str(e), str(e)
    else:
        raise AssertionError("a slide absent from the .h5 must be refused by name")


def test_an_empty_subset_is_refused(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    for names in ([], ["   "], None):
        try:
            rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                                  "RADIOGENOMICS", scope="subset", slide_names=names)
        except SystemExit as e:
            assert "at least one slide" in str(e), str(e)
        else:
            raise AssertionError(f"an empty subset ({names!r}) must be refused")


def test_an_unknown_scope_is_refused(tmp_path):
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    try:
        rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                              "RADIOGENOMICS", scope="everything")
    except SystemExit as e:
        assert "full" in str(e) and "subset" in str(e), str(e)
    else:
        raise AssertionError("an unrecognised scope must be refused, not treated as full")


def test_the_default_scope_still_registers_everything(tmp_path):
    """The companion: subset must be opt-in, or every existing caller silently
    changes behaviour."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg"), ("S1", "SLIDE-B", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-B", [(2, 2)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                                 "RADIOGENOMICS")
    assert plan["scope"] == "full"
    assert plan["requested_slides"] is None
    assert sorted(plan["slides"]) == ["SLIDE-A", "SLIDE-B"]


def test_subset_matching_is_case_insensitive(tmp_path):
    """Slide ids are upper-cased everywhere they are looked up, so a subset
    typed in the casing a human would use has to match."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", "SLIDE-A", "1_1.jpeg"), ("S1", "SLIDE-B", "2_2.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-A", [(1, 1)])
    _write_metadata(tile_dir, "Radiogenomics", "SLIDE-B", [(2, 2)])
    plan = rd.build_registration(h5_path, tile_dir, "Radiogenomics", str(h5_path),
                                 "RADIOGENOMICS", scope="subset",
                                 slide_names=["  slide-b  "])
    assert plan["slides"] == ["SLIDE-B"]


# --- the tile folder the server registers from ---------------------------
#
# register_dataset.py is handed a tile_dataset_name; the server used to take it
# only from slurm_dataset_runs.dataset_name and refuse when that was NULL
# ("register it with the CLI instead"). That is a dead end for any run tiled
# before the column existed, or tiled by hand — and one reached with the KB
# cohort key already filled in, because dataset_id is a different thing from
# this folder. The request can now carry it.


def _server():
    import tile_server_v2_ as srv
    return srv


def _run_row(tmp_path: Path, *, dataset_name=None):
    """A slurm_dataset_runs row shaped the way _registration_plan reads it,
    with a real packaged .h5 and real Stage 1 metadata under "TCGA"."""
    h5_path = tmp_path / "packaged.h5"
    _write_h5(h5_path, [("S1", SLIDE, "1_1.jpeg")])
    tile_dir = tmp_path / "tiles"
    _write_metadata(tile_dir, "TCGA", SLIDE, [(1, 1)])
    raw_dir = _write_raw_slides(tmp_path / "raw", [f"{SLIDE}.svs"])
    return {
        "h5_output_path": str(h5_path),
        "tile_dir": str(tile_dir),
        "dataset_name": dataset_name,
        "raw_dir": str(raw_dir),
        "tiling_params": None,
    }


def test_a_run_with_no_recorded_tile_folder_is_registerable_by_naming_it(tmp_path):
    srv = _server()
    row = _run_row(tmp_path, dataset_name=None)
    req = srv.RegistrationRequest(dataset_id="TCGA", tile_dataset_name="TCGA")

    plan, dataset_id, _raw = srv._registration_plan(row, req)

    assert dataset_id == "TCGA"
    assert plan["tile_dataset_name"] == "TCGA"
    # The point of the field: coordinates, which come only from that folder.
    assert len(plan["coordinates"]) == 1


def test_a_dataset_id_alone_does_not_stand_in_for_the_tile_folder(tmp_path):
    """The reported failure, exactly: a filled-in cohort key and a NULL
    dataset_name still refuses — and the message has to say which of the two
    names is missing, or it reads as "I already told you the dataset name"."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name=None)
    req = srv.RegistrationRequest(dataset_id="TCGA")

    try:
        srv._registration_plan(row, req)
    except HTTPException as e:
        assert e.status_code == 400
        assert "tile_dataset_name" in str(e.detail)
        assert "dataset_id" in str(e.detail)
    else:
        raise AssertionError("a run with no tile folder anywhere was accepted")


def test_both_names_for_the_tile_folder_are_one_field(tmp_path):
    """dataset_name (main's override) and tile_dataset_name (the older field)
    name the same folder. Either works; two different values are refused
    rather than one silently winning — the preview and the commit would
    otherwise disagree about which folder they read."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name=None)
    plan, _, _ = srv._registration_plan(row, srv.RegistrationRequest(
        dataset_id="TCGA", dataset_name="TCGA"))
    assert plan["tile_dataset_name"] == "TCGA"
    assert plan["sources"]["dataset_name"] == "supplied"

    try:
        srv._registration_plan(row, srv.RegistrationRequest(
            dataset_id="TCGA", dataset_name="TCGA", tile_dataset_name="OTHER"))
    except HTTPException as e:
        assert e.status_code == 400 and "different" in str(e.detail)
    else:
        raise AssertionError("two different tile folders were accepted")


def test_main_s_dataset_name_override_is_charset_checked_too(tmp_path):
    """The override becomes a path segment under tile_dir, whichever field
    carries it."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name=None)
    try:
        srv._registration_plan(row, srv.RegistrationRequest(
            dataset_id="TCGA", dataset_name="../TCGA"))
    except HTTPException as e:
        assert e.status_code == 400
    else:
        raise AssertionError("a tile folder name with '..' reached the filesystem")


def test_the_slurm_path_reads_the_same_files_as_the_preview(tmp_path):
    """/register-submit resolves its paths through _registration_inputs, the
    function the preview uses, so an h5_path override reaches the job too."""
    srv = _server()
    row = _run_row(tmp_path, dataset_name="TCGA")
    other_h5 = tmp_path / "elsewhere.h5"
    other_h5.write_bytes(Path(row["h5_output_path"]).read_bytes())
    inputs = srv._registration_inputs(row, srv.RegistrationRequest(
        dataset_id="TCGA", h5_path=str(other_h5)))
    assert inputs["h5_path"] == other_h5
    assert inputs["sources"]["h5_path"] == "supplied"
    assert inputs["sources"]["dataset_name"] == "run record"


def test_a_supplied_folder_name_cannot_escape_the_tile_directory(tmp_path):
    """It becomes a literal path segment under tile_dir, so it is charset-checked
    the same way /submit-dataset-job checks it."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name="TCGA")

    for bad in ("../../etc", "TCGA/../other", "/absolute", ".hidden", ""):
        req = srv.RegistrationRequest(dataset_id="TCGA", tile_dataset_name=bad)
        if not bad:
            # Empty falls through to the recorded name rather than being an
            # error — the UI's own required field is what stops a blank there.
            plan, _id, _raw = srv._registration_plan(row, req)
            assert plan["tile_dataset_name"] == "TCGA"
            continue
        try:
            srv._registration_plan(row, req)
        except HTTPException as e:
            assert e.status_code == 400
        else:
            raise AssertionError(f"{bad!r} was accepted as a tile folder name")


def test_the_supplied_name_wins_over_the_recorded_one_and_is_reported(tmp_path):
    """Overriding is deliberate — a run may have been re-tiled elsewhere — but a
    wrong override produces tiles with no coordinates rather than an error, so
    the folder actually read is carried on the plan for the preview to name."""
    srv = _server()
    row = _run_row(tmp_path, dataset_name="TCGA")
    # A real folder that simply does not hold this run's slides — the existence
    # guard passes, so what is left is the quiet failure it cannot catch.
    (Path(row["tile_dir"]) / "Radiogenomics").mkdir(parents=True, exist_ok=True)
    req = srv.RegistrationRequest(dataset_id="TCGA", tile_dataset_name="Radiogenomics")

    plan, _dataset_id, _raw = srv._registration_plan(row, req)

    assert plan["tile_dataset_name"] == "Radiogenomics"
    assert len(plan["registry"]) == 1
    assert len(plan["coordinates"]) == 0
    assert len(plan["missing_slides"]) == 1
    assert plan["missing_slides"][0].startswith(SLIDE)


def test_a_tile_folder_that_is_not_on_disk_is_refused_not_read_as_empty(tmp_path):
    """The silent version of this bug: a folder that does not exist yields no
    metadata, and registration succeeds with every tile carrying no
    coordinates. Refuse instead."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name="TCGA")
    req = srv.RegistrationRequest(dataset_id="TCGA", tile_dataset_name="Nonexistent")

    try:
        srv._registration_plan(row, req)
    except HTTPException as e:
        assert e.status_code == 400
        assert "Nonexistent" in str(e.detail)
    else:
        raise AssertionError("a tile folder that is not on disk was accepted")


def test_a_case_mismatch_names_the_folder_that_does_exist(tmp_path):
    """RADIOGENOMICS vs Radiogenomics resolves on macOS and does not on the
    cluster's Linux filesystem, so the error has to name the real one rather
    than leaving 'no such folder' to be squared with a folder that is visibly
    there."""
    from fastapi import HTTPException

    srv = _server()
    row = _run_row(tmp_path, dataset_name="TCGA")
    (Path(row["tile_dir"]) / "Radiogenomics").mkdir(parents=True, exist_ok=True)
    req = srv.RegistrationRequest(dataset_id="RADIOGENOMICS",
                                  tile_dataset_name="RADIOGENOMICS")

    try:
        srv._registration_plan(row, req)
    except HTTPException as e:
        assert "Radiogenomics" in str(e.detail)
        assert "case-sensitive" in str(e.detail)
    else:
        # macOS resolves the mismatched case, so the guard cannot fire here —
        # but then the folder really was found, which is the safe direction.
        pass


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_register_test_"))
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
