#!/usr/bin/env python3
"""Register a new dataset's identity into the Knowledge Bank, before Stage 5.

Stage 5 (load_hpc_assignments.py) only ever UPDATEs tile_registry.hpc_id — it
never creates rows. For a dataset that has never touched the KB before, that
means Stage 5's match rate is 0% by construction: there is nothing there to
UPDATE. This is that missing step. It creates the identity rows — with no
hpc_id yet — from what Stages 1–2 already wrote to disk. Stage 5 runs unchanged
after this and fills in hpc_id.

    raw slide files        ──┐                        ─► wsi_registry
                             │                        ─► wsi_metadata  (--slide-metadata)
    Stage 1 metadata CSVs  ──┼─► register_dataset.py  ─► tile_coordinates
                             │                        ─► tile_registry (no hpc_id)
    packaged .h5 (Stage 2) ──┘                        ─► dataset_config (--target-mpp)
                                                              │
                                                         Stage 5 UPDATEs hpc_id ─►
                                                         tile_registry (complete)

Why the slide tables are here and not in their own script. A registration that
wrote the tile tables but not wsi_registry passes every check this pipeline
has — Stage 5 loads, the aggregates refresh, `\\dt+` shows every table
growing — and the cohort is still invisible in the viewer, because
_load_wsi_map() in tile_server_v2_.py builds slide_id → path from wsi_registry
alone and _open_slide() 404s on anything absent from it. Splitting the two
steps would make that partial state reachable by forgetting a command, which
is the failure this codebase is written against. One command, one transaction,
all five tables or none.

Before this, the ONLY thing that ever inserted into wsi_registry was
_register_uploaded_slide() on the interactive single-slide drag-and-drop path.
No bulk Slurm dataset run has ever registered a slide.

That upload path now runs through this script too: an uploaded slide gets its
own run and its own dataset_id ("UPLOADED_<SLIDE_ID>", see
tile_server_v2_.upload_dataset_name), so it reaches the Knowledge Bank the
same way a cohort does. One consequence is worth knowing before reading a
refusal here: the upload already wrote that slide's wsi_registry row at upload
time, so the viewer could open it immediately, which means registering an
upload always finds its own cohort occupied and needs --replace. The delete
that follows is scoped to that one dataset_id, which for an upload is that one
slide.

image_index (tile_registry) / h5_index (tile_coordinates) is the tile's row
position in the packaged .h5 — the actual array index, not anything derived
from the metadata CSV, because packaging order is not guaranteed stable across
a re-package (see make_hpl_hdf5.py). Reading it back from the file is the only
way to get this right.

Everything is scoped by --dataset-id and refuses to touch another cohort's
rows. Reusing this dataset_id to re-register (the planned path once the full
14,044-slide Radiogenomics run replaces this 10-slide one) requires --replace,
which is still restricted to WHERE dataset_id = the one given — a slide_tile
or slide_id collision with a DIFFERENT dataset_id is refused as an error, not
silently reassigned, since that would mean two cohorts claiming the same tile
or the viewer opening one cohort's file for another cohort's slide.

Usage:
    # tiles only — what this script did before slide registration existed
    python register_dataset.py --h5 packaged.h5 --tile-dir /path/to/processed_tiles \\
        --tile-dataset-name Radiogenomics --dataset-id RADIOGENOMICS

    # the whole identity, which is what a new cohort actually needs
    python register_dataset.py --h5 packaged.h5 --tile-dir /path/to/processed_tiles \\
        --tile-dataset-name Radiogenomics --dataset-id RADIOGENOMICS \\
        --raw-dir /path/to/raw_slides --slide-metadata \\
        --target-mpp 1.8 --tile-size-5x 224 --commit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy import inspect as sqlalchemy_inspect

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_hpc_assignments import _LOOKUP_CHUNK, make_engine  # noqa: E402
from slide_naming import (  # noqa: E402
    file_uuid_from_raw_path,
    make_slide_tile_series,
    normalize_tile_names,
    slide_id_from_raw_path,
    tile_name_verdict,
)
from run_record import record_run  # noqa: E402
from tile_metadata import read_tile_metadata, tile_metadata_path  # noqa: E402

_TILE_COORDINATES_COLUMNS = (
    "slides", "tiles", "slide_tile", "col", "row",
    "x_5x", "y_5x", "x_native", "y_native", "h5_index", "dataset_id",
)
_TILE_REGISTRY_COLUMNS = (
    "samples", "slides", "tiles", "slide_tile", "image_index",
    "h5_source_path", "dataset_id",
)
_WSI_REGISTRY_COLUMNS = (
    "slide_id", "sample_id", "file_uuid", "filename", "hpc_path",
    "file_size_bytes", "mtime_utc", "added_at", "dataset_id",
)
_WSI_METADATA_COLUMNS = (
    "slide_id", "sample_id", "level_count", "level_dimensions_json",
    "level_downsamples_json", "mpp_x", "mpp_y", "objective_power", "vendor",
    "scanner_model", "scanner_date", "tile_width", "tile_height", "quickhash",
    "dataset_id",
)
_DATASET_CONFIG_COLUMNS = ("dataset_id", "target_mpp", "tile_size_5x_px")

# Every table this script writes, in the order it writes them. Slide identity
# first: a tile row whose slide has no wsi_registry entry is a tile the viewer
# can locate and cannot display.
_TABLES = ("wsi_registry", "wsi_metadata", "dataset_config",
           "tile_registry", "tile_coordinates")

# What OpenSlide can open. Kept explicit rather than "any file in the
# directory" so a stray .csv, .txt or a partially-transferred .svs.part is a
# slide that is reported missing, not a slide registered with a path that will
# 404 the first time somebody clicks it.
_SLIDE_SUFFIXES = (".svs", ".ndpi", ".tif", ".tiff", ".scn", ".mrxs",
                   ".vms", ".vmu", ".svslide", ".bif")


def read_h5_identity(h5_path: Path) -> pd.DataFrame:
    """samples/slides/tiles plus each row's actual position in the .h5.

    Position is read from the file, not computed, because it is the one thing
    nothing else can reconstruct: make_hpl_hdf5.py's own docstring notes
    packaging order is not guaranteed stable across a re-package.
    """
    with h5py.File(h5_path, "r") as f:
        for name in ("samples", "slides", "tiles"):
            if name not in f:
                raise SystemExit(f"{h5_path} has no '{name}' dataset — not a "
                                 f"packaged .h5 from make_hpl_hdf5.py.")
        n = f["tiles"].shape[0]
        if n == 0:
            raise SystemExit(f"{h5_path} holds zero tiles.")
        samples = [v.decode("utf-8", "replace") for v in f["samples"][:]]
        slides = [v.decode("utf-8", "replace") for v in f["slides"][:]]
        tiles_raw = f["tiles"][:]

    # A .h5 packaged before make_hpl_hdf5.py started storing "18_15.jpeg" holds
    # "18_15", which joins nothing in the KB. That used to be refused here and
    # sent to migrate_tile_names.py; the suffix is appended instead, because the
    # mapping is total and lossless (auto_tile_from_mask.py:150 writes every
    # tile as f"{col}_{row}.jpeg") and the count is reported rather than the
    # correction being made silently.
    #
    # Mixed is the exception and is still refused. It is what a resume that
    # straddled the fix leaves behind, the two sides cannot be told apart by
    # name, and appending to one of them would attach real cluster IDs to the
    # wrong tiles — the exact failure this codebase is written against.
    verdict = tile_name_verdict(tiles_raw)
    if verdict == "mixed":
        raise SystemExit(
            f"{h5_path} has SOME tile names with a file extension and some "
            f"without. That is what a packaging resume straddling the tile-name "
            f"fix leaves behind, and the two cannot be told apart by name, so "
            f"the suffix cannot be filled in. Repackage this dataset."
        )
    tiles, renamed = normalize_tile_names(tiles_raw)

    frame = pd.DataFrame({
        "samples": samples, "slides": slides, "tiles": tiles,
        "image_index": np.arange(n, dtype=np.int64),
    })
    frame["slide_tile"] = make_slide_tile_series(frame["slides"], frame["tiles"])
    # Carried on the frame rather than returned alongside it: read_h5_identity's
    # single return value is what every caller and test already expects.
    frame.attrs["tile_names_normalized"] = renamed
    return frame


def read_tile_coordinates(tile_dir: Path, tile_dataset_name: str,
                          slide_ids) -> tuple[pd.DataFrame, list[str]]:
    """Per-tile coordinates from Stage 1's metadata CSVs, for exactly the
    slides present in the .h5.

    Returns (frame, missing_slides) rather than raising on a missing slide:
    partial coverage is reported and left for the caller to decide about,
    since refusing the whole registration for one bad slide out of thousands
    would be worse than the CLAUDE.md-preferred "loud refusal" here — the
    tiles from every OTHER slide are still real and still worth registering.
    """
    frames, missing = [], []
    for slide_id in slide_ids:
        path = tile_metadata_path(tile_dir, tile_dataset_name, slide_id)
        meta = read_tile_metadata(path)
        if not meta.usable:
            missing.append(f"{slide_id} ({meta.status}: {meta.detail})")
            continue
        frames.append(_with_identity_as_text(meta.frame, path))

    if not frames:
        return pd.DataFrame(columns=["slides", "tiles", "col", "row", "x_5x",
                                     "y_5x", "x_native", "y_native", "slide_tile"]), missing

    coords = pd.concat(frames, ignore_index=True)
    # Stage 1 metadata written before the same fix carries short names too, and
    # normalising only the .h5 would leave this side short — which does not
    # fail, it comes back as tiles_with_coordinates: 0, the silent version of
    # the bug the .h5 guard used to catch loudly.
    verdict = tile_name_verdict(coords["tiles"])
    if verdict == "mixed":
        raise SystemExit(
            f"Stage 1 metadata under {tile_dir / tile_dataset_name} has some "
            f"tile names with a file extension and some without, so the suffix "
            f"cannot be filled in. Re-tile the slides this covers."
        )
    coords["tiles"], renamed = normalize_tile_names(coords["tiles"])
    coords["slide_tile"] = make_slide_tile_series(coords["slides"], coords["tiles"])
    coords.attrs["tile_names_normalized"] = renamed
    return coords, missing


def _with_identity_as_text(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """`slides` and `tiles` exactly as Stage 1's CSV spells them.

    tile_metadata.read_tile_metadata() lets pandas guess column types, which it
    must for col/row — its null check is what catches a CSV truncated
    mid-write. For the two identity columns that guess is a rewrite: an
    all-digit slide id comes back as a number ('007' -> 7) and one called 'NA'
    as NaN, so the coordinates' key is '7_1_1.JPEG' against the .h5's
    '007_1_1.JPEG' and the slide registers with tiles_with_coordinates: 0 —
    well-formed, and silently wrong. Only these two columns are re-read, as
    text; the file is small, and the length check makes certain the two reads
    line up row for row rather than assuming it.
    """
    identity = pd.read_csv(path, usecols=["slides", "tiles"], dtype=str,
                           keep_default_na=False)
    if len(identity) != len(frame) or not frame.index.equals(identity.index):
        raise SystemExit(
            f"{path}: re-reading slides/tiles as text gave {len(identity):,} "
            f"row(s) against {len(frame):,} from the metadata reader, so the "
            f"two cannot be attached to each other. Re-tile this slide.")
    frame = frame.copy()
    frame["slides"] = identity["slides"]
    frame["tiles"] = identity["tiles"]
    return frame


def find_slide_files(raw_dir: Path, slide_ids) -> tuple[dict, list[str], list[str]]:
    """Locate the raw slide file behind each slide in the .h5.

    Matching is on slide_id_from_raw_path(), the same function tile_mask.py and
    auto_tile_from_mask.py used to derive the names Stage 1 wrote — so a file
    found here is the file those tiles came from, rather than one that merely
    sorts next to them.

    Returns (slide_id -> Path, missing, ambiguous). Ambiguity is returned, not
    resolved: two files claiming one slide_id means the viewer would serve
    whichever this function happened to pick, and being wrong about which
    physical slide a cohort's tiles came from is not a tie to break silently.
    """
    wanted = {s.strip().upper(): s for s in slide_ids}
    found: dict[str, list[Path]] = {}

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SLIDE_SUFFIXES:
            continue
        key = slide_id_from_raw_path(path).strip().upper()
        if key in wanted:
            found.setdefault(key, []).append(path)

    resolved = {wanted[k]: v[0] for k, v in found.items() if len(v) == 1}
    ambiguous = [
        f"{wanted[k]} -> {', '.join(str(p) for p in v)}"
        for k, v in sorted(found.items()) if len(v) > 1
    ]
    missing = sorted(orig for k, orig in wanted.items() if k not in found)
    return resolved, missing, ambiguous


def build_wsi_registry(slide_files: dict, samples_by_slide: dict,
                       dataset_id: str) -> pd.DataFrame:
    """One wsi_registry row per slide, from the file on disk.

    slide_id is upper-cased because every reader upper-cases before looking it
    up — _load_wsi_map() and _open_slide() in tile_server_v2_.py, and
    migrate_indexes.sql normalises the column in place. Writing it any other
    way produces a row that exists and is never found.
    """
    rows = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for slide_id, path in sorted(slide_files.items()):
        stat = path.stat()
        rows.append({
            "slide_id": slide_id.strip().upper(),
            "sample_id": samples_by_slide.get(slide_id),
            "file_uuid": file_uuid_from_raw_path(path),
            "filename": path.name,
            "hpc_path": str(path.resolve()),
            "file_size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                                 .replace(tzinfo=None),
            "added_at": now,
            "dataset_id": dataset_id,
        })
    return pd.DataFrame(rows, columns=list(_WSI_REGISTRY_COLUMNS))


def read_slide_metadata(slide_files: dict, samples_by_slide: dict,
                        dataset_id: str) -> tuple[pd.DataFrame, list[str]]:
    """What OpenSlide reports about each slide, for wsi_metadata.

    Opens every slide, so it is opt-in (--slide-metadata): 14,044 headers is
    minutes of network I/O, not seconds. mpp_x is the column that stopped the
    viewer assuming every slide in every cohort was scanned at 0.252 µm/px —
    tile_server_v2_._tile_size_native() derives each slide's tile size from
    the tiles' own coordinates, and falls back to the slide's mpp. It reads
    mpp off the slide rather than out of this table, so wsi_metadata is still
    on no read path; what this docstring used to describe as hypothetical is
    the reason the numbers are worth capturing.

    A slide that fails to open is reported, not raised: one unreadable file out
    of thousands should not cost the registration of the rest, and the tile
    tables do not depend on this.
    """
    try:
        import openslide
    except ImportError as exc:
        raise SystemExit(
            f"--slide-metadata needs openslide-python ({exc}). Drop the flag to "
            f"register everything else; wsi_metadata is not on any read path."
        )

    rows, unreadable = [], []
    for slide_id, path in sorted(slide_files.items()):
        try:
            with openslide.OpenSlide(str(path)) as slide:
                props = slide.properties
                rows.append({
                    "slide_id": slide_id.strip().upper(),
                    "sample_id": samples_by_slide.get(slide_id),
                    "level_count": int(slide.level_count),
                    "level_dimensions_json": json.dumps(
                        [list(d) for d in slide.level_dimensions]),
                    "level_downsamples_json": json.dumps(
                        [float(d) for d in slide.level_downsamples]),
                    "mpp_x": _as_float(props.get(openslide.PROPERTY_NAME_MPP_X)),
                    "mpp_y": _as_float(props.get(openslide.PROPERTY_NAME_MPP_Y)),
                    "objective_power": _as_float(
                        props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)),
                    "vendor": props.get(openslide.PROPERTY_NAME_VENDOR),
                    "scanner_model": _scanner_model(props),
                    "scanner_date": _scanner_date(props),
                    "tile_width": _as_int(props.get("openslide.level[0].tile-width")),
                    "tile_height": _as_int(props.get("openslide.level[0].tile-height")),
                    "quickhash": props.get(openslide.PROPERTY_NAME_QUICKHASH1),
                    "dataset_id": dataset_id,
                })
        except Exception as exc:  # openslide raises several unrelated types
            unreadable.append(f"{slide_id} ({type(exc).__name__}: {exc})")

    return pd.DataFrame(rows, columns=list(_WSI_METADATA_COLUMNS)), unreadable


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scanner_model(props):
    """Vendor-specific, so tried per vendor rather than guessed at generically.
    None when the vendor does not publish one — which is most of them."""
    for key in ("aperio.ScanScope ID", "aperio.User", "hamamatsu.Product",
                "philips.DICOM_MANUFACTURERS_MODEL_NAME",
                "leica.device-model", "ventana.DeviceSerialNumber"):
        if props.get(key):
            return str(props[key])
    return None


def _scanner_date(props):
    """Aperio splits the scan timestamp across two properties and writes the
    date American-style. Returns None rather than a guess on anything else —
    a wrong scan date is worse than an absent one, since it would be used to
    reason about scanner drift."""
    date, time = props.get("aperio.Date"), props.get("aperio.Time")
    if not date:
        return None
    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
        try:
            stamp = datetime.strptime(date, fmt)
            break
        except ValueError:
            continue
    else:
        return None
    if time:
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(time, fmt)
                stamp = stamp.replace(hour=parsed.hour, minute=parsed.minute,
                                      second=parsed.second)
                break
            except ValueError:
                continue
    return stamp


def build_registration(h5_path: Path, tile_dir: Path, tile_dataset_name: str,
                       h5_source_path: str, dataset_id: str,
                       raw_dir: Path | None = None,
                       slide_metadata: bool = False,
                       target_mpp: float | None = None,
                       tile_size_5x_px: int | None = None,
                       scope: str = "full",
                       slide_names: list[str] | None = None) -> dict:
    """Everything preview()/commit() need, computed without touching the DB."""
    identity = read_h5_identity(h5_path)

    scope = (scope or "full").strip().lower()

    if scope not in {"full", "subset"}:
        raise SystemExit("scope must be 'full' or 'subset'")

    requested_slides = None

    if scope == "subset":
        requested_slides = [
            str(s).strip()
            for s in (slide_names or [])
            if str(s).strip()
        ]

        if not requested_slides:
            raise SystemExit(
                "Subset registration requires at least one slide ID or filename."
            )

        wanted = {s.upper() for s in requested_slides}

        available = {
            str(s).strip().upper()
            for s in identity["slides"].dropna().unique()
        }

        missing_requested = [
            s for s in requested_slides
            if s.upper() not in available
        ]

        if missing_requested:
            raise SystemExit(
                "Requested subset contains slide(s) not present in the packaged .h5: "
                + ", ".join(missing_requested[:20])
            )

        identity = identity[
            identity["slides"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(wanted)
        ].copy()

        if identity.empty:
            raise SystemExit(
                "Subset selection matched zero tiles in the packaged .h5."
            )

    slide_ids = sorted(set(identity["slides"]))
    coords, missing_slides = read_tile_coordinates(tile_dir, tile_dataset_name, slide_ids)

    merged = identity.merge(
        coords[["slide_tile", "col", "row", "x_5x", "y_5x", "x_native", "y_native"]],
        on="slide_tile", how="left", validate="one_to_one",
    )
    unmatched = merged[merged["col"].isna()]

    registry = merged[["samples", "slides", "tiles", "slide_tile", "image_index"]].copy()
    registry["h5_source_path"] = h5_source_path
    registry["dataset_id"] = dataset_id

    coordinates = merged.dropna(subset=["col"]).copy()
    coordinates = coordinates[["slides", "tiles", "slide_tile", "col", "row",
                               "x_5x", "y_5x", "x_native", "y_native", "image_index"]]
    coordinates = coordinates.rename(columns={"image_index": "h5_index"})
    coordinates["dataset_id"] = dataset_id
    for col in ("col", "row", "x_5x", "y_5x", "x_native", "y_native", "h5_index"):
        coordinates[col] = coordinates[col].astype(np.int64)

    # samples is per-tile in the .h5 but is a slide-level fact; take the first
    # and check the slide does not disagree with itself, since a slide mapped
    # to two sample ids would put half its tiles under the wrong patient.
    per_slide = identity.groupby("slides")["samples"].agg(["first", "nunique"])
    conflicting_samples = sorted(per_slide.index[per_slide["nunique"] > 1])
    samples_by_slide = per_slide["first"].to_dict()

    wsi_registry = pd.DataFrame(columns=list(_WSI_REGISTRY_COLUMNS))
    wsi_metadata = pd.DataFrame(columns=list(_WSI_METADATA_COLUMNS))
    dataset_config = pd.DataFrame(columns=list(_DATASET_CONFIG_COLUMNS))
    slides_without_files: list[str] = []
    ambiguous_slides: list[str] = []
    unreadable_slides: list[str] = []

    if raw_dir is not None:
        slide_files, slides_without_files, ambiguous_slides = find_slide_files(
            raw_dir, slide_ids)
        wsi_registry = build_wsi_registry(slide_files, samples_by_slide, dataset_id)
        if slide_metadata:
            wsi_metadata, unreadable_slides = read_slide_metadata(
                slide_files, samples_by_slide, dataset_id)

    if target_mpp is not None and tile_size_5x_px is not None:
        dataset_config = pd.DataFrame([{
            "dataset_id": dataset_id,
            "target_mpp": float(target_mpp),
            "tile_size_5x_px": int(tile_size_5x_px),
        }], columns=list(_DATASET_CONFIG_COLUMNS))

    return {
        "registry": registry,
        "coordinates": coordinates,
        "wsi_registry": wsi_registry,
        "wsi_metadata": wsi_metadata,
        "dataset_config": dataset_config,
        "slides": slide_ids,
        "scope": scope,
        "requested_slides": requested_slides,
        "missing_slides": missing_slides,
        "unmatched_tiles": unmatched["slide_tile"].tolist(),
        "slides_without_files": slides_without_files,
        "ambiguous_slides": ambiguous_slides,
        "unreadable_slides": unreadable_slides,
        "conflicting_samples": conflicting_samples,
        # How many tile names on each side had ".jpeg" appended to make the join
        # key. Reported rather than silent: this is a correction to identity,
        # and the whole argument for making it automatically is that it is
        # visible when it happens.
        "tile_names_normalized": {
            "h5": int(identity.attrs.get("tile_names_normalized", 0)),
            "coordinates": int(coords.attrs.get("tile_names_normalized", 0)),
        },
    }


def _table_exists(conn, table: str) -> bool:
    return sqlalchemy_inspect(conn).has_table(table)


# The column each table is keyed by, and the column that counts distinct
# slides in it. dataset_config has neither — it is one row per cohort.
_KEY_COLUMN = {
    "tile_registry": "slide_tile",
    "tile_coordinates": "slide_tile",
    "wsi_registry": "slide_id",
    "wsi_metadata": "slide_id",
}
_SLIDE_COLUMN = {
    "tile_registry": "slides",
    "tile_coordinates": "slides",
    "wsi_registry": "slide_id",
    "wsi_metadata": "slide_id",
}


def _existing_scope(conn, dataset_id: str) -> dict:
    """What already exists in the KB for this dataset_id, across every table
    this script writes.

    A table that is absent reports as absent rather than raising: these tables
    predate this script and a database that has not run
    migrate_kb_base_tables.sql may genuinely not have all of them. The commit
    path refuses on that; the preview should still be able to say so.
    """
    scope = {}
    for table in _TABLES:
        if not _table_exists(conn, table):
            scope[table] = {"rows": None, "slides": None, "missing": True}
            continue
        if table == "dataset_config":
            rows = conn.execute(
                text("SELECT COUNT(*) FROM dataset_config WHERE dataset_id = :d"),
                {"d": dataset_id},
            ).scalar()
            scope[table] = {"rows": rows, "slides": None, "missing": False}
            continue
        row = conn.execute(
            text(f"SELECT COUNT(*) AS n, "
                 f"COUNT(DISTINCT {_SLIDE_COLUMN[table]}) AS slides "
                 f"FROM {table} WHERE dataset_id = :d"),
            {"d": dataset_id},
        ).mappings().one()
        scope[table] = {"rows": row["n"], "slides": row["slides"], "missing": False}
    return scope


def _foreign_scope(conn, table: str, dataset_id: str, keys) -> int:
    """How many of these keys already belong to a DIFFERENT dataset_id.

    Non-zero means two cohorts are claiming the same tile, or the same slide —
    refused as an error, since a collision here means one of the two datasets
    is misidentified, not that the newer one should win. For wsi_registry the
    consequence is sharper than for the tile tables: slide_id is that table's
    whole primary key, so overwriting one would repoint the viewer at another
    cohort's file while every tile row still says otherwise.
    """
    if not _table_exists(conn, table):
        return 0
    key = _KEY_COLUMN[table]
    total = 0
    values = list(keys)
    lookup = text(
        f"SELECT COUNT(*) AS n FROM {table} "
        f"WHERE UPPER({key}) IN :keys AND dataset_id != :d"
    ).bindparams(bindparam("keys", expanding=True))
    for start in range(0, len(values), _LOOKUP_CHUNK):
        total += conn.execute(
            lookup, {"keys": [str(v).upper() for v in values[start:start + _LOOKUP_CHUNK]],
                     "d": dataset_id},
        ).scalar()
    return total


_PLAN_KEY = {
    "tile_registry": "registry",
    "tile_coordinates": "coordinates",
    "wsi_registry": "wsi_registry",
    "wsi_metadata": "wsi_metadata",
    "dataset_config": "dataset_config",
}


def preview(engine, plan: dict, dataset_id: str) -> dict:
    """What committing would do, without doing it."""
    with engine.connect() as conn:
        existing = _existing_scope(conn, dataset_id)
        foreign = {}
        for table, key in _KEY_COLUMN.items():
            frame = plan[_PLAN_KEY[table]]
            foreign[table] = (
                _foreign_scope(conn, table, dataset_id, frame[key])
                if not frame.empty else 0
            )
    return {
        "dataset_id": dataset_id,
        "slides": len(plan["slides"]),
        "tiles_in_h5": len(plan["registry"]),
        "tiles_with_coordinates": len(plan["coordinates"]),
        "slides_registered": len(plan["wsi_registry"]),
        "slides_with_metadata": len(plan["wsi_metadata"]),
        "dataset_config_rows": len(plan["dataset_config"]),
        "missing_slides": plan["missing_slides"],
        "unmatched_tiles": plan["unmatched_tiles"],
        "slides_without_files": plan["slides_without_files"],
        "ambiguous_slides": plan["ambiguous_slides"],
        "unreadable_slides": plan["unreadable_slides"],
        "conflicting_samples": plan["conflicting_samples"],
        "tile_names_normalized": plan["tile_names_normalized"],
        "existing": existing,
        "foreign_collisions": foreign,
    }


def _native(value):
    """A pandas value as something a DBAPI driver will bind.

    Building a DataFrame promotes datetimes to pandas.Timestamp and any column
    with a gap to float64, so `NULL` arrives as float('nan') and a timestamp as
    a type sqlite3 refuses outright. psycopg2 happens to accept Timestamp
    (it subclasses datetime) and would coerce nan into a numeric column as NaN
    — which is the worse outcome of the two, because it succeeds.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _insert(conn, table: str, frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    """Reflect-before-insert, same pattern as load_hpc_assignments.py: these
    tables predate this script, so a column this script assumes and the live
    table lacks must not fail the whole transaction silently — it is reported
    and the column is dropped from what's written."""
    existing_columns = {c["name"] for c in sqlalchemy_inspect(conn).get_columns(table)}
    usable = [c for c in columns if c in existing_columns]
    skipped = [c for c in columns if c not in existing_columns]
    if skipped:
        print(f"  {table}: no column(s) {skipped}; not writing them", file=sys.stderr)
    placeholders = ", ".join(f":{c}" for c in usable)
    records = [
        {k: _native(v) for k, v in record.items()}
        for record in frame[usable].to_dict("records")
    ]
    conn.execute(
        text(f"INSERT INTO {table} ({', '.join(usable)}) VALUES ({placeholders})"),
        records,
    )
    return len(frame)


_COLUMNS_FOR = {
    "tile_registry": _TILE_REGISTRY_COLUMNS,
    "tile_coordinates": _TILE_COORDINATES_COLUMNS,
    "wsi_registry": _WSI_REGISTRY_COLUMNS,
    "wsi_metadata": _WSI_METADATA_COLUMNS,
    "dataset_config": _DATASET_CONFIG_COLUMNS,
}


def commit(engine, plan: dict, dataset_id: str, replace: bool) -> dict:
    """Write every identity table this run has data for, scoped to dataset_id.

    One transaction, matching Stage 5's own reasoning: a registration that
    wrote tile_registry but not tile_coordinates (or the reverse) would leave
    the viewer able to find a tile's position but not its identity, or the
    other way round, and nothing downstream could tell that had happened. The
    same argument is why wsi_registry is in this transaction and not a separate
    command — tiles whose slide the viewer cannot open are the same class of
    half-registered cohort.

    Deleting before inserting rather than upserting, and only for this exact
    dataset_id: a --replace scoped by anything looser could touch another
    cohort's rows sharing a coincidentally identical slide_tile or slide_id.
    """
    if plan["registry"].empty:
        raise SystemExit("Nothing to register — the .h5 held no tiles.")

    writable = [t for t in _TABLES if not plan[_PLAN_KEY[t]].empty]

    with engine.begin() as conn:
        absent = [t for t in writable if not _table_exists(conn, t)]
        if absent:
            raise SystemExit(
                f"This database has no {', '.join(absent)}. Run "
                f"`psql ... -f backend/migrate_kb_base_tables.sql` first — "
                f"eight of the Knowledge Bank's tables had no CREATE TABLE in "
                f"git until that file existed, so a database built from "
                f"schema.sql is missing them."
            )

        existing = _existing_scope(conn, dataset_id)
        occupied = {t: existing[t]["rows"] for t in writable if existing[t]["rows"]}
        if occupied and not replace:
            detail = ", ".join(f"{n:,} in {t}" for t, n in occupied.items())
            raise SystemExit(
                f"{dataset_id} already has {detail}. Pass --replace to overwrite "
                f"them — this dataset_id is the intended re-registration path "
                f"once a fuller run supersedes this one, but it is never "
                f"automatic."
            )

        foreign = {}
        for table in writable:
            if table not in _KEY_COLUMN:
                continue
            n = _foreign_scope(conn, table, dataset_id,
                               plan[_PLAN_KEY[table]][_KEY_COLUMN[table]])
            if n:
                foreign[table] = n
        if foreign:
            detail = ", ".join(f"{n:,} in {t}" for t, n in foreign.items())
            raise SystemExit(
                f"{detail} already belong to a DIFFERENT dataset_id. Two cohorts "
                f"cannot claim the same tile or the same slide — this needs "
                f"investigating, not overwriting."
            )

        if replace:
            # Reverse order, so a table is never left referencing rows that
            # have already gone.
            for table in reversed(writable):
                conn.execute(text(f"DELETE FROM {table} WHERE dataset_id = :d"),
                             {"d": dataset_id})

        written = {
            table: _insert(conn, table, plan[_PLAN_KEY[table]], _COLUMNS_FOR[table])
            for table in writable
        }
    return written


def report(result: dict, commit_mode: bool) -> None:
    print(f"\ndataset_id     {result['dataset_id']}")
    print(f"slides         {result['slides']:,}")
    print(f"tiles in .h5   {result['tiles_in_h5']:,}")
    print(f"with coords    {result['tiles_with_coordinates']:,}")
    print(f"slides in wsi_registry  {result['slides_registered']:,}")
    print(f"slides with metadata    {result['slides_with_metadata']:,}")
    print(f"dataset_config rows     {result['dataset_config_rows']:,}")

    renamed = result.get("tile_names_normalized") or {}
    if any(renamed.values()):
        print(f"\ntile names            .jpeg appended to "
              f"{renamed.get('h5', 0):,} name(s) from the .h5 and "
              f"{renamed.get('coordinates', 0):,} from Stage 1's metadata, so "
              f"they match the '18_15.jpeg' form the Knowledge Bank joins on. "
              f"The artifacts on disk still hold the short form — "
              f"migrate_tile_names.py fixes them there.")

    if not result["slides_registered"]:
        print("\nNo wsi_registry rows: --raw-dir was not given. The tiles will "
              "be registered, Stage 5 will load, and the viewer will still 404 "
              "on every slide in this cohort — _open_slide() resolves paths "
              "from wsi_registry alone. Pass --raw-dir unless these slides are "
              "already registered under this dataset_id.")

    if result["conflicting_samples"]:
        print(f"\n{len(result['conflicting_samples'])} slide(s) carry more than "
              f"one sample_id in the .h5 — the first was used, which means some "
              f"of their tiles are attributed to the wrong sample:")
        for s in result["conflicting_samples"][:10]:
            print(f"  {s}")

    if result["ambiguous_slides"]:
        print(f"\nREFUSING to guess: {len(result['ambiguous_slides'])} slide id(s) "
              f"match more than one file under --raw-dir. Neither was registered:")
        for s in result["ambiguous_slides"][:10]:
            print(f"  {s}")

    if result["slides_without_files"]:
        print(f"\n{len(result['slides_without_files'])} slide(s) in the .h5 have "
              f"no raw file under --raw-dir. Their tiles are registered; the "
              f"slide itself will not open in the viewer:")
        for s in result["slides_without_files"][:10]:
            print(f"  {s}")

    if result["unreadable_slides"]:
        print(f"\n{len(result['unreadable_slides'])} slide(s) could not be opened "
              f"for metadata. They are still in wsi_registry:")
        for s in result["unreadable_slides"][:10]:
            print(f"  {s}")

    if result["missing_slides"]:
        print(f"\n{len(result['missing_slides'])} slide(s) have no usable Stage 1 "
              f"metadata — their tiles will be registered with NO tile_coordinates "
              f"row (position on the slide unknown to the viewer):")
        for s in result["missing_slides"][:10]:
            print(f"  {s}")

    if result["unmatched_tiles"]:
        print(f"\n{len(result['unmatched_tiles'])} tile(s) in the .h5 have no "
              f"matching row in Stage 1's metadata (e.g. {result['unmatched_tiles'][:3]}) "
              f"— registered in tile_registry only.")

    existing = result["existing"]
    absent = [t for t, v in existing.items() if v.get("missing")]
    if absent:
        print(f"\nThis database has no {', '.join(absent)}. Run "
              f"backend/migrate_kb_base_tables.sql before committing.")

    occupied = {t: v["rows"] for t, v in existing.items() if v["rows"]}
    if occupied:
        detail = ", ".join(f"{n:,} in {t}" for t, n in occupied.items())
        print(f"\nAlready present for this dataset_id: {detail}. --replace is "
              f"required to overwrite them.")

    collisions = {t: n for t, n in result["foreign_collisions"].items() if n}
    if collisions:
        detail = ", ".join(f"{n:,} in {t}" for t, n in collisions.items())
        print(f"\nREFUSING: {detail} already belong to a different dataset_id.")

    if not commit_mode:
        print("\nDry run — nothing written. Re-run with --commit to apply.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h5", type=Path, required=True,
                        help="The packaged .h5 from make_hpl_hdf5.py.")
    parser.add_argument("--tile-dir", type=Path, required=True,
                        help="Root directory tiles were written under (the "
                             "tile_dir passed to Stage 1).")
    parser.add_argument("--tile-dataset-name", required=True,
                        help="Folder under --tile-dir tiles live in, e.g. "
                             "Radiogenomics.")
    parser.add_argument("--dataset-id", required=True,
                        help="The KB cohort this dataset belongs to, e.g. "
                             "RADIOGENOMICS. Everything written is scoped to it, "
                             "and a --replace only ever touches its own rows.")
    parser.add_argument("--h5-source-path", default=None,
                        help="Recorded in tile_registry.h5_source_path. Defaults "
                             "to --h5.")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Directory the raw slide files live under, searched "
                             "recursively. Without it wsi_registry is not written "
                             "and the cohort's slides will not open in the viewer.")
    parser.add_argument("--slide-metadata", action="store_true",
                        help="Also read each slide's OpenSlide header into "
                             "wsi_metadata (mpp, objective power, level "
                             "dimensions). Opens every file, so it costs minutes "
                             "on a large cohort. Requires --raw-dir.")
    parser.add_argument("--target-mpp", type=float, default=None,
                        help="Microns per pixel the tiles were produced at, for "
                             "dataset_config. Use the run's recorded "
                             "tiling_params.target_mpp.")
    parser.add_argument("--tile-size-5x", type=int, default=None,
                        help="Tile edge in pixels at the target mpp, for "
                             "dataset_config. Use the run's recorded "
                             "tiling_params.target_tile_px.")
    parser.add_argument("--commit", action="store_true",
                        help="Actually write. Without it this only previews.")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite this dataset_id's existing rows. Refused "
                             "without this flag if any already exist.")
    parser.add_argument(
        "--scope",
        choices=("full", "subset"),
        default="full",
        help="Register the full packaged dataset or only selected slides.",
    )

    parser.add_argument(
        "--slide-name",
        action="append",
        dest="slide_names",
        help="Slide ID to register when --scope subset is used. "
             "Repeat for multiple slides.",
    )
    # Used when this runs as the Slurm job the server submits: the job is the
    # process that knows whether the write committed, so it is the one that
    # records it. Harmless and inert on a hand-run CLI invocation.
    parser.add_argument("--record-run", default=None, metavar="SUBMISSION_ID",
                        help="Record the outcome against this run in "
                             "slurm_dataset_runs.")
    parser.add_argument("--record-run-db", default=None, metavar="DBNAME",
                        help="Database holding slurm_dataset_runs. Run tracking "
                             "stays in production whichever Knowledge Bank the "
                             "rows go to, so this is separate from DB_NAME.")
    parser.add_argument("--record-kb-target", default=None,
                        help="Recorded as registration_kb_target, so the run "
                             "says which Knowledge Bank it filled.")
    args = parser.parse_args()

    if not args.h5.is_file():
        raise SystemExit(f"No such file: {args.h5}")
    if not args.tile_dir.is_dir():
        raise SystemExit(f"No such directory: {args.tile_dir}")
    if args.raw_dir is not None and not args.raw_dir.is_dir():
        raise SystemExit(f"No such directory: {args.raw_dir}")
    if args.slide_metadata and args.raw_dir is None:
        raise SystemExit("--slide-metadata needs --raw-dir: the metadata is read "
                         "from the slide files themselves.")
    # Refused rather than defaulted. dataset_config's two columns are NOT NULL,
    # and a row asserting the wrong tiling geometry is worse than no row —
    # every coordinate conversion downstream would trust it.
    if (args.target_mpp is None) != (args.tile_size_5x is None):
        raise SystemExit("--target-mpp and --tile-size-5x go together: "
                         "dataset_config needs both or neither.")

    if args.scope == "subset" and not args.slide_names:
        raise SystemExit(
            "--scope subset requires at least one --slide-name."
        )

    plan = build_registration(
        args.h5, args.tile_dir, args.tile_dataset_name,
        args.h5_source_path or str(args.h5), args.dataset_id,
        raw_dir=args.raw_dir,
        slide_metadata=args.slide_metadata,
        target_mpp=args.target_mpp,
        tile_size_5x_px=args.tile_size_5x,
        scope=args.scope,
        slide_names=args.slide_names,
    )
    engine = make_engine()

    if not args.commit:
        report(preview(engine, plan, args.dataset_id), commit_mode=False)
        return

    result = preview(engine, plan, args.dataset_id)  # for the report's numbers
    try:
        written = commit(engine, plan, args.dataset_id, args.replace)
    except BaseException as e:
        # Recorded before re-raising so a refusal or a crash leaves a reason on
        # the run rather than a stage that simply stopped saying anything. The
        # job's log has the traceback; this is what the UI can show.
        if args.record_run and args.record_run_db:
            record_run(args.record_run_db, args.record_run,
                       registration_error=f"{type(e).__name__}: {e}"[:2000])
        raise
    report(result, commit_mode=True)
    if args.record_run and args.record_run_db:
        record_run(
            args.record_run_db, args.record_run,
            registration_done=True,
            registration_at=datetime.now(timezone.utc),
            registration_dataset_id=args.dataset_id,
            registration_raw_dir=str(args.raw_dir) if args.raw_dir else None,
            registration_rows=json.dumps(written),
            registration_error=None,
            **({"registration_kb_target": args.record_kb_target}
               if args.record_kb_target else {}),
        )
    print("\nwritten        " + ", ".join(f"{t} +{n:,}" for t, n in written.items()))
    print("\nNext: run load_hpc_assignments.py to fill in hpc_id and the "
          "per-slide aggregates.")
    if written.get("wsi_registry"):
        print("The tile server caches wsi_registry in memory at startup "
              "(_load_wsi_map), so restart it before these slides will open.")


if __name__ == "__main__":
    raise SystemExit(main())
