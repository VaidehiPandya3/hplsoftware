"""
FastAPI Tile Server — runs on HPCC near the .svs files and PostgreSQL.

Endpoints
---------
GET  /health                         → liveness check
GET  /slides                         → list all slide IDs
GET  /slide/{slide_id}/info          → slide metadata (dimensions, levels, mpp)
GET  /dzi/{slide_id}.dzi             → Deep Zoom metadata for OpenSeadragon
GET  /dzi/{slide_id}_files/{z}/{x}_{y}.jpeg → Deep Zoom JPEG tile
GET  /debug/routes                   → list active FastAPI routes
GET  /slide/{slide_id}/thumbnail     → whole-slide JPEG preview
GET  /slide/{slide_id}/tile          → single tile at (level, x, y, w, h)
GET  /slide/{slide_id}/region        → arbitrary region in native coords
GET  /slide/{slide_id}/tiles_meta    → tile_coordinates + tile_registry + hpc join
GET  /slide/{slide_id}/adjacency     → precomputed adjacency pairs
GET  /hpc/{hpc_id}/info              → HPC dictionary + malignant/non-malignant details
GET  /hpc/{hpc_id}/survival          → survival analysis row
POST /upload-slide                   → upload + preprocess WSI, register it, and queue mask+tile pipeline
GET  /slide/{slide_id}/processing-status → poll background mask/tiling status after upload
GET  /dataset-roots                  → list submittable dataset directories under LONG_TERM_SCRATCH
POST /dataset-jobs                   → submit a dataset-wide masking+tiling Slurm array job
GET  /dataset-jobs                   → list past/active dataset job submissions
POST /dataset-jobs/{submission_id}/resume → resubmit whatever slides from a run never got tiled
POST /dataset-jobs/{submission_id}/package → user-triggered: package this run's tiles into .h5
POST /dataset-jobs/{submission_id}/extract-features → user-triggered: run the .h5 through the model
POST /dataset-jobs/{submission_id}/cancel → scancel every Slurm job for this run and mark it cancelled
GET  /dataset-jobs/{submission_id}/status → discovery/Slurm/per-slide tiling status for a dataset job
POST /query                          → full NL query → structured answer
GET  /tile_image/{slide_tile}        → H5-backed tile image by slide_tile key
"""

import getpass
import hashlib
import inspect
import io
import os
import re
import sys
import time
import json
import random
import subprocess
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import openslide
import pandas as pd
from fastapi import (BackgroundTasks, Depends, FastAPI, HTTPException, Query,
                     UploadFile, File, Form)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy import inspect as sqlalchemy_inspect
from db_url import database_url
from openslide.deepzoom import DeepZoomGenerator
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from tile_cache import TileCache
from tile_mask import run_tissue_detection
from auto_tile_from_mask import (DEFAULT_NATIVE_MPP, TARGET_MPP,
                                 TARGET_TILE_PX, native_tile_px,
                                 parse_native_mpp, tile_slide_from_mask)
from slide_naming import slide_id_from_raw_path, tiles_missing_suffix
from submit_mask_tile_slurm import submit_array as submit_dataset_array
from submit_mask_tile_slurm import (
    submit_packaging_job,
    discover_slides,
    select_slides,
    validate_unique_slide_ids,
    write_manifest,
    tiling_output_complete,
)
from make_hpl_hdf5 import _checkpoint_paths, package_slides_to_h5, hpl_h5_output_path
from find_missing_slides import find_missing_slides_detailed
from dataset_rollup import (
    coarse_run_state as _coarse_run_state,
    group_runs_by_dataset,
    job_ids as _split_job_ids,
    rollup_dataset,
)
from submit_feature_extraction import (
    submit_feature_extraction_job,
    expected_extraction_output_path,
    validate_extraction_output as _validate_extraction_output,
    _input_h5_rows as _packaged_h5_rows,
    HPL_REPO_DIR,
)

import cohort_shift
from build_hpc_reference import HPC_REFERENCE_PATH
from submit_cluster_assignment import (
    DEFAULT_VOTE_PRESET,
    VOTE_PRESETS,
    resolve_vote,
    submit_cluster_assignment_job,
    vote_flags,
)

from submit_kb_write import (
    JOB_DB_HOST_ENV,
    check_db_from_compute_node,
    probe_advice,
    resolve_job_db_host,
    submit_kb_load_job,
    submit_registration_job,
)

from submit_anorak_nf import (
    SUPERVISOR_STOP_MARKER,
    slide_list_from_directory as _anorak_directory_slide_list,
    check_submit_from_compute_node as _check_slurm_submit_from_compute_node,
    grades_csv_path as _anorak_grades_csv_path,
    read_slide_csv as _anorak_read_slide_csv,
    select_slide_rows as _anorak_select_slide_rows,
    submit_anorak_job,
    validate_anorak_output as _validate_anorak_output,
)

# Stages 1-4 as one Nextflow run (hpl-nf/, POST /pipeline-runs). The state
# module is what lets a pipeline stage stand in for a Slurm job everywhere below.
from hpl_nf_state import (
    STAGES as NF_STAGES,
    combine_head_states as _nf_combine_head_states,
    done_marker as _nf_done_marker,
    is_nf_job_id as _is_nf_job_id,
    nf_job_id as _nf_job_id,
    parse_nf_job_id as _nf_parse_job_id,
    read_head_job_ids as _nf_read_head_job_ids,
    stage_state as _nf_stage_state,
    stage_summary as _nf_stage_summary,
)
import submit_hpl_nf

from load_hpc_assignments import (
    read_assignments as _read_kb_assignments,
    inspect as _inspect_kb_load,
    load as _write_kb_load,
    compute_profiles as _compute_kb_profiles,
    _MIN_MATCH_RATE as _KB_MIN_MATCH_RATE,
)

# Registration — the step that creates the identity rows Stage 5 UPDATEs.
# Imported, not reimplemented, for the same reason as the Stage 5 functions
# above: one implementation of what makes a write to the shared KB safe.
from register_dataset import (
    build_registration as _build_registration,
    preview as _registration_preview,
    commit as _registration_commit,
)

# The stage-output validators live in stage_outputs.py so the Nextflow tasks
# (hpl-nf/bin/) can apply the very check this server gates on without
# importing this module. Bound under their old names: every call site below,
# and anything that monkeypatches them, is unchanged.
from stage_outputs import (
    ASSIGNMENT_REQUIRED_COLUMNS as _ASSIGNMENT_REQUIRED_COLUMNS,
    HPL_H5_DATASETS as _HPL_H5_DATASETS,
    validate_assignment_csv as _validate_assignment_output,
    validate_packaged_h5 as _validate_h5,
)

# sacct states that mean "still queued or actively running" — anything else
# (COMPLETED, FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, ...) is
# terminal. Used to decide when a "start next stage" button should appear.
# COMPLETING is included even though the job's steps have finished — Slurm's
# epilogue (and any of the job's own writes still flushing to a network
# filesystem) may not be done yet, so a retry-guard or tiling_complete check
# that treated COMPLETING as terminal could let a second packaging/extraction
# job start writing the same output file while the first is still finishing.
# CONFIGURING is a squeue-only state (nodes allocated, prologue still running)
# that sacct rarely surfaces — it only started mattering once squeue became the
# first source consulted in _get_slurm_job_state.
IN_FLIGHT_SLURM_STATES = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING",
}


# Configuration — edit these to match your HPCC environment

DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

# --- Knowledge Bank targets -------------------------------------------------
#
# One server process, two Knowledge Banks. "test" exists so a cohort can be
# registered, loaded, queried and looked at end to end without touching the
# database the group actually uses — which is the only way to find out that a
# registration is wrong before it is shared.
#
# Deliberately a fixed, named set rather than a database name taken from the
# request. An endpoint accepting any string would let a typo open a connection
# to something nobody meant, and would hand a caller the ability to point this
# server at an arbitrary database on the same host.
#
# PRODUCTION IS THE DEFAULT EVERYWHERE. Every parameter added for this defaults
# to "production", so an old client, a bookmarked URL, or a call that omits the
# argument behaves exactly as it did before. The failure that matters is test
# data reaching the real KB; defaulting the other way would make forgetting the
# parameter the dangerous case instead of the harmless one.
DB_NAME_TEST = os.getenv("DB_NAME_TEST", "hpl_kb_test")

KB_PRODUCTION = "production"
KB_TEST = "test"
KB_TARGETS = {KB_PRODUCTION: DB_NAME, KB_TEST: DB_NAME_TEST}

WSI_ROOT = os.getenv("WSI_ROOT", "/hpc-home/home/users/vpandya/long-term-scratch/tcga_wsi")
H5_PATH = os.getenv("H5_PATH", "/hpc-home/home/users/vpandya/long-term-scratch/Vaidehi/TCGA/hdf5_TCGA_LUAD_5x_he_train_tiles.h5")

CACHE_DIR = os.getenv("TILE_CACHE_DIR", "/tmp/hpc_tile_cache")
UPLOAD_ROOT = Path(os.getenv(
    "UPLOAD_ROOT",
    "/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi"
))
UPLOAD_RAW_DIR = UPLOAD_ROOT / "raw"
UPLOAD_METADATA_DIR = UPLOAD_ROOT / "metadata"

# Post-upload pipeline output — same defaults tile_mask.py / auto_tile_from_mask.py use standalone.
TISSUE_MASK_DIR = Path(os.getenv(
    "TISSUE_MASK_DIR",
    "/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks"
))
PROCESSED_TILES_DIR = Path(os.getenv(
    "PROCESSED_TILES_DIR",
    "/hpc-home/home/users/vpandya/long-term-scratch/processed_tiles"
))
# Fraction of a tile's area that must be tissue to keep it. Tiles are now read
# at the paper's exact 403.2um physical size (~40x the area of the old fixed
# 256-native-px tiles), so a threshold tuned for those smaller tiles may need
# lowering here — tune via env var rather than editing code.
MIN_TISSUE_PERCENT = float(os.getenv("MIN_TISSUE_PERCENT", "30.0"))

# Where packaged .h5 files land — same default submit_packaging_job() uses
# (a sibling of backend/), so single-slide and dataset-wide packaging share
# one datasets/ root.
HPL_DATASETS_ROOT = Path(os.getenv(
    "HPL_DATASETS_ROOT", str(Path(__file__).resolve().parent.parent / "model_input")
))

# Dataset-wide Slurm masking+tiling jobs (submit_mask_tile_slurm.py). A
# submittable "dataset" is any direct subdirectory of LONG_TERM_SCRATCH other
# than the pipeline's own output/working directories below — this is
# recomputed on every request rather than hand-maintained, so it can't drift.
LONG_TERM_SCRATCH = Path(os.getenv(
    "LONG_TERM_SCRATCH",
    "/hpc-home/home/users/vpandya/long-term-scratch"
))

# Stage 7, ANORAK growth-pattern grading (submit_anorak_nf.py). Both are
# deployment-level locations rather than per-run choices — the pipeline and the
# upstream clone are installed once per cluster — so they are configured here
# and the UI just says "go". ANORAK_RESULTS_ROOT holds one directory per run,
# because a Nextflow run owns its work/ cache and two runs sharing one would
# resume into each other.
ANORAK_PIPELINE_DIR = Path(os.getenv(
    "ANORAK_PIPELINE_DIR", str(Path(__file__).resolve().parent.parent / "anorak-nf")
))
ANORAK_REPO_DIR = Path(os.getenv(
    "ANORAK_REPO_DIR", str(LONG_TERM_SCRATCH / "Work" / "AIgrading")
))
ANORAK_RESULTS_ROOT = Path(os.getenv(
    "ANORAK_RESULTS_ROOT", str(LONG_TERM_SCRATCH / "anorak")
))
ANORAK_PROFILE = os.getenv("ANORAK_PROFILE", "beatson")

# The one-click pipeline (hpl-nf/, submit_hpl_nf.py). One directory per run,
# named by submission id, for the same reason as ANORAK_RESULTS_ROOT: a
# Nextflow run owns its work/ cache, and two runs sharing one would resume into
# each other. The directory is also how a stage sentinel finds its run with no
# database lookup — see _nf_run_dir.
HPL_NF_RESULTS_ROOT = Path(os.getenv(
    "HPL_NF_RESULTS_ROOT", str(LONG_TERM_SCRATCH / "hpl-nf")
))
HPL_NF_PROFILE = os.getenv("HPL_NF_PROFILE", "beatson")

# The tessellation's two fixed numbers belong to auto_tile_from_mask, which is
# what actually cuts the tiles; importing them is what keeps the box the viewer
# draws the same size as the tile underneath it.
TILE_SIZE_5X = TARGET_TILE_PX

# Fallback only. TILE_SIZE_NATIVE is the pitch of a slide scanned at 0.252
# µm/px, which is 520 of the 1,598 slides in wsi_metadata and was — until
# _tile_size_native() below — asserted of all of them. Ask that function for a
# named slide; reach for this constant only where there is no slide to ask
# about, such as a query-parameter default.
SCALE = TARGET_MPP / DEFAULT_NATIVE_MPP
TILE_SIZE_NATIVE = native_tile_px(DEFAULT_NATIVE_MPP)


# Initialising Globals 

cache: TileCache = None
_h5_handles: dict[str, "h5py.File"] = {}   # resolved .h5 path → open handle
# Everything below is keyed by KB target. It was keyed by nothing, because
# there was one database. A slide_id is only unique *within* a Knowledge Bank:
# the same id can name a different file in hpl_kb_test than in hpl_kb, so a
# cache keyed on slide_id alone would serve production's pixels for a test
# lookup — the quietest possible way to be wrong, since the image renders.
_engines: dict[str, object] = {}                    # target → Engine
_wsi_maps: dict[str, dict[str, str]] = {}           # target → {slide_id: hpc_path}
_wsi_handles: dict[tuple[str, str], openslide.OpenSlide] = {}   # (target, slide_id)
_dz_handles: dict[tuple[str, str], DeepZoomGenerator] = {}      # (target, slide_id)
#: target → its p_hpc_* column names, or None when the table is absent. The
#: probabilities themselves are no longer cached: they are read per slide.
_heatmap_columns_cache: dict[str, "list | None"] = {}
_processing_status: dict[str, dict] = {}  # slide_id → {status, stage, error, ...}
# submission_id → (succeeded, zero_tile, not_attempted) slide-id lists, once
# computed for the first time after that run's tiling is fully complete. A
# run's manifest never changes after it's written, so once tiling_complete
# is true this breakdown can never change either — safe to cache forever
# rather than re-scanning up to one file per slide on every 10s status poll,
# which was blowing past even a 60s HTTP timeout for a 14,000+ slide run.
# Lost on server restart, which just means one slow recompute, not a
# correctness issue.
_tiling_breakdown_cache: dict[str, tuple[list[str], list[str], list[str]]] = {}

# Serializes "check whether a packaging/extraction attempt is already in
# flight, then submit a new one if not" across all four submission
# endpoints below. Without this, that check-then-submit was two separate
# steps with nothing stopping two near-simultaneous requests (a double
# click, two browser tabs, or — what actually happened — repeated retries
# each hitting the guard before Slurm's own accounting had caught up on
# the previous attempt) from both passing the check and both submitting.
# Every one of these submissions writes to the exact same deterministic
# output path (real runs) or "<name>_test_sample" path (test runs), so two
# concurrent writers isn't just wasted compute, it's a genuine data race on
# the same file. Serializing all of them (even across different runs)
# costs nothing that matters here — these are rare, human-triggered
# actions, not a throughput-sensitive path.
#
# A threading.Lock() used to guard this, which only serializes within a
# single process. This server runs under multiple Uvicorn worker
# processes (see __main__ below), each with its own separate memory, so
# a plain in-process lock left the exact race above wide open across
# workers — two requests landing on different workers both saw "no
# attempt in flight" and both submitted. A Postgres advisory lock is
# process-agnostic (it's coordinated by the DB, not by memory this
# process happens to own), so it actually serializes across every worker
# talking to the same database, using the same `engine` already used
# everywhere else in this file.
_SLURM_LOCK_KEY = 927341  # arbitrary constant; identifies this one lock


@contextmanager
def _slurm_submission_lock():
    eng = _get_engine()
    conn = eng.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_lock(%s)", (_SLURM_LOCK_KEY,))
        cur.close()
        conn.commit()
        try:
            yield
        finally:
            cur = conn.cursor()
            cur.execute("SELECT pg_advisory_unlock(%s)", (_SLURM_LOCK_KEY,))
            cur.close()
            conn.commit()
    finally:
        conn.close()


def _resolve_kb_target(kb_target: str | None) -> str:
    """The canonical target name, or a 400 naming the ones that exist.

    Fixed set, not a free-form database name: an endpoint that accepted any
    string would let a typo open a connection to something nobody meant, and
    would hand a caller the ability to point this server at any database on the
    same host.
    """
    target = (kb_target or KB_PRODUCTION).strip().lower()
    if target not in KB_TARGETS:
        raise HTTPException(
            400, f"Unknown kb_target {kb_target!r}. Expected one of "
                 f"{sorted(KB_TARGETS)}.")
    return target


def kb_target_param(kb_target: str = Query(
    KB_PRODUCTION,
    description="Which Knowledge Bank to read: 'production' (hpl_kb) or "
                "'test' (hpl_kb_test). Defaults to production.")) -> str:
    """Declared once as a dependency rather than repeated on every read
    endpoint, so the set of valid targets and the default cannot drift between
    the fourteen places that honour it."""
    return _resolve_kb_target(kb_target)


def _get_engine(kb_target: str = KB_PRODUCTION):
    """One pooled engine per Knowledge Bank, created on first use.

    Lazily, and per target: the test database may not exist on a given
    deployment, and building an engine for it at import time would turn "we
    have not made hpl_kb_test yet" into a server that will not start.
    """
    target = _resolve_kb_target(kb_target)
    if target not in _engines:
        _engines[target] = create_engine(
            database_url(KB_TARGETS[target], user=DB_USER, password=DB_PASS,
                         host=DB_HOST, port=DB_PORT),
            pool_pre_ping=True,
            pool_size=5,
        )
    return _engines[target]


def _get_h5(h5_path: str | None = None):
    """An open handle to a packaged .h5, keyed by path.

    Was a single global bound to H5_PATH — one TCGA file, for every slide of
    every cohort. That is why a Radiogenomics tile rendered a TCGA image
    instead of failing: the index existed in the file, so it returned pixels.
    Callers now pass the .h5 that actually holds the slide (see
    _h5_for_slide_tile); omitting it falls back to H5_PATH so the legacy TCGA
    cohort, whose rows predate any per-dataset path, still resolves.
    """
    path = h5_path or H5_PATH
    if path not in _h5_handles:
        if not os.path.isfile(path):
            raise HTTPException(404, f"Packaged .h5 not found on disk: {path}")
        _h5_handles[path] = h5py.File(path, "r", swmr=True)
    return _h5_handles[path]


def _load_wsi_map(kb_target: str = KB_PRODUCTION) -> dict[str, str]:
    """Rebuild one target's slide_id → path map from its own wsi_registry."""
    target = _resolve_kb_target(kb_target)
    eng = _get_engine(target)
    df = pd.read_sql("SELECT slide_id, hpc_path FROM wsi_registry", eng)
    df["slide_id"] = df["slide_id"].astype(str).str.strip().str.upper()
    _wsi_maps[target] = dict(zip(df["slide_id"], df["hpc_path"]))
    return _wsi_maps[target]


def _get_wsi_map(kb_target: str = KB_PRODUCTION) -> dict[str, str]:
    """Cached per target, loaded on first use.

    Production's is warmed at startup as before. The test map is not, because
    the database may not exist — so it is built the first time something asks
    for it, and a missing database surfaces then, on a request that named it,
    rather than as a server that will not boot.
    """
    target = _resolve_kb_target(kb_target)
    if target not in _wsi_maps:
        _load_wsi_map(target)
    return _wsi_maps[target]

def _cache_key(slide_id: str, kb_target: str) -> str:
    """The TileCache is keyed by slide_id, which is only unique within one
    Knowledge Bank. Production keeps its bare slide_id so every JPEG already on
    disk stays valid; anything else is namespaced, so a test cohort cannot serve
    a production slide's cached pixels under the same id."""
    return slide_id if kb_target == KB_PRODUCTION else f"{kb_target}::{slide_id}"


# Marks an ad-hoc upload apart from the bulk cohorts (TCGA_LUAD_5x and the
# rest). "UPLOADED" on its own is what every upload used to be tagged with,
# and rows written before this still carry it — which is why the check below
# is a prefix rather than an equality.
UPLOADED_DATASET_ID = "UPLOADED"
UPLOADED_DATASET_PREFIX = "UPLOADED_"


def upload_dataset_name(slide_id: str) -> str:
    """The cohort *and* tile-folder name for one uploaded slide.

    One per slide, not one shared "UPLOADED" bucket, and that is load-bearing
    in two places rather than cosmetic:

      * register_dataset.commit() scopes everything to dataset_id, and a
        dataset_id that already holds rows can only be re-registered with
        --replace, which DELETEs them. Sharing one cohort across uploads would
        mean registering the second slide either refused or deleted the first.
      * tiles land in <PROCESSED_TILES_DIR>/<this>/<slide_id>/, so the folder
        the run records is the folder registration reads Stage 1's
        _tile_metadata.csv out of, with no special case.

    Keeping the "UPLOADED_" prefix is what preserves the original reason this
    was scoped at all: a bulk dataset submitted under a name that happens to
    match an uploaded slide_id still cannot collide with it on disk.
    """
    return f"{UPLOADED_DATASET_PREFIX}{slide_id.strip().upper()}"


def _is_upload_dataset_id(dataset_id) -> bool:
    """Whether a wsi_registry row was put there by an upload rather than by a
    curated cohort. Accepts the bare legacy "UPLOADED" as well as the
    per-slide names, so a slide uploaded before this change is still
    recognised as the caller's own upload and not as somebody else's dataset.
    """
    value = str(dataset_id or "")
    return value == UPLOADED_DATASET_ID or value.startswith(UPLOADED_DATASET_PREFIX)


def _register_uploaded_slide(slide_id: str, hpc_path: str, dataset_id: str):
    """Register or update an uploaded WSI so existing viewer endpoints can open it.

    dataset_id is written on conflict too, not just on insert. Stage 5 refuses
    outright when a slide_id it is registering already belongs to a *different*
    dataset_id ("two cohorts cannot claim the same slide"), so a row left
    tagged with a previous upload's cohort — or with the legacy bare
    "UPLOADED" — would block the registration of the very slide it describes.
    """
    slide_id = slide_id.strip().upper()
    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO wsi_registry (slide_id, hpc_path, dataset_id)
                VALUES (:slide_id, :hpc_path, :dataset_id)
                ON CONFLICT (slide_id)
                DO UPDATE SET hpc_path = EXCLUDED.hpc_path,
                              dataset_id = EXCLUDED.dataset_id
            """),
            {"slide_id": slide_id, "hpc_path": hpc_path, "dataset_id": dataset_id},
        )

    # Uploads are a production-only path: _register_uploaded_slide hard-codes
    # dataset_id='UPLOADED' and there is no upload-into-test flow.
    _load_wsi_map(KB_PRODUCTION)
    _wsi_handles.pop((KB_PRODUCTION, slide_id), None)
    _dz_handles.pop((KB_PRODUCTION, slide_id), None)


def _set_processing_status(slide_id: str, status: str, error: str | None = None):
    """Write-through: Postgres is the source of truth (survives restarts /
    would survive multiple workers); the in-memory dict is just a fast local
    cache for the common case of a single long-lived worker.
    """
    slide_id = slide_id.strip().upper()
    _processing_status[slide_id] = {"status": status, "error": error}
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    UPDATE wsi_registry
                    SET processing_status = :status,
                        processing_error = :error,
                        processing_updated_at = now()
                    WHERE slide_id = :slide_id
                """),
                {"slide_id": slide_id, "status": status, "error": error},
            )
    except Exception as e:
        print(f"[{slide_id}] failed to persist processing_status={status}: {e}")


def _get_processing_status(slide_id: str) -> dict:
    slide_id = slide_id.strip().upper()
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT processing_status, processing_error
                    FROM wsi_registry WHERE slide_id = :slide_id
                """),
                {"slide_id": slide_id},
            ).fetchone()
        if row and row[0]:
            return {"status": row[0], "error": row[1]}
    except Exception as e:
        print(f"[{slide_id}] failed to read processing_status from DB: {e}")

    return _processing_status.get(slide_id, {"status": "not_started", "error": None})


def _run_postupload_pipeline(slide_id: str, raw_path: str, submission_id: str | None = None):
    """Background job: tissue mask -> 224px tiles at Kai's 1.8um/px resolution.

    Runs after /upload-slide returns so the HTTP request doesn't block on a
    full-slide tiling pass. Progress is tracked via _set_processing_status so
    the UI can poll GET /slide/{slide_id}/processing-status.

    submission_id is the slurm_dataset_runs row /upload-slide created for this
    slide (see _start_upload_run). Recording Stages 1 and 2 against it is what
    carries an uploaded slide on into Stages 3-6 — feature extraction, cluster
    assignment, registration and the Knowledge Bank load are all keyed on a
    run, and before this an upload had none, so its tiles stopped at the .h5
    and never reached the Knowledge Bank or the viewer's overlays at all.
    None keeps this callable without one.
    """
    dataset_name = upload_dataset_name(slide_id)

    def _record(**fields):
        """Bookkeeping must never take the pipeline down with it: the tiles
        and the .h5 are real whether or not this row can be written."""
        if submission_id:
            _update_dataset_run_best_effort(submission_id, **fields)

    try:
        _set_processing_status(slide_id, "masking")
        _record(status="running")
        # Scoped under this slide's own upload dataset name rather than flat
        # under TISSUE_MASK_DIR/PROCESSED_TILES_DIR directly — those same flat
        # dirs are also where bulk dataset submissions (TCGA, Radiogenomics,
        # ...) write, scoped by their own dataset_name (see run_worker() in
        # submit_mask_tile_slurm.py). Without this, an ad-hoc upload whose
        # user-supplied slide_id happens to match a real dataset's slide_id
        # would silently overwrite that dataset's mask/tiles on disk — a
        # second, filesystem-level version of the wsi_registry collision
        # /upload-slide already guards against; that guard alone doesn't
        # cover this, since it only checks the DB row, not these directories.
        # It is also the folder registration reads this run's coordinates out
        # of, which is why it is the run's recorded dataset_name too.
        # slide_id is passed explicitly rather than left to be derived from
        # raw_path: the id here is the one the user chose and the one already
        # written to wsi_registry, so re-deriving it would be trusting a
        # filename over the record it has to agree with.
        upload_mask_dir = TISSUE_MASK_DIR / dataset_name
        mask_result = run_tissue_detection(
            slide_path=raw_path,
            output_dir=str(upload_mask_dir),
            slide_id=slide_id,
        )
        # Backstop: confirm masking actually wrote where it was told to,
        # not just that slide_id's charset was valid — see _resolve_within.
        _resolve_within(upload_mask_dir, Path(mask_result["mask_path"]))
        _resolve_within(upload_mask_dir, Path(mask_result["overlay_path"]))

        _set_processing_status(slide_id, "tiling")
        upload_tile_dir = PROCESSED_TILES_DIR / dataset_name
        tile_summary = tile_slide_from_mask(
            slide_path=raw_path,
            mask_path=mask_result["mask_path"],
            output_dir=str(upload_tile_dir),
            min_tissue_percent=MIN_TISSUE_PERCENT,
            slide_id=slide_id,
        )
        _resolve_within(upload_tile_dir, Path(tile_summary["output_dir"]))

        # Stage 1 is finished and its output is on disk. The sentinel job id is
        # what the status endpoint and the stepper read as "tiling completed"
        # — see LOCAL_JOB_ID_PREFIX — and it is written here, after the tiler
        # returned, rather than when the run row was created.
        _record(status="submitted", job_id=_local_job_id("tiling"))

        if tile_summary.get("saved_tiles", 0) == 0:
            # package_slides_to_h5 raises RuntimeError when total_tiles == 0
            # (no per-slide fallback there — it's shared with the multi-slide
            # dataset path, where "every slide produced zero tiles" is a real
            # error worth stopping on). For a single upload, tissue below
            # MIN_TISSUE_PERCENT is an expected, non-fatal outcome — skip
            # packaging instead of letting that raise get caught below and
            # reported as a packaging failure when nothing was actually wrong.
            no_tissue = ("Tiling found no tissue above the tissue threshold — "
                         "nothing to package.")
            _set_processing_status(slide_id, "done", error=no_tissue)
            # Not status="error": tiling ran and answered. The run stops here
            # because there is nothing to carry forward, and saying so on the
            # row keeps the pipeline view from offering Stage 3 a .h5 that was
            # never written.
            _record(status="completed", total_slides=1, error=no_tissue)
            return

        _set_processing_status(slide_id, "packaging")
        try:
            packaged = package_slides_to_h5(
                raw_paths=[raw_path],
                tile_dir=PROCESSED_TILES_DIR,
                # tile_dataset_name has no default and was missing entirely
                # here before — this call raised a bare TypeError on every
                # single-slide upload, always caught by the except below and
                # reported as a generic packaging failure. Matches the
                # tile_slide_from_mask output_dir above, which is where
                # tiles are actually written.
                tile_dataset_name=dataset_name,
                output_root=HPL_DATASETS_ROOT,
                dataset_name=dataset_name,
                # Same reason as the slide_id passed to run_tissue_detection
                # / tile_slide_from_mask above: the tiles were written under
                # the slide_id the user chose, and a slide_id re-derived here
                # would look for them under whatever the filename says.
                slide_ids=[slide_id],
            )
            _set_processing_status(slide_id, "done")
            # Stage 2's own sentinel, and the path Stage 3 reads. Taken from
            # what packaging returned rather than recomputed here, so the run
            # records the file that was actually written.
            _record(
                status="completed",
                total_slides=1,
                h5_job_id=_local_job_id("packaging"),
                h5_output_path=packaged["output_h5_path"],
            )
        except Exception as e:
            # Tiles are real and usable either way — a packaging failure
            # shouldn't be reported as if masking/tiling itself failed.
            message = f"Tiling succeeded, but .h5 packaging failed: {e}"
            _set_processing_status(slide_id, "done", error=message)
            # No h5_job_id: Stage 3 gates on one, and recording a sentinel for
            # a stage that raised is exactly the "plausible result" this
            # codebase refuses. The run stays open at Stage 2 instead.
            _record(status="completed", total_slides=1, error=message)
    except Exception as e:
        _set_processing_status(slide_id, "error", error=str(e))
        _record(status="error", error=str(e))


def _upload_tiling_params() -> dict:
    """What the in-process tiler will actually run this slide with.

    Read off tile_slide_from_mask()'s own signature for the same reason
    _default_tiling_params() reads submit_array()'s — a hardcoded copy drifts
    silently, and these numbers are not decoration: registration writes them
    into dataset_config, so a wrong pair claims a cohort was tessellated at a
    resolution it was not. min_tissue is the exception, because
    _run_postupload_pipeline passes MIN_TISSUE_PERCENT explicitly rather than
    taking the tiler's default.
    """
    # Which function actually owns each parameter, and under what name there:
    # the three mask_* values are run_tissue_detection's, the rest are the
    # tiler's, and min_tissue is neither — _run_postupload_pipeline passes
    # MIN_TISSUE_PERCENT rather than accepting a default.
    owners = {
        "target_mpp": (tile_slide_from_mask, "target_mpp"),
        "target_tile_px": (tile_slide_from_mask, "target_tile_px"),
        "level": (tile_slide_from_mask, "level"),
        "jpeg_quality": (tile_slide_from_mask, "jpeg_quality"),
        "mask_max_size": (run_tissue_detection, "max_size"),
        "mask_saturation": (run_tissue_detection, "saturation_threshold"),
        "mask_value": (run_tissue_detection, "value_threshold"),
    }
    params = {"min_tissue": MIN_TISSUE_PERCENT}
    for name in _TILING_PARAM_NAMES:
        if name == "min_tissue":
            continue
        owner, argument = owners.get(name, (None, None))
        parameter = (inspect.signature(owner).parameters.get(argument)
                     if owner is not None else None)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"no defaulted source for tiling parameter '{name}' on the "
                f"upload path — _upload_tiling_params needs updating."
            )
        params[name] = parameter.default
    return params


def _upload_run_for_slide(slide_id: str) -> dict:
    """The pipeline run belonging to an uploaded slide, newest first.

    Looked up by dataset_name rather than remembered in memory so it survives
    a server restart and a reloaded browser, and so re-uploading a slide_id
    (which starts a fresh run over the replaced file) resolves to the run that
    matches what is on disk now. Returns empty values rather than raising: a
    slide with no run is the ordinary state for everything that arrived before
    uploads had one.
    """
    dataset_name = upload_dataset_name(slide_id)
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT submission_id FROM slurm_dataset_runs
                    WHERE dataset_name = :dataset_name
                    ORDER BY submitted_at DESC LIMIT 1
                """),
                {"dataset_name": dataset_name},
            ).fetchone()
    except Exception as e:
        print(f"[{slide_id}] could not look up this upload's run: {e}")
        return {"submission_id": None, "dataset_name": dataset_name}
    return {"submission_id": row[0] if row else None, "dataset_name": dataset_name}


def _start_upload_run(slide_id: str, raw_path: Path) -> str | None:
    """Create the slurm_dataset_runs row a single uploaded slide runs on.

    An upload is a one-slide dataset run in every way the later stages care
    about, so it gets a real row rather than a special case: Stages 3-6 are
    keyed on submission_id, and this is what gives an uploaded slide the same
    feature extraction, cluster assignment, registration and Knowledge Bank
    load a cohort gets. Stages 1 and 2 still run in this process (a single
    slide does not need Slurm) and record themselves against the row as they
    finish — see _run_postupload_pipeline.

    raw_dir is this upload's own UUID directory, not the shared upload pool:
    registration matches raw slide files by slide_id_from_raw_path(), and a
    directory holding exactly this slide cannot produce the "two files claim
    one slide_id" ambiguity that a re-upload would otherwise create for the
    whole pool.

    Returns None rather than raising if the row cannot be written. The upload
    itself has already succeeded at this point and the slide is viewable; the
    pipeline view is the thing that degrades, and failing the request would
    throw away a file the user has already waited on.
    """
    dataset_name = upload_dataset_name(slide_id)
    submission_id = str(uuid.uuid4())
    # The manifest is how the status endpoint reads each slide's
    # _tiling_summary.json back off disk, and how a later stage knows which
    # slides this run set out to cover. Same one-path-per-line format and the
    # same directory the Slurm path writes to.
    manifest_path = Path(__file__).resolve().parent / "slurm_manifests" / \
        f"wsi_manifest_upload_{submission_id}.txt"
    try:
        write_manifest([raw_path], manifest_path)
    except OSError as e:
        print(f"[{slide_id}] could not write the upload manifest: {e}")
        return None

    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO slurm_dataset_runs
                        (submission_id, raw_dir, mask_dir, tile_dir, status,
                         is_subset, dataset_name, tiling_params, manifest_path,
                         total_slides)
                    VALUES
                        (:submission_id, :raw_dir, :mask_dir, :tile_dir, 'queued',
                         false, :dataset_name, :tiling_params, :manifest_path, 1)
                """),
                {
                    "submission_id": submission_id,
                    "raw_dir": str(raw_path.parent),
                    "mask_dir": str(TISSUE_MASK_DIR),
                    "tile_dir": str(PROCESSED_TILES_DIR),
                    "dataset_name": dataset_name,
                    "tiling_params": json.dumps(_upload_tiling_params()),
                    "manifest_path": str(manifest_path),
                },
            )
    except Exception as e:
        print(f"[{slide_id}] could not create a pipeline run for this upload: {e}")
        return None
    return submission_id


# Dataset-wide Slurm job submission and status


def _list_dataset_roots() -> list[str]:
    """Direct subdirectories of LONG_TERM_SCRATCH that are valid dataset inputs.

    Excludes the pipeline's own output/working directories so they can never
    be picked as "a dataset to tile" — recomputed fresh each call from the
    same constants those directories are actually built from, so it can't
    drift out of sync if those env vars change.
    """
    excluded = {TISSUE_MASK_DIR.name, PROCESSED_TILES_DIR.name, UPLOAD_ROOT.name}
    if not LONG_TERM_SCRATCH.is_dir():
        return []
    return sorted(
        p.name for p in LONG_TERM_SCRATCH.iterdir()
        if p.is_dir() and p.name not in excluded and not p.name.startswith(".")
    )


# Shapes sacct emits in its JobID column, all of which have to be told apart:
#
#   12345              a plain, non-array job
#   12345_7            one task of an array job
#   12345_[5-100]      a *pending* array range — one row standing for many
#                      tasks, optionally "%throttle" or a comma list
#   12345.batch        a job step; duplicates its parent's accounting and must
#                      never be counted
#
# The previous single pattern (^\d+_(\d+)\|(\S+)$) matched only the second of
# these. It silently dropped pending array ranges — so tiling could look
# finished while tasks were still queued — and, because "(\S+)$" cannot span a
# space, it also dropped every "CANCELLED by <uid>" row, hiding cancelled
# tasks from the packaging guard that is supposed to block on them. Steps were
# excluded only by accident, since "12345.batch" happens not to match; that is
# now explicit, because the patterns below deliberately accept bare job IDs.
_SACCT_ARRAY_TASK_RE = re.compile(r"^\d+_\d+$")
_SACCT_ARRAY_RANGE_RE = re.compile(r"^\d+_\[(.+?)\]$")
_SACCT_PLAIN_JOB_RE = re.compile(r"^\d+$")


def _array_range_size(spec: str) -> int:
    """How many array tasks a pending-range JobID stands for.

    "5-100" -> 96, "5,7,9" -> 3, "5-10%2" -> 6 (the %N concurrency throttle
    is not part of the task set). Counting the real span rather than treating
    the row as a single task keeps the state totals meaningful — a run with
    9,000 tasks still queued should not report one PENDING.
    """
    spec = spec.split("%", 1)[0]
    total = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, dash, high = part.partition("-")
        if dash:
            try:
                total += int(high) - int(low) + 1
                continue
            except ValueError:
                pass
        total += 1
    return max(total, 1)


def _normalise_slurm_state(state: str) -> str:
    """Bare state name, dropping any trailing detail sacct appends.

    The one that matters is "CANCELLED by 1234" — sacct records who cancelled
    a job, and callers compare against plain state names, so the suffix has to
    come off for a cancelled task to be recognised as cancelled at all.
    """
    state = state.strip()
    return state.split()[0] if state else ""

# squeue exits non-zero when asked about a job ID it has no record of. That's
# a real answer ("not live"), not a failure, and has to be told apart from an
# actual problem — see _slurm_jobs_live_states.
_SQUEUE_UNKNOWN_JOB_RE = re.compile(r"invalid job id|invalid user", re.I)


# --- stages that ran in this process, not on Slurm --------------------------
#
# A single-slide upload masks, tiles and packages inline (see
# _run_postupload_pipeline) — there is no sbatch and therefore no job id, but
# every gate downstream is written against one: "which Slurm state is this
# stage in" is how the status endpoint and the pipeline stepper decide whether
# Stage 3 may start. Recording a sentinel instead of NULL is what lets an
# uploaded slide use that machinery unchanged rather than growing a second,
# parallel set of gates that would drift from it.
#
# The sentinel is only ever written *after* the in-process stage returned, so
# reporting it as COMPLETED is a record of something that happened rather than
# a guess. It is not the whole gate either: _job_output_ready still runs the
# stage's validator (_validate_h5 opens the .h5 and reads its first and last
# row), so a stage that returned and left an unusable output is still refused.
#
# Filtering happens inside the four Slurm query helpers rather than at their
# call sites, because a sentinel reaching sacct is not a local problem: sacct
# rejects the whole call for one unknown id, so a single upload run in a
# listing would have reported "can't reach Slurm" for every real run beside it.
LOCAL_JOB_ID_PREFIX = "local:"


def _local_job_id(stage: str) -> str:
    """The id recorded for a stage this server ran itself."""
    return f"{LOCAL_JOB_ID_PREFIX}{stage}"


def _is_local_job_id(job_id) -> bool:
    return str(job_id or "").startswith(LOCAL_JOB_ID_PREFIX)


# The second kind of sentinel: a stage run inside the Nextflow pipeline
# (POST /pipeline-runs). Recorded as nf:<submission_id>:<stage> because the
# pipeline's per-task jobs are submitted by its head job, not by this server,
# and no one of them is "the stage's job". hpl_nf_state answers for it from the
# stage's own done marker plus the head job's state — see that module. Kept out
# of every Slurm query for the same reason as local:, and answered by
# _sentinel_state() rather than assumed COMPLETED, because unlike an in-process
# stage a pipeline stage can still be running, or have failed.
def _is_sentinel_job_id(job_id) -> bool:
    return _is_local_job_id(job_id) or _is_nf_job_id(job_id)


def _split_local_job_ids(job_ids: list[str]) -> tuple[list[str], list[str]]:
    """(ids Slurm knows about, sentinel ids it must never be asked about)."""
    sentinels = [j for j in job_ids if _is_sentinel_job_id(j)]
    return [j for j in job_ids if not _is_sentinel_job_id(j)], sentinels


# Head-job states per pipeline run, briefly. One /status call resolves four
# stage sentinels for the same run, and each would otherwise ask squeue about
# the same head job; a listing asks for every run at once.
_NF_HEAD_STATE_CACHE: dict[str, tuple[float, str | None]] = {}
_NF_HEAD_STATE_TTL_S = 5.0


def _nf_run_dir(submission_id: str) -> Path:
    return HPL_NF_RESULTS_ROOT / submission_id


def _nf_head_state(submission_id: str) -> str | None:
    """The pipeline run's head-job state (chain folded in), or None if unknown."""
    now = time.monotonic()
    cached = _NF_HEAD_STATE_CACHE.get(submission_id)
    if cached and now - cached[0] < _NF_HEAD_STATE_TTL_S:
        return cached[1]
    head_ids = _nf_read_head_job_ids(_nf_run_dir(submission_id))
    if not head_ids:
        # Written the moment sbatch answers. Absent means the submission is
        # still in flight in the background task, or failed before sbatch —
        # the row's status/error say which. Either way nothing is running.
        state = "PENDING" if _nf_submission_pending(submission_id) else "FAILED"
    else:
        state = _nf_combine_head_states([_get_slurm_job_state(j) for j in head_ids])
    _NF_HEAD_STATE_CACHE[submission_id] = (now, state)
    return state


def _nf_submission_pending(submission_id: str) -> bool:
    try:
        row = _get_dataset_run_row(submission_id)
    except HTTPException:
        return False
    return row.get("status") in ("queued", "discovering", "submitting")


def _sentinel_state(job_id: str) -> str | None:
    """Slurm-vocabulary state for a sentinel id of either kind."""
    if _is_local_job_id(job_id):
        # Written only after the in-process stage returned — see
        # LOCAL_JOB_ID_PREFIX. The output validator still has the final say.
        return "COMPLETED"
    try:
        submission_id, stage = _nf_parse_job_id(job_id)
    except ValueError:
        return None
    run_dir = _nf_run_dir(submission_id)
    # A finished stage answers from its marker alone. Asking Slurm (or the
    # database) about the head job would add nothing, and a listing of old
    # pipeline runs would otherwise cost a squeue per run for every poll.
    if _nf_done_marker(run_dir, stage).is_file():
        return "COMPLETED"
    return _nf_stage_state(run_dir, stage, _nf_head_state(submission_id))


def _sentinel_state_sets(sentinels: list[str]) -> dict[str, set[str]]:
    """{sentinel: {state}} for the per-job helpers. Unknown is an empty set,
    which reads as "no record" — never as a state that was not observed."""
    out: dict[str, set[str]] = {}
    for job_id in sentinels:
        state = _sentinel_state(job_id)
        out[job_id] = {state} if state else set()
    return out


def _sentinel_state_counts(sentinels: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job_id in sentinels:
        state = _sentinel_state(job_id)
        if state:
            counts[state] = counts.get(state, 0) + 1
    return counts


def _run_slurm(cmd: list[str], timeout: int) -> subprocess.CompletedProcess | None:
    """Run a read-only Slurm query. Returns None if the result can't be trusted.

    Every one of these calls used to read result.stdout without ever looking
    at returncode. A failing sacct/squeue writes its error to stderr and
    leaves stdout empty — byte-identical to "ran fine, nothing matched" — and
    callers assign those opposite meanings. "Nothing matched" is specifically
    what _job_output_ready reads as "this job aged out of the accounting
    retention window, so it finished long ago and an existing output file can
    be trusted", so a controller hiccup or an auth failure could be silently
    promoted into evidence that a job succeeded. Distinguishing them is the
    whole job of this wrapper.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[slurm] {cmd[0]} unavailable: {e}")
        return None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:200]
        print(f"[slurm] {' '.join(cmd)} exited {result.returncode}: {detail}")
        return None
    return result


def _slurm_jobs_live_states(job_ids: list[str]) -> list[str] | None:
    """States squeue reports for these job IDs right now, straight from
    the live scheduler queue — not sacct's accounting database.

    sacct goes through slurmdbd, which syncs on its own schedule and can
    lag behind a fresh sbatch submission by anywhere from seconds to
    longer; a job that was just submitted can legitimately have zero sacct
    rows yet even though it's sitting right there in the queue. squeue has
    no such lag, so it's the only reliable way to tell "sacct just hasn't
    caught up" apart from "this job genuinely isn't live" (finished long
    ago, aged out of retention). Returns None if squeue itself couldn't be
    reached (missing/timed out) — genuinely unknown, don't guess. Returns
    [] if squeue ran fine and simply has no rows for these job IDs (not
    currently queued or running, by any name).
    """
    job_ids, _local = _split_local_job_ids(job_ids)
    # A stage this server ran itself was never in the queue, and asking squeue
    # about its sentinel would fail the call for the real ids beside it.
    if not job_ids:
        return []
    cmd = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%T"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[{','.join(job_ids)}] squeue unavailable: {e}")
        return None
    if result.returncode != 0:
        # Not handled by _run_slurm, because this one call has a non-zero exit
        # that is a legitimate answer rather than a failure: squeue rejects a
        # job ID it has no record of with "Invalid job id specified", which is
        # precisely the "not currently live" result this function exists to
        # report. Everything else (controller unreachable, auth) is genuinely
        # unknown and must stay None so callers don't mistake it for proof.
        if _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            # Parse stdout anyway rather than returning [] outright. With a
            # list of IDs, squeue reports the unknown ones on stderr and still
            # prints rows for the valid ones, exiting non-zero for the whole
            # call. Returning [] here therefore threw away live rows whenever
            # a single ID in the batch had aged out of the controller —
            # claiming "nothing is running" while tiling was demonstrably
            # still going. Empty stdout still yields [], the intended answer
            # when every ID really is unknown.
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
        print(f"[{','.join(job_ids)}] squeue exited {result.returncode}: "
              f"{(result.stderr or '').strip()[:200]}")
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _slurm_controller_known_jobs(job_ids: list[str]) -> list[str] | None:
    """The subset of job_ids that slurmctld still holds a record of.

    This exists because --dependency is resolved by the controller and
    nothing else. sacct reads slurmdbd, whose retention is days to weeks;
    the controller forgets a job MinJobAge seconds after it ends (default
    300). So sacct's view and sbatch's view of "does this job exist" diverge
    within minutes of a run finishing, and a dependency on a job only sacct
    remembers is rejected outright with "Job dependency problem". Asking
    scontrol is asking the same component sbatch is about to ask, which is
    the only view that actually predicts whether the submission succeeds.

    Queried one ID at a time rather than as a list, because the answer we
    need is per-ID: a single unknown ID among live ones is exactly the mixed
    case worth resolving precisely, and a combined query collapses it into
    one pass/fail. That costs one local RPC per batch (~15 for a large
    dataset), which only happens on an explicit submit.

    A non-zero exit is a real answer here, not a failure — scontrol rejects
    an ID it has no record of with "Invalid job id specified", which is
    precisely the "controller has forgotten this" result being asked for.
    Same distinction _slurm_jobs_live_states draws, and the same regex.
    Returns None if scontrol itself is unusable, so callers can tell "the
    controller does not know these jobs" from "we could not ask" — only the
    former is grounds for dropping a dependency.
    """
    known: list[str] = []
    for job_id in job_ids:
        cmd = ["scontrol", "show", "job", job_id]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[{job_id}] scontrol unavailable: {e}")
            return None
        if result.returncode == 0:
            known.append(job_id)
            continue
        if _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            continue
        print(f"[{job_id}] scontrol exited {result.returncode}: "
              f"{(result.stderr or '').strip()[:200]}")
        return None
    return known


def _get_slurm_array_state_counts(job_ids: list[str]) -> dict[str, int] | None:
    """Aggregate task-state counts across one or more (possibly array)
    Slurm jobs, via a single combined sacct call instead of one call per
    job. A large dataset split into ~15 batches used to call sacct once
    per batch sequentially here — each querying a big array job's full
    task list — which was slow enough (SLURM controller + accounting DB
    load, thousands of task rows per call) to blow past the UI's own
    30s HTTP read timeout while a run was still actively tiling.

    Returns None if sacct itself couldn't be reached (missing binary or
    timed out) — genuinely unknown state, the caller should NOT treat this
    the same as an empty {} result. {} means BOTH sacct ran fine and
    returned zero rows for these job IDs AND squeue confirms none of them
    are currently live — for an old-enough run that combination means it's
    aged out of Slurm's accounting-DB retention window (commonly a few
    days), i.e. long finished, not "still pending." sacct alone returning
    nothing isn't enough to conclude that on its own (see
    _slurm_jobs_live_states) — conflating "aged out" with "just submitted,
    not indexed yet" used to either leave old runs permanently stuck
    showing "tiling in progress," or (worse) make a job submitted moments
    ago look instantly "complete."
    """
    job_ids, local = _split_local_job_ids(job_ids)
    # One sentinel stands for one stage — for local:, one this server ran to
    # completion itself, which counts as one COMPLETED task and is what makes
    # tiling_complete true for an uploaded slide whose Stage 1 never went
    # through sbatch. A pipeline stage counts as one task in whatever state its
    # markers and head job put it.
    local_counts = _sentinel_state_counts(local)
    if not job_ids:
        return local_counts
    result = _run_slurm(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=45,
    )
    if result is None:
        return None

    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        job_field, separator, state_field = line.strip().partition("|")
        if not separator:
            continue
        job_field = job_field.strip()
        if "." in job_field:
            # A job step ("12345.batch", "12345_5.extern"). These repeat their
            # parent's state, so counting them would inflate every total.
            continue
        state = _normalise_slurm_state(state_field)
        if not state:
            continue
        if _SACCT_ARRAY_TASK_RE.match(job_field) or _SACCT_PLAIN_JOB_RE.match(job_field):
            counts[state] = counts.get(state, 0) + 1
            continue
        pending_range = _SACCT_ARRAY_RANGE_RE.match(job_field)
        if pending_range:
            counts[state] = counts.get(state, 0) + _array_range_size(pending_range.group(1))
    if counts:
        if any(state in IN_FLIGHT_SLURM_STATES for state in counts):
            # sacct says some tasks are still queued/running. Check that
            # against the live queue before believing it: slurmdbd lags, and a
            # task that died (TIMEOUT, OOM, node failure) keeps its last
            # RUNNING row until accounting catches up. tiling_complete is
            # computed as "no in-flight state present", so one stale row kept
            # the whole run pinned at "tiling in progress" — which in turn
            # left the packaging step blocked, hiding its Full-dataset /
            # subset options entirely.
            #
            # An empty squeue result is proof of absence here (not merely a
            # missing answer): _slurm_jobs_live_states returns None when it
            # could not ask, and only [] when the scheduler answered and holds
            # none of these IDs. In that case drop the stale in-flight rows.
            # Dropping rather than guessing an outcome is deliberate — the
            # caller's fallback for thin/empty counts is to read each slide's
            # _tiling_summary.json off disk, which is ground truth about what
            # actually finished, and far better evidence than a state we would
            # otherwise have to invent.
            live_states = _slurm_jobs_live_states(job_ids)
            if live_states == []:
                stale = {s: n for s, n in counts.items() if s in IN_FLIGHT_SLURM_STATES}
                print(
                    f"[{','.join(job_ids)}] sacct reports {stale} but squeue holds none of "
                    f"these jobs — treating the accounting rows as stale."
                )
                counts = {s: n for s, n in counts.items() if s not in IN_FLIGHT_SLURM_STATES}
        return _merged_counts(counts, local_counts)

    live_states = _slurm_jobs_live_states(job_ids)
    if live_states is None:
        return None
    for state in live_states:
        counts[state] = counts.get(state, 0) + 1
    return _merged_counts(counts, local_counts)


def _merged_counts(*count_dicts: dict[str, int]) -> dict[str, int]:
    """Sum per-state counts. Kept separate so the in-process stages' own
    COMPLETED is added at every exit rather than seeded into the tally sacct's
    own answer is judged against — a real job with nothing in accounting must
    still fall through to squeue, even when a sentinel sits beside it."""
    merged: dict[str, int] = {}
    for counts in count_dicts:
        for state, n in counts.items():
            merged[state] = merged.get(state, 0) + n
    return merged


def _slurm_states_by_job(
    job_ids: list[str], *, timeout: int = 45
) -> dict[str, set[str]] | None:
    """Every state seen per job ID, from ONE sacct call.

    _get_slurm_job_state answers for a single job and _get_slurm_array_state_counts
    aggregates across jobs while discarding which job each state came from.
    Neither can label a *list* of runs: the first needs one call per run (ten
    sequential sacct calls to draw a ten-row list, on a controller already slow
    enough to have blown the UI's 30s read timeout mid-run), and the second
    cannot tell them apart afterwards.

    Keyed by base job ID, so array tasks ("123_5") and job steps ("123.batch")
    both fold into "123" — a caller asking "how is run X doing" wants the whole
    array's states together, not one row per task.

    Returns None if sacct could not be reached at all: genuinely unknown, which
    callers must not render as "finished". An empty set for a job ID means sacct
    ran and had nothing for it — aged out of retention, or too fresh to have
    been written yet, and those two are only distinguishable via squeue.

    timeout is a parameter because "one call" does not bound the work: sacct's
    cost scales with the *tasks* behind the IDs, not the IDs themselves, and
    tiling records one array job per ~1000-slide batch. A whole-history listing
    reached 321 IDs standing for tens of thousands of tasks, which took longer
    than the 45s default every single time and therefore returned None — the
    entire UI reading "can't reach Slurm" while Slurm was perfectly healthy.
    Callers spanning many runs must pass a smaller timeout and a smaller batch
    (see _listing_job_states) rather than inheriting a default sized for one.
    """
    job_ids, local = _split_local_job_ids(job_ids)
    # Sentinels answer for themselves and are kept out of the sacct call, which
    # would otherwise be rejected whole for one id it has never heard of — that
    # is the difference between one upload run being unlabelled and every run in
    # the listing reading "can't reach Slurm".
    local_states = _sentinel_state_sets(local)
    if not job_ids:
        return local_states
    result = _run_slurm(
        ["sacct", "-j", ",".join(job_ids), "--format=JobID,State",
         "--parsable2", "--noheader"],
        timeout=timeout,
    )
    if result is None:
        return None

    states: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
    states.update(local_states)
    for line in result.stdout.splitlines():
        job_field, separator, state_field = line.strip().partition("|")
        if not separator:
            continue
        job_field = job_field.strip()
        # "123.batch"/"123_5.extern" repeat their parent's state.
        job_field = job_field.split(".", 1)[0]
        base = job_field.split("_", 1)[0]
        state = _normalise_slurm_state(state_field)
        if not state:
            continue
        if base in states:
            states[base].add(state)
    return states


# Bounds for the dataset listing's Slurm lookup. The listing is polled and spans
# every run ever recorded, so it cannot afford the per-run treatment: it asks
# about recent jobs precisely and lets older ones be answered by the queue plus
# whatever is on disk.
#
# WINDOW_DAYS is Slurm's accounting retention as configured here. Asking sacct
# about a job older than that is not merely wasted work — it returns nothing, so
# the answer is identical to not having asked, at the price of the slowest part
# of the call. Set it *shorter* than the real retention rather than longer; the
# cost of being wrong in that direction is one extra squeue row, and in the
# other direction it is a job whose failure we never notice.
_LISTING_SACCT_WINDOW_DAYS = 10
_LISTING_SACCT_CHUNK = 40
_LISTING_SACCT_CHUNK_TIMEOUT = 12
_LISTING_SACCT_BUDGET_S = 24.0
_LISTING_SQUEUE_CHUNK = 100


def _squeue_states_by_job(job_ids: list[str]) -> dict[str, set[str]] | None:
    """Live queue state per base job ID — squeue only, no accounting DB.

    _slurm_jobs_live_states answers the same question but discards which job
    each state belonged to, which is fine for one run and useless for a listing
    of fifty. Chunked because the ID list runs to the hundreds here and a single
    argument that long is worth avoiding regardless of what the shell tolerates.

    A chunk that fails is skipped rather than fatal: squeue exits non-zero for a
    batch containing any ID the controller has already forgotten (see
    _slurm_jobs_live_states), which for a listing of historic runs is the normal
    case, not an error. Returns None only if *every* chunk failed, i.e. squeue
    itself is unreachable.
    """
    # A stage this server ran itself is not in the queue and never was; its
    # sentinel is answered here rather than sent to squeue, which would fail
    # the chunk it travelled in and blank every real job beside it.
    job_ids, local = _split_local_job_ids(job_ids)
    local_states: dict[str, set[str]] = _sentinel_state_sets(local)
    if not job_ids:
        return local_states

    states: dict[str, set[str]] = {job_id: set() for job_id in job_ids}
    states.update(local_states)
    any_answered = bool(local_states)
    for start in range(0, len(job_ids), _LISTING_SQUEUE_CHUNK):
        chunk = job_ids[start:start + _LISTING_SQUEUE_CHUNK]
        try:
            result = subprocess.run(
                ["squeue", "-j", ",".join(chunk), "-h", "-o", "%i|%T"],
                capture_output=True, text=True, timeout=15,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[listing] squeue unavailable: {e}")
            continue
        if result.returncode != 0 and not _SQUEUE_UNKNOWN_JOB_RE.search(result.stderr or ""):
            print(f"[listing] squeue exited {result.returncode}: "
                  f"{(result.stderr or '').strip()[:200]}")
            continue
        any_answered = True
        for line in result.stdout.splitlines():
            job_field, separator, state_field = line.strip().partition("|")
            if not separator:
                continue
            base = job_field.strip().split(".", 1)[0].split("_", 1)[0]
            state = _normalise_slurm_state(state_field)
            if state and base in states:
                states[base].add(state)
    return states if any_answered else None


def _listing_job_states(
    datasets: list[dict],
) -> tuple[dict[str, set[str]] | None, bool]:
    """Job states for every run in a dataset listing, on a time budget.

    Returns (states, complete). complete is False when some jobs were left to
    squeue alone — the answer is still usable, but a recent job that *failed*
    can read as merely "no longer queued", so the caller should say so rather
    than present it as the last word. The per-run /dataset-jobs/{id}/status
    endpoint remains the precise view, and is where the UI sends anyone who
    opens a single run.

    Two sources, deliberately:

      * squeue for every ID, because it is cheap at any list length (controller
        memory, no accounting DB) and it alone can say "this is running right
        now" — the one thing a polled listing must never get wrong.
      * sacct only for jobs from recent runs, in bounded chunks, because it is
        the expensive one and is the only way to tell COMPLETED from FAILED.

    An ID that neither source reports gets an empty set, which coarse_run_state
    reads as "no record" — aged out of retention, i.e. long finished. That is
    the same inference _get_slurm_array_state_counts already documents, and it
    is only sound because squeue was asked: without it, "nothing came back"
    would equally describe a job submitted ten seconds ago.
    """
    recent_ids: list[str] = []
    every_id: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=_LISTING_SACCT_WINDOW_DAYS)

    for dataset in datasets:
        for run in dataset["runs"]:
            ids: list[str] = []
            for field in ("job_id", "h5_job_id", "extraction_job_id", "test_h5_job_id"):
                ids.extend(_split_job_ids(run.get(field)))
            if not ids:
                continue
            every_id.extend(ids)
            submitted = _parse_timestamp(run.get("submitted_at"))
            # Unparseable timestamps count as recent: a row we cannot date is
            # more safely treated as one whose outcome still matters than as
            # one old enough to assume finished.
            if submitted is None or submitted >= cutoff:
                recent_ids.extend(ids)

    every_id = sorted(set(every_id))
    recent_ids = sorted(set(recent_ids))
    if not every_id:
        return {}, True

    live = _squeue_states_by_job(every_id)

    accounted: dict[str, set[str]] = {}
    deadline = time.monotonic() + _LISTING_SACCT_BUDGET_S
    complete = True
    sacct_reached = False
    for start in range(0, len(recent_ids), _LISTING_SACCT_CHUNK):
        chunk = recent_ids[start:start + _LISTING_SACCT_CHUNK]
        if time.monotonic() >= deadline:
            print(f"[listing] sacct budget spent; {len(recent_ids) - start} recent "
                  f"job ids left to squeue alone")
            complete = False
            break
        chunk_states = _slurm_states_by_job(
            chunk, timeout=_LISTING_SACCT_CHUNK_TIMEOUT
        )
        if chunk_states is None:
            complete = False
            continue
        sacct_reached = True
        accounted.update(chunk_states)

    if live is None and not sacct_reached:
        return None, False
    if len(recent_ids) < len(every_id):
        # Older jobs were never asked about. Honest, but not the whole story.
        complete = False

    states = {job_id: set(live.get(job_id, set()) if live else set()) for job_id in every_id}
    for job_id, seen in accounted.items():
        states.setdefault(job_id, set()).update(seen)
    return states, complete


def _parse_timestamp(value) -> datetime | None:
    """A tz-aware datetime from whatever the runs table hands back.

    Values arrive as ISO strings via pandas' to_json (which renders naive
    timestamps with a trailing Z) or as datetimes when read directly. A naive
    value is read as UTC, matching how submitted_at is written.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# _coarse_run_state is imported from dataset_rollup rather than defined here:
# the dataset rollup has to classify the same job states this server does, and
# two copies of that ordering would drift the moment one of them learned about
# a new Slurm state.


def _get_slurm_job_state(job_id: str) -> str | None:
    """Single (non-array) job's current Slurm state, e.g. for the h5-packaging
    job. squeue is asked first, sacct only as a fallback.

    That order is the whole point of this function and it used to be the other
    way round. sacct reads slurmdbd, which syncs on its own schedule; squeue
    reads slurmctld, which *is* the scheduler. Whenever the two disagree,
    squeue is right and sacct is merely stale, so consulting sacct first meant
    the UI reported a lagging accounting record in preference to the live
    queue — in both directions:

      * a job that had already died (TIMEOUT, OOM, a crash like the h5 file
        lock failure) kept its last sacct row of RUNNING, so the pipeline
        stepper showed "Running (Slurm state: RUNNING)" and hid the retry
        button, sometimes for minutes after the job was gone;
      * a job submitted moments ago has no sacct row at all yet, which the
        old code only rescued via the fallback below.

    Asking the queue first collapses both cases: if slurmctld still holds the
    job, its state is authoritative and current, full stop. squeue is also the
    cheaper of the two (no accounting DB), so this is not a cost for the
    common "is it still running?" poll.

    Whatever squeue reports is returned as-is rather than assumed to be a
    live state — a job that finished very recently can still appear in the
    queue as COMPLETED/FAILED, and that is a perfectly good terminal answer.

    Return contract is unchanged. None means neither source could be reached
    (genuinely unknown — callers must not guess). "" means squeue confirms the
    job is not in the queue AND sacct has no record of it, which for an
    old-enough job means it aged out of the accounting-DB retention window,
    i.e. long finished (see _job_output_ready).
    """
    if _is_sentinel_job_id(job_id):
        return _sentinel_state(job_id)

    live_states = _slurm_jobs_live_states([job_id])
    if live_states:
        return _normalise_slurm_state(live_states[0])

    result = _run_slurm(
        ["sacct", "-j", job_id, "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is None:
        # squeue said "not in the queue" but sacct can't say how it ended.
        # Not-in-queue is not by itself an outcome, and "" would be read as
        # "aged out, trust the output file" — so stay honestly unknown.
        return None

    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        # Normalised for the same reason as the array counts above: an
        # un-normalised "CANCELLED by 1234" matches neither "COMPLETED" nor
        # any entry in IN_FLIGHT_SLURM_STATES, so it happened to be treated as
        # failed — right answer, but by accident, and it surfaced the raw uid
        # in user-facing messages.
        if len(parts) == 2 and parts[0] == job_id:
            return _normalise_slurm_state(parts[1])

    # sacct ran and has no row for this job. Only call that "aged out" if
    # squeue actually answered; if squeue itself was unreachable (None) we
    # have two non-answers, not evidence.
    if live_states is None:
        return None
    return ""


def _find_job_id_by_name(job_name: str) -> str | None:
    """Most recent Slurm job ID currently known under this exact job name,
    or None if nothing was found (or Slurm couldn't be reached).

    Used only as a reconciliation check before submitting a *new*
    packaging/extraction job for a run whose DB row has no job_id on
    record. That "no job_id" state is ambiguous — it either means nothing
    was ever attempted, or it means sbatch already ran and the server
    crashed/restarted in the narrow window between that call returning and
    the follow-up _update_dataset_run() call persisting the id. job_name is
    scoped to this one submission (see start_packaging_job /
    start_feature_extraction_job) specifically so this lookup can tell the
    two cases apart instead of risking a duplicate sbatch.

    Unlike _get_slurm_job_state, callers here don't need to distinguish
    "unreachable" from "genuinely not found" — either way the safe, honest
    fallback is "don't know of one, go ahead and submit as normal."
    squeue is tried first (covers anything still queued/running with no
    accounting-DB lag), sacct as a fallback for a job that already
    finished before this check ran.
    """
    result = _run_slurm(["squeue", "-n", job_name, "-h", "-o", "%A"], timeout=15)
    if result is not None:
        job_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if job_ids:
            return job_ids[-1]

    result = _run_slurm(
        ["sacct", "--name", job_name, "--format=JobID,State", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is None:
        return None

    # sacct also emits job-step rows ("12345.batch", "12345.extern") for
    # the same job — only the bare-digit parent JobID identifies the actual
    # sbatch submission we'd want to reuse.
    job_ids = [
        parts[0].strip()
        for line in result.stdout.splitlines()
        if (parts := line.split("|", 1)) and parts[0].strip().isdigit()
    ]
    return job_ids[-1] if job_ids else None


def _find_job_ids_by_name_prefix(prefix: str) -> list[str]:
    """Every Slurm job ID (live or recent) whose job name is exactly
    `prefix` or starts with `prefix + "_"` — the latter to catch a
    dataset-wide tiling run split into multiple array batches, each
    suffixed "_1", "_2", ... by _submit_one_array's job_name.

    Used the same way _find_job_id_by_name is for packaging/extraction,
    but for the dataset-wide tiling submission (see _run_dataset_submission)
    — that one's job_name is only known as a prefix ahead of time, since a
    large dataset's actual batch count isn't known until submit_dataset_array
    has already discovered and split the slide list.

    Scoped to the current OS user (via -u) rather than every job on the
    cluster, both to keep squeue/sacct fast on a busy shared controller and
    because this server only ever submits jobs as this one user in the
    first place — matches every other Slurm submission in this file.
    Returns [] if nothing matches or Slurm couldn't be reached; callers
    already treat "found nothing" and "couldn't check" the same way (fall
    back to whatever state the DB already has), so this doesn't need to
    distinguish them the way _get_slurm_job_state does.
    """
    user = getpass.getuser()
    job_ids: set[str] = set()

    def _matches(name: str) -> bool:
        return name == prefix or name.startswith(prefix + "_")

    result = _run_slurm(["squeue", "-u", user, "-h", "-o", "%A|%j"], timeout=15)
    if result is not None:
        for line in result.stdout.splitlines():
            parts = line.split("|", 1)
            if len(parts) == 2 and _matches(parts[1].strip()):
                job_ids.add(parts[0].strip())

    result = _run_slurm(
        ["sacct", "-u", user, "--format=JobID,JobName", "--parsable2", "--noheader"],
        timeout=15,
    )
    if result is not None:
        for line in result.stdout.splitlines():
            parts = line.split("|", 1)
            # Job-step rows ("12345.batch") share their parent's job name —
            # only the bare-digit parent JobID is a real sbatch submission.
            if len(parts) == 2 and parts[0].strip().isdigit() and _matches(parts[1].strip()):
                job_ids.add(parts[0].strip())

    return sorted(job_ids)


def _attempt_signature(*parts) -> str:
    """Short deterministic fingerprint of a test-run's actual parameters.

    /package-test and /extract-features-test are deliberately not written
    to slurm_dataset_runs (see their docstrings — a test attempt succeeding
    or failing has no bearing on the real run), so unlike the tracked
    endpoints there's no DB row to check "is one already in flight?"
    against. This fingerprint stands in for that: it's used both to scope
    the test job's own Slurm job name (so _find_job_id_by_name can look up
    "was *this exact* test already submitted?" directly against Slurm
    itself, no DB needed) and its output filename (so two genuinely
    *different* test requests — e.g. different sample_size — land on
    different output paths instead of silently racing on the same file,
    which used to be the only thing this endpoint's docstring assumed away
    "no state to guard against").
    """
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def _h5_has_legacy_tile_names(path: Path) -> bool:
    """Whether this .h5 stores tile names without the ".jpeg" suffix.

    Deliberately *not* part of _validate_h5. The file is perfectly usable — the
    images are right, and feature extraction and cluster assignment both read it
    without caring what the name column says. The suffix only matters where the
    name becomes a join key, which is the Knowledge Bank load, and
    migrate_tile_names.py fixes it there in one command.

    Treating it as invalid blocked packaging, extraction and assignment for a
    defect none of them are affected by. So it is reported as an advisory the UI
    can show. Registration and the KB load no longer refuse over it either —
    they append the suffix themselves and report the count — so this is now
    purely informational: it says the artifact on disk still holds the short
    form, which migrate_tile_names.py is what fixes.
    """
    try:
        with h5py.File(path, "r") as f:
            if "tiles" not in f:
                return False
            rows = f["tiles"].shape[0]
            return bool(rows) and tiles_missing_suffix(f["tiles"][: min(rows, 100)])
    except (OSError, KeyError, ValueError):
        return False


def _extraction_expected_rows(row: dict) -> int | None:
    """Tiles in the .h5 this run's extraction was given, or None if that can't
    be established.

    None means "cannot check", and the row-count comparison is skipped rather
    than failed — a run whose packaged input has since been moved or deleted
    should not have a previously-good extraction start reading as incomplete.
    The narrower checks in validate_extraction_output still apply.
    """
    packaged = row.get("h5_output_path")
    if not packaged:
        return None
    packaged_path = Path(packaged)
    if not packaged_path.is_file():
        return None
    return _packaged_h5_rows(packaged_path)


def _job_output_ready(
    output_path: Path | None,
    slurm_state: str | None,
    validator: Callable[[Path], tuple[bool, str]] | None = None,
) -> bool:
    """Whether a Slurm job's output file (packaging's .h5, extraction's
    features file) is safe to treat as finished and readable.

    slurm_state == "COMPLETED" (a live sacct record) is the normal case.
    slurm_state == "" means neither sacct nor squeue (see
    _slurm_jobs_live_states) has any record of this job — for an old-enough
    job that means it's aged out of Slurm's accounting-DB retention window
    (long finished), not that it's still running or was just submitted a
    moment ago (squeue would have caught that). A non-empty file is safe to
    trust in that case too — without this, an old completed run would show
    "waiting on tiling to finish" and an eligible-for-retry prompt forever,
    since sacct can no longer vouch for it either way. slurm_state is None
    (sacct/squeue themselves failed to run) is left as not-ready — that's
    genuinely ambiguous, not a case to guess through.

    `validator` (pass _validate_h5 for packaging output) gets the final say: a
    job Slurm reports as COMPLETED can still have left a file nothing can
    read, and nothing downstream should be told it's ready until something has
    actually opened it.
    """
    if not output_path or not output_path.is_file():
        return False

    # slurm_state is None means Slurm itself could not be reached. That is
    # normally not enough to call an output ready — but it must not be an
    # automatic "no" either, or an unreachable sacct makes a *finished* run
    # look interrupted and puts a Retry button in front of a perfectly good
    # output. For packaging that retry would resubmit over a complete .h5 and
    # throw away hours of work, which is a far worse outcome than the
    # over-cautious "not ready" was ever protecting against.
    #
    # There is independent, on-disk evidence available in exactly that case:
    # packaging writes to a ".partial" sibling and os.replace()s it into place
    # only after a successful run, so the real path existing at all already
    # means a run finished, and the absence of a leftover ".partial" means no
    # other attempt is mid-write. Combined with a validator that opens the file
    # and reads its first and last row, that is strictly stronger proof than
    # the sacct row we could not fetch. Requiring a validator keeps this narrow:
    # callers with no way to check their output's integrity (no validator) get
    # the old conservative answer.
    partial_sibling = output_path.with_name(output_path.name + ".partial")
    unverifiable_but_complete = (
        slurm_state is None
        and validator is not None
        and not partial_sibling.exists()
    )

    if not unverifiable_but_complete and slurm_state != "COMPLETED" and not (
        slurm_state == "" and output_path.stat().st_size > 0
    ):
        return False
    if validator is not None:
        ok, reason = validator(output_path)
        if not ok:
            print(f"[{output_path}] output rejected as not ready: {reason}")
            return False
    if unverifiable_but_complete:
        print(
            f"[{output_path}] Slurm unreachable, but the output validates and no "
            f".partial is present — treating it as complete."
        )
    return True


def _heatmap_columns(kb_target: str = KB_PRODUCTION) -> list | None:
    """This target's p_hpc_* column names, or None if there is no heatmap table.

    Read once per target and cached, including the None: nothing in this
    repository writes tile_hpc_heatmap (see KB_TABLE_COVERAGE), so on the test
    database it is absent or empty, and a per-request probe for a table that
    will never appear is a failed query on every viewer open.
    """
    target = _resolve_kb_target(kb_target)
    if target in _heatmap_columns_cache:
        return _heatmap_columns_cache[target]
    try:
        eng = _get_engine(target)
        with eng.connect() as conn:
            probe = pd.read_sql("SELECT * FROM tile_hpc_heatmap LIMIT 0", conn)
        columns = [str(c).strip() for c in probe.columns]
        keep = [c for c in columns if c.startswith("p_hpc_")]
        _heatmap_columns_cache[target] = keep if keep and "slide_tile" in columns else None
    except Exception as e:  # noqa: BLE001 - a missing heatmap costs the overlay only
        print(f"Heatmap unavailable for {KB_TARGETS[target]}: {e}")
        _heatmap_columns_cache[target] = None
    return _heatmap_columns_cache[target]


def _heatmap_probs_for_tiles(kb_target: str, tiles: list) -> "pd.DataFrame | None":
    """One slide's heatmap rows, fetched by key instead of by reading the table.

    This used to be `SELECT * FROM tile_hpc_heatmap` — all 149 MB of it, held
    per target for the life of the process and merged into every tiles_meta
    response. It was the reason startup was slow, and once the server is run
    against the database through an SSH tunnel it is the reason the viewer does
    not open at all: 149 MB over a forwarded port cannot finish inside the
    client's 30-second read timeout, and it happens on the *first* tiles_meta
    request, so the first slide anyone opens is the one that fails.

    A slide is ~1.3k tiles out of ~2M rows, the table's primary key is
    slide_tile, and no request has ever needed another slide's probabilities.

    Matched on the exact key rather than UPPER(slide_tile), because that is what
    the primary key indexes — an expression the index cannot serve would put the
    sequential scan back, just server-side. The keys passed in come from
    tile_coordinates, already upper-cased, and both tables are written by the
    same pipeline; a slide whose heatmap rows are cased differently loses its
    overlay rather than its tile metadata, which is the same thing that happens
    today for every cohort, since nothing has written this table since the
    notebook that made it.
    """
    keep = _heatmap_columns(kb_target)
    if not keep or not tiles:
        return None
    target = _resolve_kb_target(kb_target)
    try:
        eng = _get_engine(target)
        query = text(
            f"SELECT slide_tile, {', '.join(keep)} FROM tile_hpc_heatmap "
            f"WHERE slide_tile IN :tiles"
        ).bindparams(bindparam("tiles", expanding=True))
        with eng.connect() as conn:
            df = pd.read_sql(query, conn, params={"tiles": list(tiles)})
    except Exception as e:  # noqa: BLE001 - as above: the overlay is optional
        print(f"Heatmap lookup failed for {KB_TARGETS[target]}: {e}")
        return None
    if df.empty:
        return None
    df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()
    return df


def _open_slide(slide_id: str, kb_target: str = KB_PRODUCTION) -> openslide.OpenSlide:
    target = _resolve_kb_target(kb_target)
    slide_id = slide_id.strip().upper()
    key = (target, slide_id)
    if key in _wsi_handles:
        return _wsi_handles[key]
    hpc_path = _get_wsi_map(target).get(slide_id)
    if not hpc_path:
        raise HTTPException(
            404, f"Slide {slide_id} not in wsi_registry "
                 f"({KB_TARGETS[target]}). Registering it writes that row.")
    if not os.path.isfile(hpc_path):
        raise HTTPException(404, f"SVS file not found on disk: {hpc_path}")
    slide = openslide.OpenSlide(hpc_path)
    _wsi_handles[key] = slide
    return slide


def _get_deepzoom(slide_id: str, kb_target: str = KB_PRODUCTION) -> DeepZoomGenerator:
    target = _resolve_kb_target(kb_target)
    slide_id = slide_id.strip().upper()
    key = (target, slide_id)

    if key in _dz_handles:
        return _dz_handles[key]

    slide = _open_slide(slide_id, target)

    dz = DeepZoomGenerator(
        slide,
        tile_size=256,
        overlap=1,
        limit_bounds=False,
    )

    _dz_handles[key] = dz
    return dz

def _img_to_jpeg_bytes(img, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.getvalue()


def _jpeg_response(data: bytes) -> StreamingResponse:
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        return arr
    x = arr.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    if 0.0 <= mn and mx <= 1.0:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.clip(x, 0, 255).astype(np.uint8)
    x = (x - lo) / (hi - lo) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)



# Tile pitch 

# (target, slide_id) → (pitch, source). Only a pitch read back off
# tile_coordinates is cached: that is what Stage 1 wrote and it cannot change
# afterwards. The fallbacks are deliberately recomputed, because a slide that
# has no rows yet gets one today and the real answer tomorrow, and a cached
# guess would outlive the registration that replaced it.
_tile_pitch_cache: dict[tuple[str, str], tuple[int, str]] = {}


def _pitch_from_coordinates(slide_id: str, kb_target: str) -> Optional[int]:
    """The stride Stage 1 actually used, read back off the tiles it wrote.

    auto_tile_from_mask sets x_native = col * tile_px_native, so any single
    row with col > 0 carries the pitch exactly — no mpp, no target_mpp, no
    assumption about which defaults were in force when the cohort was tiled.
    Preferred over deriving it precisely because the boxes are positioned from
    this same column: whatever this returns, the grid closes.

    None if the rows do not agree on one pitch, rather than a majority — two
    pitches in one slide is a slide tiled twice, and picking one would draw a
    grid that fits half of it.
    """
    q = text('SELECT "col", x_native FROM tile_coordinates '
             'WHERE UPPER(TRIM(slides)) = :slide_id '
             '  AND "col" > 0 AND x_native > 0 LIMIT 200')
    try:
        rows = pd.read_sql(q, _get_engine(kb_target), params={"slide_id": slide_id})
    except Exception:
        return None

    pitches = set()
    for col, x in zip(rows["col"], rows["x_native"]):
        col, x = int(col), int(x)
        if x % col:
            return None
        pitches.add(x // col)
    return pitches.pop() if len(pitches) == 1 else None


def _tile_size_native(slide_id: str, kb_target: str = KB_PRODUCTION,
                      slide: "openslide.OpenSlide | None" = None) -> tuple[int, str]:
    """This slide's tile size in native pixels, and where the number came from.

    Both callers of the number draw with it — the grid overlay sizes its
    rectangles and the tile inspector crops its region — so being wrong here
    is silent in exactly the way this codebase's failures usually are: the
    overlay renders, every box lands on a real tile, and the boxes are the
    wrong size.
    """
    target = _resolve_kb_target(kb_target)
    slide_id = slide_id.strip().upper()
    key = (target, slide_id)
    if key in _tile_pitch_cache:
        return _tile_pitch_cache[key]

    pitch = _pitch_from_coordinates(slide_id, target)
    if pitch is not None:
        _tile_pitch_cache[key] = (pitch, "tile_coordinates")
        return _tile_pitch_cache[key]

    if slide is None:
        try:
            slide = _open_slide(slide_id, target)
        except HTTPException:
            slide = None
    mpp = parse_native_mpp(
        slide.properties.get("openslide.mpp-x")) if slide is not None else None
    if mpp is None:
        return TILE_SIZE_NATIVE, f"default mpp {DEFAULT_NATIVE_MPP}"
    return native_tile_px(mpp), "slide mpp"


# Grid / adjacency helpers 


def _grid_xy(x_native, y_native, pitch: int = TILE_SIZE_NATIVE):
    gx = int(float(x_native) // float(pitch))
    gy = int(float(y_native) // float(pitch))
    return gx, gy


def _neighbors_8(gx, gy):
    return [
        (gx - 1, gy - 1), (gx, gy - 1), (gx + 1, gy - 1),
        (gx - 1, gy),                     (gx + 1, gy),
        (gx - 1, gy + 1), (gx, gy + 1), (gx + 1, gy + 1),
    ]


def _compute_adjacency(df_slide: pd.DataFrame, tile_size_native: Optional[int] = None):
    """Which HPCs touch which, over the tile grid.

    col/row are the grid, when the caller supplies them: the tiler wrote them
    as x_native // tile_px_native and they need no rederiving. Dividing
    x_native by a pitch that is not this slide's does not fail, it drifts —
    one cell per 1600/(pitch-1600) columns — until two tiles share a cell and
    one of them is dropped from the map, so neighbours are lost and gained
    with no sign that the grid was ever wrong.
    """
    df2 = df_slide.copy()
    df2["hpc_id"] = pd.to_numeric(df2["hpc_id"], errors="coerce")
    df2 = df2.dropna(subset=["hpc_id", "x_native", "y_native"])
    df2["hpc_id"] = df2["hpc_id"].astype(int)
    id_col = "slide_tile" if "slide_tile" in df2.columns else "tiles"
    pitch = int(tile_size_native or TILE_SIZE_NATIVE)
    has_grid = "col" in df2.columns and "row" in df2.columns

    pos_to_row = {}
    for _, r in df2.iterrows():
        if has_grid and pd.notna(r["col"]) and pd.notna(r["row"]):
            gx, gy = int(r["col"]), int(r["row"])
        else:
            gx, gy = _grid_xy(r["x_native"], r["y_native"], pitch)
        if (gx, gy) not in pos_to_row:
            pos_to_row[(gx, gy)] = r

    pair_edge_counts: dict[tuple, int] = {}
    tile_has_neighbor_pair: dict[tuple, dict] = {}
    visited_edges = set()

    for (gx, gy), r in pos_to_row.items():
        a = int(r["hpc_id"])
        tile_a = str(r[id_col])
        for nb in _neighbors_8(gx, gy):
            r2 = pos_to_row.get(nb)
            if r2 is None:
                continue
            b = int(r2["hpc_id"])
            if a == b:
                continue
            tile_b = str(r2[id_col])
            p = (a, b) if a < b else (b, a)
            edge_key = tuple(sorted([(gx, gy), nb]))
            if (p, edge_key) in visited_edges:
                continue
            visited_edges.add((p, edge_key))
            pair_edge_counts[p] = pair_edge_counts.get(p, 0) + 1
            if p not in tile_has_neighbor_pair:
                tile_has_neighbor_pair[p] = {"a_touch": set(), "b_touch": set()}
            if a < b:
                tile_has_neighbor_pair[p]["a_touch"].add(tile_a)
                tile_has_neighbor_pair[p]["b_touch"].add(tile_b)
            else:
                tile_has_neighbor_pair[p]["a_touch"].add(tile_b)
                tile_has_neighbor_pair[p]["b_touch"].add(tile_a)

    # Convert sets → lists for JSON serialisation
    serialisable = {}
    for pair_key, sets in tile_has_neighbor_pair.items():
        k = f"{pair_key[0]}_{pair_key[1]}"
        serialisable[k] = {
            "a_touch": sorted(sets["a_touch"]),
            "b_touch": sorted(sets["b_touch"]),
        }
    pair_counts = {f"{a}_{b}": cnt for (a, b), cnt in pair_edge_counts.items()}
    return pair_counts, serialisable


# App lifecycle


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LOADED TILE SERVER WITH DZI SUPPORT:", __file__)
    # Production only. The test database is optional — warming it here would
    # make "hpl_kb_test does not exist yet" a server that refuses to start,
    # rather than a 500 on the one request that asked for it. Both are built
    # lazily on first use (_get_wsi_map / _heatmap_columns).
    _load_wsi_map(KB_PRODUCTION)
    # Column names only — a LIMIT 0. Reading the whole 149 MB table here is
    # what made startup slow, and it bought nothing: the probabilities are
    # now fetched per slide, by primary key.
    _heatmap_columns(KB_PRODUCTION)
    yield
    for handle in _h5_handles.values():
        try:
            handle.close()
        except Exception:
            pass
    _h5_handles.clear()
    _dz_handles.clear()
    for slide in _wsi_handles.values():
        slide.close()
    _wsi_handles.clear()

app = FastAPI(title="HPC Tile Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = TileCache(cache_dir=CACHE_DIR)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health(kb_target: str = Depends(kb_target_param)):
    return {
        "status": "ok",
        "kb_target": kb_target,
        "database": KB_TARGETS[kb_target],
        "slides_loaded": len(_get_wsi_map(kb_target)),
        "kb_targets": sorted(KB_TARGETS),
    }


@app.get("/debug/routes")
def debug_routes():
    return {
        "file": __file__,
        "routes": sorted([getattr(route, "path", str(route)) for route in app.routes]),
    }


@app.get("/slides")
def list_slides(kb_target: str = Depends(kb_target_param)):
    return {"slides": sorted(_get_wsi_map(kb_target).keys())}

@app.post("/upload-slide")
async def upload_slide(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    slide_id: str = Form(None),
    confirm_overwrite: bool = Form(False),
):
    original_filename = Path(file.filename or "uploaded_slide").name
    suffix = Path(original_filename).suffix.lower()

    allowed_suffixes = {".svs", ".ndpi", ".tif", ".tiff", ".isyntax"}

    if suffix not in allowed_suffixes:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Allowed types: {sorted(allowed_suffixes)}",
        )

    safe_user_slide_id = (slide_id or "").strip().upper()

    if not safe_user_slide_id:
        safe_user_slide_id = f"UPLOAD_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif not _DATASET_NAME_RE.match(safe_user_slide_id):
        # slide_id still becomes a literal filename component below (not the
        # directory itself — see internal_id further down) and a primary
        # key in wsi_registry, and it's used as-is by mask/tile output paths
        # for this upload (tissue_masks/UPLOADED/<slide_id>_..., etc.) — a
        # value containing "/" or ".." would let those land outside their
        # intended directories. Same charset _sanitize_dataset_name enforces
        # for dataset folder names below, for the same reason. Belt-and-
        # suspenders: _resolve_within also asserts this below rather than
        # relying on this check alone.
        raise HTTPException(
            400,
            f"Invalid slide_id '{safe_user_slide_id}': use letters, numbers, '.', '_', or '-' "
            "only, and don't start with one of those.",
        )

    # A slide_id colliding with an existing registry entry from outside this
    # ad-hoc-upload pool (e.g. a real curated TCGA slide) would otherwise let
    # any upload silently repoint that ID's hpc_path at whatever file was just
    # uploaded — every existing viewer/pipeline call for that ID would then
    # serve the wrong slide, with nothing in the response signaling that a
    # collision (not a fresh registration) just happened. Not overridable by
    # confirm_overwrite — this is never the intended "replace my own test
    # upload" case, so there's no confirmation that makes it safe to proceed.
    eng = _get_engine()
    with eng.connect() as conn:
        existing = conn.execute(
            text("SELECT dataset_id FROM wsi_registry WHERE slide_id = :slide_id"),
            {"slide_id": safe_user_slide_id},
        ).fetchone()
    if existing and not _is_upload_dataset_id(existing[0]):
        raise HTTPException(
            409,
            {
                "error": "slide_id_conflict",
                "slide_id": safe_user_slide_id,
                "existing_dataset_id": existing[0],
                "message": (
                    f"slide_id '{safe_user_slide_id}' is already registered under dataset "
                    f"'{existing[0]}' — choose a different slide_id instead of overwriting it."
                ),
            },
        )

    # Re-using an ID from a previous ad-hoc upload is allowed (DO UPDATE in
    # _register_uploaded_slide) — that's the intended "replace my own test
    # upload" path — but it silently overwrote that upload's file, tissue
    # mask, tiles, and packaged .h5 with no warning. Now a hard stop unless
    # the caller already confirmed: first attempt gets a structured 409 the
    # UI can render as a warning with an explicit "yes, overwrite" action,
    # rather than the overwrite just happening.
    if existing and not confirm_overwrite:
        raise HTTPException(
            409,
            {
                "error": "slide_id_exists",
                "slide_id": safe_user_slide_id,
                "message": (
                    f"slide_id '{safe_user_slide_id}' already exists from a previous upload. "
                    "Uploading again will overwrite its saved file, tissue mask, tiles, and "
                    "packaged .h5. Resubmit with confirm_overwrite=true to proceed."
                ),
            },
        )

    internal_id = str(uuid.uuid4())
    safe_filename = original_filename.replace(" ", "_")

    UPLOAD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Storage is keyed by internal_id — server-generated, never user
    # input — not by safe_user_slide_id. Charset validation above already
    # makes slide_id safe to use as a path segment, but the actual
    # collision-safety/traversal guarantee here comes from a UUID directory
    # neither the user nor a future bug in that validation can influence;
    # slide_id is embedded only in the filename *inside* that directory, so
    # anyone browsing UPLOAD_RAW_DIR can still immediately tell which slide
    # a given upload is — it's just no longer what determines *where* the
    # file lands. _resolve_within is the explicit backstop: even given all
    # of the above, refuse rather than silently writing outside
    # UPLOAD_RAW_DIR if something upstream is ever wrong.
    try:
        save_dir = _resolve_within(UPLOAD_RAW_DIR, UPLOAD_RAW_DIR / internal_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        # "{slide_id}_{uuid}_{filename}" is the shape slide_naming.py parses,
        # and parsing it back is not cosmetic: registration finds each slide's
        # raw file by slide_id_from_raw_path(), so a name this cannot be
        # recovered from registers the cohort with no wsi_registry row and the
        # viewer 404s on every tile of a slide that is sitting right there.
        # Without the uuid the pattern does not match and the whole stem —
        # slide id, original filename and all — is read back as the slide id.
        save_path = _resolve_within(
            save_dir, save_dir / f"{safe_user_slide_id}_{internal_id}_{safe_filename}")
    except PathEscapeError as e:
        raise HTTPException(500, f"Refusing to save upload: {e}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Failed to save uploaded file: {e}")
    finally:
        await file.close()

    validation = {
        "openslide_readable": False,
        "error": None,
    }

    slide_info_payload = None
    metadata_path = None
    # Bound before the try below, which can leave via its own except before
    # ever reaching _start_upload_run — the response reads it either way.
    submission_id = None
    status = "uploaded"
    try:
        # Context manager, not a bare OpenSlide(...) + a .close() call at
        # the end of the block — several things between open and close can
        # legitimately raise (metadata_dir.mkdir()/json.dump() on a full
        # disk, _resolve_within() on a PathEscapeError, even a malformed
        # slide's own property/dimension reads), and a bare .close() placed
        # after all of that is simply never reached if any of it throws —
        # the outer except below still catches the error and returns
        # normally, but the OpenSlide handle (an open fd plus, for a large
        # WSI, a substantial mmap'd region) leaks until GC eventually gets
        # to it, if it ever does. __exit__ runs regardless of how the block
        # exits, so this can't leak the same way.
        with openslide.OpenSlide(str(save_path)) as slide:
            slide_info_payload = {
                "slide_id": safe_user_slide_id,
                "filename": original_filename,
                "stored_path": str(save_path),
                "level_count": slide.level_count,
                "level_dimensions": [
                    {"width": int(w), "height": int(h)}
                    for w, h in slide.level_dimensions
                ],
                "mpp_x": slide.properties.get("openslide.mpp-x"),
                "mpp_y": slide.properties.get("openslide.mpp-y"),
                "vendor": slide.properties.get("openslide.vendor"),
                "objective_power": slide.properties.get("openslide.objective-power"),
                "uploaded_at": datetime.now().isoformat(),
            }

            # No thumbnail generated here — tile_mask.py's masking step (the
            # very next thing the background pipeline does) already produces
            # one as a byproduct of computing the tissue mask
            # (tissue_masks/UPLOADED/{slide_id}_thumbnail.png), and that step
            # starts essentially immediately after this request returns.
            # Generating a second one here would just duplicate it for a few
            # seconds' head start that isn't worth the redundancy.
            # Same internal_id-keyed layout as save_path above, for the same
            # reason — slide_id stays readable in the filename, but isn't what
            # determines the path.
            metadata_dir = _resolve_within(UPLOAD_METADATA_DIR, UPLOAD_METADATA_DIR / internal_id)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = _resolve_within(metadata_dir, metadata_dir / f"{safe_user_slide_id}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(slide_info_payload, f, indent=2)

        # The cohort this slide belongs to from here on: its own, one per
        # upload. Written now so the viewer can open the slide immediately,
        # and matching what Stage 5 will register it under — Stage 5 refuses
        # to register a slide that already belongs to a different dataset_id.
        _register_uploaded_slide(safe_user_slide_id, str(save_path),
                                 upload_dataset_name(safe_user_slide_id))

        validation["openslide_readable"] = True

        _set_processing_status(safe_user_slide_id, "queued")
        submission_id = _start_upload_run(safe_user_slide_id, save_path)
        background_tasks.add_task(_run_postupload_pipeline, safe_user_slide_id,
                                  str(save_path), submission_id)

    except Exception as e:
        validation["error"] = str(e)
        status = "viewer_ready"
    return {
        "internal_id": internal_id,
        "slide_id": safe_user_slide_id,
        "filename": original_filename,
        "stored_path": str(save_path),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "status": status,
        "processing_status": _get_processing_status(safe_user_slide_id),
        "validation": validation,
        "slide_info": slide_info_payload,
        # The run the rest of the pipeline is driven from. None means the row
        # could not be written (see _start_upload_run) — masking and tiling
        # still run, but this slide has no pipeline view to carry it into the
        # Knowledge Bank, and the UI should say so rather than show nothing.
        "submission_id": submission_id,
        "dataset_name": upload_dataset_name(safe_user_slide_id),
        "next_step": "Tissue masking + tiling started in the background — poll /slide/{slide_id}/processing-status.",
    }


@app.get("/slide/{slide_id}/processing-status")
def slide_processing_status(slide_id: str):
    """Masking/tiling/packaging progress, plus the run those stages belong to.

    submission_id is here as well as in the upload response because the two
    are read at different times: the upload response is gone once a browser
    reloads, and this is what the UI polls. Without it the pipeline view
    would only be reachable for as long as the page that started the upload
    stayed open.
    """
    payload = dict(_get_processing_status(slide_id))
    payload.update(_upload_run_for_slide(slide_id))
    return payload


# Every submit_array() argument that changes what the tiles themselves look
# like — as opposed to how the job is scheduled (cpus, memory, partition,
# batch_size) or which slides are picked (sample_size, slide_names). These are
# what has to be identical between an original run and any resume of it, since
# a dataset half-tiled at one threshold and half at another is not one dataset.
#
# jpeg_quality is included: it only affects the on-disk JPEGs and not the .h5's
# uncompressed pixels, but it does change the image data those pixels are
# decoded from, so a resume at a different quality is still a mixed dataset.
_TILING_PARAM_NAMES = (
    "min_tissue",
    "target_mpp",
    "target_tile_px",
    "level",
    "jpeg_quality",
    "mask_max_size",
    "mask_saturation",
    "mask_value",
)


def _default_tiling_params() -> dict:
    """Current defaults, read off submit_array()'s own signature.

    Introspection rather than a hardcoded copy specifically so this cannot
    drift from the function it feeds: changing a default in
    submit_mask_tile_slurm.py updates what gets recorded here automatically,
    and a renamed or removed parameter fails loudly at import instead of
    silently recording a value nothing uses.
    """
    signature = inspect.signature(submit_dataset_array)
    defaults = {}
    for name in _TILING_PARAM_NAMES:
        parameter = signature.parameters.get(name)
        if parameter is None or parameter.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"submit_array() no longer has a defaulted '{name}' parameter — "
                f"_TILING_PARAM_NAMES needs updating."
            )
        defaults[name] = parameter.default
    return defaults


def _resolve_tiling_params(req: "DatasetJobRequest") -> dict:
    """The tiling parameters this submission will actually run with.

    Precedence: an explicit tiling_params block (how a resume passes the
    original run's recorded values through) over the defaults, and an
    explicitly-set min_tissue over both — min_tissue predates tiling_params as
    a top-level request field and the UI still sends it that way.

    model_fields_set is what makes that last part work: min_tissue has a
    default on the model, so its presence in the request is the only way to
    tell "the user chose 30.0" from "the user said nothing and 30.0 is the
    default". Without that distinction a resume could not avoid overriding the
    recorded value with a default that merely looks deliberate.
    """
    params = _default_tiling_params()
    if req.tiling_params:
        params.update(
            {k: v for k, v in req.tiling_params.items() if k in _TILING_PARAM_NAMES}
        )
    if "min_tissue" in req.model_fields_set:
        params["min_tissue"] = req.min_tissue
    return params


def _row_tiling_params(row) -> dict | None:
    """Tiling parameters recorded for a run, or None if it predates the column.

    None is returned rather than the defaults so callers can tell "this run
    used these values" from "nobody knows what this run used" — only the former
    is grounds for claiming a resume reproduces the original.
    """
    try:
        recorded = row["tiling_params"]
    except (KeyError, IndexError):
        # Server pointed at a database without the migration applied.
        return None
    if not recorded:
        return None
    if isinstance(recorded, str):
        # psycopg2 without the JSONB adapter registered hands back raw text.
        try:
            recorded = json.loads(recorded)
        except json.JSONDecodeError:
            return None
    if not isinstance(recorded, dict):
        return None
    return {k: v for k, v in recorded.items() if k in _TILING_PARAM_NAMES} or None


class DatasetJobRequest(BaseModel):
    dataset_path: str
    max_concurrent: int = 10
    min_tissue: float = 30.0
    # Set by resume_dataset_job (and the UI's full-directory run) to reproduce
    # an earlier run's tiling exactly. Unset on a fresh submission, which then
    # takes the current defaults plus whatever min_tissue the user chose.
    tiling_params: Optional[dict] = None
    sample_size: Optional[int] = None
    slide_names: Optional[list[str]] = None
    partition: Optional[str] = None  # None -> Slurm's own default partition
    notify_email: Optional[str] = None  # Slurm's own END/FAIL notification, no attachments
    # Folder tiles land in under PROCESSED_TILES_DIR, e.g. "TCGA" or
    # "Radiogenomics" — lets the UI reuse an existing dataset folder or name
    # a new one. None falls back to dataset_path's own folder name.
    dataset_name: Optional[str] = None


@app.get("/dataset-roots")
def list_dataset_roots():
    """Top-level directories currently under LONG_TERM_SCRATCH, for reference
    only — POST /dataset-jobs accepts any path (including nested ones), it
    doesn't require picking from this list.
    """
    return {"root": str(LONG_TERM_SCRATCH), "datasets": _list_dataset_roots()}


@app.get("/dataset-jobs/{submission_id}/cohort-shift-readiness")
def cohort_shift_readiness(submission_id: str):
    """Can the cohort check run for this run yet, and if not, what is missing?

    So the UI can state the prerequisite before anyone clicks, rather than
    turning a 400 into a wall of text. The profile is a one-off ~20 minute
    leave-one-out over the reference and is then reused by every dataset, so the
    common case is that it already exists and this returns ready.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = (Path(row["assignment_output_path"])
                if row.get("assignment_output_path") else None)
    recorded = row.get("assignment_vote") or ""
    preset = DEFAULT_VOTE_PRESET
    for name in VOTE_PRESETS:
        if recorded.startswith(name):
            preset = name
            break
    reference = Path(row.get("assignment_reference") or HPC_REFERENCE_PATH)
    profile_path = cohort_shift.default_profile_path(reference, preset)

    flags = " ".join(vote_flags(**resolve_vote(preset)))
    return {
        "submission_id": submission_id,
        "ready": bool(csv_path and csv_path.is_file() and profile_path.is_file()),
        "has_assignments": bool(csv_path and csv_path.is_file()),
        "assignments_path": str(csv_path) if csv_path else None,
        "has_profile": profile_path.is_file(),
        "profile_path": str(profile_path),
        "vote_preset": preset,
        # No assumption about the deployment's cwd: absolute paths throughout,
        # since a relative one is what broke the acceptance test's container.
        "build_profile_command": (
            f"python validate_reference.py --reference {reference} "
            f"--sample 200000 {flags} --save-profile {profile_path}"
        ),
    }


class CohortShiftRequest(BaseModel):
    # Defaults to this run's tracked assignment CSV. Overridable for the same
    # reason Stage 5's csv_path is: Stage 4's test mode records no path.
    csv_path: str | None = None
    top_slides: int = 10


@app.post("/dataset-jobs/{submission_id}/cohort-shift")
def check_cohort_shift(submission_id: str, req: CohortShiftRequest):
    """Is this dataset's tissue represented in the reference at all?

    Read-only and cheap — it reads two columns of the assignments CSV and
    compares them against precomputed reference quantiles. No search, no GPU.

    Deliberately separate from /kb-load-preview even though both are read-only
    checks on the same CSV. Stage 5's preview answers "will this load cleanly",
    which is about naming and coverage. This answers "should it be loaded at
    all", which is about whether the cluster IDs mean anything for this cohort.
    A clean preview and a shifted cohort is a perfectly possible combination,
    and the whole point is that it looks fine.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = Path(req.csv_path) if req.csv_path else (
        Path(row["assignment_output_path"]) if row.get("assignment_output_path")
        else None)
    if csv_path is None:
        raise HTTPException(
            400,
            "This run has no assignment CSV yet, and no csv_path was given. "
            "Run Stage 4 first, or pass the path of a CSV from its test mode.",
        )
    if not csv_path.is_file():
        raise HTTPException(400, f"{csv_path} does not exist.")

    # The profile has to describe the vote that produced this CSV. The run record
    # is the only place that is written down -- the CSV cannot carry it, since
    # load_hpc_assignments.py identifies its cluster column by elimination.
    recorded = row.get("assignment_vote") or ""
    preset = DEFAULT_VOTE_PRESET
    for name in VOTE_PRESETS:
        if recorded.startswith(name):
            preset = name
            break
    reference = Path(row.get("assignment_reference") or HPC_REFERENCE_PATH)
    profile_path = cohort_shift.default_profile_path(reference, preset)
    if not profile_path.is_file():
        raise HTTPException(
            400,
            f"No reference profile for the '{preset}' vote at {profile_path}. "
            f"Build it once per (reference, vote) pair — it needs a leave-one-out "
            f"run over the reference, which takes ~20 minutes and is then reused "
            f"by every dataset:\n"
            f"  python validate_reference.py --reference {reference} "
            f"--sample 200000 --save-profile {profile_path}\n"
            f"adding the same vote flags the assignment used.",
        )

    try:
        profile = cohort_shift.load_profile(profile_path)
        frame = pd.read_csv(csv_path, usecols=lambda c: c in (
            "neighbor_distance", "vote_margin", "slides"))
        result = cohort_shift.compare(frame, profile)
    except SystemExit as e:
        # load_profile and compare refuse with SystemExit; that is the caller's
        # to fix, not a fault.
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Cohort-shift check failed: {e}")

    level, sentence = cohort_shift.verdict(result)
    return {
        "submission_id": submission_id,
        "csv_path": str(csv_path),
        "profile_path": str(profile_path),
        "vote_preset": preset,
        "recorded_vote": recorded or None,
        "profile_vote": profile.get("vote"),
        "level": level,
        "verdict": sentence,
        "slide_spread": cohort_shift.slide_concentration(result),
        # Trimmed to what the UI draws: the full per-slide list can be thousands
        # of rows and nothing renders them all.
        **{key: value for key, value in result.items() if key != "per_slide"},
        "per_slide": (result.get("per_slide") or [])[:max(req.top_slides, 0)],
        "n_slides": len(result.get("per_slide") or []),
    }


@app.get("/vote-presets")
def list_vote_presets():
    """The named vote configurations Stage 4 can run, for the UI to offer.

    Served rather than hardcoded in the UI for the same reason they live in one
    module: a preset is seven numbers, and a second copy of them is a copy that
    can drift. The UI showing "97.27%" beside a configuration that is no longer
    that configuration is precisely the kind of confidently-wrong display this
    codebase is written against.
    """
    return {
        "default": DEFAULT_VOTE_PRESET,
        "presets": {
            name: {
                "label": spec["label"],
                "accuracy": spec["accuracy"],
                "summary": spec["summary"],
                "why": spec["why"],
                "flags": spec["flags"],
            }
            for name, spec in VOTE_PRESETS.items()
        },
    }


@app.get("/tile-dataset-names")
def list_tile_dataset_names():
    """Existing folders directly under PROCESSED_TILES_DIR, e.g. ["TCGA",
    "Radiogenomics"] — lets the UI offer "add to an existing dataset folder"
    as a dropdown instead of everyone retyping the name by hand and risking
    a near-miss (e.g. "Radiogenomic" splitting off a second folder).
    """
    if not PROCESSED_TILES_DIR.is_dir():
        return {"dataset_names": []}
    return {
        "dataset_names": sorted(
            p.name for p in PROCESSED_TILES_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    }


_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sanitize_dataset_name(name: str) -> str:
    """Validate a user-supplied dataset folder name.

    This becomes a literal path segment under PROCESSED_TILES_DIR
    (tile_dir/<name>/<slide_id>/...), so it's restricted to a safe charset
    rather than trusted as-is — a name like "../../etc" or containing "/"
    would otherwise let a submission write tiles outside PROCESSED_TILES_DIR.
    """
    name = name.strip()
    if not name:
        raise ValueError("Dataset folder name cannot be empty.")
    if not _DATASET_NAME_RE.match(name):
        raise ValueError(
            f"Invalid dataset folder name '{name}': use letters, numbers, "
            "'.', '_', or '-' only, and don't start with one of those."
        )
    return name


class PathEscapeError(RuntimeError):
    """A resolved path landed outside the directory it was supposed to be
    confined to. Should be unreachable in practice — every caller of
    _resolve_within already validates its inputs before building the path
    (charset-checked slide_id/dataset_name, or a server-generated UUID) —
    but this is the structural backstop for when that validation has a bug,
    gets skipped, or a future code path forgets it. Whoever catches this
    should treat it as "refuse and report," never "strip and continue."
    """


def _resolve_within(base: Path, path: Path) -> Path:
    """Resolve `path` and assert it's actually inside `base`.

    This is the last line of defense, not the primary one — charset
    validation (_sanitize_dataset_name, the slide_id check in
    /upload-slide) is what should actually stop a "../" or absolute-path
    value from ever reaching here. This exists for the case that
    validation misses something: even if a bad value somehow got this far,
    the write/read still can't land outside where it's supposed to.
    """
    resolved_base = base.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_base and resolved_base not in resolved_path.parents:
        raise PathEscapeError(
            f"Resolved path {resolved_path} escapes permitted directory {resolved_base}"
        )
    return resolved_path


def _resolve_dataset_path(user_path: str) -> Path:
    """Resolve a user-supplied absolute path.

    Not restricted to any particular root (by request) — accepts any
    location the server process can read. Still rejects empty input and
    pipeline-owned output directories, since submitting those as "a
    dataset" is always a mistake regardless of where they live. Submitting
    this path triggers a real sbatch job, so treat this input as trusted —
    anyone with UI access can point it at any directory the server can see.
    """
    user_path = user_path.strip()
    if not user_path:
        raise ValueError("Dataset path cannot be empty.")

    expanded = Path(user_path).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Dataset path must be absolute (start with '/'): got '{user_path}'. "
            "A relative path would silently resolve against the server's own "
            "working directory, not where you meant."
        )

    candidate = expanded.resolve()

    excluded_dirs = {TISSUE_MASK_DIR.resolve(), PROCESSED_TILES_DIR.resolve(), UPLOAD_ROOT.resolve()}
    for excluded in excluded_dirs:
        if candidate == excluded or excluded in candidate.parents:
            raise ValueError(
                f"'{user_path}' is a pipeline-owned output directory, not a dataset input."
            )

    if not candidate.is_dir():
        raise ValueError(f"Directory not found: {candidate}")

    return candidate


def _record_run_job(
    submission_id: str,
    stage: str,
    job_id: str | None,
    output_path: str | None = None,
    params: dict | None = None,
) -> None:
    """Append one submitted Slurm job to a run's history.

    Purely additive alongside the single-slot columns on slurm_dataset_runs,
    which stay authoritative for gating (see migrate_dataset_run_jobs.sql).

    Never raises. Every caller is on the far side of a successful sbatch, so a
    bookkeeping failure must not become a 500 — that would tell the caller
    nothing was queued while the job runs anyway, which is worse than a gap in
    the history. ON CONFLICT DO NOTHING makes it safe to call again for a job
    already recorded, which the tiling path does after recovering job ids by
    name.
    """
    if not job_id:
        return
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO slurm_dataset_run_jobs
                        (submission_id, stage, job_id, output_path, params)
                    VALUES (:submission_id, :stage, :job_id, :output_path, :params)
                    ON CONFLICT (submission_id, stage, job_id) DO NOTHING
                """),
                {
                    "submission_id": submission_id,
                    "stage": stage,
                    "job_id": str(job_id),
                    "output_path": str(output_path) if output_path else None,
                    # Serialised here rather than relying on the driver's dict
                    # adaptation, which differs between psycopg2 and psycopg3 —
                    # same reason tiling_params is written this way.
                    "params": json.dumps(params) if params is not None else None,
                },
            )
    except Exception as e:
        print(f"[warn] could not record {stage} job {job_id} for {submission_id}: {e}")


def _run_job_history(submission_id: str) -> list[dict]:
    """Every recorded Slurm job for a run, newest first, with live state.

    States for the whole history come from one sacct call via
    _slurm_states_by_job — a per-row lookup would be one call per attempt, and
    this is rendered inside a run's panel where several attempts are normal.
    """
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT stage, job_id, output_path, params, submitted_at
                    FROM slurm_dataset_run_jobs
                    WHERE submission_id = :submission_id
                    ORDER BY submitted_at DESC, id DESC
                """),
                {"submission_id": submission_id},
            ).mappings().fetchall()
    except Exception as e:
        # The table may not exist yet if the code is deployed before the
        # migration is run. An empty history is the right degradation; failing
        # the whole status response is not.
        print(f"[warn] could not read job history for {submission_id}: {e}")
        return []

    every_id: list[str] = []
    for row in rows:
        every_id.extend(j for j in str(row["job_id"]).split(",") if j)
    states = _slurm_states_by_job(sorted(set(every_id)))

    history = []
    for row in rows:
        ids = [j for j in str(row["job_id"]).split(",") if j]
        if states is None:
            state = "unknown"
        else:
            seen: set[str] = set()
            for job_id in ids:
                seen |= states.get(job_id.split("_", 1)[0], set())
            state = _coarse_run_state(seen)
        params = row["params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = None
        history.append({
            "stage": row["stage"],
            "job_id": row["job_id"],
            "batch_count": len(ids),
            "output_path": row["output_path"],
            "params": params,
            "submitted_at": row["submitted_at"].isoformat() if row["submitted_at"] else None,
            "slurm_state": state,
        })
    return history


# Columns the run record needs before a stage can record where it wrote. They
# live in PRODUCTION for every target, because run tracking is not split across
# databases — so a deployment that migrated hpl_kb_test but forgot hpl_kb would
# register into test successfully and then fail writing the run row, leaving the
# rows in place and the run claiming it never registered. Checked at preview.
_RUN_TRACKING_COLUMNS = ("registration_done", "registration_at",
                         "registration_dataset_id", "registration_raw_dir",
                         "registration_rows", "registration_kb_target")


def _missing_run_tracking_columns() -> list[str]:
    """Which of the above slurm_dataset_runs does not have, in production."""
    try:
        existing = {c["name"] for c in sqlalchemy_inspect(
            _get_engine(KB_PRODUCTION)).get_columns("slurm_dataset_runs")}
    except Exception:
        return []
    return [c for c in _RUN_TRACKING_COLUMNS if c not in existing]


def _update_dataset_run(submission_id: str, **fields):
    eng = _get_engine()
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    with eng.begin() as conn:
        conn.execute(
            text(f"UPDATE slurm_dataset_runs SET {set_clause} WHERE submission_id = :submission_id"),
            {**fields, "submission_id": submission_id},
        )


def _update_dataset_run_best_effort(submission_id: str, **fields) -> None:
    """Like _update_dataset_run, but a missing column is a caption the UI does
    not get rather than a failed request.

    For bookkeeping columns added by a migration that may not have been applied
    yet. Every caller is on the far side of a successful sbatch, so the same
    rule as _record_run_job applies: telling the caller nothing was queued while
    the job runs anyway is worse than a gap in the record. The failure is logged
    rather than swallowed, so a missing migration is findable.
    """
    try:
        _update_dataset_run(submission_id, **fields)
    except Exception as e:
        print(f"[run {submission_id}] could not record {list(fields)}: {e}. "
              f"If this is an undefined column, apply the matching "
              f"backend/migrate_*.sql; the run itself is unaffected.",
              flush=True)


def _effective_h5_dataset_name(base_name: str, is_subset: bool) -> str:
    """Full-dataset runs keep the plain name; subset runs (random sample or
    specific slides) get a distinct "_subset_N" suffix so a validation run
    doesn't silently overwrite a previous run's .h5 at the same path —
    h5py.File(path, 'w') always clobbers whatever's already there.

    Numbered by scanning what's already on disk under HPL_DATASETS_ROOT
    rather than querying the DB, since the filesystem is the actual source
    of truth for "would this collide with something that already exists."
    """
    if not is_subset:
        return base_name

    existing = 0
    if HPL_DATASETS_ROOT.is_dir():
        for entry in HPL_DATASETS_ROOT.iterdir():
            match = re.match(rf"^{re.escape(base_name)}_subset_(\d+)$", entry.name)
            if match:
                existing = max(existing, int(match.group(1)))
    return f"{base_name}_subset_{existing + 1}"


def _run_dataset_submission(submission_id: str, raw_dir: str, req: DatasetJobRequest):
    """Background job: discover every slide under raw_dir (can be slow on a
    large or network-mounted dataset) and submit the Slurm array, updating
    slurm_dataset_runs as it progresses. Runs after /dataset-jobs already
    returned submission_id, so the HTTP request isn't held open for however
    long the directory walk takes.
    """
    try:
        _update_dataset_run(submission_id, status="discovering")

        def _persist_plan(plan: dict):
            # Runs the instant submit_dataset_array has written the combined
            # manifest, before any sbatch call — i.e. before the part of that
            # call that can take minutes on a large dataset split into
            # several batches (each retrying through Slurm controller
            # congestion, see _run_sbatch_with_retry). Persisting the
            # manifest/slide-count now, not only after submit_dataset_array
            # fully returns, means a crash mid-submission still leaves this
            # row with everything dataset_job_status() needs except job_id
            # — which the job-name lookup below can recover directly from
            # Slurm — instead of the run looking permanently lost with no
            # manifest to even resume from.
            _update_dataset_run(
                submission_id,
                status="submitting",
                manifest_path=plan["manifest_path"],
                total_slides=plan["slides_found"],
            )

        result = submit_dataset_array(
            raw_dir=Path(raw_dir),
            mask_dir=TISSUE_MASK_DIR,
            tile_dir=PROCESSED_TILES_DIR,
            max_concurrent=req.max_concurrent,
            # Every tile-affecting parameter, from the same resolved dict that
            # was written to the row — including min_tissue, which used to be
            # passed on its own. Passing them as a unit is what guarantees a
            # resume runs the original values: there is no second code path
            # here that could quietly reintroduce a default.
            **(req.tiling_params or _default_tiling_params()),
            sample_size=req.sample_size,
            slide_names=req.slide_names,
            partition=req.partition,
            notify_email=req.notify_email,
            dataset_name=req.dataset_name,
            # Scoped to this submission so _find_job_ids_by_name_prefix can
            # recover it (and any per-batch job under this name, e.g.
            # "..._1", "..._2") if this background task never gets to write
            # job_id itself.
            job_name=f"wsi_mask_tile_{submission_id}",
            on_planned=_persist_plan,
        )
        job_ids = result.get("job_ids") or []
        failed_batch_count = result.get("failed_batch_count", 0)
        if job_ids:
            # Stored comma-joined in the existing job_id column — one batch
            # (the common case) looks exactly as it always did; multiple
            # batches (a large dataset split to avoid overwhelming Slurm's
            # controller with one giant array) list all of them. A batch
            # can fail even after its own retries without aborting the rest
            # — surface that here as a warning-style note, not a hard error,
            # since whatever did submit is still real, running work.
            partial_failure_note = (
                f"{failed_batch_count} of {result['batch_count']} batches failed to submit "
                f"after retries — {len(job_ids)} succeeded and are running; missing slides "
                f"would need a follow-up submission."
                if failed_batch_count else None
            )
            _update_dataset_run(
                submission_id,
                status="submitted",
                job_id=",".join(job_ids),
                manifest_path=result["manifest_path"],
                total_slides=result["slides_found"],
                error=partial_failure_note,
            )
            _record_run_job(
                submission_id, "tiling", ",".join(job_ids),
                output_path=result["manifest_path"],
                params={
                    "slides": result["slides_found"],
                    "batches": len(job_ids),
                    "failed_batches": failed_batch_count,
                    "tiling_params": req.tiling_params,
                },
            )
            # Packaging (and later, feature extraction) no longer auto-chain
            # from here — each stage now needs an explicit "start" click from
            # the UI once the previous stage is confirmed done. See
            # POST /dataset-jobs/{id}/package and .../extract-features.
        else:
            batch_errors = "; ".join(
                b.get("error") or b.get("sbatch_stdout") or "no job id"
                for b in result.get("batches", [])
            )
            _update_dataset_run(
                submission_id, status="error",
                error=f"sbatch did not return any job IDs. {batch_errors}".strip(),
            )
    except Exception as e:
        _update_dataset_run(submission_id, status="error", error=str(e))


def _record_resume_lineage(submission_id: str, parent_submission_id: str) -> None:
    """Note that this run was created by resuming another.

    A separate best-effort UPDATE rather than a column in the INSERT, and it
    never raises, for the same reason _record_run_job doesn't: this is
    bookkeeping on the far side of a decision that has already been made. If
    migrate_dataset_runs_lineage.sql has not been applied yet, folding this
    into the INSERT would make every resume fail outright — trading a missing
    display detail for a broken pipeline stage.

    GET /datasets does not depend on this. It groups runs by
    (raw_dir, dataset_name), which resume reuses, so lineage only sharpens the
    picture from "these runs share a directory" to "this one continued that
    one".
    """
    try:
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text(
                    "UPDATE slurm_dataset_runs "
                    "SET resumed_from_submission_id = :parent "
                    "WHERE submission_id = :submission_id"
                ),
                {"parent": parent_submission_id, "submission_id": submission_id},
            )
    except Exception as e:
        print(
            f"[warn] could not record that {submission_id} resumed "
            f"{parent_submission_id}: {e}"
        )


def _start_dataset_submission(
    raw_dir: Path,
    req: DatasetJobRequest,
    background_tasks: BackgroundTasks,
    resumed_from_submission_id: str | None = None,
) -> dict:
    """Create a new slurm_dataset_runs row and kick off the background
    pipeline for it. Shared by POST /dataset-jobs (a fresh submission) and
    the /resume endpoint (a follow-up submission for whatever a previous
    run's slides are still missing) — both are "start a submission for this
    raw_dir with this req," just with req.slide_names populated differently.

    resumed_from_submission_id is what tells those two apart afterwards. Both
    land in the same table looking identical, so without it a resume and a
    deliberate second run over the same directory are indistinguishable.
    """
    submission_id = str(uuid.uuid4())
    is_subset = bool(req.sample_size or req.slide_names)

    # Falls back to raw_dir's own folder name when the caller (or the UI's
    # "use default" state) didn't pick one — same default submit_array()
    # itself uses, kept here too so the resolved name gets stored on the row
    # and every later stage (resume, packaging, status) reads it back
    # instead of recomputing it and potentially drifting if raw_dir's own
    # name ever gets reused for a different dataset.
    try:
        dataset_name = _sanitize_dataset_name(req.dataset_name) if req.dataset_name else raw_dir.name
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Resolved once, here, then both persisted and handed to submit_array — so
    # the row records exactly what ran rather than a second, independently
    # computed guess at it. A resume reads these back and passes them straight
    # through, which is the whole point: nothing downstream re-derives them.
    tiling_params = _resolve_tiling_params(req)
    req = req.model_copy(
        update={"dataset_name": dataset_name, "tiling_params": tiling_params}
    )

    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO slurm_dataset_runs
                    (submission_id, raw_dir, mask_dir, tile_dir, status,
                     is_subset, partition, notify_email, dataset_name,
                     tiling_params)
                VALUES
                    (:submission_id, :raw_dir, :mask_dir, :tile_dir, 'queued',
                     :is_subset, :partition, :notify_email, :dataset_name,
                     :tiling_params)
            """),
            {
                "submission_id": submission_id,
                "raw_dir": str(raw_dir),
                "mask_dir": str(TISSUE_MASK_DIR),
                "tile_dir": str(PROCESSED_TILES_DIR),
                "is_subset": is_subset,
                "partition": req.partition,
                "notify_email": req.notify_email,
                "dataset_name": dataset_name,
                # Serialised here rather than relying on the driver's dict
                # adaptation, which differs between psycopg2 and psycopg3.
                "tiling_params": json.dumps(tiling_params),
            },
        )

    # After the INSERT, so a database without the lineage migration still gets
    # a fully working run out of this function.
    if resumed_from_submission_id:
        _record_resume_lineage(submission_id, resumed_from_submission_id)

    background_tasks.add_task(_run_dataset_submission, submission_id, str(raw_dir), req)

    return {"submission_id": submission_id, "status": "queued", "raw_dir": str(raw_dir), "dataset_name": dataset_name}


@app.post("/dataset-jobs")
def create_dataset_job(req: DatasetJobRequest, background_tasks: BackgroundTasks):
    """Queue a dataset-wide masking+tiling Slurm array job.

    dataset_path is resolved and validated server-side via
    _resolve_dataset_path — never trust that the client only ever sends a
    safe path, since this ultimately triggers a real sbatch submission.

    Discovering slides and submitting to Slurm both happen in a background
    task, not here — a recursive walk of a large/network-mounted dataset can
    take far longer than a client is willing to hold an HTTP request open
    for. This returns immediately with a submission_id to poll instead.
    """
    try:
        raw_dir = _resolve_dataset_path(req.dataset_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return _start_dataset_submission(raw_dir, req, background_tasks)


@app.post("/dataset-jobs/{submission_id}/resume")
def resume_dataset_job(submission_id: str, background_tasks: BackgroundTasks):
    """Find whatever slides from a previous submission never got tiled
    (checked against the filesystem — a slide counts as done if it has a
    real _tile_metadata.csv, regardless of what Slurm's job records say)
    and queue a new submission for just those, reusing the same raw_dir.

    This is how "the whole dataset ended partway through" gets resolved
    without manually figuring out which Slurm batch failed — every run,
    however it stopped, can be resumed the same way.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    _refuse_if_pipeline_run(row, "resuming its tiling")
    if not row["manifest_path"]:
        raise HTTPException(400, "This run never got far enough to have a manifest to resume from.")

    manifest_path = Path(row["manifest_path"])
    tile_dir = Path(row["tile_dir"])
    if not manifest_path.is_file():
        raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

    dataset_name = _row_dataset_name(row)
    # Detailed variant so the response can distinguish slides that were never
    # attempted from ones whose tile metadata is corrupt (both get re-tiled)
    # and from ones that legitimately produced no tissue (which never should
    # be, since re-running yields the same empty result).
    breakdown = find_missing_slides_detailed(manifest_path, tile_dir, dataset_name)
    missing_raw_paths, total = breakdown["missing"], breakdown["total"]

    if not missing_raw_paths:
        # "Missing" here only means "never produced a _tile_metadata.csv at
        # all" — it does NOT mean every slide has real tiles. A slide that
        # ran and legitimately saved zero tiles (no tissue passed the
        # threshold) still writes that file, so it doesn't show up as
        # missing here, but it's not "done" in any meaningful sense either.
        # Say so explicitly instead of claiming they "have tiles," which is
        # false for that case and contradicts the zero-tile breakdown shown
        # elsewhere in the status view.
        return {
            "resumed": False,
            "message": (
                f"Nothing to resubmit — every one of the {total} slides in this run was "
                f"already attempted. (Some may have saved zero tiles; resubmitting won't "
                f"change that — see the zero-tile breakdown in the status view above.)"
            ),
        }

    missing_slide_ids = [slide_id_from_raw_path(p) for p in missing_raw_paths]

    # The parameters the original run actually tiled with, so the resumed
    # slides come out identical to the ones already on disk. Note min_tissue is
    # deliberately NOT passed as a top-level field: doing so would put it in
    # model_fields_set and let it override the recorded block (see
    # _resolve_tiling_params). None here — a run predating the tiling_params
    # column — falls back to current defaults, reported below so the caller
    # knows the resume is not a guaranteed reproduction.
    recorded_tiling_params = _row_tiling_params(row)

    resume_req = DatasetJobRequest(
        dataset_path=row["raw_dir"],
        slide_names=missing_slide_ids,
        partition=row["partition"],
        notify_email=row["notify_email"],
        tiling_params=recorded_tiling_params,
        # Re-run into the same folder tiles already live in, not whatever
        # raw_dir.name would resolve to by default — matters if the
        # original submission was given a custom dataset_name.
        dataset_name=dataset_name,
    )
    result = _start_dataset_submission(
        Path(row["raw_dir"]),
        resume_req,
        background_tasks,
        # Persisted, not just echoed back: this reply is the only place the
        # relationship existed before, so reloading the page lost it.
        resumed_from_submission_id=submission_id,
    )
    result.update({
        "resumed": True,
        "resumed_from_submission_id": submission_id,
        # What the resumed slides will be tiled with, and whether that is the
        # original run's own recorded settings or a fallback. The caller should
        # surface the fallback case: mixing thresholds within one dataset is
        # exactly the failure this is meant to prevent, and silently defaulting
        # would reintroduce it for every pre-migration run.
        "tiling_params": _resolve_tiling_params(resume_req),
        "tiling_params_source": "original_run" if recorded_tiling_params else "defaults",
        "missing_slide_count": len(missing_raw_paths),
        "never_attempted_count": len(breakdown["never_attempted"]),
        "corrupt_metadata_count": len(breakdown["corrupt"]),
        "zero_tile_count": len(breakdown["zero_tile"]),
        "total_in_original_manifest": total,
    })
    return result


# --- the one-click pipeline: Stages 1-4 as one Nextflow run ------------------
#
# POST /pipeline-runs replaces clicking through tiling, packaging, extraction
# and classification one at a time. It creates an ordinary slurm_dataset_runs
# row, so every later stage, the listing, the rollup and the stepper see an
# ordinary run — the difference is that its four stage job-id columns hold
# nf:<submission_id>:<stage> sentinels, answered by hpl_nf_state from the
# stage's done marker and the head job (see _sentinel_state). Output paths are
# recorded at submission, so _job_output_ready and the validators gate a
# pipeline stage exactly as they gate a clicked one.
#
# No migration: that the tiling job id is a sentinel IS the record that a run
# is a pipeline run, and everything else it needs lives in its own directory
# (HPL_NF_RESULTS_ROOT/<submission_id>: run_config.json, head_job_ids, stage
# markers, nextflow.log).
#
# Registration and the KB load are deliberately outside it. They write the
# shared Knowledge Bank, and Stage 6 never commits without a dry run a person
# has looked at.


class PipelineRunRequest(DatasetJobRequest):
    """A new run, Stages 1-4 in one submission. Every field a later stage used
    to ask for at its own button is asked for here, once."""

    # Stage 3. A deployment setting (HPL_CHECKPOINT), not something the UI
    # asks for: the reference's clusters were built from this checkpoint's
    # embeddings. Overridable per request for a deliberate comparison.
    checkpoint: str | None = None
    # How many slides tile at once (HPL_NF_MAX_TILING); the per-stage form's 10
    # would take weeks on a large cohort.
    max_concurrent: int = submit_hpl_nf.DEFAULT_MAX_TILING
    model: str = "BarlowTwins_3"
    marker: str = "he"
    # Split the encode across N GPU tasks — the lever that scales extraction,
    # since reads are single-threaded per process (see CLAUDE.md).
    extraction_shards: int = 1

    # Stage 4. Same meaning and defaults as ClusterAssignmentRequest.
    reference: str | None = None
    vote_preset: str = DEFAULT_VOTE_PRESET
    distance_weighted: bool | None = None
    distance_power: float | None = None
    class_weighted: bool | None = None
    local_scaling: int | None = None
    adaptive_margin: float | None = None
    adaptive_k: int | None = None
    assignment_shards: int = 1
    device: str = "auto"

    # The head job, as for ANORAK: how many head jobs to chain (the head plus
    # standbys that resume it if it ends unfinished), and its walltime — a
    # partition property, so overridable here rather than only in the server's
    # environment. None takes HPL_NF_CHAIN / HPL_NF_HEAD_TIME_LIMIT.
    chain: int | None = None
    time_limit: str | None = None

    # A random subset is sampled with a recorded seed, so the same slides can
    # be asked for again — the rule ANORAK's subset follows. None chooses one.
    seed: int | None = None

    # Package without slides that fail to tile. Off by default, as the
    # per-stage path's afterok was; on, the failures are named on the tiling
    # step rather than silently absent from the .h5.
    allow_incomplete: bool = False

    # When this run's output paths already hold another run's complete
    # outputs (a full run over a dataset an earlier run packaged), move them
    # into a superseded-<date> folder beside them instead of refusing. On by
    # default, because the UI's one click has no other way forward: it is never
    # a delete, and is refused while any recorded job may still be writing them.
    move_existing_outputs: bool = True

    def vote_overrides(self) -> dict:
        return {
            "distance_weighted": self.distance_weighted,
            "distance_power": self.distance_power,
            "class_weighted": self.class_weighted,
            "local_scaling": self.local_scaling,
            "adaptive_margin": self.adaptive_margin,
            "adaptive_k": self.adaptive_k,
        }


def _is_pipeline_row(row) -> bool:
    return _is_nf_job_id(row.get("job_id"))


def _refuse_if_pipeline_run(row, action: str) -> None:
    """The per-stage endpoints stay for runs that were clicked through; a
    pipeline run's Stages 1-4 belong to its pipeline. Submitting one of them by
    hand would overwrite the stage's sentinel with a job id the pipeline knows
    nothing about, and the next -resume would redo or contradict it."""
    if _is_pipeline_row(row):
        raise HTTPException(
            400,
            f"This run's Stages 1-4 are driven by its Nextflow pipeline, so {action} "
            f"cannot be started on its own. If the pipeline stopped, resume it "
            f"(POST /dataset-jobs/{row['submission_id']}/pipeline-resume); it "
            f"re-runs only what did not finish.",
        )


def _run_pipeline_submission(submission_id: str, raw_dir: str,
                             req: PipelineRunRequest, config: dict, nf_params: dict) -> None:
    """Background half of POST /pipeline-runs: find the slides, write the
    manifest, submit the head job. Discovery is a recursive walk that can take
    minutes on a large network-mounted directory, which is why the request
    returned before it started — the same split /dataset-jobs makes."""
    try:
        _update_dataset_run(submission_id, status="discovering")
        pool = discover_slides(Path(raw_dir))
        seed = req.seed
        if req.sample_size and seed is None:
            seed = random.randrange(2 ** 31)
        slides = select_slides(
            pool,
            sample_size=req.sample_size,
            slide_names=req.slide_names,
            random_seed=seed,
        )
        if not slides:
            raise ValueError(f"No supported slide files under {raw_dir}.")
        validate_unique_slide_ids(slides)
        write_manifest(slides, Path(config["manifest"]))
        _update_dataset_run(submission_id, status="submitting", total_slides=len(slides))

        # Stop pressed while the directory walk ran: nothing is queued yet, and
        # submitting now would start a run its owner has already cancelled.
        if _get_dataset_run_row(submission_id).get("status") == "cancelled":
            return

        # How the slides were chosen, in the run's own config — what makes a
        # subset reproducible, and a result explainable, months later.
        config["selection"] = {
            "scope": ("subset" if req.sample_size else
                      "slides" if req.slide_names else "full"),
            "slides": len(slides),
            "pool": len(pool),
            "sample_size": req.sample_size,
            "seed": seed,
        }
        result = submit_hpl_nf.submit_pipeline(
            config, nf_params,
            profile=HPL_NF_PROFILE,
            job_name=f"hpl_nf_{submission_id}",
            notify_email=req.notify_email,
            chain=req.chain or submit_hpl_nf.DEFAULT_CHAIN,
            time_limit=(req.time_limit or "").strip() or None,
        )
        _update_dataset_run(
            submission_id, status="submitted",
            error=result.get("chain_error"),
        )
        _record_run_job(
            submission_id, "pipeline",
            ",".join([result["nf_job_id"], *result.get("chain_job_ids", [])]),
            output_path=result["out_dir"],
            params={
                "slides": len(slides),
                "checkpoint": config["extraction"]["checkpoint"],
                "reference": config["assignment"]["reference"],
                "vote": config["assignment"]["vote"],
                "extraction_shards": config["extraction"]["shards"],
                "assignment_shards": config["assignment"]["shards"],
                "device": config["assignment"]["device"],
                "chain": result.get("chain"),
            },
        )
    except Exception as e:  # noqa: BLE001 - recorded on the row, where the UI shows it
        _update_dataset_run(submission_id, status="error", error=str(e))


@app.get("/pipeline-defaults")
def pipeline_defaults():
    """What a one-click pipeline run uses, so the UI can say so beside the
    button: every setting is the server's, and nothing is asked for but a
    dataset path."""
    import submit_cluster_assignment as sca
    return {
        "checkpoint": submit_hpl_nf.DEFAULT_CHECKPOINT,
        "reference": str(sca._reference_path(None)),
        "vote_preset": DEFAULT_VOTE_PRESET,
        "max_tiling": submit_hpl_nf.DEFAULT_MAX_TILING,
        "min_tissue": _default_tiling_params().get("min_tissue"),
        "tile_root": str(PROCESSED_TILES_DIR),
        "h5_root": str(HPL_DATASETS_ROOT),
        "runs_root": str(HPL_NF_RESULTS_ROOT),
        "head_jobs": submit_hpl_nf.DEFAULT_CHAIN,
    }


@app.get("/pipeline-submit-check")
def check_pipeline_submit(partition: str | None = None):
    """Can a compute node run sbatch? The pipeline's head job submits every
    task itself, and one that cannot starts, submits nothing and waits out its
    walltime. Same probe as /anorak-submit-check."""
    return _check_slurm_submit_from_compute_node(partition)


@app.post("/pipeline-runs")
def create_pipeline_run(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Start Stages 1-4 as one Nextflow run: the UI's single "Run pipeline" click.

    Everything that can be refused is refused here, synchronously, before a row
    exists — a checkpoint typo is a 400 now, not a failed GPU task after hours
    of tiling. Slide discovery and the sbatch happen in the background, and the
    returned submission_id is polled through /dataset-jobs/{id}/status like any
    other run.
    """
    try:
        raw_dir = _resolve_dataset_path(req.dataset_path)
        dataset_name = (_sanitize_dataset_name(req.dataset_name)
                        if req.dataset_name else raw_dir.name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    submission_id = str(uuid.uuid4())
    is_subset = bool(req.sample_size or req.slide_names)
    tiling_params = _resolve_tiling_params(req)
    h5_dataset_name = _effective_h5_dataset_name(dataset_name, is_subset)
    out_dir = _nf_run_dir(submission_id)

    with _slurm_submission_lock():
        try:
            config, nf_params = submit_hpl_nf.resolve_run(
                submission_id=submission_id,
                out_dir=out_dir,
                manifest=out_dir / "manifest.txt",
                raw_dir=raw_dir,
                mask_dir=TISSUE_MASK_DIR,
                tile_dir=PROCESSED_TILES_DIR,
                output_root=HPL_DATASETS_ROOT,
                tile_dataset_name=dataset_name,
                h5_dataset_name=h5_dataset_name,
                tiling_params=tiling_params,
                checkpoint=req.checkpoint or submit_hpl_nf.DEFAULT_CHECKPOINT,
                model=req.model,
                marker=req.marker,
                reference=Path(req.reference) if req.reference else None,
                vote_preset=req.vote_preset,
                vote_overrides=req.vote_overrides(),
                extraction_shards=req.extraction_shards,
                assignment_shards=req.assignment_shards,
                device=req.device,
                cpu_partition=req.partition,
                max_tiling_forks=req.max_concurrent,
                allow_incomplete=req.allow_incomplete,
            )
            targets = {
                "h5": Path(config["packaging"]["h5_path"]),
                "projections": Path(config["extraction"]["output_path"]),
                "assignments": Path(config["assignment"]["out_csv"]),
            }
            try:
                submit_hpl_nf.refuse_foreign_outputs(targets)
            except FileExistsError:
                if not req.move_existing_outputs:
                    raise
                _refuse_if_outputs_in_use(targets)
                # Recorded with the run, so where its predecessor's outputs
                # went is answerable from the run itself.
                config["superseded"] = submit_hpl_nf.move_outputs_aside(
                    targets, datetime.now().strftime("%Y%m%d-%H%M%S"))
        except (ValueError, FileNotFoundError, FileExistsError,
                NotADirectoryError, KeyError) as e:
            raise HTTPException(400, str(e))
        except SystemExit as e:
            # resolve_vote / vote_flags refuse an inert vote this way.
            raise HTTPException(400, str(e))

        out_dir.mkdir(parents=True, exist_ok=True)
        eng = _get_engine()
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO slurm_dataset_runs
                        (submission_id, raw_dir, mask_dir, tile_dir, status,
                         is_subset, partition, notify_email, dataset_name,
                         tiling_params, manifest_path,
                         job_id, h5_job_id, h5_output_path,
                         extraction_job_id, extraction_output_path, extraction_checkpoint,
                         assignment_job_id, assignment_output_path, assignment_reference)
                    VALUES
                        (:submission_id, :raw_dir, :mask_dir, :tile_dir, 'queued',
                         :is_subset, :partition, :notify_email, :dataset_name,
                         :tiling_params, :manifest_path,
                         :job_id, :h5_job_id, :h5_output_path,
                         :extraction_job_id, :extraction_output_path, :extraction_checkpoint,
                         :assignment_job_id, :assignment_output_path, :assignment_reference)
                """),
                {
                    "submission_id": submission_id,
                    "raw_dir": str(raw_dir),
                    "mask_dir": str(TISSUE_MASK_DIR),
                    "tile_dir": str(PROCESSED_TILES_DIR),
                    "is_subset": is_subset,
                    "partition": req.partition,
                    "notify_email": req.notify_email,
                    "dataset_name": dataset_name,
                    "tiling_params": json.dumps(tiling_params),
                    "manifest_path": config["manifest"],
                    "job_id": _nf_job_id(submission_id, "tiling"),
                    "h5_job_id": _nf_job_id(submission_id, "packaging"),
                    "h5_output_path": config["packaging"]["h5_path"],
                    "extraction_job_id": _nf_job_id(submission_id, "extraction"),
                    "extraction_output_path": config["extraction"]["output_path"],
                    "extraction_checkpoint": config["extraction"]["checkpoint"],
                    "assignment_job_id": _nf_job_id(submission_id, "assignment"),
                    "assignment_output_path": config["assignment"]["out_csv"],
                    "assignment_reference": config["assignment"]["reference"],
                },
            )
    # Best-effort, like the per-stage endpoint: this column arrives with a
    # migration that may not be applied, and it must not cost the run.
    _update_dataset_run_best_effort(submission_id, assignment_vote=config["assignment"]["vote"])

    req = req.model_copy(update={"dataset_name": dataset_name, "tiling_params": tiling_params})
    background_tasks.add_task(
        _run_pipeline_submission, submission_id, str(raw_dir), req, config, nf_params,
    )
    return {
        "submission_id": submission_id,
        "status": "queued",
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        "out_dir": str(out_dir),
        "h5_output_path": config["packaging"]["h5_path"],
        "extraction_output_path": config["extraction"]["output_path"],
        "assignment_output_path": config["assignment"]["out_csv"],
        "gpu_gres": config["extraction"]["gres"],
        "gpu_gres_reason": config["extraction"]["gres_reason"],
        "device": config["assignment"]["device"],
        "device_reason": config["assignment"]["device_reason"],
        "vote": config["assignment"]["vote"],
        "superseded": config.get("superseded") or [],
    }


def _refuse_if_outputs_in_use(targets: dict[str, Path]) -> None:
    """Refuse to move outputs any recorded run may still be writing.

    Every run that records one of these paths is asked for its stage's state
    (sentinels included, so a live pipeline run counts). In flight, or
    unknown because Slurm cannot be reached, is a refusal: moving a file out
    from under a writer is the one thing worse than refusing the run.
    """
    columns = {"h5": ("h5_output_path", "h5_job_id"),
               "projections": ("extraction_output_path", "extraction_job_id"),
               "assignments": ("assignment_output_path", "assignment_job_id")}
    eng = _get_engine()
    with eng.connect() as conn:
        for key, (path_col, job_col) in columns.items():
            rows = conn.execute(
                text(f"SELECT submission_id, {job_col} AS job_id FROM slurm_dataset_runs "
                     f"WHERE {path_col} = :path AND {job_col} IS NOT NULL"),
                {"path": str(targets[key])},
            ).mappings().fetchall()
            for row in rows:
                for job_id in _split_job_ids(row["job_id"]) or [row["job_id"]]:
                    state = _get_slurm_job_state(job_id)
                    if state is None or state in IN_FLIGHT_SLURM_STATES:
                        shown = state or "unknown — Slurm unreachable"
                        raise HTTPException(
                            400,
                            f"Run {row['submission_id']} may still be writing "
                            f"{targets[key]} (state: {shown}). Stop it, or wait for "
                            f"it, before moving its outputs aside.")


class PipelineResumeRequest(BaseModel):
    chain: int | None = None
    time_limit: str | None = None
    # Switch on "package without slides that fail to tile" for the resumed
    # run — the recovery for a run that stopped on unreadable slides.
    allow_incomplete: bool | None = None


@app.post("/dataset-jobs/{submission_id}/pipeline-resume")
def resume_pipeline_run(submission_id: str, req: PipelineResumeRequest | None = None):
    """Resubmit a stopped pipeline run's head job with -resume.

    Nextflow skips every task it recorded as done, and each task that does run
    first checks whether its output already exists and validates — so this
    re-runs exactly what did not finish, with the run's own recorded config.
    Refused while the head job (or a standby) is still alive: two head
    processes in one work directory is the collision --signal and the watchdog
    exist to prevent.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not _is_pipeline_row(row):
            raise HTTPException(400, "This run was not started as a pipeline run.")
        out_dir = _nf_run_dir(submission_id)
        try:
            config = submit_hpl_nf.read_run_config(out_dir)
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"Cannot read this run's config in {out_dir}: {e}")
        if not Path(config["manifest"]).is_file():
            raise HTTPException(
                400, "This run never got as far as writing its slide manifest — "
                     "start a new pipeline run instead.")

        _NF_HEAD_STATE_CACHE.pop(submission_id, None)
        head_state = _nf_head_state(submission_id)
        if head_state is None:
            raise HTTPException(503, "Couldn't reach Slurm to confirm the pipeline "
                                     "has stopped — try again shortly.")
        if head_state in IN_FLIGHT_SLURM_STATES:
            raise HTTPException(400, f"The pipeline is still running (head job "
                                     f"{head_state}). Stop it first to restart it.")
        if all((_nf_stage_summary(out_dir, head_state)[stage]["state"] == "COMPLETED")
               for stage in NF_STAGES):
            raise HTTPException(400, "Every stage of this pipeline already finished.")

        try:
            config, nf_params = submit_hpl_nf.resume_params(
                config, allow_incomplete=req.allow_incomplete if req else None)
        except ValueError as e:
            raise HTTPException(400, str(e))
        try:
            result = submit_hpl_nf.submit_pipeline(
                config, nf_params,
                profile=HPL_NF_PROFILE,
                resume=True,
                job_name=f"hpl_nf_{submission_id}",
                notify_email=row.get("notify_email"),
                chain=(req.chain if req and req.chain else submit_hpl_nf.DEFAULT_CHAIN),
                time_limit=((req.time_limit or "").strip() or None) if req else None,
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))
        _NF_HEAD_STATE_CACHE.pop(submission_id, None)
        _update_dataset_run(submission_id, status="submitted", error=result.get("chain_error"))
        _record_run_job(
            submission_id, "pipeline",
            ",".join([result["nf_job_id"], *result.get("chain_job_ids", [])]),
            output_path=result["out_dir"], params={"resume": True},
        )
        return {"submission_id": submission_id, "resumed": True, **result}


def _pipeline_status(submission_id: str) -> dict:
    """The pipeline block of /status: head job, per-stage state, and — once it
    has stopped short — the end of nextflow.log, which is where it says why."""
    out_dir = _nf_run_dir(submission_id)
    head_state = _nf_head_state(submission_id)
    stages = _nf_stage_summary(out_dir, head_state)
    finished = all(stages[s]["state"] == "COMPLETED" for s in NF_STAGES)
    try:
        config = submit_hpl_nf.read_run_config(out_dir)
    except (OSError, ValueError):
        config = {}
    stopped = head_state not in IN_FLIGHT_SLURM_STATES and head_state is not None
    return {
        "mode": "nextflow",
        "out_dir": str(out_dir),
        "head_job_ids": _nf_read_head_job_ids(out_dir),
        "head_state": head_state,
        "stages": stages,
        "finished": finished,
        "resumable": stopped and not finished and bool(config),
        "log_path": str(out_dir / "nextflow.log"),
        "log_tail": submit_hpl_nf.log_tail(out_dir) if stopped and not finished else None,
        # Written by the supervisor when a run ends for good, so no standby
        # resumes it — the same marker, and the same reading, as ANORAK's.
        "stop_reason": _anorak_stop_reason(out_dir) if stopped and not finished else None,
        "selection": config.get("selection"),
        "report_path": str(out_dir / "pipeline_info" / "report.html"),
        "settings": {
            "checkpoint": config.get("extraction", {}).get("checkpoint"),
            "extraction_shards": config.get("extraction", {}).get("shards"),
            "gpu_gres": config.get("extraction", {}).get("gres"),
            "reference": config.get("assignment", {}).get("reference"),
            "vote": config.get("assignment", {}).get("vote"),
            "assignment_shards": config.get("assignment", {}).get("shards"),
            "device": config.get("assignment", {}).get("device"),
            "allow_incomplete": bool(config.get("allow_incomplete")),
        } if config else None,
    }


def _row_test_packaging_params(row) -> dict | None:
    """What the recorded test packaging job actually sampled.

    Lets the UI say "the job below is from an earlier setup" after a reload has
    thrown away the form state it used to compare against — previously that
    warning only worked within a single session, which is the one case where the
    user could still remember what they had typed.
    """
    recorded = row.get("test_h5_params")
    if not recorded:
        return None
    if isinstance(recorded, str):
        try:
            recorded = json.loads(recorded)
        except json.JSONDecodeError:
            return None
    return recorded if isinstance(recorded, dict) else None


def _row_dataset_name(row) -> str:
    """The dataset folder this run's tiles live under. Rows created before
    the dataset_name column existed have it NULL — fall back to raw_dir's
    own folder name, which is what those runs actually used at the time.
    """
    return row["dataset_name"] or Path(row["raw_dir"]).name


def _get_dataset_run_row(submission_id: str) -> dict:
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()
    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    return dict(row)


def _tiled_coverage(
    raw_dir: Path, tile_dir: Path, dataset_name: str, *, strict: bool = False
) -> dict:
    """What is actually tiled on disk right now, for every slide in raw_dir.

    The run's manifest says what a submission set out to do; this says what
    exists. They diverge constantly and the difference is what the UI needs:
    a subset run's manifest lists 30 slides, but the dataset folder may hold
    tiles for all 400 because earlier runs (or a resume, or a hand-run job)
    filled it in. Deciding what can be packaged from the manifest alone means
    refusing to package tiles that are sitting right there.

    "Tiled" delegates to tiling_output_complete() — the same function
    submit_mask_tile_slurm.py's worker uses to decide whether to skip a slide —
    so the two cannot drift apart as that test is tightened.

    strict=False (the default) is the one deliberate difference. The full test
    also parses the tile-metadata CSV to confirm its row count matches the
    summary's saved_tiles, which costs a whole-file parse per slide: fine for
    the single slide a worker is deciding about, ruinous across 14,000 of them
    on cephfs while a user waits. The JSON checks (parseable summary,
    saved_tiles present and sane) are kept, since those files are small.
    #
    The asymmetry that leaves is worth being explicit about: a slide whose CSV
    stopped short of its summary counts as tiled here, and the worker would
    re-tile it. That is the safe direction — the worker has the final say and
    redoes the work — but it means this count can be marginally optimistic
    after an interrupted tiling run. strict=True removes that gap at the cost
    of reading every CSV, which is why it's opt-in and never used by anything
    polling: it's for the one moment someone wants to sign off on a dataset
    being finished before committing GPU hours to it.

    Cost is a stat plus a small JSON read per slide over a network filesystem.
    That's why this lives behind its own endpoint rather than in the 10s poll.
    """
    slide_dataset_dir = tile_dir / dataset_name
    tiled: list[str] = []
    untiled: list[str] = []
    for slide_path in discover_slides(raw_dir):
        slide_id = slide_id_from_raw_path(slide_path)
        slide_tile_dir = slide_dataset_dir / slide_id
        metadata = slide_tile_dir / f"{slide_id}_tile_metadata.csv"
        summary = slide_tile_dir / f"{slide_id}_tiling_summary.json"
        complete = tiling_output_complete(
            metadata, summary, slide_tile_dir, verify_row_count=strict
        )
        (tiled if complete else untiled).append(str(slide_path))
    return {
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        "strict": strict,
        "slides_in_directory": len(tiled) + len(untiled),
        "slides_tiled": len(tiled),
        "slides_untiled": len(untiled),
        "tiled_paths": tiled,
        # Capped: this is for showing the user which slides still need work,
        # and a full list on a 14,000-slide directory is neither useful in the
        # UI nor cheap to ship on every poll.
        "untiled_sample": [Path(p).name for p in untiled[:50]],
    }


@app.get("/dataset-jobs/{submission_id}/jobs")
def dataset_job_history(submission_id: str):
    """Every Slurm job this run has submitted, across all stages, newest first.

    Separate from /status because it costs an sacct call and answers a different
    question: /status says what the run can do next, this says what it has
    already tried. Repackaging and repeated test packaging are invisible in
    /status by design — those fields hold only the latest attempt.
    """
    _get_dataset_run_row(submission_id)      # 404s for an unknown run
    return {"submission_id": submission_id, "jobs": _run_job_history(submission_id)}


@app.get("/dataset-jobs/{submission_id}/tiled-coverage")
def tiled_coverage(submission_id: str):
    """Live, on-disk answer to "how much of this directory is actually tiled?"

    Deliberately not folded into /status: it stats two files per slide over
    cephfs, and /status is polled every 10s by every open tab.
    """
    row = _get_dataset_run_row(submission_id)
    return _tiled_coverage(
        Path(row["raw_dir"]), Path(row["tile_dir"]), _row_dataset_name(row)
    )


def _runs_for_directory(raw_dir: Path, dataset_name: str) -> list[dict]:
    """Every recorded run that tiled into this (raw_dir, dataset_name) pair.

    Filtering on dataset_name in Python rather than SQL because
    _row_dataset_name() has to resolve a NULL column back to raw_dir's folder
    name for rows predating that column — a WHERE clause would silently drop
    exactly those older runs, which are the ones most likely to hold the
    tiles nobody remembers submitting.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT * FROM slurm_dataset_runs WHERE raw_dir = :raw_dir "
                "ORDER BY submitted_at ASC"
            ),
            {"raw_dir": str(raw_dir)},
        ).mappings().fetchall()
    return [dict(row) for row in rows if _row_dataset_name(row) == dataset_name]


def _tiling_readiness(
    raw_dir: Path, tile_dir: Path, dataset_name: str, *, strict: bool = False
) -> dict:
    """One answer to "is tiling finished for this whole directory?", across
    every run that ever tiled into it.

    The per-run endpoints cannot answer this, and that is the point. A run's
    /status reports the run's own manifest — correct, but a 30-slide subset
    run reporting 30/30 reads as "the dataset is tiled" when 14,000 slides
    sit beside it untouched. Directories here get filled in by several runs
    plus resumes plus hand-run jobs, so the only trustworthy scope is the
    directory, and the only trustworthy authority is the disk.

    Disk decides what is *done*; Slurm decides what that means about what is
    *left*. Untiled slides with jobs still running is a wait; the same
    untiled slides with nothing running is a stall needing resubmission, and
    those two need to be distinguishable without reading sacct by hand.

    sacct being unreachable is reported as its own verdict rather than
    folded into either. _get_slurm_array_state_counts returns None for
    "genuinely unknown" and {} for "ran fine, nothing in flight" — collapsing
    those would let a controller outage read as "nothing is running, so this
    has stalled" and send someone off to resubmit work that is mid-flight.
    """
    coverage = _tiled_coverage(raw_dir, tile_dir, dataset_name, strict=strict)
    runs = _runs_for_directory(raw_dir, dataset_name)

    job_ids: list[str] = []
    for run in runs:
        job_ids.extend(j for j in (run["job_id"] or "").split(",") if j)
    job_ids = sorted(set(job_ids))

    state_counts = _get_slurm_array_state_counts(job_ids)
    slurm_known = state_counts is not None
    in_flight = bool(
        slurm_known and set(state_counts) & IN_FLIGHT_SLURM_STATES
    )

    untiled = coverage["slides_untiled"]
    total = coverage["slides_in_directory"]

    if not total:
        verdict = "no_slides"
        message = f"No supported WSI files found under {raw_dir}."
    elif not untiled:
        verdict = "complete"
        message = (
            f"All {total} slides in {raw_dir} have tiles on disk"
            f"{' (row counts verified)' if strict else ''}."
        )
    elif in_flight:
        verdict = "in_progress"
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled; "
            f"tiling jobs are still running."
        )
    elif not slurm_known:
        verdict = "unknown"
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled, but sacct "
            f"could not be reached — cannot confirm whether the remaining "
            f"{untiled} are still being worked on. Retry before resubmitting."
        )
    elif not runs:
        # No tracked run has ever targeted this directory, so there is nothing
        # to have stalled and nothing to resume. Kept distinct from "stalled"
        # because the two need opposite actions — submit a first run here,
        # versus resume an existing one — and because a brand-new dataset
        # being told it "needs resubmitting" is the kind of wrong-but-plausible
        # message that sends someone hunting for a run that never existed.
        verdict = "not_started"
        if coverage["slides_tiled"]:
            message = (
                f"{coverage['slides_tiled']} of {total} slides already have "
                f"tiles, but no tracked run targeted this directory — they came "
                f"from a hand-run job or a different raw_dir. Submit a run "
                f"(POST /dataset-jobs) to tile the remaining {untiled}."
            )
        else:
            message = (
                f"None of the {total} slides in {raw_dir} have been tiled, and "
                f"no run has been submitted for it. Submit one with "
                f"POST /dataset-jobs."
            )
    else:
        verdict = "stalled"
        # Name a real run to resume rather than a {submission_id} placeholder —
        # the most recent one, since _runs_for_directory orders oldest-first.
        message = (
            f"{coverage['slides_tiled']} of {total} slides tiled and no tiling "
            f"job is running — the remaining {untiled} need resubmitting "
            f"(POST /dataset-jobs/{runs[-1]['submission_id']}/resume)."
        )

    return {
        "raw_dir": str(raw_dir),
        "dataset_name": dataset_name,
        # The plain yes/no. Deliberately only true on full coverage, so it
        # cannot be satisfied by a subset run finishing its own manifest.
        "tiling_done": verdict == "complete",
        "verdict": verdict,
        "message": message,
        "strict": strict,
        "slides_in_directory": total,
        "slides_tiled": coverage["slides_tiled"],
        "slides_untiled": untiled,
        "untiled_sample": coverage["untiled_sample"],
        # tiled_paths is omitted: 14,000 absolute paths is a payload nobody
        # reading a verdict wants. /tiled-coverage still returns it for the
        # callers that package from it.
        "runs_checked": len(runs),
        "runs": [
            {
                "submission_id": run["submission_id"],
                "status": run["status"],
                "is_subset": run["is_subset"],
                "total_slides": run["total_slides"],
                "job_id": run["job_id"],
                "submitted_at": str(run["submitted_at"]) if run["submitted_at"] else None,
            }
            for run in runs
        ],
        "slurm_job_ids": job_ids,
        "slurm_state_counts": state_counts,
        "slurm_reachable": slurm_known,
    }


@app.get("/tiling-readiness")
def tiling_readiness(
    raw_dir: Optional[str] = Query(
        None, description="Directory of WSIs to check. Omit if passing submission_id."
    ),
    dataset_name: Optional[str] = Query(
        None, description="Tile folder under PROCESSED_TILES_DIR. Defaults to raw_dir's name."
    ),
    submission_id: Optional[str] = Query(
        None, description="Resolve raw_dir/dataset_name from an existing run instead."
    ),
    strict: bool = Query(
        False,
        description="Also verify each slide's tile-metadata row count against its "
                    "summary. Authoritative but reads every CSV — minutes on a "
                    "14,000-slide directory. Use before committing to GPU hours.",
    ),
):
    """Is tiling actually finished for a whole directory, across every run?

    Not scoped to one submission, unlike /dataset-jobs/{id}/tiled-coverage —
    pass submission_id only as a convenient way to name the directory, and
    the answer still covers everything in it.

    Same cost profile as /tiled-coverage (two files stat'd per slide over
    cephfs, more with strict=true), so this is a button, not a poll.
    """
    if submission_id:
        row = _get_dataset_run_row(submission_id)
        resolved_raw = Path(row["raw_dir"])
        resolved_tile = Path(row["tile_dir"])
        resolved_dataset = dataset_name or _row_dataset_name(row)
    elif raw_dir:
        # Through the same resolver POST /dataset-jobs uses, so the path this
        # looks up is byte-identical to what that endpoint stored. A bare
        # Path(raw_dir) matched nothing for "~/x", a trailing slash, or a
        # symlink — _runs_for_directory compares raw_dir as an exact string, so
        # every run would silently drop out and a fully-tiled directory could
        # report as never started.
        try:
            resolved_raw = _resolve_dataset_path(raw_dir)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # PROCESSED_TILES_DIR is this server's current default. A run records
        # its own tile_dir, so if one was submitted when that pointed
        # elsewhere, pass submission_id instead and the row decides.
        resolved_tile = PROCESSED_TILES_DIR
        resolved_dataset = dataset_name or resolved_raw.name
    else:
        raise HTTPException(400, "Provide raw_dir or submission_id.")

    return _tiling_readiness(
        resolved_raw, resolved_tile, resolved_dataset, strict=strict
    )


@app.post("/dataset-jobs/{submission_id}/package")
def start_packaging_job(
    submission_id: str,
    allow_incomplete: bool = Query(
        False,
        description="Package even though some slides failed tiling. Switches the Slurm "
                    "dependency from afterok to afterany and accepts a dataset with holes.",
    ),
    scope: str = Query(
        "run",
        description="'run' packages this run's own manifest (the default, unchanged). "
                    "'tiled' packages every slide in the raw directory that has tiles on "
                    "disk right now, regardless of which run produced them.",
    ),
    resume: Optional[bool] = Query(
        None,
        description="true continues a previous attempt's checkpoint (and fails if there "
                    "is none); false discards it and repackages from scratch. Omit only "
                    "for non-interactive callers — the UI always sends an explicit "
                    "choice, because silently continuing an earlier attempt is a "
                    "decision the user should be making.",
    ),
):
    """Manually start .h5 packaging for a run whose tiling has already been
    submitted. This used to auto-chain via Slurm's --dependency the moment
    tiling was submitted; now it only happens when the user clicks the
    "Start packaging" button in the UI, once they've confirmed tiling is
    actually done. The submitted packaging job still carries its own
    --dependency on every tiling batch job as a safety net, in case this gets
    clicked slightly before the last batch finishes.

    That dependency is afterok by default (see submit_packaging_job), so a run
    with failed tiling tasks is refused here with a 400 rather than submitted
    as a job Slurm could never start. Pass allow_incomplete=true to package
    anyway, knowing the resulting .h5 will be missing those slides.

    Not gated on row["status"] == "submitted" — cancelling a run only flips
    its status column, it doesn't touch the manifest or the tiles already
    written to disk, so packaging the whole thing is still valid (and
    useful) for a cancelled run whose tiling had already finished. job_id
    existing is what actually means tiling was submitted in the first place.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        _refuse_if_pipeline_run(row, "packaging")
        if not row["job_id"]:
            raise HTTPException(400, "Tiling hasn't been submitted yet for this run.")

        tile_dataset_name = _row_dataset_name(row)

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path:
            raise HTTPException(400, "This run never got far enough to have a manifest.")
        # Only scope="run" actually reads this file (below, for the sbatch job
        # and the pool-size count). scope="tiled" writes a brand new manifest
        # from live disk coverage a few lines down, using only this path's
        # parent directory — checking the original file exists here refused a
        # "package the whole directory" request over a manifest it was never
        # going to open, whenever the run's own manifest had since been
        # deleted or moved.
        if scope != "tiled" and not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")
        job_ids = [j for j in row["job_id"].split(",") if j]

        # Scope is resolved up front, before the retry guard below, because it
        # decides *which output file* this submission is about. Resolving it
        # afterwards meant the guard always judged the run's recorded path: a
        # subset run whose own 30-slide .h5 had completed refused a scope="tiled"
        # request with "Packaging has already completed for this run", even
        # though the full-coverage .h5 it was actually asking for did not exist.
        if scope == "tiled":
            # Package what is on disk, not what this run set out to do. A
            # subset run's manifest is 30 slides even when the dataset folder
            # holds tiles for the whole directory — put there by an earlier
            # run, a resume, or a hand-run job. Packaging from the manifest
            # then ignores tiles sitting right there, and no amount of
            # re-running this run widens it, because its manifest was fixed at
            # submission time.
            coverage = _tiled_coverage(
                Path(row["raw_dir"]), Path(row["tile_dir"]), tile_dataset_name
            )
            if not coverage["slides_tiled"]:
                raise HTTPException(
                    400, f"No slide in {row['raw_dir']} has tiles on disk yet."
                )
            # The real-time completeness gate. Disk is the authority, not
            # sacct: a slide either has its tiles or it does not, whatever
            # Slurm remembers about the job that was meant to produce them.
            if coverage["slides_untiled"] and not allow_incomplete:
                raise HTTPException(
                    400,
                    {
                        "error": "tiling_incomplete",
                        "submission_id": submission_id,
                        "slides_tiled": coverage["slides_tiled"],
                        "slides_untiled": coverage["slides_untiled"],
                        "slides_in_directory": coverage["slides_in_directory"],
                        "untiled_sample": coverage["untiled_sample"],
                        "message": (
                            f"{coverage['slides_untiled']} of "
                            f"{coverage['slides_in_directory']} slides in this directory "
                            f"have no tiles on disk. Tile them first, or package the "
                            f"{coverage['slides_tiled']} that do."
                        ),
                    },
                )
            manifest_path = manifest_path.parent / (
                f"wsi_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_tiled_{tile_dataset_name}.txt"
            )
            write_manifest([Path(p) for p in coverage["tiled_paths"]], manifest_path)
            # Full coverage of the dataset folder is by definition not a
            # subset, so it takes the unsuffixed name and its own output
            # directory — deliberately leaving any "_subset_N" .h5 this run
            # already produced untouched instead of overwriting it.
            dataset_name = _effective_h5_dataset_name(tile_dataset_name, False)
            # Nothing to wait on: the tiles already exist. Keeping this run's
            # tiling job IDs as a dependency would only expose the submission
            # to slurmctld having forgotten them.
            job_ids = []

        # Only pick a *new* "_subset_N" the first time this run packages.
        # _effective_h5_dataset_name numbers by scanning HPL_DATASETS_ROOT, so
        # calling it again on a retry returns the *next* N — i.e. a different
        # output directory than the one this run already started writing to.
        # make_hpl_hdf5.py's resume checkpoint sidecars are keyed off that
        # output path, so a retry under a freshly-numbered name silently
        # abandoned the previous attempt's progress (potentially hours of
        # decoded tiles) and restarted from zero, and left the run's recorded
        # h5_output_path pointing at an orphaned directory. That's the exact
        # opposite of what retrying a timed-out packaging job should do.
        # Reading the name back off the recorded path keeps it stable across
        # retries — the same approach start_feature_extraction_job already
        # takes, for the same reason. Full (non-subset) runs were unaffected
        # either way, since their name never had a suffix to renumber.
        # scope="tiled" already chose its own name above and must not be
        # overridden by whatever this run last recorded.
        if scope != "tiled":
            recorded_name = (
                Path(row["h5_output_path"]).parent.name if row["h5_output_path"] else None
            )
            if recorded_name and not (
                # One case where the recorded name must NOT be reused: a subset
                # run that previously packaged with scope="tiled" has the
                # *unsuffixed* full-dataset directory on record. Reading it back
                # for a plain scope="run" retry would write this run's 30-slide
                # manifest over the full-coverage .h5. Fall through to the
                # computed subset name, which is where a run-scoped package
                # belongs.
                row["is_subset"] and recorded_name == tile_dataset_name
            ):
                dataset_name = recorded_name
            else:
                dataset_name = _effective_h5_dataset_name(
                    tile_dataset_name, bool(row["is_subset"])
                )
        # Scoped to this one submission so _find_job_id_by_name can recover
        # it below — must match the job_name passed to submit_packaging_job
        # further down.
        job_name = f"hpl_h5_package_{submission_id}"

        if not row["h5_job_id"]:
            # No h5_job_id on record is ambiguous, not proof nothing was
            # attempted: a prior call here could have had sbatch succeed and
            # then the server crash/restart before _update_dataset_run()
            # right below it ran, leaving Postgres unaware of a job that's
            # actually live (or already finished) in Slurm. Check by this
            # submission's own job name before trusting the DB's silence —
            # without this, that crash window turns into a duplicate sbatch
            # submission every time it's hit.
            recovered_job_id = _find_job_id_by_name(job_name)
            if recovered_job_id:
                recovered_output_path = str(hpl_h5_output_path(HPL_DATASETS_ROOT, dataset_name))
                _update_dataset_run(
                    submission_id, h5_job_id=recovered_job_id, h5_output_path=recovered_output_path,
                )
                row["h5_job_id"] = recovered_job_id
                row["h5_output_path"] = recovered_output_path

        if row["h5_job_id"]:
            # Same reasoning as feature extraction's retry guard below: only
            # block a new attempt if the prior one actually succeeded or is
            # still genuinely running — otherwise a single failed packaging
            # attempt (e.g. the empty-CSV bug, or a TIMEOUT kill mid-write)
            # permanently locks the run out of ever packaging again. Note the
            # readiness check is Slurm state *plus* a real HDF5 read (see
            # _validate_h5), not file existence: packaging now stages to a
            # ".partial" sibling and renames on success, so a killed attempt
            # leaves nothing at this path — but a stale .h5 from an earlier
            # successful run would still be sitting there, and existence alone
            # would read that as "this attempt already completed."
            #
            # The comparison is against the output *this* submission would
            # write, not the one the row happens to remember. The row has a
            # single h5_job_id/h5_output_path slot shared by both scopes, so a
            # subset run that finished its own 30-slide .h5 was blocking a
            # scope="tiled" request for a completely different, non-existent
            # file. Only a recorded job pointing at the same target can say
            # anything about this one.
            target_output = hpl_h5_output_path(HPL_DATASETS_ROOT, dataset_name)
            recorded_output = Path(row["h5_output_path"]) if row["h5_output_path"] else None

            if recorded_output == target_output:
                prior_state = _get_slurm_job_state(row["h5_job_id"])
                if _job_output_ready(target_output, prior_state, validator=_validate_h5):
                    raise HTTPException(400, "Packaging has already completed for this run.")
                if prior_state in IN_FLIGHT_SLURM_STATES:
                    raise HTTPException(
                        400,
                        f"Packaging is already running for this run "
                        f"(Slurm state: {prior_state}).",
                    )
                # Otherwise the prior attempt failed/was cancelled/timed out (or
                # its state is unknown) — fall through and submit a fresh one.
            elif _job_output_ready(target_output, "", validator=_validate_h5):
                # A different target, but a complete and readable .h5 is already
                # sitting at it — from an earlier run, or an earlier scope. Slurm
                # state is irrelevant (no job on this row produced it), so "" is
                # passed deliberately: the file itself is the evidence. Refuse
                # rather than silently overwrite something valid.
                raise HTTPException(
                    400,
                    f"A complete .h5 already exists at {target_output}. Delete or move "
                    f"it first if you want to rebuild it.",
                )

        # With afterok, submitting while any tiling task has failed produces a
        # job whose dependency can never be satisfied. --kill-on-invalid-dep
        # makes Slurm kill it rather than queue it forever, but the user would
        # still just see packaging vanish with no explanation. Refuse up front
        # with something actionable instead. Deliberately conservative: only
        # states we positively observed as failures count, so an unreachable
        # sacct (None) or an unparsed state never blocks a legitimate submit.
        tiling_states = _get_slurm_array_state_counts(job_ids)
        if not allow_incomplete:
            failed_states = {
                state: n for state, n in (tiling_states or {}).items()
                if state != "COMPLETED" and state not in IN_FLIGHT_SLURM_STATES
            }
            if failed_states:
                raise HTTPException(
                    400,
                    {
                        "error": "tiling_incomplete",
                        "submission_id": submission_id,
                        "failed_task_states": failed_states,
                        "message": (
                            f"{sum(failed_states.values())} tiling task(s) did not complete "
                            f"successfully ({failed_states}). Packaging now would produce an "
                            f".h5 missing those slides. Resume the run to retry them, or "
                            f"resubmit with allow_incomplete=true to package without them."
                        ),
                    },
                )

        # Drop the dependency once tiling is terminal, because by then it can
        # only hurt. sbatch resolves --dependency against slurmctld, which
        # forgets a job MinJobAge seconds after it ends (default 300), whereas
        # the state check above reads sacct, whose retention is days. Between
        # those two windows sits the common case: tiling finished yesterday,
        # sacct still reports every task COMPLETED so nothing above objects,
        # and then sbatch rejects the whole submission with "Job dependency
        # problem" because slurmctld no longer recognises the IDs. Packaging
        # was unreachable for exactly the runs most ready to be packaged.
        #
        # This does not weaken the afterok guarantee. afterok exists to stop
        # packaging from starting while tiling is still going or after it
        # failed, and both of those are decided above from sacct state — the
        # dependency is a redundant second opinion here, and a stale one.
        #
        # Conservative on purpose: None means sacct was unreachable, so the
        # states are genuinely unknown and the dependency stays as the
        # backstop. An empty dict is different — per
        # _get_slurm_array_state_counts, that means sacct answered AND squeue
        # confirms nothing is live, i.e. the run aged out of accounting
        # retention entirely, which is the strongest evidence available that
        # it is long finished (and guarantees slurmctld has forgotten it too).
        if tiling_states is not None and not any(
            state in IN_FLIGHT_SLURM_STATES for state in tiling_states
        ):
            job_ids = []
        else:
            # Either sacct couldn't be reached (None — state genuinely
            # unknown) or it reports tasks still in flight. Neither is proof
            # the dependency is submittable, because sacct is not the
            # component that resolves it. Ask the controller, which is: any
            # ID it no longer holds makes the dependency unsatisfiable no
            # matter what sacct believes, while a genuinely running job is
            # always known to it, so filtering here can never drop a
            # dependency that was doing real work. If scontrol itself is
            # unusable (None) nothing is known for certain and the IDs stay
            # untouched — a rejected submit is a better failure than
            # packaging that quietly starts before tiling has finished.
            known_job_ids = _slurm_controller_known_jobs(job_ids)
            if known_job_ids is not None:
                job_ids = known_job_ids

        try:
            packaging_result = submit_packaging_job(
                manifest_path=manifest_path,
                tile_dir=Path(row["tile_dir"]),
                dataset_name=dataset_name,
                tile_dataset_name=tile_dataset_name,
                depends_on_job_ids=job_ids,
                allow_incomplete=allow_incomplete,
                resume=resume,
                partition=row["partition"],
                notify_email=row["notify_email"],
                job_name=job_name,
                # Explicit rather than left to submit_packaging_job's own
                # default (backend_dir.parent / "model_input") — that
                # default happens to match HPL_DATASETS_ROOT's own default
                # today, but only by coincidence; without this, overriding
                # HPL_DATASETS_ROOT would silently only affect single-slide
                # uploads (_run_postupload_pipeline passes it explicitly)
                # and not dataset-wide packaging, splitting the two onto
                # different output roots.
                output_root=HPL_DATASETS_ROOT,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to submit packaging job: {e}")

        _update_dataset_run(
            submission_id,
            h5_job_id=packaging_result.get("h5_job_id"),
            h5_output_path=packaging_result.get("h5_output_path"),
        )
        _record_run_job(
            submission_id, "packaging", packaging_result.get("h5_job_id"),
            output_path=packaging_result.get("h5_output_path"),
            params={"scope": scope, "resume": resume, "allow_incomplete": allow_incomplete},
        )
        return {"submission_id": submission_id, **packaging_result}


def _count_lines(path: Path) -> int:
    """Line count without holding the file in memory.

    The completed-tiles checkpoint is one label per line and can reach
    millions of lines / hundreds of MB on a full run, so this reads in
    blocks rather than splitlines()-ing the lot.
    """
    total = 0
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                return total
            total += block.count(b"\n")


# How long a .partial can go untouched before it stops counting as evidence of
# a live writer. Generous on purpose: packaging alternates between long decode
# batches (a whole batch of tiles is decoded by the worker pool before anything
# is written) and bursts of writes, so short gaps are normal and a tight window
# would flap between "running" and "stalled" on an entirely healthy job.
_PACKAGING_ACTIVE_WINDOW_SECONDS = 15 * 60


def _packaging_write_activity(final_h5_path: Path) -> dict:
    """Whether something is currently writing this run's .h5, judged from disk.

    Exists because Slurm state is not always available (see
    _get_slurm_job_state) and is never available *promptly* — a job submitted
    seconds ago may have no accounting row at all. The .partial's mtime has
    neither problem: packaging touches it continuously while it runs, and the
    file only exists between the start of an attempt and the os.replace() that
    completes it.

    Costs one stat() on one file, which is what makes it safe to include in the
    10s status poll. Returns partial_* as None/False rather than raising when
    the file isn't there, since "no .partial" is a normal state (not started, or
    already finished and renamed).
    """
    partial_path = final_h5_path.with_name(final_h5_path.name + ".partial")
    try:
        stat = partial_path.stat()
    except OSError:
        return {
            "h5_partial_exists": False,
            "h5_partial_bytes": 0,
            "h5_partial_seconds_since_write": None,
            "h5_packaging_active": False,
        }
    seconds_since_write = max(0.0, time.time() - stat.st_mtime)
    return {
        "h5_partial_exists": True,
        "h5_partial_bytes": stat.st_size,
        "h5_partial_seconds_since_write": round(seconds_since_write, 1),
        "h5_packaging_active": seconds_since_write < _PACKAGING_ACTIVE_WINDOW_SECONDS,
    }


@app.get("/dataset-jobs/{submission_id}/packaging-progress")
def packaging_progress(
    submission_id: str,
    exact: bool = Query(
        True,
        description="Count the completed-tiles checkpoint exactly (hundreds of MB on a "
                    "large run). Pass false for live polling, which estimates tiles "
                    "written from the .partial's size instead — one stat() rather than "
                    "a full file read.",
    ),
):
    """How far an interrupted packaging attempt actually got.

    Deliberately a separate endpoint rather than more fields on
    /status: answering it means counting the lines of a checkpoint file
    that can be hundreds of MB, and /status is polled every 10s by every
    open browser tab. This is only called when someone actually opens the
    packaging step in the UI.

    "resumable" is the question the UI is really asking — whether clicking
    package again would continue the previous attempt or silently start
    from zero. That's true exactly when a .partial and its checkpoint are
    both still on disk; package_slides_to_h5 then validates the recorded
    run identity itself and falls back to a fresh run if anything moved.
    """
    row = _get_dataset_run_row(submission_id)
    output_path = row["h5_output_path"]
    if not row["h5_job_id"] or not output_path:
        return {"state": "not_started", "resumable": False}

    final_path = Path(output_path)
    partial_path = final_path.with_name(final_path.name + ".partial")
    ckpt = _checkpoint_paths(final_path)
    slurm_state = _get_slurm_job_state(row["h5_job_id"])

    activity = _packaging_write_activity(final_path)

    info = {
        "output_path": str(final_path),
        "slurm_state": slurm_state,
        "partial_exists": activity["h5_partial_exists"],
        "partial_bytes": activity["h5_partial_bytes"],
        "seconds_since_write": activity["h5_partial_seconds_since_write"],
        "writing_now": activity["h5_packaging_active"],
        # The finished article, so the UI can distinguish "no output yet" from
        # "output exists but hasn't validated".
        "final_exists": final_path.is_file(),
        "final_bytes": final_path.stat().st_size if final_path.is_file() else 0,
        "resumable": False,
        "tiles_done": None,
        "tiles_done_is_estimate": False,
        "tiles_total": None,
        "percent": None,
        "bytes_per_tile": None,
        "skipped_tiles": None,
    }

    if _job_output_ready(final_path, slurm_state, validator=_validate_h5):
        info["state"] = "complete"
        return info
    # Disk activity outranks a missing Slurm state. A job whose .partial was
    # touched moments ago is running, whatever sacct does or doesn't know — this
    # is what stops a freshly-started job from being reported as "interrupted".
    if slurm_state in IN_FLIGHT_SLURM_STATES or info["writing_now"]:
        info["state"] = "running"
    else:
        info["state"] = "interrupted"

    # tiles_total comes from the run config the attempt itself wrote, so it
    # reflects what that attempt was actually packaging rather than a count
    # recomputed now from possibly-changed tile directories.
    tile_size = None
    img_compression = None
    if ckpt["config"].is_file():
        try:
            config = json.loads(ckpt["config"].read_text())
            info["tiles_total"] = config.get("total_tiles")
            tile_size = config.get("tile_size")
            img_compression = config.get("img_compression")
        except (json.JSONDecodeError, OSError):
            pass

    # Every tile occupies exactly tile_size**2 * 3 bytes in the .h5 (uint8, one
    # tile per chunk) ONLY when the img dataset is uncompressed — that fixed
    # width is what makes the .partial's size a usable proxy for tiles
    # written, costing one stat() instead of a full checkpoint read. Once
    # img_compression is set (see make_hpl_hdf5.py's _IMG_COMPRESSION), each
    # tile's compressed chunk size varies with how much actual detail is in
    # that tile, so this proxy has no fixed divisor to use and is left unset
    # rather than reported as a number that quietly drifts from reality.
    if tile_size and not img_compression:
        info["bytes_per_tile"] = int(tile_size) ** 2 * 3

    if exact and ckpt["completed"].is_file():
        try:
            info["tiles_done"] = _count_lines(ckpt["completed"])
        except OSError:
            pass
    elif info["bytes_per_tile"] and info["partial_bytes"]:
        # Slightly low: HDF5's own metadata (b-tree nodes, the superblock)
        # shares the file, so dividing overstates nothing. Flagged as an
        # estimate so the UI never presents it as a tile-accurate figure.
        info["tiles_done"] = info["partial_bytes"] // info["bytes_per_tile"]
        info["tiles_done_is_estimate"] = True

    # Tiles the attempt gave up on (unreadable/corrupt JPEGs). Small file, and
    # worth surfacing: they count as done for resume purposes but never make it
    # into the .h5, so a run can legitimately finish short of tiles_total.
    if ckpt["skipped"].is_file():
        try:
            info["skipped_tiles"] = _count_lines(ckpt["skipped"])
        except OSError:
            pass

    if info["tiles_done"] is not None and info["tiles_total"]:
        info["percent"] = round(
            100.0 * min(info["tiles_done"], info["tiles_total"]) / info["tiles_total"], 1
        )

    # Resumability is about the checkpoint, so it needs a real count — an
    # estimate from file size says nothing about whether the checkpoint exists.
    # When polling cheaply, fall back to the checkpoint merely being non-empty.
    if info["tiles_done_is_estimate"]:
        try:
            checkpoint_has_content = (
                ckpt["completed"].is_file() and ckpt["completed"].stat().st_size > 0
            )
        except OSError:
            checkpoint_has_content = False
    else:
        checkpoint_has_content = bool(info["tiles_done"])
    info["resumable"] = bool(info["partial_exists"] and checkpoint_has_content)
    return info


class PackagingTestRequest(BaseModel):
    sample_size: Optional[int] = None
    slide_names: Optional[list[str]] = None
    random_seed: Optional[int] = None
    # Which pool the sample is drawn from. "run" keeps the original behaviour
    # (this run's own manifest); "tiled" draws from every slide in the raw
    # directory that has tiles on disk, whichever run produced them — mirroring
    # the same option on /package. Without this a test run could never exceed
    # its run's manifest, so a 30-slide subset run capped every test sample at
    # 30 however large a number the UI offered to accept.
    scope: str = "run"


@app.post("/dataset-jobs/{submission_id}/package-test")
def start_test_packaging_job(submission_id: str, req: PackagingTestRequest):
    """Package a chosen subset of slides into a separately-named test .h5 —
    sanity-check packaging (and a downstream feature-extraction checkpoint)
    against a sample before committing to a multi-hour run over the whole
    dataset. Deliberately NOT tracked on the run's own
    h5_job_id/h5_output_path — a test run succeeding or failing has no
    bearing on whether the real packaging run is allowed to proceed, and
    vice versa (the retry-guard above only ever looks at h5_job_id, which
    this never touches).

    scope="tiled" widens the pool past this run's manifest to everything
    tiled on disk. That is the difference between "test the pipeline on 3
    slides" and "test it on 3500 of the 14,000 I actually have", and only
    the former was previously possible from a subset run.
    """
    if not req.sample_size and not req.slide_names:
        raise HTTPException(400, "Provide sample_size or slide_names for a test run.")
    if req.scope not in ("run", "tiled"):
        raise HTTPException(400, f"scope must be 'run' or 'tiled', got '{req.scope}'.")

    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        if not row["job_id"]:
            # Not status == "submitted" specifically — a cancelled run's tiling
            # batches may well have already finished, and cancelling only flips
            # the status column, leaving the manifest/tiles on disk untouched.
            # job_id existing at all is what actually means tiling was submitted.
            raise HTTPException(400, "Tiling hasn't been submitted yet for this run.")

        manifest_path = Path(row["manifest_path"]) if row["manifest_path"] else None
        if not manifest_path:
            raise HTTPException(400, "This run never got far enough to have a manifest.")
        # Same reasoning as /package: scope="tiled" writes its own fresh
        # manifest from live disk coverage below and only needs this path's
        # parent directory, so the original file's existence is irrelevant to
        # it — only scope="run" actually opens it.
        if req.scope != "tiled" and not manifest_path.is_file():
            raise HTTPException(400, f"Manifest no longer exists on disk: {manifest_path}")

        job_ids = [j for j in row["job_id"].split(",") if j]
        # Same slurmctld-forgot-the-job-IDs problem the real /package endpoint
        # handles — see the long note there. afterany is no more resolvable
        # than afterok once the IDs have aged out of the controller, so a test
        # package against a finished run failed at sbatch for a dependency
        # that had nothing left to wait for.
        test_tiling_states = _get_slurm_array_state_counts(job_ids)
        if test_tiling_states is not None and not any(
            state in IN_FLIGHT_SLURM_STATES for state in test_tiling_states
        ):
            job_ids = []
        else:
            known_job_ids = _slurm_controller_known_jobs(job_ids)
            if known_job_ids is not None:
                job_ids = known_job_ids

        tile_dataset_name = _row_dataset_name(row)

        if req.scope == "tiled":
            # Draw from what is on disk instead of this run's fixed manifest.
            # Tiles already exist, so there is nothing to depend on — keeping
            # this run's tiling job IDs would only expose the submission to
            # slurmctld having forgotten them, same as /package's scope="tiled".
            coverage = _tiled_coverage(
                Path(row["raw_dir"]), Path(row["tile_dir"]), tile_dataset_name
            )
            if not coverage["slides_tiled"]:
                raise HTTPException(
                    400, f"No slide in {row['raw_dir']} has tiles on disk yet."
                )
            manifest_path = manifest_path.parent / (
                f"wsi_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_testpool_{tile_dataset_name}.txt"
            )
            write_manifest([Path(p) for p in coverage["tiled_paths"]], manifest_path)
            job_ids = []
            pool_size = coverage["slides_tiled"]
        else:
            pool_size = sum(
                1 for line in manifest_path.read_text().splitlines() if line.strip()
            )

        # Checked here, before the signature and the Slurm lookup, so an
        # over-large request is rejected against the pool the user actually
        # asked for rather than surfacing later from select_slides with no
        # mention of which pool it measured.
        if req.sample_size and req.sample_size > pool_size:
            pool_label = (
                "tiled-on-disk" if req.scope == "tiled" else "run's manifest"
            )
            widen_hint = (
                ""
                if req.scope == "tiled"
                else (
                    " Retry with scope='tiled' to draw from every slide tiled "
                    "on disk instead of just this run's."
                )
            )
            raise HTTPException(
                400,
                {
                    "error": "sample_larger_than_pool",
                    "submission_id": submission_id,
                    "scope": req.scope,
                    "pool_size": pool_size,
                    "requested": req.sample_size,
                    "message": (
                        f"Asked for {req.sample_size} slides but the "
                        f"{pool_label} pool holds {pool_size}.{widen_hint}"
                    ),
                },
            )

        # A random draw with no seed is a different set of slides every time, so
        # leaving the seed out of the signature made two genuinely different
        # attempts share one output path — and, because the Slurm lookup below
        # treats a matching name as "already done", made re-rolling a random
        # sample impossible: same N came back as "already completed" while never
        # having packaged those slides. Resolving a seed here keeps the
        # signature honest, and makes the draw reproducible, which it never was.
        # The cost is that an unseeded double-click submits two jobs instead of
        # being deduped; pass an explicit random_seed to get the old idempotency.
        resolved_seed = req.random_seed
        if resolved_seed is None and req.sample_size:
            resolved_seed = random.randrange(1_000_000_000)

        base_dataset_name = _effective_h5_dataset_name(
            tile_dataset_name,
            # scope="tiled" is drawn from full coverage rather than this run's
            # subset, so it doesn't inherit the run's "_subset_N" naming.
            False if req.scope == "tiled" else bool(row["is_subset"]),
        )
        # Signature-suffixed rather than a fixed "_test_sample" name: this
        # endpoint has no DB row to guard against, so the *filename itself*
        # is what has to keep two different test attempts (different
        # scope/sample_size/slide_names/seed) from clobbering each other's
        # output, and keeps a re-run of an explicitly-seeded attempt idempotent
        # (same signature -> same path -> caught by the Slurm lookup below)
        # rather than piling up duplicate jobs.
        signature = _attempt_signature(
            "package-test", submission_id, req.scope, req.sample_size,
            sorted(req.slide_names or []), resolved_seed,
        )
        test_dataset_name = f"{base_dataset_name}_test_sample_{signature}"
        job_name = f"hpl_h5_package_test_{signature}"

        existing_job_id = _find_job_id_by_name(job_name)
        if existing_job_id:
            existing_output = hpl_h5_output_path(HPL_DATASETS_ROOT, test_dataset_name)
            existing_state = _get_slurm_job_state(existing_job_id)
            if _job_output_ready(existing_output, existing_state, validator=_validate_h5):
                raise HTTPException(
                    400, "A test packaging run with these exact parameters has already completed."
                )
            if existing_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    "A test packaging run with these exact parameters is already in progress "
                    f"(Slurm state: {existing_state}).",
                )
            # Otherwise that attempt failed/was cancelled/its state is
            # unknown — fall through and let this submit a fresh one.

        try:
            result = submit_packaging_job(
                manifest_path=manifest_path,
                tile_dir=Path(row["tile_dir"]),
                dataset_name=test_dataset_name,
                tile_dataset_name=tile_dataset_name,
                depends_on_job_ids=job_ids,
                partition=row["partition"],
                notify_email=row["notify_email"],
                sample_size=req.sample_size,
                slide_names=req.slide_names,
                random_seed=resolved_seed,
                job_name=job_name,
                # A test run packages a deliberately-chosen handful of slides,
                # so a partial dataset is the entire point — afterok would
                # make it hostage to every unrelated slide in the run having
                # tiled cleanly, which defeats the purpose of a quick check.
                allow_incomplete=True,
                # Same as the real /package endpoint — keep test packaging
                # on the same output root as everything else instead of
                # silently falling back to submit_packaging_job's own default.
                output_root=HPL_DATASETS_ROOT,
            )
        except ValueError as e:
            # select_slides() rejecting the request — asking for more slides
            # than this run's manifest holds, or naming slides that aren't in
            # it. That's the caller's input, not a server fault, and it used to
            # come back as a 500 the UI could only show as a generic failure.
            raise HTTPException(
                400,
                {
                    "error": "invalid_slide_selection",
                    "submission_id": submission_id,
                    "manifest_slides": row["total_slides"],
                    "message": str(e),
                },
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test packaging job: {e}")

        # Recorded on the run so the job survives a page reload. Written to
        # test_h5_* rather than h5_job_id/h5_output_path on purpose: those gate
        # the packaging retry guard and stage 3's h5_ready, and a subset sample
        # must not satisfy either. This only makes the test job *visible* — it
        # still has no bearing on whether real packaging may proceed.
        #
        # Failure here is logged, not raised: the Slurm job is already submitted
        # by this point, and turning a bookkeeping error into a 500 would leave
        # the caller believing nothing was queued while the job ran anyway.
        try:
            _update_dataset_run(
                submission_id,
                test_h5_job_id=result.get("h5_job_id"),
                test_h5_output_path=result.get("h5_output_path"),
                test_h5_submitted_at=datetime.now(timezone.utc),
                test_h5_params=json.dumps({
                    "scope": req.scope,
                    "sample_size": req.sample_size,
                    "slide_names": req.slide_names,
                    "random_seed": resolved_seed,
                    "pool_size": pool_size,
                }),
            )
        except Exception as e:
            print(f"[warn] test packaging job {result.get('h5_job_id')} submitted "
                  f"but not recorded on {submission_id}: {e}")

        # The history row is what makes a *second* test packaging visible: the
        # test_h5_* columns above hold only the latest attempt, so without this
        # an earlier sample and its .h5 path disappear the moment another runs.
        _record_run_job(
            submission_id, "packaging_test", result.get("h5_job_id"),
            output_path=result.get("h5_output_path"),
            params={
                "scope": req.scope,
                "sample_size": req.sample_size,
                "slide_names": req.slide_names,
                "random_seed": resolved_seed,
                "pool_size": pool_size,
            },
        )

        # scope/pool_size/random_seed are echoed back so the UI can state what
        # was actually drawn and from where. random_seed especially: it's the
        # only record of which slides a random sample picked, and without it
        # a test .h5 worth investigating couldn't be reproduced.
        return {
            "submission_id": submission_id,
            "scope": req.scope,
            "pool_size": pool_size,
            "random_seed": resolved_seed,
            **result,
        }


@app.get("/dataset-jobs/{submission_id}/package-test-status")
def test_packaging_status(submission_id: str, job_id: str, output_path: str):
    """Status for one ad-hoc test packaging job. submission_id isn't
    actually looked up here — test runs aren't persisted to the DB at all
    (see start_test_packaging_job) — it's kept in the path purely for
    routing consistency with the other per-run endpoints. job_id and
    output_path are whatever start_test_packaging_job returned; the caller
    (the UI) is responsible for remembering them between polls.
    """
    state = _get_slurm_job_state(job_id)
    ready = _job_output_ready(Path(output_path), state, validator=_validate_h5)
    return {"job_id": job_id, "slurm_state": state, "ready": ready, "output_path": output_path}


class FeatureExtractionRequest(BaseModel):
    checkpoint: str
    model: str = "BarlowTwins_3"
    marker: str = "he"


@app.post("/dataset-jobs/{submission_id}/extract-features")
def start_feature_extraction_job(submission_id: str, req: FeatureExtractionRequest):
    """Manually start Stage 2 (running the packaged .h5 through Kai's frozen
    self-supervised encoder), once the user confirms the .h5 is ready and
    supplies a checkpoint path. Never auto-triggered — the checkpoint is a
    per-run input only the user knows, so this was always going to need a
    manual step, not just the "click to proceed" gating the other stages
    also now use.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        _refuse_if_pipeline_run(row, "feature extraction")
        if not row["h5_job_id"] or not row["h5_output_path"]:
            raise HTTPException(400, "Packaging hasn't been started for this run yet.")

        h5_path = Path(row["h5_output_path"])
        # Read the dataset_name back out of the .h5 path itself (its parent dir
        # name) rather than recomputing via _effective_h5_dataset_name a second
        # time — that function numbers "_subset_N" by scanning the filesystem,
        # so calling it again here (after packaging already created that dir)
        # could silently pick a different, wrong number than packaging actually
        # used.
        dataset_name = h5_path.parent.name
        # Scoped to this one submission so _find_job_id_by_name can recover
        # it below — must match the job_name passed to
        # submit_feature_extraction_job further down.
        job_name = f"hpl_feature_extraction_{submission_id}"

        if not row["extraction_job_id"]:
            # See the matching comment in start_packaging_job — "no
            # extraction_job_id on record" doesn't prove nothing was
            # attempted, it's also what a crash between sbatch succeeding
            # and the _update_dataset_run() call right after it looks like.
            recovered_job_id = _find_job_id_by_name(job_name)
            if recovered_job_id:
                recovered_output_path = str(
                    expected_extraction_output_path(HPL_REPO_DIR, req.model, dataset_name, h5_path)
                )
                _update_dataset_run(
                    submission_id,
                    extraction_job_id=recovered_job_id,
                    extraction_output_path=recovered_output_path,
                )
                row["extraction_job_id"] = recovered_job_id
                row["extraction_output_path"] = recovered_output_path

        if row["extraction_job_id"]:
            # A previous attempt exists — only block starting another one if
            # that attempt actually succeeded or is still in flight. Blocking
            # unconditionally here would mean a single failed attempt (wrong
            # conda setup, bad checkpoint path, whatever) permanently locks out
            # ever retrying feature extraction for this run. File existence
            # alone isn't enough to mean "succeeded" either — same reasoning as
            # packaging above, an output file created early and then left
            # behind by a killed/timed-out attempt would otherwise look done.
            prior_output = Path(row["extraction_output_path"]) if row["extraction_output_path"] else None
            prior_state = _get_slurm_job_state(row["extraction_job_id"])
            # Validated, not just existence-checked: the encoder creates its
            # output file before encoding anything, so a killed attempt leaves
            # one behind that would otherwise read as a completed extraction
            # and permanently block this run at "already completed".
            # Bound to the input's tile count for the same reason the status
            # payload is: an output covering a fraction of the slides passes
            # every internal check, and refusing the retry with "already
            # completed" is precisely how a partial run becomes permanent.
            prior_expected = _extraction_expected_rows(row)
            if _job_output_ready(
                prior_output, prior_state,
                validator=lambda p: _validate_extraction_output(p, expected_rows=prior_expected),
            ):
                raise HTTPException(400, "Feature extraction has already completed for this run.")
            if prior_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Feature extraction is already running for this run (Slurm state: {prior_state}).",
                )
            # Otherwise the prior attempt failed/was cancelled/timed out (or its
            # state is unknown) — fall through and let this submit a fresh one.

        # Packaging must have genuinely COMPLETED *and* produced a readable
        # .h5 before extraction is allowed to start. File existence was the
        # entire check here, which was never sufficient — and is now actively
        # misleading in the opposite direction too, since packaging writes to
        # a ".partial" sibling and renames on success: a run still in flight
        # leaves nothing at this path, while a *stale* .h5 from an earlier
        # successful run does, and would have been accepted as this run's
        # output. Feature extraction is a long GPU job; discovering the input
        # was truncated hours in is the failure this prevents.
        h5_state = _get_slurm_job_state(row["h5_job_id"])
        if not _job_output_ready(h5_path, h5_state, validator=_validate_h5):
            if h5_state is None:
                raise HTTPException(
                    503,
                    "Couldn't reach Slurm to confirm packaging finished — try again shortly.",
                )
            if h5_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Packaging is still running (Slurm state: {h5_state}) — wait for it to "
                    "finish before extracting features.",
                )
            if not h5_path.is_file():
                raise HTTPException(
                    400,
                    f"Packaging has not produced its .h5 (Slurm state: "
                    f"{h5_state or 'no Slurm record'}): {h5_path}",
                )
            valid, reason = _validate_h5(h5_path)
            if not valid:
                raise HTTPException(
                    400,
                    f"The packaged .h5 is not usable ({reason}) — re-run packaging for this "
                    f"run before extracting features: {h5_path}",
                )
            raise HTTPException(
                400,
                f"Packaging did not complete successfully (Slurm state: "
                f"{h5_state or 'no Slurm record'}) — re-run packaging first.",
            )

        checkpoint = req.checkpoint.strip()
        if not checkpoint:
            raise HTTPException(400, "Checkpoint path is required.")

        try:
            result = submit_feature_extraction_job(
                real_hdf5_path=h5_path,
                checkpoint=checkpoint,
                dataset_name=dataset_name,
                model=req.model,
                marker=req.marker,
                notify_email=row["notify_email"],
                job_name=job_name,
            )
        except (NotADirectoryError, FileExistsError, FileNotFoundError) as e:
            # Misconfiguration and already-done are user-fixable states, not
            # server faults — 400 with the message the submitter composed.
            # FileNotFoundError covers a missing Singularity SIF / binary.
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit feature extraction job: {e}")

        _update_dataset_run(
            submission_id,
            extraction_job_id=result.get("extraction_job_id"),
            extraction_output_path=result.get("expected_output_path"),
            extraction_checkpoint=checkpoint,
        )
        _record_run_job(
            submission_id, "extraction", result.get("extraction_job_id"),
            output_path=result.get("expected_output_path"),
            params={"checkpoint": checkpoint, "model": req.model, "marker": req.marker},
        )
        return {"submission_id": submission_id, **result}


class FeatureExtractionTestRequest(BaseModel):
    h5_path: str
    checkpoint: str
    model: str = "BarlowTwins_3"
    marker: str = "he"


@app.post("/dataset-jobs/{submission_id}/extract-features-test")
def start_test_feature_extraction_job(submission_id: str, req: FeatureExtractionTestRequest):
    """Run feature extraction against an arbitrary .h5 — typically a
    test-sample .h5 from /package-test above — rather than this run's own
    tracked h5_output_path. Lets a checkpoint get validated against a
    small sample before running it over the full dataset's .h5, which can
    take hours. Deliberately NOT tracked on the run's own
    extraction_job_id, for the same reason as /package-test.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)  # 404s if the run doesn't exist

        h5_path = Path(req.h5_path)
        if not h5_path.is_file():
            raise HTTPException(400, f".h5 file not found: {h5_path}")
        # No Slurm job to check against here (the caller supplies an arbitrary
        # path), so validating the file itself is the only gate available —
        # and the one that matters, since the usual reason to point this at a
        # hand-picked .h5 is that a packaging run was interrupted.
        valid, reason = _validate_h5(h5_path)
        if not valid:
            raise HTTPException(400, f"Not a usable packaged .h5 ({reason}): {h5_path}")

        checkpoint = req.checkpoint.strip()
        if not checkpoint:
            raise HTTPException(400, "Checkpoint path is required.")

        dataset_name = h5_path.parent.name
        signature = _attempt_signature(
            "extract-test", submission_id, str(h5_path), checkpoint, req.model, req.marker,
        )
        job_name = f"hpl_feature_extraction_test_{signature}"

        existing_job_id = _find_job_id_by_name(job_name)
        if existing_job_id:
            existing_output = expected_extraction_output_path(HPL_REPO_DIR, req.model, dataset_name, h5_path)
            existing_state = _get_slurm_job_state(existing_job_id)
            # The test path knows its input directly rather than through the
            # run row, so bind the count from the .h5 the caller handed us.
            existing_expected = _packaged_h5_rows(h5_path)
            if _job_output_ready(
                existing_output, existing_state,
                validator=lambda p: _validate_extraction_output(p, expected_rows=existing_expected),
            ):
                raise HTTPException(
                    400, "A test feature-extraction run with these exact parameters has already completed."
                )
            if existing_state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    "A test feature-extraction run with these exact parameters is already in "
                    f"progress (Slurm state: {existing_state}).",
                )
            # Otherwise that attempt failed/was cancelled/its state is
            # unknown — fall through and let this submit a fresh one.

        try:
            result = submit_feature_extraction_job(
                real_hdf5_path=h5_path,
                checkpoint=checkpoint,
                dataset_name=dataset_name,
                model=req.model,
                marker=req.marker,
                notify_email=row["notify_email"],
                job_name=job_name,
            )
        except (NotADirectoryError, FileExistsError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit test feature extraction job: {e}")

        # Recorded even though this endpoint still writes nothing to
        # slurm_dataset_runs. That invariant is about *gating* — a test attempt
        # must not satisfy extraction_ready — and history does not gate
        # anything, so the two are not in tension. Before this, a test
        # extraction existed only in the caller's session state.
        _record_run_job(
            submission_id, "extraction_test", result.get("extraction_job_id"),
            output_path=result.get("expected_output_path"),
            params={
                "h5_path": str(h5_path),
                "checkpoint": checkpoint,
                "model": req.model,
                "marker": req.marker,
            },
        )
        return {"submission_id": submission_id, **result}


@app.get("/dataset-jobs/{submission_id}/extract-features-test-status")
def test_feature_extraction_status(submission_id: str, job_id: str, output_path: str):
    """Status for one ad-hoc test extraction job — same pattern as
    /package-test-status, nothing persisted server-side."""
    state = _get_slurm_job_state(job_id)
    out_path = Path(output_path)
    ready = _job_output_ready(out_path, state, validator=_validate_extraction_output)
    payload = {"job_id": job_id, "slurm_state": state, "ready": ready, "output_path": output_path}
    if not ready and out_path.is_file():
        # The file existing while the job is finished means an attempt died
        # partway. Surfacing the reason here is what stops the UI showing a
        # bare "not ready" for a job Slurm already called done.
        payload["extraction_invalid_reason"] = _validate_extraction_output(out_path)[1] or None
    return payload


class ClusterAssignmentRequest(BaseModel):
    # Optional so the UI can just say "go": the reference is a deployment-level
    # setting, not a per-run choice, and defaulting to the configured one keeps
    # the common case a single click. Overridable because comparing two
    # references is a real thing to want to do.
    reference: str | None = None
    k: int | None = None
    overwrite: bool = False

    # Which vote to use, by name. Defaults to the tuned one here rather than in
    # submit_cluster_assignment_job, so that queueing the measured
    # configuration is explicit in the request while no programmatic caller of
    # the submitter changes behaviour just by being upgraded.
    vote_preset: str = DEFAULT_VOTE_PRESET
    # Overrides on the preset. None means "leave the preset alone" — an unset
    # field in a JSON body must not erase the preset it was sent with.
    distance_weighted: bool | None = None
    distance_power: float | None = None
    class_weighted: bool | None = None
    local_scaling: int | None = None
    adaptive_margin: float | None = None
    adaptive_k: int | None = None

    def vote_kwargs(self) -> dict:
        return {
            "vote_preset": self.vote_preset,
            "distance_weighted": self.distance_weighted,
            "distance_power": self.distance_power,
            "class_weighted": self.class_weighted,
            "local_scaling": self.local_scaling,
            "adaptive_margin": self.adaptive_margin,
            "adaptive_k": self.adaptive_k,
        }


class ClusterAssignmentTestRequest(ClusterAssignmentRequest):
    # An arbitrary projections .h5, typically the one a test extraction wrote.
    projections_h5: str


def _assignment_output_path(projections_h5: Path, dataset_name: str) -> Path:
    """Where a run's assignment CSV goes: beside its projections file.

    Not in backend/ or a shared results dir — the CSV is only meaningful
    together with the embeddings it was computed from, and keeping the two
    adjacent is what makes a stale pair obvious instead of plausible.
    """
    return projections_h5.parent / f"{dataset_name}_hpc_assignments.csv"


# --- Stage 7 run state -------------------------------------------------------
# One place decides what ANORAK's head job is doing, and both /status and the
# submit endpoint ask it. They used to decide separately, and each deferred to
# the other: the UI sent overwrite=True on every click because "the in-flight
# check is the server's", and the server skipped that check whenever overwrite
# was set. So a second head job could be queued over a live one, rewriting
# slide_list.csv under it, and the row then tracked the new job while the
# original ran on unobserved.


def _anorak_job_ids(row: dict) -> list[str]:
    """Every head job the latest submission queued: the head first, then any
    --chain standbys. Stored comma-joined in anorak_job_id, the same shape the
    pipeline run records its chain in, because a standby PENDING on its
    predecessor is as much "this run is in flight" as the head itself is."""
    return [j.strip() for j in str(row.get("anorak_job_id") or "").split(",") if j.strip()]


def _anorak_run_state(row: dict) -> str | None:
    """The run's Slurm state across its head job and any chain standbys.

    None means at least one of them could not be asked about and none is known
    to be in flight — genuinely unknown, and callers must treat it that way
    rather than as "stopped". Any job in flight makes the run in flight. A
    chain succeeds when any member COMPLETED (the rest are cleared by
    --kill-on-invalid-dep, so their CANCELLED says nothing about the run).
    Otherwise the run ended the way its last member that actually ran did.
    """
    ids = _anorak_job_ids(row)
    if not ids:
        return None
    states = [_get_slurm_job_state(job_id) for job_id in ids]
    for state in states:
        if state in IN_FLIGHT_SLURM_STATES:
            return state
    if any(state is None for state in states):
        return None
    if "COMPLETED" in states:
        return "COMPLETED"
    ran = [s for s in states if s and not s.startswith("CANCELLED")]
    return ran[-1] if ran else states[0]


def _anorak_submit_blocker(row: dict, state: str | None) -> tuple[int, str] | None:
    """(HTTP status, reason) why a new ANORAK submission must be refused right
    now, or None. overwrite does not enter into it: overwrite replaces a
    *finished* run's outputs, and nothing replaces a live head job's."""
    ids = _anorak_job_ids(row)
    if not ids:
        return None
    if state is None:
        # Refused rather than guessed through. A live head job read as stopped
        # is exactly the case this guard exists for, and "Slurm is unreachable"
        # is when that misreading happens.
        return 503, (
            f"Couldn't reach Slurm to confirm ANORAK job {', '.join(ids)} has "
            f"stopped, so a new submission is refused: if it is still running, a "
            f"second head job would rewrite its slide list and share its work "
            f"directory. Try again once squeue/sacct answer."
        )
    if state in IN_FLIGHT_SLURM_STATES:
        return 400, (
            f"ANORAK is already running for this run (job {', '.join(ids)}, "
            f"state {state}). Two head processes on one output directory would "
            f"resume into each other's cache. scancel it first if you mean to "
            f"replace it."
        )
    return None


def _as_utc(value) -> datetime | None:
    """A recorded TIMESTAMPTZ as an aware datetime, whether the driver handed
    back a datetime or an ISO string."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def _anorak_output_ready(row: dict, grades: Path | None,
                         state: str | None) -> tuple[bool, str | None]:
    """(ready, reason-if-not) for this run's grading table.

    Not _job_output_ready, whose Slurm-unreachable shortcut is sound only for
    stages that write a ".partial" and rename it into place: there, the real
    file existing with no .partial beside it proves a run finished. ANORAK
    writes no .partial, so the same shortcut read a grading table left by an
    *earlier* attempt in this out_dir as the current one's — "complete",
    whenever sacct was unreachable, while the new run was still going.

    None is therefore never ready. "" (aged out of sacct: long finished)
    accepts a table only if it is newer than this submission, since age is all
    that tells an old attempt's table from this one's. COMPLETED goes straight
    to the validator — a resume whose grading task was cached legitimately
    leaves the table's mtime where it was — which is the check that compares
    the table with the slide list this run was given.
    """
    if not grades or not grades.is_file():
        return False, None
    if state is None or state in IN_FLIGHT_SLURM_STATES:
        return False, None
    if state not in ("COMPLETED", ""):
        return False, None
    if state == "":
        submitted = _as_utc(row.get("anorak_submitted_at"))
        written = datetime.fromtimestamp(grades.stat().st_mtime, tz=timezone.utc)
        if submitted is None or written < submitted:
            return False, (
                f"{grades.name} was written {written:%Y-%m-%d %H:%M} UTC, before "
                f"this attempt was submitted"
                + (f" ({submitted:%Y-%m-%d %H:%M} UTC)" if submitted else
                   " (no submission time is recorded)")
                + " — it is an earlier attempt's table, and Slurm no longer has "
                  "a record of how this one ended."
            )
    ok, reason = _validate_anorak_output(grades)
    return ok, (None if ok else reason)


def _anorak_status_fields(row: dict) -> dict:
    """Stage 7's block of /status. Also carries the server's own verdict on
    whether a submission would be accepted (anorak_submit_blocked), so neither
    UI keeps a copy of the in-flight rule that can drift from this one — the
    Streamlit set once lacked CONFIGURING, and showed a Retry form over a head
    job Slurm was still starting."""
    # Written by a failed submission even when no job id was ever recorded,
    # so it is reported outside the job block below.
    fields = {"anorak_error": row.get("anorak_error")}
    if not _anorak_job_ids(row):
        return fields
    anorak_out = Path(row["anorak_out_dir"]) if row.get("anorak_out_dir") else None
    anorak_grades = _anorak_grades_csv_path(anorak_out) if anorak_out else None
    anorak_state = _anorak_run_state(row)
    ready, not_ready_reason = _anorak_output_ready(row, anorak_grades, anorak_state)
    blocker = _anorak_submit_blocker(row, anorak_state)
    fields.update(
        anorak_ready=ready,
        anorak_slurm_state=anorak_state,
        anorak_in_flight=anorak_state in IN_FLIGHT_SLURM_STATES,
        anorak_state_unknown=anorak_state is None,
        anorak_submit_blocked=blocker[1] if blocker else None,
        anorak_job_id=row.get("anorak_job_id"),
        anorak_out_dir=row.get("anorak_out_dir"),
        anorak_grades_csv=str(anorak_grades) if anorak_grades else None,
        anorak_slide_list=row.get("anorak_slide_list"),
        anorak_scope=row.get("anorak_scope"),
        anorak_sample_size=row.get("anorak_sample_size"),
        anorak_seed=row.get("anorak_seed"),
        anorak_slides=row.get("anorak_slides"),
        anorak_stop_reason=_anorak_stop_reason(anorak_out),
        anorak_tumour_verified=_anorak_tumour_verified(anorak_out),
    )
    # Shown only once the job is known to have stopped and the output is not
    # usable — while it is running (or its state is unknown) there is no
    # finished table to complain about.
    if (not ready and anorak_grades and anorak_state is not None
            and anorak_state not in IN_FLIGHT_SLURM_STATES):
        fields["anorak_invalid_reason"] = (
            not_ready_reason or _validate_anorak_output(anorak_grades)[1] or None
        )
    return fields


def _anorak_tumour_verified(out_dir: Path | None) -> bool | None:
    """Whether this ANORAK run's slides were a tumour-slide list (True), every
    slide in the directory (False), or unknown (None, a run from before this
    was recorded)."""
    try:
        selection = json.loads((out_dir / "slide_list.selection.json").read_text(encoding="utf-8"))
    except (TypeError, OSError, ValueError):
        return None
    return selection.get("tumour_verified", True)


def _anorak_retry_seed(row: dict, slides_csv: Path, sample_size: int | None) -> int | None:
    """The seed a subset submission with no seed of its own should use.

    None — draw fresh — unless this run's previous attempt was itself a random
    subset. Then a blank seed on "Retry" used to draw a *different* sample into
    the same output directory: a retry that grades other slides than the
    attempt it is retrying, with resume mixing the two in one cache. So the
    recorded seed is reused, and only after re-drawing with it from the list
    given now reproduces the recorded slide_list.csv exactly — a regenerated
    source list or a different sample size under the same seed is a different
    sample, and that is refused rather than quietly submitted.
    """
    if row.get("anorak_scope") != "subset":
        return None
    recorded_seed = row.get("anorak_seed")
    recorded_list = Path(row["anorak_slide_list"]) if row.get("anorak_slide_list") else None
    how_to_choose = (
        "Enter a seed: the previous attempt's"
        + (f" ({recorded_seed})" if recorded_seed is not None else "")
        + " to repeat its sample, or any other to draw a new one on purpose."
    )
    if recorded_seed is None or recorded_list is None or not recorded_list.is_file():
        raise HTTPException(400, (
            f"This run's previous attempt was a random subset, but its "
            f"{'seed' if recorded_seed is None else 'slide list'} is not on "
            f"record, so a blank seed cannot repeat it and would silently draw "
            f"different slides. {how_to_choose}"
        ))
    # Read the way submit_anorak_job reads, so the re-draw sees the same rows
    # it will — a type-inferring read turns sample "007" into 7.
    try:
        chosen, _ = _anorak_select_slide_rows(
            _anorak_read_slide_csv(slides_csv), scope="subset",
            sample_size=sample_size, seed=int(recorded_seed),
        )
        previous = _anorak_read_slide_csv(recorded_list)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "slide_id" in chosen.columns and "slide_id" in previous.columns:
        same = chosen["slide_id"].tolist() == previous["slide_id"].tolist()
    else:
        same = chosen.reset_index(drop=True).equals(previous)
    if not same:
        raise HTTPException(400, (
            f"This run's previous attempt was a random subset "
            f"({len(previous)} slides, seed {recorded_seed}; list at "
            f"{recorded_list}). Seed {recorded_seed} over {slides_csv} at "
            f"{sample_size} slides does not reproduce it — the source list or "
            f"the sample size has changed — and a blank seed would draw yet "
            f"another sample. {how_to_choose}"
        ))
    return int(recorded_seed)


class AnorakRequest(BaseModel):
    """Stage 7: ANORAK growth-pattern grading over this run's slides."""

    # The filtered slide list. Optional so the UI can offer the cohort's own
    # tumour-slide list by default, since that is what this stage is for:
    # ANORAK segments lung-adenocarcinoma growth patterns, which means nothing
    # on a slide carrying no tumour.
    slides_csv: str | None = None

    # "full" runs every slide in that list; "subset" samples it at random.
    # A subset is what a test run is — the pipeline is unchanged, only the
    # number of slides differs, so a subset that works is evidence about the
    # full run in a way a separate test mode would not be.
    scope: str = "full"
    sample_size: int | None = None
    seed: int | None = None

    # Continue the cached run by default. Nextflow keys its cache on task
    # inputs, so a resubmission after a fixed container or a raised time limit
    # re-runs only what actually failed.
    resume: bool = True

    # Replace a run that already produced a valid grading table. Only that:
    # it never lets a submission through while the previous head job is in
    # flight or its state is unknown — see _anorak_submit_blocker.
    overwrite: bool = False

    # Walltime for the head job, in Slurm format. Optional because it is a
    # deployment-level setting (ANORAK_HEAD_TIME_LIMIT) rather than a per-run
    # choice — but overridable, because it is capped by the partition's MaxTime
    # and a submission over that ceiling is rejected outright rather than
    # trimmed to fit.
    time_limit: str | None = None

    # Head jobs in all: the first plus chain-1 standbys, each starting only if
    # the one before ended non-zero and resuming it. Without a standby, a head
    # job that reaches its walltime or runs out of watchdog restarts ends the
    # run with nothing to take over. 1 here, as on the command line, so an old
    # client gets what it always got; both UIs send 2. Needs resume, which
    # submit_anorak_job enforces (a 400 below).
    chain: int = 1


def _anorak_stop_reason(out_dir: Path | None) -> str | None:
    """What nf_supervise.stop says, or None.

    The supervisor writes it when a run ends for good — Nextflow's own failure,
    a scancel — so that no chain standby resumes it. Without this the stage
    showed a bare FAILED and the reason sat in a file nobody was pointed at.
    The submitter clears it on the next submission.
    """
    if out_dir is None:
        return None
    marker = out_dir / SUPERVISOR_STOP_MARKER
    try:
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text or f"{marker} exists but is empty"


@app.get("/anorak-submit-check")
def check_anorak_submit(partition: str | None = None):
    """Can a compute node reach the Slurm controller?

    The Nextflow head job submits every task itself. On a cluster where compute
    nodes cannot submit, it starts, submits nothing, and waits until its own
    time limit — no error and no children, which reads as a busy queue. The
    login node can always submit, so nothing about this server's own ability to
    run sbatch answers the question; this asks a real compute node.
    """
    return _check_slurm_submit_from_compute_node(partition)


@app.post("/dataset-jobs/{submission_id}/anorak")
def start_anorak_job(submission_id: str, req: AnorakRequest):
    """Stage 7: submit the ANORAK Nextflow pipeline for this run."""
    return _submit_anorak(submission_id, req)


def _submit_anorak(submission_id: str, req: AnorakRequest, *, tumour_verified: bool = True):
    """Submit the ANORAK Nextflow pipeline for a run (Stage 7, or an ANORAK run).

    tumour_verified=False only for POST /anorak-runs' own directory-wide list
    (slide_list_from_directory), never for a list a caller supplied.

    Gated on nothing this pipeline produces. ANORAK does its own tiling at its
    own resolution (0.44 um/px against HPL's 1.8) and reads the raw slides, so
    it shares no artifact with Stages 1-6 and could in principle run first. It
    sits last because what makes it worth running is the *slide list*: the
    cohort's tumour slides, which come from the cluster composition Stages 4-6
    produce. A run whose slide list is chosen some other way is free to submit
    it whenever.

    What is checked is the slide list itself. A path that does not exist, or a
    sample larger than the list holds, is refused here rather than by a head
    job an hour into the queue.

    And, before anything touches the run's directory, the previous head job:
    submit_anorak_job rewrites slide_list.csv first thing, so a submission
    refused any later than here has already changed a live run's input.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)

        state = _anorak_run_state(row)
        blocker = _anorak_submit_blocker(row, state)
        if blocker:
            raise HTTPException(*blocker)
        if not req.overwrite and _anorak_job_ids(row):
            out = Path(row["anorak_out_dir"]) if row.get("anorak_out_dir") else None
            if _anorak_output_ready(row, _anorak_grades_csv_path(out) if out else None,
                                    state)[0]:
                raise HTTPException(
                    400,
                    "This run already has a valid ANORAK grading table. Submit "
                    "with overwrite to run it again — the new run publishes "
                    "over it.",
                )

        slides_csv = Path(req.slides_csv) if req.slides_csv else None
        if slides_csv is None:
            raise HTTPException(
                400,
                "This stage needs a slide list. Point it at the output of "
                "select_tumour_slides.py (optionally filtered by "
                "filter_slides_by_tile_count.py) — ANORAK only means anything "
                "on slides that carry tumour.",
            )
        if not slides_csv.is_file():
            raise HTTPException(400, f"No such slide list: {slides_csv}")

        raw_dir = Path(row["raw_dir"]) if row.get("raw_dir") else None
        if not raw_dir or not raw_dir.is_dir():
            raise HTTPException(
                400, f"This run's raw slide directory is not readable: {raw_dir}"
            )

        seed = req.seed
        if req.scope == "subset" and seed is None:
            seed = _anorak_retry_seed(row, slides_csv, req.sample_size)

        # One directory per run, never shared: a Nextflow run owns its work/
        # cache, and two runs pointed at one would resume into each other's
        # tasks — silently, since a cache hit looks exactly like a fast task.
        out_dir = ANORAK_RESULTS_ROOT / (_row_dataset_name(row) or submission_id) / submission_id

        try:
            result = submit_anorak_job(
                slides_csv=slides_csv,
                raw_dir=raw_dir,
                out_dir=out_dir,
                pipeline_dir=ANORAK_PIPELINE_DIR,
                anorak_dir=ANORAK_REPO_DIR,
                scope=req.scope,
                sample_size=req.sample_size,
                seed=seed,
                profile=ANORAK_PROFILE,
                resume=req.resume,
                notify_email=row.get("notify_email"),
                job_name=f"anorak_{submission_id}",
                time_limit=req.time_limit,
                chain=req.chain,
                tumour_verified=tumour_verified,
            )
        except ValueError as e:
            # A bad scope, a sample larger than the list, a missing pipeline
            # directory: all the caller's to fix, and all worth reading.
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            # Recorded on the row, not only returned: sbatch refused *after*
            # submit_anorak_job rewrote slide_list.csv, so the directory no
            # longer holds the list the recorded attempt ran on, and the next
            # person to open this run needs to be told why.
            message = f"Failed to submit the ANORAK pipeline: {e}"
            _update_dataset_run_best_effort(submission_id, anorak_error=message)
            raise HTTPException(500, message)

        selection = result["selection"]
        job_ids = ([result["anorak_job_id"], *result.get("chain_job_ids", [])]
                   if result.get("anorak_job_id") else [])
        error = result.get("chain_error")
        if not job_ids:
            # Nothing to poll, so nothing for the in-flight guard to see: say
            # so on the row rather than let the stage read as never run.
            error = ("sbatch reported no job id for the head job, so this run "
                     "cannot tell whether it is running — check squeue for "
                     f"anorak_{submission_id} before submitting again.")
        _update_dataset_run_best_effort(
            submission_id,
            anorak_job_id=",".join(job_ids) or None,
            anorak_error=error,
            anorak_submitted_at=datetime.now(timezone.utc),
            anorak_out_dir=result["out_dir"],
            anorak_slide_list=result["slides_csv"],
            anorak_scope=selection["scope"],
            anorak_sample_size=selection.get("sample_size"),
            anorak_seed=selection.get("seed"),
            anorak_slides=selection["slides"],
        )
        _record_run_job(
            submission_id, "anorak", ",".join(job_ids) or None,
            output_path=result["grades_csv"],
            params={
                "scope": selection["scope"],
                "slides": selection["slides"],
                "sample_size": selection.get("sample_size"),
                "seed": selection.get("seed"),
                "pool": selection.get("pool"),
                "source_csv": selection.get("source_csv"),
                "resume": req.resume,
                "chain": req.chain,
            },
        )
        return {
            "submission_id": submission_id,
            "anorak_job_id": result.get("anorak_job_id"),
            "out_dir": result["out_dir"],
            "slide_list": result["slides_csv"],
            "grades_csv": result["grades_csv"],
            "selection": selection,
        }


# --- ANORAK on its own: POST /anorak-runs -------------------------------------
#
# The UI's "Run ANORAK" beside "Run HPL", for someone who wants growth-pattern
# grading without the HPL pipeline. It is its own run — a slurm_dataset_runs
# row whose status is ANORAK_ONLY_STATUS and whose Stages 1-6 were never
# started — so it is listed, polled, stopped and kept in history like any
# other, and the ANORAK submission itself is Stage 7's own code path
# (_submit_anorak), with every check that carries.
#
# The one thing it adds is a slide list when none is given: every slide in the
# directory, grouped into tumours by the rule HPL packaging uses, with tumour
# status recorded as unverified — the recorded choice for a cohort nobody has
# selected tumour slides from yet. A tumour-slide list, when given, is checked
# exactly as Stage 7 checks one.
ANORAK_ONLY_STATUS = "anorak_only"


class AnorakRunRequest(BaseModel):
    dataset_path: str
    dataset_name: str | None = None
    # Optional: select_tumour_slides.py's output. Blank grades every slide.
    slides_csv: str | None = None
    # A test run: a random sample of this many slides, with a recorded seed.
    sample_size: int | None = None
    seed: int | None = None
    chain: int = 2
    time_limit: str | None = None


@app.post("/anorak-runs")
def create_anorak_run(req: AnorakRunRequest):
    try:
        raw_dir = _resolve_dataset_path(req.dataset_path)
        dataset_name = (_sanitize_dataset_name(req.dataset_name)
                        if req.dataset_name else raw_dir.name)
    except ValueError as e:
        raise HTTPException(400, str(e))

    submission_id = str(uuid.uuid4())
    slides_csv = (req.slides_csv or "").strip() or None
    listed: dict = {}
    # Built before the row exists, so a refusal (no readable slides, two files
    # with one name) leaves nothing behind.
    if slides_csv is None:
        try:
            frame, listed = _anorak_directory_slide_list(raw_dir)
        except ValueError as e:
            raise HTTPException(400, str(e))
        # Beside the run's output directory, never inside it: the submitter
        # refuses a source list inside out_dir as a previous attempt's output.
        source = ANORAK_RESULTS_ROOT / dataset_name / f"{submission_id}.all_slides.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(source, index=False)
        slides_csv = str(source)

    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO slurm_dataset_runs
                    (submission_id, raw_dir, mask_dir, tile_dir, status, is_subset,
                     dataset_name)
                VALUES
                    (:submission_id, :raw_dir, :mask_dir, :tile_dir, :status,
                     :is_subset, :dataset_name)
            """),
            {"submission_id": submission_id, "raw_dir": str(raw_dir),
             "mask_dir": str(TISSUE_MASK_DIR), "tile_dir": str(PROCESSED_TILES_DIR),
             "status": ANORAK_ONLY_STATUS, "is_subset": bool(req.sample_size),
             "dataset_name": dataset_name},
        )

    anorak_req = AnorakRequest(
        slides_csv=slides_csv,
        scope="subset" if req.sample_size else "full",
        sample_size=req.sample_size,
        seed=req.seed,
        chain=req.chain,
        time_limit=req.time_limit,
    )
    try:
        result = _submit_anorak(submission_id, anorak_req,
                                tumour_verified=req.slides_csv is not None
                                and bool(req.slides_csv.strip()))
    except HTTPException as e:
        # The status stays ANORAK_ONLY_STATUS (it is the run's kind); the
        # refusal is recorded where the run's panel shows it.
        _update_dataset_run_best_effort(submission_id, error=str(e.detail)[:2000])
        raise
    return {
        **result,
        "dataset_name": dataset_name,
        "tumour_verified": bool((req.slides_csv or "").strip()),
        "skipped_unsupported": listed.get("skipped_unsupported", []),
    }


@app.post("/dataset-jobs/{submission_id}/anorak-resume")
def resume_anorak_run(submission_id: str):
    """Resubmit a stopped ANORAK run exactly as it was: the same source list,
    scope, sample, seed and tumour-verified choice, read back from the run's
    own slide_list.selection.json, with -resume so only unfinished slides run.
    """
    row = _get_dataset_run_row(submission_id)
    out = Path(row["anorak_out_dir"]) if row.get("anorak_out_dir") else None
    selection_path = out / "slide_list.selection.json" if out else None
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (AttributeError, OSError, ValueError):
        raise HTTPException(400, "This run never recorded which slides it was given, "
                                 "so it cannot be resumed as it was — start a new run.")
    source = selection.get("source_csv")
    if not source or not Path(source).is_file():
        raise HTTPException(400, f"The slide list this run was given is gone: {source}")
    anorak_req = AnorakRequest(
        slides_csv=source,
        scope=selection.get("scope", "full"),
        sample_size=selection.get("sample_size"),
        seed=selection.get("seed"),
        resume=True,
        chain=2,
    )
    return _submit_anorak(submission_id, anorak_req,
                          tumour_verified=bool(selection.get("tumour_verified", True)))


@app.post("/dataset-jobs/{submission_id}/assign-clusters")
def start_cluster_assignment_job(submission_id: str, req: ClusterAssignmentRequest):
    """Stage 4: assign HPL cluster IDs to this run's embeddings by k-NN vote.

    Gated on extraction having produced a *valid* output rather than merely
    having run. Assigning clusters to a projections file that extraction left
    half-written would read zero rows as embeddings and return cluster IDs for
    them — confidently, since every tile gets a nearest neighbour however
    meaningless the vector.
    """
    with _slurm_submission_lock():
        row = _get_dataset_run_row(submission_id)
        _refuse_if_pipeline_run(row, "cluster assignment")
        if not row["extraction_job_id"] or not row["extraction_output_path"]:
            raise HTTPException(400, "Feature extraction hasn't been started for this run yet.")

        projections = Path(row["extraction_output_path"])
        # Bound to the packaged input's tile count. Unbound, this accepts an
        # extraction that covered a fraction of the slides — and Stage 4 would
        # then write a complete, well-formed assignments CSV for that fraction,
        # which Stage 5 loads into the KB as if it were the whole dataset. The
        # per-slide aggregates come out of that silently wrong, with nothing
        # downstream able to tell.
        ok, reason = _validate_extraction_output(
            projections, expected_rows=_extraction_expected_rows(row)
        )
        if not ok:
            raise HTTPException(
                400,
                f"Feature extraction has not produced a usable output yet ({reason}). "
                f"Cluster assignment reads those embeddings, so it would produce IDs "
                f"for rows the encoder never wrote.",
            )

        if row.get("assignment_job_id") and not req.overwrite:
            state = _get_slurm_job_state(row["assignment_job_id"])
            if state in IN_FLIGHT_SLURM_STATES:
                raise HTTPException(
                    400,
                    f"Cluster assignment is already running for this run "
                    f"(job {row['assignment_job_id']}, state {state}).",
                )

        dataset_name = projections.parent.name if projections.parent.name else submission_id
        out_csv = _assignment_output_path(projections, _row_dataset_name(row) or dataset_name)

        try:
            result = submit_cluster_assignment_job(
                projections_h5=projections,
                out_csv=out_csv,
                reference=Path(req.reference) if req.reference else None,
                k=req.k,
                notify_email=row["notify_email"],
                job_name=f"hpl_assign_{submission_id}",
                overwrite=True,  # decided above; the stage is cheap to redo
                **req.vote_kwargs(),
            )
        except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
            # Same split as the other stages: a missing reference or an input
            # that isn't a projections file is the user's to fix, not a fault.
            raise HTTPException(400, str(e))
        except SystemExit as e:
            # resolve_vote/vote_flags refuse an unknown preset or a combination
            # that cannot do anything. That is the caller's to fix, and it must
            # not surface as a 500 — the whole point of refusing before the
            # queue is that the reason reaches whoever asked.
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to submit cluster assignment job: {e}")

        _update_dataset_run(
            submission_id,
            assignment_job_id=result.get("assignment_job_id"),
            assignment_output_path=result.get("out_csv"),
            assignment_reference=result.get("reference_path"),
        )
        # Separate call, best-effort: this column arrives with
        # migrate_dataset_runs_assignment_vote.sql, and folding it into the
        # update above would make an unapplied migration lose the job id the
        # whole stage gates on.
        _update_dataset_run_best_effort(
            submission_id, assignment_vote=result.get("vote"),
        )
        _record_run_job(
            submission_id, "assignment", result.get("assignment_job_id"),
            output_path=result.get("out_csv"),
            params={
                "reference": result.get("reference_path"),
                "reference_rows": result.get("reference_rows"),
                "n_clusters": result.get("n_clusters"),
                # Which vote produced this CSV. The CSV itself cannot carry it:
                # load_hpc_assignments.py identifies the cluster column by
                # elimination, so an extra column there breaks Stage 5. This is
                # the only place two CSVs from one reference but different votes
                # are distinguishable.
                "vote": result.get("vote"),
                "vote_preset": result.get("vote_preset"),
                "vote_flags": result.get("vote_flags"),
            },
        )
        return {"submission_id": submission_id, **result}


@app.post("/dataset-jobs/{submission_id}/assign-clusters-test")
def start_test_cluster_assignment_job(submission_id: str, req: ClusterAssignmentTestRequest):
    """Assign clusters for an arbitrary projections .h5 — typically the output
    of a test extraction — without touching this run's tracked Stage 4 state.

    Deliberately writes nothing to slurm_dataset_runs, for the same reason
    /extract-features-test does not: a test attempt must never satisfy
    assignment_ready and let the run look further along than it is.
    """
    projections = Path(req.projections_h5)
    ok, reason = _validate_extraction_output(projections)
    if not ok:
        raise HTTPException(400, f"{projections} is not a usable projections file ({reason}).")

    out_csv = projections.with_name(f"{projections.stem}_hpc_assignments.csv")
    try:
        result = submit_cluster_assignment_job(
            projections_h5=projections,
            out_csv=out_csv,
            reference=Path(req.reference) if req.reference else None,
            k=req.k,
            job_name=f"hpl_assign_test_{submission_id}",
            overwrite=True,
            **req.vote_kwargs(),
        )
    except (FileNotFoundError, FileExistsError, ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    except SystemExit as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to submit test cluster assignment job: {e}")

    _record_run_job(
        submission_id, "assignment_test", result.get("assignment_job_id"),
        output_path=result.get("out_csv"),
        params={"projections_h5": str(projections),
                "reference": result.get("reference_path"),
                "vote": result.get("vote"),
                "vote_preset": result.get("vote_preset")},
    )
    return {"submission_id": submission_id, **result}


def _kb_load_source_csv(row: dict, override_path: str | None = None) -> Path:
    """This run's assignment CSV, or an explicit override path.

    The override exists for output from Stage 4's "Test on a sample .h5"
    mode: that mode deliberately never records an output path on
    slurm_dataset_runs (see start_test_cluster_assignment_job), so there is no
    tracked path here to fall back to for it. Given one explicitly, it is
    validated exactly the same way — Stage 5 does not get a looser bar just
    because the CSV did not come from this run's own tracking.

    Gated on the same validator the /status endpoint uses for
    assignment_ready — a CSV that exists but is a stub (interrupted run, half
    the columns) is not a state Stage 5 can load from, whatever the DB row's
    assignment_job_id says.
    """
    if override_path:
        csv_path = Path(override_path)
    else:
        path = row.get("assignment_output_path")
        if not path:
            raise HTTPException(400, "Cluster assignment hasn't been run for this dataset yet.")
        csv_path = Path(path)

    ok, reason = _validate_assignment_output(csv_path)
    if not ok:
        source = "This CSV" if override_path else "This run's assignment output"
        # Name the path. "no output file" without it is unactionable — the
        # commonest cause is a typed path that does not exist, and the message
        # was giving no way to tell that apart from a corrupt file.
        detail = (
            f"{source} isn't usable ({reason}):\n  {csv_path}\n\n"
            f"Stage 5 loads exactly what's there, so it refuses to read a partial "
            f"or missing CSV rather than loading whatever rows happen to be there."
        )
        if reason == "no output file":
            detail += (
                f"\n\nNothing exists at that path. Check it with "
                f"`ls -l {csv_path}`. If you meant the output of "
                f"migrate_tile_names.py, that file is only written when it is run "
                f"with --commit — a dry run reports what it would do and writes "
                f"nothing."
            )
        elif reason.startswith("missing column"):
            detail += (
                "\n\nIf the columns look like data values, this CSV has no header "
                "row. migrate_tile_names.py restores one, and doing so also "
                "recovers the first tile, which a headerless read silently drops."
            )
        raise HTTPException(400, detail)
    return csv_path


class KbLoadPreviewRequest(BaseModel):
    # See RegistrationRequest.kb_target.
    kb_target: str = KB_PRODUCTION
    # min_margin previews compute_profiles()'s own exclusion (see
    # load_hpc_assignments.py) — leave-one-out validation put tiles below 0.1
    # vote_margin at 57% correct and 0.1-0.25 at 76%, against 92%+ once margin
    # clears 0.25, so this is how many of those the CSV actually holds before
    # anyone decides whether to exclude them.
    min_margin: float = 0.0
    # See _kb_load_source_csv — output from Stage 4's "Test on a sample .h5"
    # mode has no tracked path, so it has to be given explicitly.
    csv_path: str | None = None


@app.post("/dataset-jobs/{submission_id}/kb-load-preview")
def preview_kb_load(submission_id: str, req: KbLoadPreviewRequest = KbLoadPreviewRequest()):
    """Stage 5, dry-run half: what loading an assignment CSV into the
    Knowledge Bank would do, computed without changing anything.

    Same report load_hpc_assignments.py prints for --dry-run (its default
    posture), just returned as JSON instead of stdout — this endpoint calls
    the identical read_assignments()/inspect() pair the CLI does, so the two
    can never disagree about what a load would do.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = _kb_load_source_csv(row, override_path=req.csv_path)

    try:
        frame, cluster_column = _read_kb_assignments(csv_path)
    except SystemExit as e:
        raise HTTPException(400, str(e))

    report = _inspect_kb_load(_get_engine(req.kb_target), frame, cluster_column,
                              min_margin=req.min_margin)
    match_rate = report["matched"] / report["rows"] if report["rows"] else 0.0
    report.update({
        "cluster_column": cluster_column,
        "match_rate": match_rate,
        "min_match_rate": _KB_MIN_MATCH_RATE,
        "would_refuse_low_match": match_rate < _KB_MIN_MATCH_RATE,
        # Tracked run state is meaningless for an explicit csv_path — it may
        # be a completely different run's test output, so showing this run's
        # own kb_load_done/at/rows next to it would claim a connection that
        # is not there.
        "already_loaded": bool(row.get("kb_load_done")) if not req.csv_path else None,
        "kb_load_at": (row["kb_load_at"].isoformat() if row.get("kb_load_at") else None)
                      if not req.csv_path else None,
        "kb_load_rows": row.get("kb_load_rows") if not req.csv_path else None,
        "kb_load_reference": row.get("kb_load_reference") if not req.csv_path else None,
    })
    return report


class KbLoadRequest(BaseModel):
    # See RegistrationRequest.kb_target.
    kb_target: str = KB_PRODUCTION
    # Mirrors load_hpc_assignments.py's CLI flags. cancer_type stays optional
    # rather than guessed — hpl_profile_summary.cancer_type is left unset when
    # omitted, same as the CLI, rather than this endpoint inventing a value the
    # CLI never would.
    cancer_type: str | None = None
    allow_unknown_clusters: bool = False
    skip_profiles: bool = False
    # See KbLoadPreviewRequest.min_margin — this is the value that actually
    # gets applied to hpl_profile_proportion/summary, not just previewed.
    min_margin: float = 0.0
    # See _kb_load_source_csv. This still performs a real write to the KB —
    # unlike Stage 4's own test mode, there is no throwaway version of
    # "filling the knowledge bank". What it skips is recording kb_load_done
    # against this particular run, since the CSV may not be this run's own.
    csv_path: str | None = None


@app.post("/dataset-jobs/{submission_id}/kb-load")
def commit_kb_load(submission_id: str, req: KbLoadRequest):
    """Stage 5: write cluster assignments into tile_registry, plus the
    hpl_profile_proportion/summary aggregates the chatbot and HPC panels
    actually read.

    This is load_hpc_assignments.py's --commit path, called in-process rather
    than reimplemented — every guard the CLI enforces (95% match rate, unknown
    cluster IDs) applies here unchanged, because it runs the same function.
    That is deliberate: the one script that mutates the shared KB should have
    exactly one implementation of what makes a load safe to commit, not one
    for the terminal and a looser one for the UI.
    """
    row = _get_dataset_run_row(submission_id)
    csv_path = _kb_load_source_csv(row, override_path=req.csv_path)

    try:
        frame, cluster_column = _read_kb_assignments(csv_path)
    except SystemExit as e:
        raise HTTPException(400, str(e))

    eng = _get_engine(req.kb_target)
    report = _inspect_kb_load(eng, frame, cluster_column, min_margin=req.min_margin)
    match_rate = report["matched"] / report["rows"] if report["rows"] else 0.0

    problems = []
    if match_rate < _KB_MIN_MATCH_RATE:
        problems.append(
            f"only {match_rate * 100:.1f}% of rows match tile_registry (need "
            f"{_KB_MIN_MATCH_RATE * 100:.0f}%). The usual cause is a slide-naming "
            f"difference between the .h5 and the registry, not missing tiles."
        )
    if report["unknown_clusters"] and not req.allow_unknown_clusters:
        problems.append(
            f"{len(report['unknown_clusters'])} cluster ID(s) have no hpc_dictionary "
            f"row: {report['unknown_clusters'][:10]}. Those tiles would show a cluster "
            f"with no pattern or malignancy annotation. Pass allow_unknown_clusters if "
            f"that is intended."
        )
    if problems:
        raise HTTPException(400, "Refusing to load: " + "; ".join(problems))

    profiles = None
    if not req.skip_profiles:
        profiles = _compute_kb_profiles(frame, cluster_column, req.cancer_type,
                                        min_margin=req.min_margin)

    updated = _write_kb_load(eng, frame, cluster_column, profiles=profiles)
    reference = report["reference"]

    if not req.csv_path:
        # Only this run's own tracked assignment marks the run's KB-load
        # state. An explicit csv_path may be a different run's test output
        # entirely (see _kb_load_source_csv), so recording it here would
        # claim progress this run never actually made.
        _update_dataset_run(
            submission_id,
            kb_load_done=True,
            kb_load_at=datetime.now(timezone.utc),
            kb_load_rows=updated,
            kb_load_reference=reference,
            kb_load_kb_target=_resolve_kb_target(req.kb_target),
        )

    return {
        "submission_id": submission_id,
        "kb_target": _resolve_kb_target(req.kb_target),
        "database": KB_TARGETS[_resolve_kb_target(req.kb_target)],
        "updated_rows": updated,
        "matched": report["matched"],
        "reference": reference,
        "profiles_written": profiles is not None,
        "min_margin": req.min_margin,
        "excluded_from_aggregates": report["excluded_from_aggregates"],
        "recorded_on_run": req.csv_path is None,
    }


# --- Registration: the identity rows Stage 5 needs to exist -----------------
#
# Every argument register_dataset.py takes is already recorded on the run —
# h5_output_path, tile_dir, dataset_name, raw_dir, tiling_params — so these
# endpoints derive them rather than asking the UI to retype paths that the
# server already knows and could be typed wrong. dataset_id is the exception:
# it is the KB cohort key, it outlives the run, and a re-run under a new
# submission must be able to target the same cohort.


class RegistrationRequest(BaseModel):
    # Defaults to the run's own dataset_name, upper-cased, but stays settable:
    # dataset_name is a folder on scratch and dataset_id is the KB's cohort
    # key, and a second, fuller run of the same cohort has a different folder
    # and the same key.
    dataset_id: str | None = None
    # The folder under tile_dir this run's tiles live in — register_dataset.py
    # reads Stage 1's per-slide _tile_metadata.csv out of it, so without it
    # there are no coordinates for any tile. It is normally recorded on the run
    # (slurm_dataset_runs.dataset_name), but a run submitted before that column
    # existed — or tiled by hand, outside /submit-dataset-job — has it NULL.
    # Those were refused outright ("register it with the CLI instead"), which is
    # a dead end reached with the KB cohort key already filled in, because that
    # key is a different thing from this folder. Supplying it here is the fix.
    tile_dataset_name: str | None = None
    # Which Knowledge Bank to write into. Defaults to production, so a client
    # that does not know about this field cannot land a cohort in the wrong
    # database by omission.
    kb_target: str = KB_PRODUCTION
    scope: str = "full"
    slide_names: list[str] | None = None
    # --- overrides for what the run record does not hold ---------------------
    #
    # Registration reads these four off slurm_dataset_runs, which is right when
    # the run was driven through this UI. Runs that predate a column, or work
    # done on the cluster before the pipeline existed, have no dataset_name and
    # sometimes no h5_output_path — and the only advice the endpoint could give
    # was to leave the UI and use the CLI.
    #
    # None means "take it from the run", which is what every existing caller
    # sends. Supplied values are used as given and are still checked to exist,
    # so an override can be wrong in ways that are noticed, not in ways that
    # register the wrong cohort quietly.
    dataset_name: str | None = None
    raw_dir: str | None = None
    tile_dir: str | None = None
    h5_path: str | None = None
    # Opens every slide file, so it is opt-in for the same reason as the CLI
    # flag: 14,044 headers is minutes, not seconds.
    slide_metadata: bool = False
    # Written only when both are present, matching the CLI. Defaults come from
    # the run's recorded tiling_params, which is where the numbers actually
    # used live — not from the server's module constants, which assume one
    # geometry for every cohort (see TILE_SIZE_5X / SCALE).
    write_dataset_config: bool = True
    replace: bool = False


def _registration_inputs(row, req: "RegistrationRequest") -> dict:
    """The four paths registration reads, from the run record or the request.

    Shared by the in-server registration (_registration_plan) and the Slurm one
    (/register-submit), so the two cannot read different files for one run.
    Returns h5_path, tile_dir, dataset_name, raw_dir and a `sources` map saying,
    per input, whether the value came from the run or was supplied — an override
    is a chance to register the wrong directory, so the preview shows it.
    """
    sources = {}

    def _resolve(name, override, recorded):
        value = (override or "").strip() if isinstance(override, str) else override
        if value:
            sources[name] = "supplied"
            return value
        sources[name] = "run record"
        return recorded

    packaged = _resolve("h5_path", req.h5_path, row.get("h5_output_path"))
    if not packaged:
        raise HTTPException(400,
            "This run has no packaged .h5 recorded. Finish Stage 2, or give the "
            "path to an .h5 packaged elsewhere in 'Packaged .h5' below.")
    h5_path = Path(packaged)
    if not h5_path.is_file():
        raise HTTPException(400, f"No .h5 at {h5_path} ({sources['h5_path']}).")

    tile_dir = Path(_resolve("tile_dir", req.tile_dir,
                             row.get("tile_dir") or str(PROCESSED_TILES_DIR)))
    if not tile_dir.is_dir():
        raise HTTPException(400, f"No such directory: {tile_dir} ({sources['tile_dir']}).")

    # The tile folder: dataset_name, or tile_dataset_name — the same thing under
    # the name some clients send, accepted so neither breaks, and refused if the
    # two disagree. A supplied name is charset-checked before anything else,
    # because it becomes a literal path segment under tile_dir.
    by_name = (req.dataset_name or "").strip()
    by_tile_name = (req.tile_dataset_name or "").strip()
    if by_name and by_tile_name and by_name != by_tile_name:
        raise HTTPException(400,
            f"dataset_name ({by_name!r}) and tile_dataset_name ({by_tile_name!r}) "
            f"name different tile folders; send one.")
    supplied_name = by_name or by_tile_name
    if supplied_name:
        try:
            supplied_name = _sanitize_dataset_name(supplied_name)
        except ValueError as e:
            raise HTTPException(400, str(e))
    dataset_name = _resolve("dataset_name", supplied_name or None, row.get("dataset_name"))
    if not dataset_name:
        raise HTTPException(400,
            "This run has no recorded dataset_name, so the folder its tiles live "
            "under is not known. Choose it as 'Tile folder' (over the API, send "
            "tile_dataset_name) — the directory under "
            f"{tile_dir} holding one folder per slide, e.g. 'Radiogenomics'. It is "
            "not the same as dataset_id: that is the Knowledge Bank cohort key, "
            "this is the directory on disk.")

    # Refuse a folder that is not on disk, rather than reading no metadata out
    # of it. A wrong name does not fail anywhere downstream — it registers every
    # tile with no coordinates, which is the shape of a successful run. Case is
    # the likely way to get it wrong: macOS matches RADIOGENOMICS to
    # Radiogenomics and the cluster's Linux filesystem does not.
    if not (tile_dir / dataset_name).is_dir():
        near = [p.name for p in tile_dir.iterdir()
                if p.is_dir() and p.name.lower() == dataset_name.lower()]
        hint = (f" Did you mean '{near[0]}'? Folder names are case-sensitive here."
                if near else
                " It is a folder name, not a path — it is joined onto the tile "
                "root — and registration reads every tile's coordinates from it, "
                "so continuing would register tiles with no coordinates at all.")
        raise HTTPException(
            400, f"No tile folder '{dataset_name}' under {tile_dir} "
                 f"({sources['dataset_name']}).{hint}")

    raw_override = _resolve("raw_dir", req.raw_dir, row.get("raw_dir"))
    raw_dir = Path(raw_override) if raw_override else None
    if raw_dir is not None and not raw_dir.is_dir():
        # Reported rather than fatal: the tile tables are still registerable,
        # and saying so is more useful than refusing everything because the
        # slides have been moved off scratch.
        raw_dir = None


    return {"h5_path": h5_path, "tile_dir": tile_dir, "dataset_name": dataset_name,
            "raw_dir": raw_dir, "sources": sources}


def _registration_plan(row, req: "RegistrationRequest"):
    """Build register_dataset.py's plan from _registration_inputs()."""
    inputs = _registration_inputs(row, req)
    h5_path, tile_dir = inputs["h5_path"], inputs["tile_dir"]
    dataset_name, raw_dir, sources = inputs["dataset_name"], inputs["raw_dir"], inputs["sources"]

    dataset_id = (req.dataset_id or dataset_name).strip().upper()
    scope = (req.scope or "full").strip().lower()

    if scope not in {"full", "subset"}:
        raise HTTPException(
            400,
            "scope must be 'full' or 'subset'",
        )

    slide_names = None

    if scope == "subset":
        slide_names = [
            str(s).strip()
            for s in (req.slide_names or [])
            if str(s).strip()
        ]

        if not slide_names:
            raise HTTPException(
                400,
                "Subset registration requires at least one slide ID or filename.",
            )
    target_mpp = tile_px = None
    if req.write_dataset_config:
        params = _row_tiling_params(row) or {}
        target_mpp = params.get("target_mpp")
        tile_px = params.get("target_tile_px")

    plan = _build_registration(
        h5_path, tile_dir, dataset_name, str(h5_path), dataset_id,
        raw_dir=raw_dir,
        slide_metadata=req.slide_metadata,
        target_mpp=target_mpp,
        tile_size_5x_px=tile_px,
        scope=scope,
        slide_names=slide_names,
    )
    # Carried on the plan so both endpoints can report what was actually read,
    # and where each value came from — an override is a chance to register the
    # wrong directory, so it has to be visible before anything is written.
    plan["tile_dataset_name"] = dataset_name
    plan["sources"] = sources
    plan["resolved"] = {
        "h5_path": str(h5_path),
        "tile_dir": str(tile_dir),
        "dataset_name": dataset_name,
        "raw_dir": str(raw_dir) if raw_dir else None,
    }
    return plan, dataset_id, raw_dir


@app.post("/dataset-jobs/{submission_id}/register-preview")
def preview_registration(submission_id: str, req: RegistrationRequest):
    """What registering this run would write, without writing it.

    Same preview-then-commit shape as Stage 5, and for the same reason: this
    writes to the shared Knowledge Bank, and the failure it guards against is
    a registration that succeeds against the wrong cohort.
    """
    row = _get_dataset_run_row(submission_id)
    try:
        plan, dataset_id, raw_dir = _registration_plan(row, req)
        report = _registration_preview(_get_engine(req.kb_target), plan, dataset_id)
    except SystemExit as e:
        raise HTTPException(400, str(e))

    collisions = sum(report["foreign_collisions"].values())
    occupied = {t: v["rows"] for t, v in report["existing"].items() if v["rows"]}
    report.update({
        "submission_id": submission_id,
        "kb_target": _resolve_kb_target(req.kb_target),
        "database": KB_TARGETS[_resolve_kb_target(req.kb_target)],
        # What will actually be read, and where each value came from.
        "resolved": plan.get("resolved"),
        "sources": plan.get("sources"),
        "raw_dir": str(raw_dir) if raw_dir else None,
        # Which tile folder the coordinates were read from. Named in the report
        # because a wrong folder does not fail — it comes back as tiles with no
        # coordinates, which reads like missing Stage 1 output.
        "tile_dataset_name": plan["tile_dataset_name"],
        "missing_tables": [t for t, v in report["existing"].items() if v.get("missing")],
        # The two states the UI has to gate its button on, computed here so the
        # rule lives next to the guard that enforces it rather than being
        # reimplemented in two frontends.
        "would_refuse_collision": bool(collisions),
        "needs_replace": bool(occupied) and not req.replace,
        # Reported here, before anything is written, because the write that
        # would fail happens after the rows are already committed.
        "missing_run_tracking_columns": _missing_run_tracking_columns(),
        "already_registered": bool(row.get("registration_done")),
        "registration_at": (row["registration_at"].isoformat()
                            if row.get("registration_at") else None),
    })
    return report


@app.post("/dataset-jobs/{submission_id}/register")
def commit_registration(submission_id: str, req: RegistrationRequest):
    """Create this dataset's identity rows in the Knowledge Bank.

    register_dataset.py's --commit path, called in-process rather than
    reimplemented, for the reason commit_kb_load gives: the guards that make a
    write to the shared KB safe should have one implementation, not a strict
    one for the terminal and a looser one for the UI. Every refusal here —
    a cohort collision, an existing registration without replace, a base table
    the database does not have — is raised by the same function the CLI runs.
    """
    row = _get_dataset_run_row(submission_id)
    try:
        plan, dataset_id, raw_dir = _registration_plan(row, req)
        written = _registration_commit(_get_engine(req.kb_target), plan, dataset_id,
                                       req.replace)
    except SystemExit as e:
        # register_dataset.py refuses by raising SystemExit with the reason.
        # 400 rather than 500: every one of those is a decision for the
        # operator, not a server fault.
        raise HTTPException(400, str(e))

    _update_dataset_run(
        submission_id,
        registration_done=True,
        registration_at=datetime.now(timezone.utc),
        registration_dataset_id=dataset_id,
        registration_raw_dir=str(raw_dir) if raw_dir else None,
        registration_rows=json.dumps(written),
        # Run tracking stays in production for every stage, by design. That
        # makes "registration_done" ambiguous on its own — it does not say
        # which Knowledge Bank the rows went into — so record the target.
        registration_kb_target=_resolve_kb_target(req.kb_target),
    )

    # wsi_registry is cached in memory at startup, so a slide registered now is
    # not openable until the map is rebuilt. Doing it here rather than telling
    # the user to restart the server: _load_wsi_map() is one query and this is
    # the only place that invalidates it outside the upload path.
    if written.get("wsi_registry"):
        _load_wsi_map(req.kb_target)

    return {
        "submission_id": submission_id,
        "dataset_id": dataset_id,
        "tile_dataset_name": plan["tile_dataset_name"],
        "tile_names_normalized": plan["tile_names_normalized"],
        "kb_target": _resolve_kb_target(req.kb_target),
        "database": KB_TARGETS[_resolve_kb_target(req.kb_target)],
        "written": written,
        "slides_without_files": plan["slides_without_files"],
        "ambiguous_slides": plan["ambiguous_slides"],
        "unreadable_slides": plan["unreadable_slides"],
        "conflicting_samples": plan["conflicting_samples"],
        "missing_slides": plan["missing_slides"],
        "unmatched_tiles": len(plan["unmatched_tiles"]),
    }


# --- Stages 5 and 6 as Slurm jobs -------------------------------------------
#
# The same two writes, handed to Slurm instead of run inside the request. The
# endpoints above still exist and still work; these are for a cohort big enough
# that the write outliving the server matters. What the job runs is the CLI, so
# every guard is the same code — a low match rate, a cohort collision, an
# existing registration without replace are all refused inside the job exactly
# as they are refused inline.
#
# Preview first, the same as before. These endpoints deliberately do not run the
# plan: building it is most of the work, and doing it twice would put the cost
# back in the request that this exists to get it out of.


def _refuse_if_job_in_flight(row, job_column: str, label: str) -> None:
    job_id = row.get(job_column)
    if not job_id:
        return
    state = _get_slurm_job_state(job_id)
    if state in IN_FLIGHT_SLURM_STATES:
        raise HTTPException(
            400,
            f"{label} is already queued or running for this run (job {job_id}, "
            f"state {state}). Cancel it first if you mean to replace it.",
        )


@app.post("/dataset-jobs/{submission_id}/register-submit")
def submit_registration(submission_id: str, req: RegistrationRequest):
    """Queue Stage 5 on Slurm. Returns immediately with a job id."""
    row = _get_dataset_run_row(submission_id)
    target = _resolve_kb_target(req.kb_target)

    inputs = _registration_inputs(row, req)
    packaged, tile_dir = inputs["h5_path"], inputs["tile_dir"]
    dataset_name, raw_dir = inputs["dataset_name"], inputs["raw_dir"]
    params = _row_tiling_params(row) or {} if req.write_dataset_config else {}

    with _slurm_submission_lock():
        _refuse_if_job_in_flight(row, "registration_job_id", "Registration")
        try:
            result = submit_registration_job(
                submission_id=submission_id,
                h5=Path(packaged),
                tile_dir=tile_dir,
                tile_dataset_name=dataset_name,
                dataset_id=(req.dataset_id or dataset_name).strip().upper(),
                db_name=KB_TARGETS[target],
                run_db_name=KB_TARGETS[KB_PRODUCTION],
                raw_dir=raw_dir,
                slide_metadata=req.slide_metadata,
                target_mpp=params.get("target_mpp"),
                tile_size_5x_px=params.get("target_tile_px"),
                scope=(req.scope or "full").strip().lower(),
                slide_names=req.slide_names,
                replace=req.replace,
                notify_email=row.get("notify_email"),
            )
        except ValueError as e:
            # resolve_job_db_host's refusal, which is the one worth reading in
            # full — it names the variable to set and how to test it.
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Failed to submit registration job: {e}")

        _update_dataset_run_best_effort(
            submission_id,
            registration_job_id=result.get("registration_job_id"),
            registration_submitted_at=datetime.now(timezone.utc),
            registration_log_path=result.get("registration_log_path"),
            registration_error=None,
            registration_kb_target=target,
        )
        _record_run_job(submission_id, "registration",
                        result.get("registration_job_id"),
                        params={"dataset_id": req.dataset_id,
                                "tile_dataset_name": dataset_name,
                                "kb_target": target})
    return {"submission_id": submission_id, "kb_target": target,
            "database": KB_TARGETS[target], **result}


@app.post("/dataset-jobs/{submission_id}/kb-load-submit")
def submit_kb_load(submission_id: str, req: KbLoadRequest):
    """Queue Stage 6 on Slurm. Returns immediately with a job id."""
    row = _get_dataset_run_row(submission_id)
    target = _resolve_kb_target(req.kb_target)
    csv_path = _kb_load_source_csv(row, override_path=req.csv_path)

    with _slurm_submission_lock():
        _refuse_if_job_in_flight(row, "kb_load_job_id", "The Knowledge Bank load")
        try:
            result = submit_kb_load_job(
                submission_id=submission_id,
                csv_path=Path(csv_path),
                db_name=KB_TARGETS[target],
                run_db_name=KB_TARGETS[KB_PRODUCTION],
                cancer_type=req.cancer_type,
                allow_unknown_clusters=req.allow_unknown_clusters,
                skip_profiles=req.skip_profiles,
                min_margin=req.min_margin,
                notify_email=row.get("notify_email"),
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Failed to submit Knowledge Bank load job: {e}")

        _update_dataset_run_best_effort(
            submission_id,
            kb_load_job_id=result.get("kb_load_job_id"),
            kb_load_submitted_at=datetime.now(timezone.utc),
            kb_load_log_path=result.get("kb_load_log_path"),
            kb_load_error=None,
            kb_load_kb_target=target,
        )
        _record_run_job(submission_id, "kb_load", result.get("kb_load_job_id"),
                        params={"csv_path": str(csv_path), "kb_target": target,
                                "min_margin": req.min_margin})
    return {"submission_id": submission_id, "kb_target": target,
            "database": KB_TARGETS[target], "csv_path": str(csv_path), **result}


@app.get("/kb-job-db-check")
def kb_job_db_check(kb_target: str = KB_PRODUCTION):
    """Can a compute node reach *and use* Postgres? The whole feature rests on it.

    Read-only and slow (it queues a one-second srun), so the UI asks only when
    the operator clicks — but it is here rather than in a runbook because the
    answer is cluster configuration nobody can infer from the server, which
    reaches the database over a socket or localhost quite happily.

    `usable` is the field to believe: `reachable` is only the TCP handshake, and
    a socket that opens and then fails authentication is the state this check
    exists to catch. kb_target picks which database is probed, because the test
    Knowledge Bank not existing on that server looks identical from here.
    """
    try:
        result = check_db_from_compute_node(
            database=KB_TARGETS[_resolve_kb_target(kb_target)])
        return {**result, "advice": probe_advice(result)}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except subprocess.TimeoutExpired:
        raise HTTPException(
            504, "The probe job did not run within 5 minutes — the queue is busy "
                 "rather than the database unreachable. Try again, or run "
                 "`python submit_kb_write.py --check-db` yourself.")
    except FileNotFoundError:
        raise HTTPException(400, "srun is not on this machine's PATH, so there "
                                 "is no Slurm to submit to.")


@app.post("/dataset-jobs/{submission_id}/cancel")
def cancel_dataset_job(submission_id: str):
    """Cancel every Slurm job (all tiling batches + the packaging job, if
    any) associated with this submission via scancel, and mark it cancelled.
    scancel on a job that's already finished is a harmless no-op, so this
    doesn't need to know which of the jobs are still actually running.
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")

    if row.get("status") == ANORAK_ONLY_STATUS:
        # An ANORAK run: its head chain, then what it had queued, found by its
        # work directory exactly as for the HPL pipeline. The status stays
        # ANORAK_ONLY_STATUS — it is what makes the run an ANORAK run — and the
        # head job's CANCELLED state is what says it was stopped.
        out = Path(row["anorak_out_dir"]) if row.get("anorak_out_dir") else None
        try:
            outcome = (submit_hpl_nf.cancel_run(out, _split_job_ids(row.get("anorak_job_id")))
                       if out else {"cancelled_job_ids": [], "errors": []})
        except (OSError, subprocess.SubprocessError) as e:
            outcome = {"cancelled_job_ids": [], "errors": [str(e)]}
        return {
            "submission_id": submission_id,
            "cancelled_job_ids": outcome["cancelled_job_ids"],
            "scancel_error": "; ".join(outcome["errors"]) or None,
        }

    if _is_nf_job_id(row["job_id"]):
        # A pipeline run: the head chain, then whatever it had queued (found by
        # work directory, as the watchdog finds them). Its stage columns hold
        # sentinels, which scancel would reject along with every real id.
        try:
            outcome = submit_hpl_nf.cancel_run(_nf_run_dir(submission_id))
        except (OSError, subprocess.SubprocessError) as e:
            outcome = {"cancelled_job_ids": [], "errors": [str(e)]}
        _NF_HEAD_STATE_CACHE.pop(submission_id, None)
        _update_dataset_run(submission_id, status="cancelled")
        return {
            "submission_id": submission_id,
            "cancelled_job_ids": outcome["cancelled_job_ids"],
            "scancel_error": "; ".join(outcome["errors"]) or None,
        }

    job_ids_to_cancel = [j for j in (row["job_id"] or "").split(",") if j]
    if row["h5_job_id"]:
        job_ids_to_cancel.append(row["h5_job_id"])
    if row["extraction_job_id"]:
        job_ids_to_cancel.append(row["extraction_job_id"])
    # A stage this server ran inline has already finished and has no Slurm job
    # to cancel; scancel would reject the whole call for its sentinel and take
    # the run's real jobs down with it into scancel_error.
    job_ids_to_cancel, _local = _split_local_job_ids(job_ids_to_cancel)

    cancelled = []
    scancel_error = None
    if job_ids_to_cancel:
        try:
            subprocess.run(
                ["scancel", *job_ids_to_cancel],
                capture_output=True, text=True, timeout=15, check=True,
            )
            cancelled = job_ids_to_cancel
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            scancel_error = str(e)

    _update_dataset_run(submission_id, status="cancelled")

    return {
        "submission_id": submission_id,
        "cancelled_job_ids": cancelled,
        "scancel_error": scancel_error,
    }


@app.get("/dataset-jobs")
def list_dataset_jobs(
    with_state: bool = Query(
        False,
        description="Also resolve each run's live Slurm state (one sacct call for "
                    "the whole list). Off by default so the plain listing stays a "
                    "single DB query.",
    ),
    limit: int = Query(
        25, ge=1, le=200,
        description="How many recent runs to resolve state for. Only applies with "
                    "with_state=true.",
    ),
):
    """Past/active dataset job submissions, most recent first.

    with_state=true adds, per run:
      stage        which pipeline stage it has reached, from the row alone
      slurm_state  running / pending / complete / failed / no record / unknown

    Resolved for the whole list in ONE sacct call (see _slurm_states_by_job) —
    a per-run lookup would be one call each, and this endpoint is what draws the
    "Recent dataset jobs" list on every page render.
    """
    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at DESC", eng
    )
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    if not with_state:
        return rows

    considered = rows[:limit]
    wanted: list[str] = []
    for row in considered:
        # The stage each run has reached decides which job ID says whether it is
        # busy. Reporting the tiling array's state for a run that finished
        # tiling hours ago and is now packaging would describe the wrong job.
        for field in ("extraction_job_id", "h5_job_id", "job_id"):
            value = row.get(field)
            if value:
                wanted.extend(j for j in str(value).split(",") if j)
                break

    states = _slurm_states_by_job(sorted(set(wanted)))

    for row in considered:
        stage, job_ids = None, []
        if row.get("extraction_job_id"):
            stage, job_ids = "extracting features", [row["extraction_job_id"]]
        elif row.get("h5_job_id"):
            stage, job_ids = "packaging", [row["h5_job_id"]]
        elif row.get("job_id"):
            stage = "tiling"
            job_ids = [j for j in str(row["job_id"]).split(",") if j]

        row["stage"] = stage or (row.get("status") or "queued")

        # A row status of error/cancelled is a decision already recorded about
        # the run and outranks whatever Slurm remembers about its jobs — a
        # cancelled run's batches may well read COMPLETED.
        if row.get("status") in ("error", "cancelled"):
            row["slurm_state"] = row["status"]
        elif not job_ids:
            row["slurm_state"] = row.get("status") or "queued"
        elif states is None:
            row["slurm_state"] = "unknown"
        else:
            seen: set[str] = set()
            for job_id in job_ids:
                for part in str(job_id).split(","):
                    if part:
                        seen |= states.get(part.split("_", 1)[0], set())
            # seen passed as-is: empty means sacct ran and had no rows for these
            # jobs ("no record" — aged out, or too fresh), which is a different
            # answer from sacct being unreachable ("unknown") handled above.
            row["slurm_state"] = _coarse_run_state(seen)

    return rows


def _packaging_scopes_by_run() -> dict[str, set[str]]:
    """Which packaging scopes each run has used, from the job history.

    Needed because "did this run package the whole dataset?" is not answerable
    from slurm_dataset_runs alone. A run flagged is_subset normally produces a
    `_subset_N` .h5, but the same run packaged with scope="tiled" packages
    every slide with tiles on disk and gets the plain, unsuffixed name — a full
    dataset .h5 produced by a subset run. Only the recorded params say which
    happened.

    One query for the whole table: it holds a handful of rows per run, and the
    alternative is a lookup per dataset on an endpoint the UI polls.
    """
    scopes: dict[str, set[str]] = {}
    try:
        eng = _get_engine()
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT submission_id, params FROM slurm_dataset_run_jobs "
                    "WHERE stage = 'packaging'"
                )
            ).mappings().fetchall()
    except Exception as e:
        # Same degradation as _run_job_history: the table may not exist yet if
        # the code is deployed before migrate_dataset_run_jobs.sql is run. No
        # history means no scope overrides, which lands on is_subset alone —
        # conservative (a subset run's .h5 won't be claimed as the dataset's).
        print(f"[warn] could not read packaging scopes: {e}")
        return scopes

    for row in rows:
        params = row["params"]
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                continue
        if not isinstance(params, dict):
            continue
        scope = params.get("scope")
        if scope:
            scopes.setdefault(str(row["submission_id"]), set()).add(str(scope))
    return scopes


@app.get("/datasets")
def list_datasets(
    dataset_name: Optional[str] = Query(
        None,
        description="Return only this dataset. Required alongside raw_dir when "
                    "asking for coverage.",
    ),
    raw_dir: Optional[str] = Query(
        None,
        description="Return only the dataset tiled from this directory. Two runs "
                    "can share a dataset_name from different raw directories.",
    ),
    coverage: bool = Query(
        False,
        description="Also walk the filesystem for the authoritative count of how "
                    "many slides actually have tiles. Stats two files per slide "
                    "over cephfs, so this is a button, not a poll — and only "
                    "allowed for a single named dataset.",
    ),
):
    """Every dataset, with all of its runs rolled up into one pipeline state.

    This is the dataset-level counterpart to /dataset-jobs, which lists runs.
    The distinction matters because a resume does not continue a run, it starts
    a new one (see POST /dataset-jobs/{id}/resume) — so a dataset that took
    three resumes to tile is four rows there, none of which can say whether the
    dataset is finished. Here they are one entry with one answer and one
    next_action.

    Cost: one query for the runs, one for the packaging scopes, and a bounded
    Slurm lookup (see _listing_job_states) — squeue for everything, sacct only
    for recent jobs and only within a time budget. That is what makes it safe
    to poll. coverage=true is the exception and is deliberately restricted to a
    single dataset.
    """
    if coverage and not (dataset_name and raw_dir):
        raise HTTPException(
            400,
            "coverage=true needs both dataset_name and raw_dir — it walks the "
            "filesystem per slide, so it is not run across every dataset at once.",
        )

    eng = _get_engine()
    df = pd.read_sql(
        "SELECT * FROM slurm_dataset_runs ORDER BY submitted_at ASC", eng
    )
    rows = json.loads(df.to_json(orient="records", date_format="iso"))

    grouped = group_runs_by_dataset(rows)
    if dataset_name:
        grouped = [d for d in grouped if d["dataset_name"] == dataset_name]
    if raw_dir:
        wanted = raw_dir.rstrip("/")
        grouped = [d for d in grouped if d["raw_dir"].rstrip("/") == wanted]

    job_states, states_complete = _listing_job_states(grouped)

    scopes = _packaging_scopes_by_run()

    resolved = []
    for dataset in grouped:
        dataset_coverage = None
        if coverage:
            runs = dataset["runs"]
            # tile_dir is recorded per run and PROCESSED_TILES_DIR may have
            # moved since; the run's own value is what its tiles were written
            # under. Newest run wins, matching what a resume would use.
            tile_dir = Path(runs[-1]["tile_dir"]) if runs and runs[-1].get("tile_dir") else PROCESSED_TILES_DIR
            dataset_coverage = _tiling_readiness(
                Path(dataset["raw_dir"]), tile_dir, dataset["dataset_name"]
            )
        resolved.append(
            rollup_dataset(
                dataset["raw_dir"],
                dataset["dataset_name"],
                dataset["runs"],
                job_states=job_states,
                # One stat per artifact (not per slide), which is what keeps
                # this pollable. See _artifact_status for why existence at the
                # final path is trustworthy: packaging stages to `.partial`
                # and only os.replace()s on success.
                path_exists=lambda p: bool(p) and Path(p).is_file(),
                packaging_scopes=scopes,
                coverage=dataset_coverage,
            )
        )

    return {
        "datasets": resolved,
        "slurm_reachable": job_states is not None,
        # False means some jobs were answered by the live queue alone, so a
        # recent failure can be showing here as "no longer queued". Open the run
        # for the precise answer — /dataset-jobs/{id}/status asks accounting
        # about that one run and can afford to.
        "slurm_states_complete": states_complete,
    }


@app.get("/dataset-jobs/{submission_id}/status")
def dataset_job_status(submission_id: str):
    """Status for one dataset submission.

    While status is queued/discovering/error-with-no-job-yet, this just
    reflects the slurm_dataset_runs row — there's no Slurm job yet to ask
    sacct about. Once a job_id exists, this layers on live Slurm array state
    (via sacct) cross-referenced against each slide's actual
    tiling_summary.json, so a slide that "completed" with zero saved tiles
    shows up distinctly from one that genuinely succeeded. This is keyed off
    job_id existing, not off status == "submitted" — cancelling a run only
    flips its status column, it doesn't touch the manifest or the tiles
    already written to disk, so a cancelled run whose tiling batches had
    already finished should still report tiling_complete accurately (the UI
    uses this to offer test packaging/extraction against a cancelled run's
    already-tiled slides).
    """
    eng = _get_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM slurm_dataset_runs WHERE submission_id = :submission_id"),
            {"submission_id": submission_id},
        ).mappings().fetchone()

    if not row:
        raise HTTPException(404, f"No dataset job found with id {submission_id}")
    row = dict(row)

    if not row["job_id"] and row.get("status") != ANORAK_ONLY_STATUS:
        # No job_id on record doesn't prove tiling was never submitted — it's
        # also what a crash between submit_dataset_array's sbatch call(s) and
        # _run_dataset_submission's own follow-up DB write looks like (see
        # that function's _persist_plan). Recover by this submission's own
        # job name before assuming the row's stuck status column (e.g. still
        # "discovering") reflects reality.
        recovered = _find_job_ids_by_name_prefix(f"wsi_mask_tile_{submission_id}")
        if recovered:
            recovered_job_id = ",".join(recovered)
            _update_dataset_run(submission_id, status="submitted", job_id=recovered_job_id)
            row["job_id"] = recovered_job_id
            if row["status"] not in ("cancelled", "error"):
                row["status"] = "submitted"

    base = {
        "submission_id": submission_id,
        "status": row["status"],
        "error": row["error"],
        "raw_dir": row["raw_dir"],
        "job_id": row["job_id"],
        "total_slides": row["total_slides"],
        # Whether this run's manifest is itself a subset of the raw directory
        # (submitted with sample_size / slide_names) rather than every slide in
        # it. Exposed because the packaging step's "Full dataset" option means
        # "every slide in *this run's manifest*", which for a subset run is not
        # the full dataset at all — without this the UI had no way to say so,
        # and a run created as a 30-slide sample looked identical to one over
        # the whole directory right up until the .h5 came out short.
        "is_subset": bool(row["is_subset"]),
        # Needed by the UI to offer "run the whole directory" as a *new* run
        # that reuses this one's tile output folder — the already-tiled slides
        # are only skipped when the dataset name matches (the skip check in
        # submit_mask_tile_slurm.py is scoped to tile_dir/<dataset_name>/).
        "dataset_name": _row_dataset_name(row),
        "partition": row["partition"],
        "notify_email": row["notify_email"],
        # What this run tiled with, so a derived submission (resume, or the
        # packaging step's full-directory run) can reuse it instead of asking
        # the user to retype settings it has no way to verify. None means the
        # run predates the tiling_params column — the UI must say so rather
        # than presenting the current defaults as this run's settings.
        "tiling_params": _row_tiling_params(row),
        "h5_job_id": row["h5_job_id"],
        "h5_output_path": row["h5_output_path"],
        "extraction_job_id": row["extraction_job_id"],
        "extraction_output_path": row["extraction_output_path"],
        "extraction_checkpoint": row["extraction_checkpoint"],
    }

    # A pipeline run's Stages 1-4 are one Nextflow run; its stage columns hold
    # sentinels the helpers below resolve like any job id, and this block says
    # the rest — the head job, the per-stage markers, and why it stopped.
    base["pipeline"] = _pipeline_status(submission_id) if _is_pipeline_row(row) else None
    # What kind of run this is, for the UI: the one-click HPL pipeline, an
    # ANORAK run on its own, or a run from before either.
    base["run_kind"] = ("hpl_pipeline" if _is_pipeline_row(row)
                        else "anorak" if row.get("status") == ANORAK_ONLY_STATUS
                        else "hpl_legacy")

    if row["h5_job_id"]:
        # Three independent conditions, none sufficient alone: the file is
        # present (packaging stages to a ".partial" and only renames on
        # success, so this now genuinely means "a run finished"), the Slurm
        # job reached COMPLETED specifically (not merely a terminal state —
        # FAILED/CANCELLED/TIMEOUT are terminal and mean it did not work), and
        # the .h5 actually opens and reads back (see _validate_h5 — a file can
        # be complete by both of the first two measures and still be
        # unreadable after a bad disk or an interrupted copy).
        h5_path = Path(row["h5_output_path"]) if row["h5_output_path"] else None
        h5_state = _get_slurm_job_state(row["h5_job_id"])
        base["h5_ready"] = _job_output_ready(h5_path, h5_state, validator=_validate_h5)
        base["h5_slurm_state"] = h5_state
        # Surfaced so the UI can say *why* a COMPLETED packaging job still
        # isn't usable, rather than showing a silent "not ready" forever.
        if not base["h5_ready"] and h5_path and h5_path.is_file():
            base["h5_invalid_reason"] = _validate_h5(h5_path)[1] or None
        # Advisory, not a blocker: the run proceeds normally, but the KB load
        # at the end will need migrate_tile_names.py first.
        if base["h5_ready"] and h5_path:
            base["h5_legacy_tile_names"] = _h5_has_legacy_tile_names(h5_path)
        # Slurm-independent evidence that packaging is alive: the .partial is
        # being written to right now. This is the only thing that answers "did
        # my job actually start?" when sacct/squeue can't be reached — without
        # it, h5_slurm_state comes back None and a job that had been running
        # happily for an hour was reported as "interrupted (no Slurm record)".
        # One stat() on one file, so it's cheap enough for the 10s poll.
        if not base["h5_ready"] and h5_path:
            base.update(_packaging_write_activity(h5_path))

    # Test packaging, reported the same way as the real thing so the UI can
    # show a Slurm state ticking and then a finished .h5, rather than the
    # nothing it had once a reload discarded its session_state.
    #
    # row.get() rather than row[...]: this reads columns added by
    # migrate_dataset_runs_test_packaging.sql, and the code may well be deployed
    # before the migration is run. A missing column should degrade to "no test
    # job recorded", not 500 every status poll for every run.
    test_job_id = row.get("test_h5_job_id")
    if test_job_id:
        test_path = Path(row["test_h5_output_path"]) if row.get("test_h5_output_path") else None
        test_state = _get_slurm_job_state(test_job_id)
        base["test_h5_job_id"] = test_job_id
        base["test_h5_output_path"] = row.get("test_h5_output_path")
        base["test_h5_slurm_state"] = test_state
        # Same three-condition test as real packaging (present, COMPLETED,
        # actually readable) — a test .h5 that cannot be opened is no more
        # usable for a checkpoint trial than a real one.
        base["test_h5_ready"] = _job_output_ready(
            test_path, test_state, validator=_validate_h5
        )
        base["test_h5_params"] = _row_test_packaging_params(row)
        if not base["test_h5_ready"] and test_path and test_path.is_file():
            base["test_h5_invalid_reason"] = _validate_h5(test_path)[1] or None
        if not base["test_h5_ready"] and test_path:
            # Prefixed, so a test job's write activity can't be mistaken for the
            # real packaging job's in the same payload.
            activity = _packaging_write_activity(test_path)
            base.update({f"test_{k}": v for k, v in activity.items()})

    if row["extraction_job_id"]:
        ext_path = Path(row["extraction_output_path"]) if row["extraction_output_path"] else None
        ext_state = _get_slurm_job_state(row["extraction_job_id"])
        # Bind the input's tile count into the validator. Without it this only
        # checks the file is internally consistent, and a run that encoded a
        # fraction of the slides — an array job where most tasks died, a
        # sharded run merged from the shards that happened to finish — writes a
        # perfectly self-consistent file and reads as "features ready". That
        # is the whole failure mode this pipeline is written against: right
        # shape, right dtype, no missing values, a third of the data.
        expected = _extraction_expected_rows(row)
        base["extraction_ready"] = _job_output_ready(
            ext_path, ext_state,
            validator=lambda p: _validate_extraction_output(p, expected_rows=expected),
        )
        base["extraction_slurm_state"] = ext_state
        base["extraction_expected_rows"] = expected
        if not base["extraction_ready"] and ext_path and ext_path.is_file():
            # Mirrors h5_invalid_reason / test_h5_invalid_reason above: an
            # output that exists but doesn't validate is the single most
            # confusing state to show without a reason attached.
            base["extraction_invalid_reason"] = _validate_extraction_output(
                ext_path, expected_rows=expected
            )[1] or None

    if row.get("assignment_job_id"):
        asg_path = Path(row["assignment_output_path"]) if row.get("assignment_output_path") else None
        asg_state = _get_slurm_job_state(row["assignment_job_id"])
        base["assignment_ready"] = _job_output_ready(
            asg_path, asg_state, validator=_validate_assignment_output
        )
        base["assignment_slurm_state"] = asg_state
        base["assignment_output_path"] = row.get("assignment_output_path")
        base["assignment_reference"] = row.get("assignment_reference")
        # .get, so a deployment that has not applied
        # migrate_dataset_runs_assignment_vote.sql reports None rather than 500.
        base["assignment_vote"] = row.get("assignment_vote")
        if not base["assignment_ready"] and asg_path and asg_path.is_file():
            base["assignment_invalid_reason"] = _validate_assignment_output(asg_path)[1] or None

    # Stage 7, ANORAK. There is no done flag: this stage writes files rather
    # than committing to the Knowledge Bank, so "finished" is answerable by
    # reading the grading table it publishes, and a flag kept beside that table
    # could disagree with it. .get throughout, so a deployment without
    # migrate_dataset_runs_anorak.sql reports the stage as never run instead of
    # 500-ing.
    base.update(_anorak_status_fields(row))

    # Registration. Like Stage 5, in-process and all-or-nothing, so a single
    # boolean is the whole state (see migrate_dataset_runs_registration.sql).
    # .get throughout: a deployment that has not applied that migration reports
    # None and the UI shows the step as never run, rather than 500-ing.
    base["registration_done"] = bool(row.get("registration_done"))
    base["registration_at"] = (row["registration_at"].isoformat()
                               if row.get("registration_at") else None)
    base["registration_dataset_id"] = row.get("registration_dataset_id")
    # Defaults to production for runs registered before targets existed, which
    # is what they did.
    base["registration_kb_target"] = row.get("registration_kb_target") or KB_PRODUCTION
    base["registration_rows"] = row.get("registration_rows")
    # Since this stage can be submitted to Slurm, "done" is no longer the whole
    # state: a job can be queued, running, or finished-without-committing. The
    # boolean still means committed — the job sets it — and these say what is
    # happening when it is not set yet.
    base["registration_job_id"] = row.get("registration_job_id")
    base["registration_log_path"] = row.get("registration_log_path")
    base["registration_error"] = row.get("registration_error")
    base["registration_slurm_state"] = (
        _get_slurm_job_state(row["registration_job_id"])
        if row.get("registration_job_id") else None
    )
    # Registration reads tile identity out of the packaged .h5, so it is gated
    # on Stage 2 rather than on Stage 4 — it does not need an assignment, and
    # making it wait for one would keep Stage 5 blocked behind a step it could
    # have finished hours earlier.
    base["registration_ready"] = bool(base.get("h5_ready"))

    # Stage 6. kb_load_done still means committed and nothing else — whichever
    # way the load ran, in-process or as the Slurm job below.
    base["kb_load_done"] = bool(row.get("kb_load_done"))
    base["kb_load_at"] = row["kb_load_at"].isoformat() if row.get("kb_load_at") else None
    base["kb_load_rows"] = row.get("kb_load_rows")
    base["kb_load_reference"] = row.get("kb_load_reference")
    base["kb_load_kb_target"] = row.get("kb_load_kb_target") or KB_PRODUCTION
    base["kb_load_job_id"] = row.get("kb_load_job_id")
    base["kb_load_log_path"] = row.get("kb_load_log_path")
    base["kb_load_error"] = row.get("kb_load_error")
    base["kb_load_slurm_state"] = (
        _get_slurm_job_state(row["kb_load_job_id"])
        if row.get("kb_load_job_id") else None
    )
    base["kb_targets"] = sorted(KB_TARGETS)

    if not row["job_id"] or not row["manifest_path"]:
        return base

    manifest_path = Path(row["manifest_path"])
    tile_dir = Path(row["tile_dir"]) / _row_dataset_name(row)

    slide_paths: list[Path] = []
    if manifest_path.exists():
        slide_paths = [
            Path(line.strip())
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]

    # job_id is comma-joined when a large dataset was split across multiple
    # Slurm array batches — one combined sacct call across all of them
    # (task index 0 of batch 1 and task index 0 of batch 2 are different
    # slides, so per-index detail can't be merged, only summed counts per
    # state — which is all this endpoint actually needs).
    job_ids = [jid.strip() for jid in row["job_id"].split(",") if jid.strip()]
    raw_state_counts = _get_slurm_array_state_counts(job_ids)
    # None means sacct itself failed (missing/timed out) — genuinely
    # unknown, don't guess. {} means sacct ran fine and just has no rows
    # for these job IDs, which for an old run means they've aged out of
    # its accounting-DB retention window, not that they're still pending.
    sacct_unreachable = raw_state_counts is None
    slurm_state_counts = raw_state_counts or {}

    # Whether the "Start packaging" button should appear: every tiling batch
    # has reported a terminal sacct state (nothing still pending/running).
    tiling_complete = bool(slurm_state_counts) and not (
        set(slurm_state_counts) & IN_FLIGHT_SLURM_STATES
    )

    # Reading every slide's _tiling_summary.json is the expensive part of
    # this endpoint (up to one file per slide, over a network filesystem) —
    # the UI only ever displays succeeded/zero_tile/not_attempted once
    # tiling_complete is true, so skip it entirely while still tiling
    # instead of redoing it on every 10s auto-refresh for no visible
    # benefit. Once complete, cache the result forever (see
    # _tiling_breakdown_cache above) instead of re-scanning on every future
    # poll too — together these were the reason status checks kept timing
    # out (first at 30s, then even at 60s) for a 14,000+ slide run.
    #
    # Also run this scan as a fallback when slurm_state_counts came back
    # truly empty — meaning both sacct AND squeue (see
    # _slurm_jobs_live_states) confirmed nothing, not just sacct having a
    # rough moment or a fresh submission it hasn't indexed yet (squeue
    # would have caught that live). Not when sacct itself was unreachable
    # — that's still ambiguous, and doing a full disk scan on every 10s
    # poll of an actively-tiling 14,000-slide run whenever sacct has a
    # rough moment would reintroduce exactly the timeout problem above.
    # A real "nothing, confirmed by both" response means this job predates
    # sacct's retention window — every slide having its completion marker
    # on disk is what actually proves tiling finished in that case.
    succeeded, zero_tile, not_attempted = [], [], []
    need_breakdown = tiling_complete or (not slurm_state_counts and not sacct_unreachable)
    if need_breakdown:
        cached = _tiling_breakdown_cache.get(submission_id)
        if cached is not None:
            succeeded, zero_tile, not_attempted = cached
        else:
            for slide_path in slide_paths:
                slide_id = slide_id_from_raw_path(slide_path)
                summary_path = tile_dir / slide_id / f"{slide_id}_tiling_summary.json"
                if not summary_path.exists():
                    not_attempted.append(slide_id)
                    continue
                try:
                    summary = json.loads(summary_path.read_text())
                except (json.JSONDecodeError, OSError):
                    not_attempted.append(slide_id)
                    continue
                if summary.get("saved_tiles", 0) > 0:
                    succeeded.append(slide_id)
                else:
                    zero_tile.append(slide_id)

        if not tiling_complete and not slurm_state_counts and not sacct_unreachable and slide_paths:
            # Not conditioned on not_attempted being zero — a slide that
            # crashed without ever writing a summary file, or was one of a
            # handful the array genuinely never got to before being
            # cancelled, is still "tiling done" in the sense that matters
            # here: sacct having zero rows already proves this job isn't
            # live or pending anymore (see the comment above), so nothing
            # further is ever going to attempt these slides on its own.
            # Surfacing them as a count the user can see and choose to
            # ignore (the "Ignore N incomplete slide(s)" button below) is
            # correct; silently refusing to ever show a packaging option
            # at all just because a couple of slides never produced output
            # is not.
            tiling_complete = True

        # Cached forever once final — except a pipeline run whose tiling stage
        # ended any way but COMPLETED, which a pipeline resume can still finish.
        if tiling_complete and not (
            _is_pipeline_row(row) and slurm_state_counts != {"COMPLETED": 1}
        ):
            _tiling_breakdown_cache[submission_id] = (succeeded, zero_tile, not_attempted)

    base.update({
        "tiling_complete": tiling_complete,
        # Distinguishes "tiling is genuinely still running" from "we could not
        # ask Slurm". Both used to arrive at the UI as tiling_complete=False,
        # which it rendered as "waiting on tiling" — a positive claim the
        # server had no evidence for, and one that left packaging blocked with
        # no way forward for as long as sacct stayed unreachable.
        "slurm_unreachable": sacct_unreachable,
        "slurm_state_counts": slurm_state_counts,
        "attempted": len(succeeded) + len(zero_tile),
        "succeeded": len(succeeded),
        "zero_tile_slides": zero_tile,
        "not_yet_attempted": len(not_attempted),
    })
    return base


@app.get("/slide/{slide_id}/info")
def slide_info(slide_id: str, kb_target: str = Depends(kb_target_param)):
    slide = _open_slide(slide_id, kb_target)
    dims = slide.level_dimensions
    # Per slide, not per deployment: every overlay in both UIs sizes its boxes
    # from this one number, and the tile grid it has to line up with was cut at
    # a pitch derived from this slide's own mpp.
    tile_size_native, pitch_source = _tile_size_native(
        slide_id, kb_target, slide=slide)
    return {
        "slide_id": slide_id.upper(),
        "level_count": slide.level_count,
        "level_dimensions": [{"width": w, "height": h} for w, h in dims],
        "mpp_x": slide.properties.get("openslide.mpp-x"),
        "mpp_y": slide.properties.get("openslide.mpp-y"),
        "vendor": slide.properties.get("openslide.vendor"),
        "objective_power": slide.properties.get("openslide.objective-power"),
        "tile_size_5x": TILE_SIZE_5X,
        "scale_5x_to_native": tile_size_native / TILE_SIZE_5X,
        "tile_size_native": tile_size_native,
        # Named so a misdrawn grid is a question with an answer. "default mpp"
        # means neither the tiles nor the slide said, and the grid is a guess.
        "tile_size_native_source": pitch_source,
    }



@app.get("/dzi/{slide_id}.dzi")
def dzi_metadata(slide_id: str, kb_target: str = Depends(kb_target_param)):
    dz = _get_deepzoom(slide_id, kb_target)
    dzi_xml = dz.get_dzi("jpeg")

    return StreamingResponse(
        io.BytesIO(dzi_xml.encode("utf-8")),
        media_type="application/xml",
    )


@app.get("/dzi/{slide_id}_files/{level}/{col}_{row}.jpeg")
def dzi_tile(
    slide_id: str,
    level: int,
    col: int,
    row: int,
    quality: int = Query(90, ge=10, le=100),
    kb_target: str = Depends(kb_target_param),
):
    dz = _get_deepzoom(slide_id, kb_target)

    try:
        tile = dz.get_tile(level, (col, row)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            404,
            f"Could not read DZI tile level={level}, col={col}, row={row}: {e}",
        )

    return _jpeg_response(_img_to_jpeg_bytes(tile, quality=quality))

@app.get("/slide/{slide_id}/thumbnail")
def slide_thumbnail(
    slide_id: str,
    max_width: int = Query(3000, ge=100, le=8000),
    quality: int = Query(85, ge=10, le=100),
    kb_target: str = Depends(kb_target_param),
):
    slide_id = slide_id.strip().upper()
    ckey = _cache_key(slide_id, kb_target)
    cached = cache.get(ckey, kind="thumbnail", level=None, x=None, y=None,
                       width=max_width, height=None)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    slide = _open_slide(slide_id, kb_target)
    w0, h0 = slide.level_dimensions[0]
    thumb_height = int(h0 * (max_width / w0))
    thumb = slide.get_thumbnail((max_width, thumb_height)).convert("RGB")
    cache.put(thumb, ckey, quality=quality, kind="thumbnail", level=None,
              x=None, y=None, width=max_width, height=None)
    return _jpeg_response(_img_to_jpeg_bytes(thumb, quality))


@app.get("/slide/{slide_id}/tile")
def slide_tile(
    slide_id: str,
    level: int = Query(0, ge=0),
    x: int = Query(..., ge=0),
    y: int = Query(..., ge=0),
    w: int = Query(256, ge=64, le=2048),
    h: int = Query(256, ge=64, le=2048),
    quality: int = Query(85, ge=10, le=100),
    kb_target: str = Depends(kb_target_param),
):
    """Read a tile at (x, y) in *level* coordinates, return JPEG."""
    slide_id = slide_id.strip().upper()
    ckey = _cache_key(slide_id, kb_target)
    cached = cache.get(ckey, kind="tile", level=level, x=x, y=y, width=w, height=h)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    slide = _open_slide(slide_id, kb_target)
    if level >= slide.level_count:
        raise HTTPException(400, f"Level {level} out of range (max {slide.level_count - 1})")

    ds = slide.level_downsamples[level]
    origin_x = int(x * ds)
    origin_y = int(y * ds)
    region = slide.read_region((origin_x, origin_y), level, (w, h)).convert("RGB")
    cache.put(region, ckey, quality=quality, kind="tile", level=level,
              x=x, y=y, width=w, height=h)
    return _jpeg_response(_img_to_jpeg_bytes(region, quality))


@app.get("/slide/{slide_id}/region")
def slide_region(
    slide_id: str,
    x: int = Query(..., description="Native-coordinate X"),
    y: int = Query(..., description="Native-coordinate Y"),
    w: int = Query(TILE_SIZE_NATIVE),
    h: int = Query(TILE_SIZE_NATIVE),
    level: int = Query(0, ge=0),
    quality: int = Query(85),
    kb_target: str = Depends(kb_target_param),
):
    """Read an arbitrary region in native (level-0) coordinates."""
    slide_id = slide_id.strip().upper()
    ckey = _cache_key(slide_id, kb_target)
    cached = cache.get(ckey, kind="region", level=level, x=x, y=y, width=w, height=h)
    if cached:
        return _jpeg_response(_img_to_jpeg_bytes(cached, quality))

    # kb_target, not the default: the cache key above is already per target, so
    # without it this endpoint reads test's cache and production's pixels — and
    # for a slide registered only in test, 404s on a request the rest of the
    # viewer answered.
    slide = _open_slide(slide_id, kb_target)
    ds = slide.level_downsamples[level]
    read_w = int(w / ds)
    read_h = int(h / ds)
    region = slide.read_region((x, y), level, (read_w, read_h)).convert("RGB")
    cache.put(region, ckey, quality=quality, kind="region", level=level,
              x=x, y=y, width=w, height=h)
    return _jpeg_response(_img_to_jpeg_bytes(region, quality))


@app.get("/slide/{slide_id}/tiles_meta")
def slide_tiles_meta(slide_id: str, kb_target: str = Depends(kb_target_param)):
    """Return tile coordinates + HPC labels + heatmap probs for a slide (JSON)."""
    slide_id = slide_id.strip().upper()
    eng = _get_engine(kb_target)
    q = text("""
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
        -- UPPER(TRIM(...)), not UPPER(...): migrate_indexes.sql §8 indexes
        -- UPPER(TRIM(slides)) and the planner matches expressions rather
        -- than values, so the bare UPPER() here could not use the index and
        -- scanned all of tile_coordinates — 18.5M rows once Radiogenomics
        -- was registered, which is a viewer that times out at 30s.
        WHERE UPPER(TRIM(tc.slides)) = :slide_id
    """)
    df = pd.read_sql(q, eng, params={"slide_id": slide_id})
    df.columns = df.columns.astype(str).str.strip()

    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()

    # Merge heatmap probs — this target's, not the process's. Merging
    # production's numbers into a test cohort's tiles would render an overlay
    # for tiles they were never computed for.
    probs = _heatmap_probs_for_tiles(
        kb_target, df["slide_tile"].tolist() if "slide_tile" in df.columns else [])
    if probs is not None and "slide_tile" in df.columns:
        df = df.merge(probs, on="slide_tile", how="left")

    # Replace NaN with None for JSON
    df = df.where(df.notna(), None)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/slide/{slide_id}/adjacency")
def slide_adjacency(slide_id: str, kb_target: str = Depends(kb_target_param)):
    slide_id = slide_id.strip().upper()
    eng = _get_engine(kb_target)
    q = text("""
        SELECT tc.slide_tile, tc."col", tc."row", tc.x_native, tc.y_native,
               tr.hpc_id
        FROM tile_coordinates tc
        LEFT JOIN tile_registry tr ON UPPER(tr.slide_tile) = UPPER(tc.slide_tile)
        -- UPPER(TRIM(...)), not UPPER(...): migrate_indexes.sql §8 indexes
        -- UPPER(TRIM(slides)) and the planner matches expressions rather
        -- than values, so the bare UPPER() here could not use the index and
        -- scanned all of tile_coordinates — 18.5M rows once Radiogenomics
        -- was registered, which is a viewer that times out at 30s.
        WHERE UPPER(TRIM(tc.slides)) = :slide_id
    """)
    df = pd.read_sql(q, eng, params={"slide_id": slide_id})
    if df.empty:
        return {"pair_edge_counts": {}, "tile_neighbor_pairs": {}}
    pair_counts, tile_pairs = _compute_adjacency(
        df, _tile_size_native(slide_id, kb_target)[0])
    return {"pair_edge_counts": pair_counts, "tile_neighbor_pairs": tile_pairs}


@app.get("/hpc/{hpc_id}/info")
def hpc_info(hpc_id: int, kb_target: str = Depends(kb_target_param)):
    eng = _get_engine(kb_target)
    with eng.connect() as conn:
        base = conn.execute(
            text("SELECT * FROM hpc_dictionary WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()
        if not base:
            raise HTTPException(404, f"HPC {hpc_id} not found")
        result = dict(base._mapping)

        mal = conn.execute(
            text("SELECT * FROM hpc_malignant_details WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()
        non = conn.execute(
            text("SELECT * FROM hpc_non_malignant_details WHERE hpc_id = :h LIMIT 1"), {"h": str(hpc_id)}
        ).fetchone()

    result["malignant_details"] = dict(mal._mapping) if mal else None
    result["non_malignant_details"] = dict(non._mapping) if non else None
    return result


@app.get("/hpc/{hpc_id}/survival")
def hpc_survival(hpc_id: int, kb_target: str = Depends(kb_target_param)):
    eng = _get_engine(kb_target)
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM hpc_survival_analysis WHERE hpc_id = :h LIMIT 1"),
            {"h": str(hpc_id)}
        ).fetchone()
    if not row:
        raise HTTPException(404, f"No survival data for HPC {hpc_id}")
    return dict(row._mapping)


# The image dataset's name inside a packaged .h5. Ours is "img"
# (make_hpl_hdf5.py:631); the legacy TCGA file this endpoint was written
# against uses "train_img", HPL's own convention. Tried in order rather than
# hardcoded, because hardcoding either one makes the other cohort 500.
_H5_IMAGE_DATASETS = ("img", "train_img")


@app.get("/tile_image/{slide_tile}")
def tile_image_by_key(slide_tile: str, quality: int = Query(85),
                      kb_target: str = Depends(kb_target_param)):
    """Return the H5-backed tile image for a slide_tile key.

    The .h5 comes from the tile's own row, not from the module-level H5_PATH.
    That constant is one TCGA file, and it was being used for every slide of
    every cohort — so a Radiogenomics tile did not fail, it rendered whatever
    TCGA tile happened to sit at the same index. Registration records
    h5_source_path per tile precisely so this is answerable from the data.
    """
    slide_tile = slide_tile.strip().upper()
    eng = _get_engine(kb_target)
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT image_index, h5_source_path FROM tile_registry "
                 "WHERE UPPER(slide_tile) = :st LIMIT 1"),
            {"st": slide_tile}
        ).fetchone()
    if not row or row.image_index is None:
        raise HTTPException(404, f"No H5 index for {slide_tile}")

    idx = int(row.image_index)
    # Falls back to H5_PATH only for rows that predate registration — the
    # hand-loaded TCGA cohort has no h5_source_path of its own.
    f = _get_h5(row.h5_source_path)
    ds = None
    for name in _H5_IMAGE_DATASETS:
        if name in f:
            ds = f[name]
            break
    if ds is None:
        raise HTTPException(
            500, f"{f.filename} has none of {list(_H5_IMAGE_DATASETS)} — "
                 f"not a packaged tile .h5.")
    if idx < 0 or idx >= ds.shape[0]:
        raise HTTPException(400, f"Index {idx} out of range")

    arr = _to_uint8(ds[idx])
    from PIL import Image as PILImage
    if arr.ndim == 2:
        img = PILImage.fromarray(arr)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        img = PILImage.fromarray(arr if arr.shape[-1] != 1 else arr[:, :, 0])
    else:
        raise HTTPException(500, f"Unexpected tile shape {arr.shape}")
    return _jpeg_response(_img_to_jpeg_bytes(img, quality))


def _hpc_reference_maps(kb_target: str) -> tuple[dict[int, str], set[int]]:
    """(hpc_id -> hpc_title, {every hpc_id in the dictionary}) for this target.

    build_query_plan_v25 needs both: hpc_title_map for fuzzy title→id matching
    ("the tumour budding cluster" -> HPC 40), valid_hpc_ids so
    validate_plan_entities can drop an HPC number the user typed that the
    dictionary has never heard of, rather than pass it on to a DB query that
    will not know it either. Small (71 rows) and static enough that this
    endpoint reads it fresh per request rather than caching it — Streamlit's
    own version (load_hpc_titles) does cache, but with an ordinary 300s TTL,
    not never; here it is one cheap query, not a network round trip.
    """
    eng = _get_engine(kb_target)
    with eng.connect() as conn:
        rows = conn.execute(
            text("SELECT hpc_id, hpc_title FROM hpc_dictionary ORDER BY hpc_id")
        ).fetchall()
    title_map: dict[int, str] = {}
    valid_ids: set[int] = set()
    for row in rows:
        try:
            hid = int(row.hpc_id)
        except (TypeError, ValueError):
            continue
        valid_ids.add(hid)
        title = str(row.hpc_title or "").strip()
        if title:
            title_map[hid] = title
    return title_map, valid_ids


class QueryRequest(BaseModel):
    query: str
    slide_id: Optional[str] = None
    # Defaults to production for the same reason every other kb_target does:
    # an older client that does not send it keeps working, and forgetting it
    # is the harmless case.
    kb_target: str = KB_PRODUCTION
    # The last few chat turns, {"role": "user"|"assistant", "content": str} —
    # explain_answer() reads up to 6 of them for follow-up questions ("what
    # about HPC 12 instead?"). Optional: a caller with no history yet, or one
    # that doesn't track it, gets the same answer minus that context.
    history: Optional[list[dict]] = None
    # Mirrors app_v28.py's session_context dict (active_slide, viewer_open,
    # selected_hpc, highlight_mode) — passed to the planner so "this slide" /
    # "that HPC" can resolve against whatever the client currently has open.
    # The server has no session of its own to read this from; the client's
    # own UI state is the only place it exists.
    session_context: Optional[dict] = None


@app.post("/query")
def handle_query(req: QueryRequest):
    """Full NL query pipeline, server-side — completing the move this
    endpoint's docstring has described since it was a stub: "the Streamlit
    client still runs fetch_answer_from_db locally... in Phase 2 you move
    that logic here too." This is Phase 2.

    Reuses app/query_planner_v25.py, nlp_enrich_v25.py and llm_layer_v25.py
    directly (sys.path trick, mirroring app_v28.py's own reach into backend/
    for malignancy.py/db_url.py) rather than re-implementing query planning —
    all three are pure functions with no Streamlit dependency, confirmed by
    reading them, so importing them here carries no risk of relying on
    Streamlit's script-run context outside one. detect_entity_patterns is the
    same story: it's imported from hpc_chat_handlers_v23 (which does import
    streamlit at module level — that import alone is harmless, only *calling*
    st.* outside a run is not) because the function itself makes no st.* call.

    The DB-query handlers (handle_tile/handle_slide/handle_hpc/...) are NOT
    reused the same way: those interleave SQL with real st.image/
    st.session_state calls, which is exactly the undefined-outside-Streamlit
    behaviour this file exists to avoid. backend/chat_answers.py is a
    hand-kept-in-sync twin of just their SQL and text formatting — see that
    file's own docstring for what that tradeoff costs.

    llm_layer_v25 calls Ollama (`import ollama`, no API key involved) for the
    planner and the explanation step; both already degrade gracefully to a
    regex-only plan / the raw structured answer if Ollama is not reachable
    from wherever this server runs, so an unreachable Ollama makes chat less
    fluent, not broken. Both imports are deferred to request time — a
    dependency and a network service this one endpoint needs, not the whole
    tile server.
    """
    target = _resolve_kb_target(req.kb_target)
    eng = _get_engine(target)

    app_dir = str(Path(__file__).resolve().parent.parent / "app")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from query_planner_v25 import build_query_plan_v25
    from llm_layer_v25 import explain_answer, should_fetch_from_db
    from hpc_chat_handlers_v23 import detect_entity_patterns

    import chat_answers

    slide_list = sorted(_get_wsi_map(target).keys())
    hpc_title_map, valid_hpc_ids = _hpc_reference_maps(target)

    plan = build_query_plan_v25(
        req.query,
        slide_list=slide_list,
        hpc_title_map=hpc_title_map,
        valid_hpc_ids=valid_hpc_ids,
        session_context=req.session_context,
    )

    structured_answer = None
    tile_images: list[dict] = []
    if should_fetch_from_db(plan):
        query_for_db = plan.get("enriched_query") or req.query
        try:
            polarity = chat_answers.classify_malignancy_polarity(query_for_db)
            detected = detect_entity_patterns(query_for_db)
            structured_answer, tile_images = chat_answers.fetch_answer_from_db(
                query_for_db, eng, detected, polarity,
            )
        except Exception as e:
            structured_answer = f"⚠️ DB query failed: {e}"

    try:
        final_answer = explain_answer(
            plan, structured_answer, user_query=req.query, history=req.history or [],
        )
    except Exception:
        final_answer = structured_answer or "Hi! How can I help you?"

    return {
        "plan": plan,
        "slide_id": req.slide_id,
        "kb_target": target,
        "structured_answer": structured_answer,
        "final_answer": final_answer,
        "evidence": {"tile_images": tile_images},
    }


# ---------------------------------------------------------------------------
# Run with:  uvicorn <this file's name>:app --host 0.0.0.0 --port 8000 --workers 2
# or just:   python <this file>.py   (which derives the module name itself)
# For local dev with autoreload (single worker only — Uvicorn doesn't
# support reload + multiple workers):  UVICORN_RELOAD=true python <this file>.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    # Derived from this file's own name, never written out.
    #
    # Uvicorn re-imports the app from this string in a subprocess, for reload
    # and for every worker, so the string has to name whatever file is actually
    # being run. Hardcoding it has now failed twice in different ways: first as
    # "tile_server:app", which pointed at an older file with none of the
    # upload/dataset-job/Slurm endpoints, so multi-worker runs silently served
    # stale code; then as "tile_server_v2_:app" on a deployment where the file
    # had been renamed, where every worker died on ImportError and the parent
    # respawned it in a loop that printed nothing but "Could not import module".
    #
    # Path(__file__).stem cannot disagree with the file it is in, which is the
    # only property that matters here.
    module_name = Path(__file__).stem
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run(
        f"{module_name}:app",
        host="0.0.0.0",
        port=8000,
        workers=1 if reload else 2,
        reload=reload,
    )
