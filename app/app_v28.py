import streamlit as st
import pandas as pd
from pathlib import Path
from sqlalchemy import bindparam, create_engine, inspect, text
import re
import html
import numpy as np
import io
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
# from streamlit_image_zoom import image_zoom
import streamlit.components.v1 as components
import colorsys
import hashlib
from query_planner_v25 import build_query_plan_v25
from plan_query import save_query_plan_to_file
from llm_layer_v25 import explain_answer, llm_enabled, llm_planner_enabled, should_fetch_from_db
from ui_actions_v25 import ViewerContext, apply_plan_to_session
import os
import time
import json
import requests
from urllib.parse import quote
from api_client import TileServerClient
from hpc_chat_handlers_v23 import detect_entity_patterns as chat_detect_entity_patterns
from hpc_chat_handlers_v23 import fetch_answer_from_db as chat_fetch_answer_from_db

detect_entity_patterns = chat_detect_entity_patterns

# backend/ holds the definitions the pipeline and the UI have to agree on. This
# one is what hpc_dictionary.malignant means — a loosely typed column whose type
# kb_live_schema_2026-08-26.txt never captured — and it is now read by
# backend/select_tumour_slides.py to decide which slides ANORAK runs on at all.
# Imported rather than reimplemented because this file already held three
# normalisations of that column and they did not agree: color_for_malignant()
# below did not recognise "malignant"/"non-malignant", so a dictionary row
# spelled that way rendered grey in the viewer while the tile filter counted it
# correctly.
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from malignancy import describe_malignant, malignant_flag  # noqa: E402
from db_url import database_url, safe_text  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="HPC Chatbot", layout="wide")
st.title("HPC Chatbot")


# ---------------------------------------------------------------------------
# Tile server client (talks to FastAPI on HPCC)
# ---------------------------------------------------------------------------
TILE_SERVER_URL = os.getenv("TILE_SERVER_URL", "http://localhost:8000")

# The same server, as the *browser* has to reach it — which is not always the
# same address this process reaches it at. Nearly every call here is made
# server-side by requests, but OpenSeadragon's are made by the viewer's browser,
# so over an SSH tunnel the two differ: this process talks to localhost:8000 on
# the machine Streamlit runs on, while the laptop has the tile server forwarded
# to some other local port (localhost:8001, the same way the database is on
# 5433 rather than 5432). Pointing TILE_SERVER_URL at the browser's port would
# break every server-side call; leaving the viewer on this one leaves it
# fetching a port nothing is listening on, and the only symptom is
# OpenSeadragon's "failed to open the DZI source".
#
# Defaults to TILE_SERVER_URL, so a deployment where both sides see the same
# address needs no configuration:
#     export TILE_SERVER_BROWSER_URL=http://localhost:8001
TILE_SERVER_BROWSER_URL = os.getenv("TILE_SERVER_BROWSER_URL", TILE_SERVER_URL)

# ---------------------------------------------------------------------------
# Knowledge Bank target
# ---------------------------------------------------------------------------
# Chosen before anything else is built, because both the API client and the
# direct SQLAlchemy engine below have to agree on it. If the selector were
# rendered further down the sidebar, the engine would already have been created
# against whatever the previous rerun chose, and the viewer would read one
# database while the chatbot read the other — with nothing on screen to say so.
KB_PRODUCTION = "production"
KB_TEST = "test"
KB_DATABASES = {KB_PRODUCTION: os.getenv("HPL_DB_NAME", "hpl_kb"),
                KB_TEST: os.getenv("HPL_DB_NAME_TEST", "hpl_kb_test")}

_kb_labels = {KB_PRODUCTION: f"Production — {KB_DATABASES[KB_PRODUCTION]}",
              KB_TEST: f"Test — {KB_DATABASES[KB_TEST]}"}

kb_target = st.sidebar.radio(
    "Knowledge Bank",
    [KB_PRODUCTION, KB_TEST],
    format_func=lambda t: _kb_labels[t],
    key="kb_target",
    help="Everything downstream follows this: registration and the cluster-"
         "assignment load write here, and the slide viewer, HPC panels and "
         "chatbot read from here. Pipeline execution and run history always "
         "stay in production — a run is one run regardless of which Knowledge "
         "Bank it filled.",
)
# Every @st.cache_data reader in this file is keyed on its arguments, and most
# of them take none — load_hpc_titles(), load_slide_list(), load_valid_hpc_ids()
# and the rest would happily serve production's rows for the whole 300s ttl
# after a switch to test. Clearing on change is one line and covers readers
# added later; adding a kb_target argument to each would have to be remembered
# eight times and then again for the ninth.
if st.session_state.get("_kb_target_active") not in (None, kb_target):
    st.cache_data.clear()
st.session_state["_kb_target_active"] = kb_target

if kb_target == KB_TEST:
    st.sidebar.warning(
        f"Reading and writing **{KB_DATABASES[KB_TEST]}**. Production is untouched."
    )

client = TileServerClient(TILE_SERVER_URL, kb_target=kb_target)


def _post_upload_slide(file_bytes: bytes, file_name: str, file_type: str,
                        slide_id: str, confirm_overwrite: bool) -> dict:
    """POST to /upload-slide and classify the outcome.

    Returns {"kind": "success", "payload": ...} on 200, {"kind":
    "needs_confirm", "message": ...} on a 409 the caller can resolve by
    resubmitting with confirm_overwrite=True (slide_id reused from a
    previous upload), or {"kind": "error", "message": ...} for anything
    else (network failure, timeout, a 409 the user can't just confirm
    through — e.g. slide_id collides with a real dataset slide — or any
    other non-2xx status).
    """
    try:
        response = requests.post(
            f"{TILE_SERVER_URL}/upload-slide",
            files={"file": (file_name, file_bytes, file_type or "application/octet-stream")},
            data={"slide_id": slide_id, "confirm_overwrite": "true" if confirm_overwrite else "false"},
            timeout=600,
        )
    except requests.exceptions.ConnectionError:
        return {"kind": "error", "message": f"Could not connect to tile server at {TILE_SERVER_URL}."}
    except requests.exceptions.Timeout:
        return {"kind": "error", "message": "Upload timed out. Try a smaller test file first, or increase the backend/proxy timeout."}
    except Exception as e:
        return {"kind": "error", "message": f"Upload failed: {e}"}

    if response.status_code == 200:
        return {"kind": "success", "payload": response.json()}

    if response.status_code == 409:
        detail = {}
        try:
            detail = response.json().get("detail") or {}
        except Exception:
            pass
        message = (detail.get("message") if isinstance(detail, dict) else None) or response.text
        if isinstance(detail, dict) and detail.get("error") == "slide_id_exists":
            return {"kind": "needs_confirm", "message": message}
        return {"kind": "error", "message": message}

    return {"kind": "error", "message": f"Upload failed with status {response.status_code}: {response.text}"}


def _handle_upload_result(result: dict):
    if result["kind"] == "success":
        payload = result["payload"]
        st.success("Slide uploaded successfully.")
        st.json(payload)

        returned_slide_id = str(payload.get("slide_id") or "").strip().upper()
        if returned_slide_id:
            st.session_state.active_slide = returned_slide_id
            st.session_state.viewer_open = False
            st.session_state.processing_slide_id = returned_slide_id

        st.session_state.pop("wsi_upload_pending_file", None)
        st.session_state.pop("wsi_upload_conflict", None)

    elif result["kind"] == "needs_confirm":
        # Stash the message; the pending file bytes are already in
        # wsi_upload_pending_file from the triggering submit below. Actually
        # rendering the warning + confirm/cancel buttons happens further
        # down, after this function returns, so it shows up in this same
        # script run rather than needing another rerun first.
        st.session_state.wsi_upload_conflict = {"message": result["message"]}

    else:
        st.error(result["message"])
        st.session_state.pop("wsi_upload_pending_file", None)
        st.session_state.pop("wsi_upload_conflict", None)


def render_wsi_upload_panel():
    """Upload a new WSI to the FastAPI backend.

    The backend queues tissue masking + tiling as a background job right
    after saving the file; this panel polls /processing-status for progress.

    Re-using a slide_id from a previous upload is allowed, but the backend
    now hard-stops the first attempt (409 slide_id_exists) rather than
    silently overwriting that upload's file/mask/tiles/.h5 — this renders
    that as a warning with an explicit "Yes, overwrite" action instead of
    just letting the overwrite happen.
    """
    with st.sidebar.expander("Upload new WSI", expanded=False):
        st.caption("Step 1: Click 'Browse files' and select a WSI from your computer")

        uploaded_file = st.file_uploader(
            "Browse for WSI file",
            type=["svs", "ndpi", "tif", "tiff", "isyntax"],
            key="wsi_upload_file",
        )

        user_slide_id = st.text_input(
            "Slide ID optional",
            placeholder="Example: TCGA-XX-XXXX-DX1",
            key="wsi_upload_slide_id",
        )

        if st.button("Send selected file to server", key="wsi_upload_submit", use_container_width=True):
            if uploaded_file is None:
                st.error("Please choose a WSI file first.")
            else:
                pending = {
                    "file_bytes": uploaded_file.getvalue(),
                    "file_name": uploaded_file.name,
                    "file_type": uploaded_file.type or "application/octet-stream",
                    "slide_id": (user_slide_id or "").strip(),
                }
                st.session_state.wsi_upload_pending_file = pending
                _handle_upload_result(_post_upload_slide(**pending, confirm_overwrite=False))

        conflict = st.session_state.get("wsi_upload_conflict")
        if conflict:
            st.warning(conflict["message"])
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Yes, overwrite", key="wsi_upload_confirm_overwrite", use_container_width=True):
                    pending = st.session_state.get("wsi_upload_pending_file")
                    if pending:
                        # No st.rerun() here — _handle_upload_result already
                        # renders the success/error message inline within
                        # this same click-triggered run; rerunning would
                        # immediately wipe it before the user ever sees it.
                        _handle_upload_result(_post_upload_slide(**pending, confirm_overwrite=True))
                    else:
                        st.error("Original file is no longer available — please re-select it and try again.")
                        st.session_state.pop("wsi_upload_conflict", None)
            with cancel_col:
                if st.button("Cancel", key="wsi_upload_cancel", use_container_width=True):
                    st.session_state.pop("wsi_upload_conflict", None)
                    st.session_state.pop("wsi_upload_pending_file", None)
                    st.rerun()

        processing_slide_id = st.session_state.get("processing_slide_id")
        if processing_slide_id:
            st.divider()
            st.caption(f"Background processing: **{processing_slide_id}**")

            try:
                st.image(client.get_thumbnail(processing_slide_id, max_width=600))
            except Exception:
                pass  # slide isn't openable yet or thumbnail generation hiccuped — not worth blocking on

            status_payload = None
            try:
                status_payload = client.get_processing_status(processing_slide_id)
            except Exception as e:
                st.warning(f"Could not fetch processing status: {e}")

            stage = (status_payload or {}).get("status", "unknown")
            if stage == "done":
                st.success("Tissue masking + tiling complete. Tiles are ready.")
                packaging_error = (status_payload or {}).get("error")
                if packaging_error:
                    st.warning(packaging_error)
                if st.button("Dismiss", key="wsi_processing_dismiss_done"):
                    del st.session_state["processing_slide_id"]
            elif stage == "error":
                st.error(f"Processing failed: {(status_payload or {}).get('error')}")
                if st.button("Dismiss", key="wsi_processing_dismiss_error"):
                    del st.session_state["processing_slide_id"]
            else:
                st.info(f"Status: **{stage}** (queued → masking → tiling → packaging → done)")
                st.button("Refresh status", key="wsi_processing_refresh")

            _render_upload_pipeline(processing_slide_id, status_payload or {})


def _render_upload_pipeline(slide_id: str, status_payload: dict):
    """The rest of the pipeline for an uploaded slide: Stages 3-7, rendered by
    the same stepper a dataset run uses.

    An upload used to stop at its .h5. Masking, tiling and packaging all ran
    and then nothing else could: feature extraction, cluster classification,
    registration and the Knowledge Bank load are keyed on a run, and an upload
    had none — so an uploaded slide could be looked at, but never carried any
    HPC labels, never appeared in hpl_profile_*, and the chatbot could not
    answer a single question about it. The backend now creates a one-slide run
    for every upload (see _start_upload_run in tile_server_v2_.py), and this is
    where the user drives it: the same seven steps, with Stages 1 and 2 already
    finished by the upload itself.
    """
    submission_id = status_payload.get("submission_id")
    if not submission_id:
        st.warning(
            "This upload has no pipeline run on record, so it stops at the "
            "tiles. Re-upload the slide to create one; without it the slide "
            "can be viewed but cannot reach the Knowledge Bank."
        )
        return

    st.divider()
    st.caption(
        f"**Rest of the pipeline for {slide_id}** — cohort "
        f"`{status_payload.get('dataset_name') or ''}`. Steps 1 and 2 were done "
        f"by the upload itself; run 3 onwards here to get this slide's tiles "
        f"into the Knowledge Bank and its HPC overlay into the viewer."
    )
    # Registration always asks for Replace on an uploaded slide, and the
    # refusal it comes from names a row count rather than a reason. Said here,
    # next to the step, because the step itself is shared with cohorts where
    # the same message means something else entirely.
    st.caption(
        "Step 5 will report this cohort as already occupied and ask for "
        "**Replace** — that is the slide's own registry row, written at upload "
        "time so the viewer could open it straight away. Replace rewrites "
        "exactly this slide's rows and touches no other cohort."
    )
    _render_job_progress(
        {"submission_id": submission_id, "total_slides": 1, "submitted_at": ""},
        key_prefix=f"upload_{slide_id}_",
    )


def _job_label(job: dict) -> str:
    """Short human label for a job row: batch count if array-split, the
    single Slurm job ID if not, or a submission-id fallback before any
    Slurm IDs exist yet (still discovering)."""
    job_id_field = job.get("job_id") or ""
    job_id_list = [j for j in job_id_field.split(",") if j]
    if len(job_id_list) > 1:
        return f"{len(job_id_list)} batches"
    if job_id_list:
        return job_id_list[0]
    return f"submission {job.get('submission_id', '')[:8]}"


# A run's state is written out as a word wherever it is shown; there is no
# icon for it. (These were coloured-circle emoji, which the UI no longer uses.)
_RUN_STATE_ICONS: dict[str, str] = {}


_STAGE_LABELS = {
    "tiling": "Tiling",
    "packaging": "Packaging",
    "packaging_test": "Packaging (test)",
    "extraction": "Feature extraction",
    "extraction_test": "Feature extraction (test)",
    "assignment": "Cluster assignment",
    "assignment_test": "Cluster assignment (test)",
}


def _describe_job_params(stage: str, params: dict | None) -> str:
    """The one line that tells two attempts at the same stage apart.

    Without this a run with three test packagings shows three rows differing
    only by job id, which is not the thing anyone is trying to distinguish.
    """
    if not params:
        return ""
    bits = []
    if stage == "tiling":
        if params.get("slides") is not None:
            bits.append(f"{int(params['slides']):,} slides")
        if params.get("batches"):
            bits.append(f"{params['batches']} batches")
        if params.get("failed_batches"):
            bits.append(f"{params['failed_batches']} batch(es) failed to submit")
    elif stage in ("packaging", "packaging_test"):
        if params.get("sample_size"):
            bits.append(f"{params['sample_size']} slides")
        if params.get("scope"):
            bits.append("from everything tiled" if params["scope"] == "tiled"
                        else "from this run")
        if params.get("pool_size"):
            bits.append(f"pool {params['pool_size']}")
        if params.get("random_seed") is not None:
            bits.append(f"seed {params['random_seed']}")
        if params.get("slide_names"):
            bits.append(f"{len(params['slide_names'])} named slides")
        if params.get("allow_incomplete"):
            bits.append("allow-incomplete")
    elif stage in ("extraction", "extraction_test"):
        if params.get("checkpoint"):
            bits.append(Path(params["checkpoint"]).name)
        if params.get("h5_path"):
            bits.append(f"on {Path(params['h5_path']).name}")
    return " · ".join(str(b) for b in bits)


def _render_job_history(submission_id: str, key_prefix: str):
    """Every Slurm job this run has submitted, across all stages.

    The run's own status fields hold one attempt per stage, so repackaging and
    repeated test packaging were previously invisible — this is the only place
    an earlier attempt (and the .h5 path it produced) can still be seen.
    """
    try:
        history = client.get_dataset_job_history(submission_id).get("jobs", [])
    except Exception as e:
        st.caption(f"Couldn't load job history: {e}")
        return

    if not history:
        st.caption(
            "No Slurm jobs recorded for this run yet. Runs submitted before the "
            "job-history migration have none — their current stage still shows above."
        )
        return

    for job in history:
        stage = job.get("stage", "?")
        state = (job.get("slurm_state") or "unknown").lower()
        when = (job.get("submitted_at") or "")[:16].replace("T", " ")
        batches = job.get("batch_count") or 1

        header = f"{_STAGE_LABELS.get(stage, stage)} · {state}"
        if batches > 1:
            header += f" · {batches} batches"
        if when:
            header += f" · {when}"
        st.markdown(f"**{header}**")

        detail = _describe_job_params(stage, job.get("params"))
        if detail:
            st.caption(detail)
        if job.get("output_path"):
            # Shown for every attempt, not just the current one: an earlier test
            # .h5 is still on disk and still usable for a checkpoint trial.
            st.caption(f"→ `{job['output_path']}`")


def _find_existing_job_for_path(dataset_path: str):
    """Most recent *actionable* submission (if any) whose raw_dir exactly
    matches dataset_path — skips cancelled/errored ones, since those are
    dead ends with no next step. Without skipping them, a cancelled
    duplicate submitted after the real run (e.g. from a bad "Resume"
    click) would shadow the actual, still-relevant run further back in
    the list, showing a dead end instead of real progress.

    Without this function at all, re-entering a path that was already
    submitted just looks like a blank form — nothing distinguishes "brand
    new dataset" from "this was already tiled, packaged, etc." — which is
    exactly what led to a duplicate full-dataset re-tiling run getting
    submitted by accident in the first place.
    """
    normalized = dataset_path.strip().rstrip("/")
    if not normalized:
        return None
    try:
        jobs = client.list_dataset_jobs()
    except Exception:
        return None
    for job in jobs:
        if (job.get("raw_dir") or "").rstrip("/") != normalized:
            continue
        if job.get("status") in ("cancelled", "error"):
            continue
        return job
    return None


def _dataset_key(dataset: dict) -> str:
    """Stable identity for a dataset across polls.

    Both halves are needed: two runs can tile the same raw directory into
    different output folders, and two directories can share a folder name.
    """
    return f"{dataset.get('raw_dir', '')}::{dataset.get('dataset_name', '')}"


def _dataset_overall_state(dataset: dict) -> str:
    """One step state standing in for the whole pipeline, for a picker row.

    Ordered by what someone scanning the list needs to spot first: something
    broken, then something waiting on them, then something running. A dataset
    is only "done" when all three of its steps are.
    """
    states = [step["state"] for step in dataset.get("steps", [])]
    for state in ("failed", "attention", "action", "running"):
        if state in states:
            return state
    return "done" if states and all(s == "done" for s in states) else "blocked"


def _dataset_option_label(dataset: dict) -> str:
    """One line identifying a dataset and saying where its pipeline has got to.

    Deliberately carries the current step rather than just a name: the whole
    point of this list is to answer "which of these still needs something from
    me" without opening each one.
    """
    icon = _STEP_ICON.get(_dataset_overall_state(dataset), "")
    bits = [f"{icon} {dataset.get('dataset_name') or '?'}"]

    total = dataset.get("total_slides")
    if total:
        bits.append(f"{int(total):,} slides")

    # Name the first step that isn't finished — that is the dataset's actual
    # position in the pipeline. All three done means there's nothing to name.
    for step in dataset.get("steps", []):
        if step["state"] != "done":
            bits.append(f"{step['title'].split('. ', 1)[-1].lower()}: {step['summary']}")
            break
    else:
        bits.append("complete")

    return " · ".join(bits)


def _apply_stored_coverage(dataset: dict) -> dict:
    """Overlay a previously fetched filesystem coverage check onto live data.

    The 10s poll deliberately does not walk the filesystem, so its tiling line
    reports only what Slurm says about the runs' own manifests. Once someone
    has paid for a real coverage check, that answer is better and should
    survive the next poll — but only for tiling. Packaging and extraction stay
    live, because the stored copy of those goes stale within seconds.
    """
    stored = st.session_state.get(f"dataset_coverage_{_dataset_key(dataset)}")
    if not stored:
        return dataset

    merged = dict(dataset)
    checked = stored.get("rollup", {})
    merged["tiling"] = checked.get("tiling", dataset["tiling"])
    merged["steps"] = [merged["tiling"], *dataset["steps"][1:]]
    for field in ("slides_tiled", "slides_untiled", "total_slides"):
        if checked.get(field) is not None:
            merged[field] = checked[field]
    merged["coverage_checked_at"] = stored.get("at")
    return merged


def _report_resume_result(result: dict) -> bool:
    """Say what a resume actually queued. Returns whether anything was queued.

    Shared by the two places a resume can be triggered — a run's tiling step
    and the dataset rollup's next-action button. Kept in one function because
    of the tiling-settings warning below: a second copy of this reporting is a
    second place for that warning to be forgotten, and it guards against
    exactly the failure resuming is supposed to prevent.
    """
    if not result.get("resumed"):
        st.info(result.get("message", "Nothing missing."))
        return False

    corrupt = result.get("corrupt_metadata_count") or 0
    extra = f" ({corrupt} had corrupt metadata)" if corrupt else ""
    st.success(
        f"Queued {result['missing_slide_count']} slides{extra} of "
        f"{result['total_in_original_manifest']} as new submission "
        f"{result['submission_id']}."
    )

    # A resume that quietly used different tiling settings than the slides
    # already on disk is the failure this reports on, so the fallback case is a
    # warning rather than a footnote.
    params = result.get("tiling_params") or {}
    if result.get("tiling_params_source") == "original_run":
        st.caption("Tiled with this run's own recorded settings.")
    elif params:
        st.warning(
            "This run has no recorded tiling settings, so the resumed "
            "slides used current defaults — check these match what the "
            "original run used, or they'll differ from the tiles already "
            "on disk."
        )
    if params:
        with st.expander("Tiling settings used for the resumed slides"):
            st.code(
                "\n".join(f"{k} = {v}" for k, v in sorted(params.items())),
                language=None,
            )
    return True


def _render_dataset_rollup(dataset: dict):
    """The whole pipeline for one dataset, and the single next thing to do.

    Every step is listed whether or not it is the active one, for the same
    reason _pipeline_steps lists all three for a run: "what is finished and
    what is left" cannot be answered from a screen that only draws the current
    stage. The difference here is scope — these steps are the aggregate across
    every run that has touched this dataset, which is the only level at which
    the question makes sense once a resume has forked the run in two.
    """
    st.caption(f"`{dataset.get('raw_dir', '')}`")

    provenance = []
    if dataset.get("has_full_run"):
        provenance.append("full-dataset run")
    if dataset.get("has_subset_run"):
        provenance.append("subset run(s)")
    if dataset.get("run_count"):
        provenance.append(f"{dataset['run_count']} run(s) total")
    if dataset.get("cancelled_run_count"):
        # Said out loud rather than silently dropped: the runs below are
        # filtered, and a count that doesn't add up invites the suspicion that
        # something has gone missing.
        provenance.append(f"{dataset['cancelled_run_count']} cancelled (hidden below)")
    if provenance:
        st.caption(" · ".join(provenance))

    for step in dataset.get("steps", []):
        icon = _STEP_ICON.get(step["state"], "")
        st.markdown(f"{icon} **{step['title']}** — {step['summary']}")

    if dataset.get("coverage_checked_at"):
        st.caption(f"Tiling coverage checked from disk at {dataset['coverage_checked_at']}.")

    _render_deliverables(dataset)
    _render_next_action(dataset)


_DELIVERABLE_ICON = {
    "ready": "[Ready]",
    "running": "[Packaging]",
    "interrupted": "[Interrupted]",
}


def _render_deliverables(dataset: dict):
    """Every .h5 this dataset has produced or attempted.

    The three steps above compress the dataset to one answer per stage, which
    is what makes them scannable and also what makes them lossy: they name a
    single .h5 as "the" packaging output. A directory routinely has several —
    a full one, and a subset per experiment — and packaging a large one costs
    hours of cluster time, so a finished file that isn't listed is a file
    someone re-makes. This is the list that stops that happening.

    Attempts and slide counts come from the run records, not from opening the
    files, so they describe what was asked for rather than what is inside. Where
    the attempts disagreed, all the counts are shown rather than the likeliest.
    """
    deliverables = dataset.get("deliverables") or []
    if not deliverables:
        return

    st.markdown("**.h5 files**")
    for item in deliverables:
        icon = _DELIVERABLE_ICON.get(item.get("status"), "")
        bits = []

        slides = item.get("total_slides")
        if slides is not None:
            bits.append(f"{int(slides):,} slides")
        others = [s for s in item.get("slides_seen") or [] if s != slides]
        if others:
            bits.append("also tried at " + ", ".join(f"{s:,}" for s in others))

        submitted = (item.get("submitted_at") or "")[:10]
        if submitted:
            bits.append(submitted)
        if (item.get("attempts") or 1) > 1:
            bits.append(f"{item['attempts']} attempts")
        if item.get("status") == "running":
            bits.append(f"packaging now ({item.get('slurm_state')})")
        elif item.get("status") == "interrupted":
            bits.append("no file on disk")

        st.markdown(f"{icon} `{item.get('name')}`")
        if bits:
            st.caption(" · ".join(bits))

    trials = [t for t in dataset.get("tests") or [] if t.get("status") == "ready"]
    if trials:
        st.caption(
            f"{len(trials)} test .h5 also on disk — usable for a checkpoint "
            f"trial without repackaging."
        )


def _render_next_action(dataset: dict):
    """Where this dataset stands, in one line — no button.

    This used to be the one button that moved a dataset forward a stage at a
    time (resume tiling, start packaging, start extraction). Stages 1-4 are
    one pipeline run now, started from the button above, which reuses every
    slide already tiled; so the per-stage actions are gone and this only says
    what state the history is in.
    """
    action = dataset.get("next_action") or {}
    kind = action.get("kind")
    if kind in ("complete", "none", "wait"):
        prefix = ""
        st.caption(f"{prefix}{action.get('label', '')} — {action.get('detail', '')}")
    elif kind == "submit":
        st.caption("Nothing has been run for this dataset yet.")
    elif kind:
        st.caption(
            f"Earlier runs stopped at: {action.get('label', '').rstrip(' →')}. Run the "
            f"pipeline above to carry this dataset through — it reuses every slide "
            f"already tiled."
        )


def _render_dataset_runs(dataset: dict):
    """Every run this dataset has had — HPL, ANORAK and runs from before the
    pipeline, cancelled ones included — one shown at a time, newest first."""
    runs = list(reversed(dataset.get("runs", [])))
    if not runs:
        st.caption("No runs for this dataset yet.")
        return
    key = _dataset_key(dataset)
    options = [r["submission_id"] for r in runs]
    by_id = {r["submission_id"]: r for r in runs}
    widget_key = f"dataset_run_pick_{key}"
    if st.session_state.get(widget_key) not in options:
        st.session_state.pop(widget_key, None)
    chosen = st.selectbox(
        "Run",
        options,
        key=widget_key,
        format_func=lambda sid: _dataset_run_label(by_id[sid]),
    )
    _render_job_progress(by_id[chosen], key_prefix=f"ds_{key}_")


def _dataset_run_label(run: dict) -> str:
    """One run, labelled so it can be told apart from its siblings.

    Runs over one dataset differ by when they went in, how many slides they
    covered, and whether they were a resume — which is exactly what a reader
    is choosing between.
    """
    state = (run.get("slurm_state") or run.get("status") or "unknown").lower()
    bits = [(run.get('submitted_at') or '')[:16].replace('T', ' ')]

    bits.append("ANORAK" if _is_anorak_run(run) else "HPL" if _is_pipeline_run(run)
                else "HPL (before the pipeline)")
    total = run.get("total_slides")
    if total is not None:
        bits.append(f"{int(total):,} slides" + (" (subset)" if run.get("is_subset") else ""))
    if run.get("resumed_from_submission_id"):
        bits.append("resume")
    bits.append(state)
    return " · ".join(bits)


def _datasets_under_path(datasets: list[dict], path: str) -> list[dict]:
    """The datasets built from this raw directory.

    A list rather than one, because a directory is not a dataset: the same
    Radiogenomics folder has been packaged into Radiogenomics, and into
    _subset_1, _2 and _3. Someone typing the path wants all of them.

    Prefix matching is deliberately not done. A path is either the directory a
    run recorded or it isn't; treating /users/vpandya as a match for
    /users/vpandya/Radiogenomics would answer a question about one dataset with
    every dataset underneath it.
    """
    wanted = path.strip().rstrip("/")
    return [d for d in datasets if str(d.get("raw_dir", "")).rstrip("/") == wanted]


@st.fragment(run_every="10s")
def _render_dataset_workspace(path: str):
    """Everything known about the dataset(s) at this path, and what's pending.

    Path-first, because that is how the work is actually organised: you have a
    directory of slides on the HPC and you want to know what has been done to
    it. Selecting from a list of datasets the server already knows about only
    answers that question for directories that have been through here before,
    and puts a lookup between the reader and an answer they could have had
    immediately.

    Polls on a 10s fragment so a step finishing shows up without needing an
    unrelated widget interaction to force a redraw.
    """
    try:
        payload = client.list_datasets()
    except Exception as e:
        st.warning(f"Could not load datasets: {e}")
        return

    datasets = payload.get("datasets", [])
    if not payload.get("slurm_reachable", True):
        st.warning(
            "Slurm isn't answering, so the states below come from what's on "
            "disk only. A job could be running right now without showing here."
        )
    elif not payload.get("slurm_states_complete", True):
        st.caption(
            "Older jobs were checked against the live queue only — open a run "
            "for its exact outcome."
        )

    if path:
        matches = _datasets_under_path(datasets, path)
        if not matches:
            st.info(
                "No runs recorded for this path yet. The submit form below "
                "will start the first one."
            )
            return
        dataset = _pick_dataset(matches, key_suffix="at_path")
    elif datasets:
        st.caption("Or pick one of the datasets already on record:")
        dataset = _pick_dataset(datasets, key_suffix="known")
    else:
        st.caption(
            "No dataset runs on record yet. Enter a path and submit one below "
            "and it will appear here with its progress."
        )
        return

    dataset = _apply_stored_coverage(dataset)

    # The newest HPL run and the newest ANORAK run are what someone opening
    # this path is following; every run, those included, is in History.
    key = _dataset_key(dataset)
    runs = dataset.get("runs", [])
    latest_hpl = [r for r in runs if _is_pipeline_run(r)]
    latest_anorak = [r for r in runs if _is_anorak_run(r)]
    if latest_hpl:
        st.markdown("**Latest HPL run**")
        _render_job_progress(latest_hpl[-1], key_prefix=f"hpl_{key}_")
    if latest_anorak:
        st.markdown("**Latest ANORAK run**")
        _render_job_progress(latest_anorak[-1], key_prefix=f"an_{key}_")
    with st.expander("History", expanded=False):
        _render_dataset_rollup(dataset)
        st.divider()
        _render_dataset_runs(dataset)


def _is_pipeline_run(run: dict) -> bool:
    return str(run.get("job_id") or "").startswith("nf:")


def _is_anorak_run(run: dict) -> bool:
    """An ANORAK run on its own (POST /anorak-runs)."""
    return run.get("status") == "anorak_only" or run.get("run_kind") == "anorak"


def _is_upload_run(status: dict) -> bool:
    """An uploaded slide's one-slide run: Stages 1-2 ran in-process (local:)."""
    return str(status.get("job_id") or "").startswith("local:")


def _pick_dataset(datasets: list[dict], *, key_suffix: str) -> dict:
    """Choose among datasets, without making a choice out of a single option.

    The selectbox is skipped for one dataset because a control with one setting
    is not a choice, it is an extra click before the thing you asked for.
    """
    by_key = {_dataset_key(d): d for d in datasets}
    if len(by_key) == 1:
        return next(iter(by_key.values()))

    chosen_key = st.selectbox(
        "Dataset",
        list(by_key),
        key=f"dataset_picker_{key_suffix}",
        format_func=lambda k: _dataset_option_label(by_key[k]),
        help=(
            "Every run that tiled into the same folder from the same directory, "
            "rolled into one pipeline — including the extra runs each resume "
            "creates."
        ),
    )
    return by_key[chosen_key]


def render_dataset_job_panel():
    """Submit a whole dataset (already staged on the HPC) for masking +
    tiling via a Slurm array job, and show status for jobs anyone has
    submitted — this reads from the shared Postgres-backed run record, not
    per-session state, so it survives page reloads and other users' sessions.
    """
    with st.sidebar.expander("Process a dataset", expanded=False):
        # One path box for the whole panel, above both halves. It used to sit
        # inside the submit form, which made "look at what exists" and "start
        # something new" two separate journeys that happened to need the same
        # piece of information typed once each.
        dataset_path = st.text_input(
            "Dataset path",
            key="dataset_job_path",
            help=(
                "Full absolute path on the HPC filesystem, e.g. "
                "/mnt/cephfs-lts/long-term-scratch/users/vpandya/Radiogenomics. "
                "Enter one to see everything already done to it; leave empty to "
                "browse what's on record."
            ),
        ).strip()

        # The two one-click runs first; what has already been done to the
        # dataset follows.
        _render_run_buttons(dataset_path)
        st.divider()
        _render_dataset_workspace(dataset_path)


def _render_run_buttons(dataset_path: str):
    """Run HPL, and below it Run ANORAK — each one click, each on its own.

    Nothing is asked for but the dataset path. HPL runs on the server's own
    settings (GET /pipeline-defaults), shown beside its button. ANORAK grades
    every slide in the directory unless a tumour-slide list is given. One
    shared option makes either a test run on a random, seeded subset.
    """
    if not dataset_path:
        st.caption("Enter a dataset path above.")
        return

    subset = st.checkbox(
        "Test on a random subset", value=False, key="run_subset",
        help="Run on a random sample of the directory's slides instead of all of "
             "them — the same pipeline, fewer slides. The seed is recorded, so the "
             "same sample can be asked for again.",
    )
    sample_size = seed = None
    if subset:
        columns = st.columns(2)
        sample_size = int(columns[0].number_input(
            "Slides", min_value=1, value=10, step=1, key="run_subset_n"))
        seed_text = columns[1].text_input("Seed (optional)", value="", key="run_subset_seed")
        seed = int(seed_text) if seed_text.strip().isdigit() else None

    # --- HPL -------------------------------------------------------------------
    st.markdown("**Run HPL** (tiling, packaging, feature extraction, classification)")
    try:
        defaults = client.get_pipeline_defaults()
    except Exception as e:
        defaults = None
        st.caption(f"Could not load HPL's settings ({e}); the server applies them anyway.")
    if defaults:
        name = Path(dataset_path.rstrip("/")).name or "?"
        st.caption(
            f"Tiles into `{defaults['tile_root']}/{name}`, the .h5 into "
            f"`{defaults['h5_root']}/{name}` · checkpoint `{defaults['checkpoint']}` · "
            f"reference `{defaults['reference']}` · {defaults['vote_preset']} vote · "
            f"{defaults['max_tiling']} slides at a time · {defaults['min_tissue']:g}% "
            f"minimum tissue. Slides already tiled are reused; earlier outputs in the "
            f"way are moved aside, never deleted."
        )
    if st.button("Run HPL", key="run_hpl", type="primary", use_container_width=True):
        try:
            result = client.start_pipeline_run(
                dataset_path=dataset_path, sample_size=sample_size, seed=seed)
        except requests.exceptions.HTTPError as e:
            st.error(f"Refused: {_http_detail(e)}")
        except Exception as e:
            st.error(f"Submission failed: {e}")
        else:
            st.success(f"HPL queued — run {result['submission_id']} for "
                       f"'{result.get('dataset_name')}'. Its progress appears below.")
            for entry in result.get("superseded") or []:
                st.caption(f"Moved aside: {entry['from']} -> {entry['to']}")

    # --- ANORAK ------------------------------------------------------------------
    st.markdown("**Run ANORAK** (growth-pattern grading)")
    slides_csv = st.text_input(
        "Tumour-slide list (optional)", value="", key="run_anorak_csv",
        help="select_tumour_slides.py's output, optionally filtered by "
             "filter_slides_by_tile_count.py. Blank grades every slide in the "
             "directory.",
    ).strip()
    if not slides_csv:
        st.caption(
            "Blank: every slide in the directory is graded, grouped into tumours by "
            "the part of the slide name before the first space. Tumour status is not "
            "checked, so non-tumour slides are graded too — give a tumour-slide list "
            "for a proper run."
        )
    if st.button("Run ANORAK", key="run_anorak", type="primary", use_container_width=True):
        try:
            result = client.start_anorak_run(
                dataset_path, slides_csv=slides_csv or None,
                sample_size=sample_size, seed=seed)
        except requests.exceptions.HTTPError as e:
            st.error(f"Refused: {_http_detail(e)}")
        except Exception as e:
            st.error(f"Submission failed: {e}")
        else:
            selection = result.get("selection") or {}
            st.success(f"ANORAK queued — run {result['submission_id']}, "
                       f"{selection.get('slides')} slides. Its progress appears below.")
            if result.get("skipped_unsupported"):
                st.caption(f"{len(result['skipped_unsupported'])} .scn slide(s) left out: "
                           f"ANORAK cannot read that format.")


def _dataset_job_display_stage(status: dict) -> str:
    """Collapse the raw status fields into one short label, used to detect
    genuine state changes (for toasts) without caring about every field.

    Every stage past tiling now requires a conscious click in the UI rather
    than auto-chaining, so this also distinguishes "done, waiting for a
    click" (tiling done / h5 ready) from "actually running the next stage"
    (packaging / extracting features)."""
    stage = status.get("status")
    if stage in ("queued", "discovering", "error", "cancelled"):
        return stage
    if stage == "submitted":
        if status.get("extraction_ready"):
            return "features ready"
        if status.get("extraction_job_id"):
            return "extracting features"
        if status.get("h5_ready"):
            return "h5 ready"
        if status.get("h5_job_id"):
            return "packaging"
        if status.get("tiling_complete"):
            return "tiling done"
        return "tiling"
    return stage or "unknown"


# Step state -> the icon shown in the pipeline list. The point of the list is
# that you can see, at a glance and without clicking anything, which stages are
# finished and which are outstanding — previously the UI rendered only the
# current stage's widget, so "what's left?" could not be answered without
# reading the code.
_STEP_ICON = {
    "done": "[Done]",
    "running": "[Running]",
    "action": "[Ready]",             # ready for you to start
    "attention": "[Needs attention]",  # finished or stalled, but needs a decision
    "failed": "[Failed]",
    "blocked": "[Waiting]",          # can't start yet, an earlier step must finish
}
# Must match IN_FLIGHT_SLURM_STATES in tile_server_v2_.py. It lacked
# CONFIGURING (nodes allocated, prologue still running — squeue reports it
# first now), so a job Slurm was starting read here as "did not finish" and
# got a Retry button. Stages whose submit guard matters (ANORAK) also get the
# server's own verdict in /status rather than trusting this copy.
_SLURM_IN_FLIGHT = {
    "PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING", "CONFIGURING",
}


def _error_detail(e) -> tuple[dict | None, str]:
    """Split a FastAPI HTTPError into (structured detail, message).

    Several endpoints raise HTTPException with a dict body rather than a
    string — the packaging guard's "tiling_incomplete" is the one that
    matters here, since the UI needs to offer a specific follow-up action
    for it rather than printing raw JSON at the user.
    """
    response = getattr(e, "response", None)
    if response is None:
        return None, str(e)
    try:
        detail = response.json().get("detail")
    except Exception:
        return None, response.text
    if isinstance(detail, dict):
        return detail, detail.get("message") or str(detail)
    return None, str(detail) if detail else response.text


def _human_bytes(n: int) -> str:
    step = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if step < 1024 or unit == "TB":
            return f"{step:.0f} {unit}" if unit == "B" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{step:.1f} TB"


def _render_start_packaging_button(
    submission_id: str,
    key_prefix: str,
    button_label: str = "Start packaging (.h5)",
    allow_incomplete: bool = False,
    resume: bool | None = None,
    disabled: bool = False,
):
    """Button for the packaging stage. Covers a first attempt, a retry, and a
    resume identically — the server decides which it is (it only blocks when
    the prior attempt genuinely succeeded or is still running, and picks up
    from the checkpoint when a .partial is present).

    A refusal because tiling didn't fully succeed is caught specially: that
    is now an expected, recoverable answer rather than an error, so it offers
    the two real choices (fix the tiling, or package without those slides)
    instead of showing the raw 400.
    """
    state_key = f"{key_prefix}pkg_incomplete_{submission_id}"

    if st.button(
        button_label,
        key=f"{key_prefix}dataset_job_package_{submission_id}",
        disabled=disabled,
    ):
        try:
            client.start_packaging_job(
                submission_id, allow_incomplete=allow_incomplete, resume=resume,
            )
            st.session_state.pop(state_key, None)
            st.success("Packaging job queued.")
            st.rerun()
        except requests.exceptions.HTTPError as e:
            detail, message = _error_detail(e)
            if detail and detail.get("error") == "tiling_incomplete":
                # Stash it so the warning survives the rerun this button click
                # ends in, rather than flashing once and vanishing.
                st.session_state[state_key] = detail
            else:
                st.error(f"Failed to start packaging: {message}")
        except Exception as e:
            st.error(f"Failed to start packaging: {e}")

    blocked = st.session_state.get(state_key)
    if blocked:
        st.warning(blocked.get("message", "Some tiling tasks did not complete."))
        failed_states = blocked.get("failed_task_states") or {}
        if failed_states:
            st.caption(
                "Failed tiling tasks: "
                + ", ".join(f"{n}x {state}" for state, n in failed_states.items())
            )
        st.caption(
            "Either resume the tiling step above to retry those slides, or "
            "package without them — the .h5 will be missing their tiles."
        )
        if st.button(
            "Package anyway (accept missing slides)",
            key=f"{key_prefix}dataset_job_package_force_{submission_id}",
        ):
            try:
                # Carries the same resume decision the caller made — this is the
                # "package anyway" variant of the very same submission, so it
                # must not quietly fall back to auto-resume.
                client.start_packaging_job(
                    submission_id, allow_incomplete=True, resume=resume,
                )
                st.session_state.pop(state_key, None)
                st.success("Packaging job queued (incomplete dataset accepted).")
                st.rerun()
            except requests.exceptions.HTTPError as e:
                st.error(f"Failed to start packaging: {_error_detail(e)[1]}")
            except Exception as e:
                st.error(f"Failed to start packaging: {e}")


def _render_extract_features_form(submission_id: str, key_prefix: str, button_label: str = "Start feature extraction"):
    """Checkpoint input + submit button for Stage 2 (feature extraction).
    Used both for a run's first attempt and for retrying one that failed —
    a failed attempt (e.g. the conda-activation bug) used to be a
    permanent dead end since the backend blocked resubmitting once any
    extraction_job_id existed; now it only blocks a genuinely still-running
    or already-succeeded attempt, so this same form covers both cases.
    """
    checkpoint = st.text_input(
        "Model checkpoint path",
        key=f"{key_prefix}dataset_job_checkpoint_{submission_id}",
        placeholder="/path/to/BarlowTwins_3.ckt",
    )
    if st.button(button_label, key=f"{key_prefix}dataset_job_extract_{submission_id}"):
        if not checkpoint.strip():
            st.error("Enter a checkpoint path first.")
        else:
            try:
                client.start_feature_extraction(submission_id, checkpoint.strip())
                st.success("Feature extraction job queued.")
                st.rerun()
            except requests.exceptions.HTTPError as e:
                detail = e.response.text if e.response is not None else str(e)
                st.error(f"Failed to start feature extraction: {detail}")
            except Exception as e:
                st.error(f"Failed to start feature extraction: {e}")


def _test_packaging_note(status: dict) -> str:
    """A short ' · test subset: ...' clause for the packaging step's summary,
    or '' if no test/subset packaging job has ever been submitted for this
    run. See the "test_note" comment in _pipeline_steps for why this exists:
    the full-dataset h5_* fields alone can't tell "never touched packaging"
    apart from "already checked it works on a subset", and those read as the
    same "ready to start" line without this.
    """
    if not status.get("test_h5_job_id"):
        return ""
    if status.get("test_h5_ready"):
        return " · test subset: ready"
    test_state = status.get("test_h5_slurm_state")
    if test_state in _SLURM_IN_FLIGHT:
        return f" · test subset: running ({test_state})"
    return f" · test subset: {test_state or 'interrupted'}"


def _pipeline_steps(status: dict) -> list[dict]:
    """Classify all three pipeline stages at once, from a single /status
    response, so the UI can show the whole pipeline rather than only whichever
    stage happens to be current.

    This is the core of the redesign. The old renderer walked a nested
    if/else and drew exactly one stage's widget, which meant the two questions
    people actually ask — what has finished, and what is still outstanding —
    could not be answered from the screen at all. Every stage now reports a
    state and a one-line summary whether or not it is the active one.
    """
    stage = status.get("status")
    total = status.get("total_slides")
    succeeded = status.get("succeeded")

    # --- 1. Tiling ---------------------------------------------------------
    if stage in ("queued", "discovering"):
        tiling = ("running", "discovering slides")
    elif stage == "error":
        tiling = ("failed", (status.get("error") or "failed")[:70])
    elif status.get("tiling_complete"):
        not_attempted = status.get("not_yet_attempted") or 0
        if not_attempted:
            tiling = ("attention", f"{succeeded}/{total} slides · {not_attempted} never ran")
        elif succeeded is not None:
            tiling = ("done", f"{succeeded}/{total} slides have tiles")
        else:
            tiling = ("done", "complete")
    elif status.get("slurm_unreachable"):
        # Not "running" — the server could not reach Slurm at all, so it has no
        # idea whether tiling is going or finished weeks ago. Saying "in
        # progress" here asserted the former on no evidence.
        tiling = ("attention", "can't reach Slurm — state unknown")
    else:
        counts = status.get("slurm_state_counts") or {}
        in_flight = sum(n for s, n in counts.items() if s in _SLURM_IN_FLIGHT)
        tiling = ("running", f"{in_flight:,} task(s) in flight" if in_flight else "in progress")

    # --- 2. Packaging ------------------------------------------------------
    # test_note surfaces a "Test on a subset" packaging job even when no
    # full-dataset job exists yet. Without it, a run where someone had only
    # ever tried the subset option looked identical to one where packaging
    # had never been touched at all — "ready to start", as if the subset
    # job's progress or completed .h5 weren't sitting right there. It's
    # appended to every no-full-job branch below, never to done/running,
    # since once a full-dataset .h5 exists that's the state that matters.
    test_note = _test_packaging_note(status)
    h5_state = status.get("h5_slurm_state")
    if status.get("h5_ready"):
        # Legacy tile names are noted, not treated as a failure: the .h5 is
        # usable and every stage that reads it — including registration and the
        # KB load, which append the suffix themselves — handles it. Worth saying
        # anyway, because the file on disk still holds the short form and
        # anything reading it outside this pipeline will see that.
        packaging = ("done", ".h5 ready (legacy tile names; handled on load)"
                             if status.get("h5_legacy_tile_names") else ".h5 ready")
    elif status.get("h5_job_id"):
        if h5_state in _SLURM_IN_FLIGHT:
            packaging = ("running", f"running ({h5_state})")
        elif status.get("h5_packaging_active"):
            # The .partial was written to recently, so packaging is running even
            # though Slurm won't say so — either sacct is unreachable or the job
            # is too new to have an accounting row. Without this a job that had
            # been running for an hour was labelled "interrupted (no Slurm
            # record)", complete with a retry button.
            written = _human_bytes(status.get("h5_partial_bytes") or 0)
            packaging = ("running", f"writing ({written} so far)")
        elif status.get("h5_invalid_reason"):
            packaging = ("attention", "finished but .h5 unusable")
        else:
            packaging = ("attention", f"interrupted ({h5_state or 'no Slurm record'})")
    elif status.get("tiling_complete"):
        packaging = ("action", f"ready to start{test_note}")
    elif status.get("slurm_unreachable"):
        # "blocked" is a claim that an earlier stage is still working, and it
        # takes the start button away. With Slurm unreachable that claim is
        # unfounded and the lockout is permanent — a run whose tiling finished
        # long ago could never be packaged for as long as sacct stayed down.
        # Downgrade to a decision the user is allowed to make.
        packaging = ("attention", f"can't reach Slurm — tiling state unknown{test_note}")
    else:
        packaging = ("blocked", f"waiting on tiling{test_note}")

    # --- 3. Feature extraction --------------------------------------------
    ext_state = status.get("extraction_slurm_state")
    expected_rows = status.get("extraction_expected_rows")
    if status.get("extraction_ready"):
        extraction = ("done", f"features ready ({expected_rows:,} tiles)"
                              if expected_rows else "features ready")
    elif status.get("extraction_job_id"):
        if ext_state in _SLURM_IN_FLIGHT:
            extraction = ("running", f"running ({ext_state})")
        elif status.get("extraction_invalid_reason"):
            # Slurm can call a job COMPLETED while its output is unusable —
            # saying "did not finish (COMPLETED)" for that is the confusing
            # read this branch exists to avoid.
            #
            # A short output is the case worth naming in the one-line summary
            # rather than leaving to the expandable detail: it is the failure
            # that otherwise looks exactly like success, and the count is the
            # whole point. "Incomplete" alone sent someone on to the next stage
            # believing 14,044 slides had been encoded when 3,500 had.
            reason = status.get("extraction_invalid_reason") or ""
            shortfall = re.search(r"has ([\d,]+) embeddings but the input \.h5 has "
                                  r"([\d,]+) tiles", reason)
            if shortfall:
                got = int(shortfall.group(1).replace(",", ""))
                want = int(shortfall.group(2).replace(",", ""))
                extraction = ("attention",
                              f"incomplete — {got:,} of {want:,} tiles encoded "
                              f"({got / want * 100:.0f}%)")
            else:
                extraction = ("attention", "finished but the output is incomplete")
        else:
            extraction = ("attention", f"did not finish ({ext_state or 'no Slurm record'})")
    elif status.get("h5_ready"):
        extraction = ("action", "ready to start")
    else:
        extraction = ("blocked", "waiting on packaging")

    # --- 4. Cluster assignment --------------------------------------------
    # Gated on extraction_ready rather than on extraction_job_id: this stage
    # reads the embeddings, so an extraction that ran but left an unusable file
    # is not a state from which assignment can proceed.
    asg_state = status.get("assignment_slurm_state")
    if status.get("assignment_ready"):
        assignment = ("done", "clusters assigned")
    elif status.get("assignment_job_id"):
        if asg_state in _SLURM_IN_FLIGHT:
            assignment = ("running", f"running ({asg_state})")
        elif status.get("assignment_invalid_reason"):
            assignment = ("attention", "finished but the output is incomplete")
        else:
            assignment = ("attention", f"did not finish ({asg_state or 'no Slurm record'})")
    elif status.get("extraction_ready"):
        assignment = ("action", "ready to start")
    else:
        assignment = ("blocked", "waiting on feature extraction")

    # --- 5. Registration -----------------------------------------------------
    # Gated on h5_ready, not on the assignment: registration reads tile
    # identity out of the packaged .h5 and the raw slides, and needs no cluster
    # labels at all. Making it wait for Stage 4 would keep Stage 6 blocked
    # behind a step that could have finished hours earlier.
    #
    # This step is new, and everything before it is unchanged, so a run that
    # completed before it existed shows it as "action" — which is correct.
    # Those runs did not register; their tiles reached the KB by hand or not at
    # all.
    if status.get("registration_done"):
        rows = status.get("registration_rows") or {}
        if isinstance(rows, dict) and rows:
            summary = ", ".join(f"{n:,} {t.replace('_', ' ')}" for t, n in rows.items())
        else:
            summary = "registered"
        registration = ("done", summary[:70])
    elif status.get("registration_slurm_state") in _SLURM_IN_FLIGHT:
        # A Slurm-backed write in flight. Without this the stage reads "ready to
        # register" while a job is actively writing, which invites a second one.
        registration = ("running", f"running ({status['registration_slurm_state']})")
    elif status.get("registration_job_id"):
        registration = ("attention", "a job ended without committing")
    elif status.get("registration_ready"):
        registration = ("action", "ready to register")
    else:
        registration = ("blocked", "waiting on packaging")

    # --- 6. Knowledge Bank load ---------------------------------------------
    # Gated on assignment_ready, not assignment_job_id, for the same reason
    # Stage 4 gates on extraction_ready: this stage reads the CSV Stage 4
    # wrote, so a finished-but-unusable one is not a state it can load from.
    #
    # Also gated on registration, and this is the whole point of the step
    # above: load_hpc_assignments only UPDATEs, so without identity rows its
    # match rate is 0% and it refuses. Before this, that refusal was the first
    # sign anything was wrong, and it named a match rate rather than a missing
    # step.
    if status.get("kb_load_done"):
        rows = status.get("kb_load_rows")
        kb_load = ("done", f"{rows:,} tiles in the KB" if rows is not None else "loaded")
    elif status.get("kb_load_slurm_state") in _SLURM_IN_FLIGHT:
        kb_load = ("running", f"running ({status['kb_load_slurm_state']})")
    elif status.get("kb_load_job_id"):
        kb_load = ("attention", "a job ended without committing")
    elif not status.get("assignment_ready"):
        kb_load = ("blocked", "waiting on cluster classification")
    elif not status.get("registration_done"):
        kb_load = ("blocked", "waiting on registration")
    else:
        kb_load = ("action", "ready to load")

    # --- 7. ANORAK growth-pattern grading ------------------------------------
    # Not gated on anything this pipeline produces: ANORAK does its own tiling
    # at its own resolution and reads the raw slides, so it shares no artifact
    # with Stages 1-6. What it needs from them is the *slide list* — the
    # cohort's tumour slides, which come from the cluster composition Stage 6
    # loads. So this is "action" as soon as there is something to grade, and
    # the form says which list it is about to use rather than assuming one.
    if status.get("anorak_ready"):
        scope = status.get("anorak_scope")
        slides = status.get("anorak_slides")
        detail = f"{slides:,} slides" if slides else "complete"
        if scope == "subset":
            detail += f" (random subset, seed {status.get('anorak_seed')})"
        anorak = ("done", f"growth patterns graded · {detail}")
    elif status.get("anorak_in_flight") or status.get("anorak_slurm_state") in _SLURM_IN_FLIGHT:
        # The head job being alive is all this says. It submits a job per slide
        # per stage itself, so its own state carries no progress — which is why
        # the summary names the pipeline rather than a percentage it does not
        # have.
        anorak = ("running", f"pipeline running ({status['anorak_slurm_state']})")
    elif status.get("anorak_job_id") and status.get("anorak_state_unknown"):
        # Not "did not finish": Slurm could not be asked, so the head job may
        # well be alive — and the server refuses a new submission until it can.
        anorak = ("attention", "state unknown — Slurm unreachable")
    elif status.get("anorak_job_id"):
        if status.get("anorak_invalid_reason"):
            anorak = ("attention", "finished without a usable grading table")
        else:
            anorak = ("attention",
                      f"did not finish ({status.get('anorak_slurm_state') or 'no Slurm record'})")
    elif status.get("kb_load_done"):
        anorak = ("action", "ready to run")
    else:
        # Deliberately not "blocked": a slide list chosen some other way is a
        # perfectly good input, and blocking would hide the form that takes one.
        anorak = ("action", "ready to run (needs a tumour-slide list)")

    if status.get("pipeline"):
        tiling, packaging, extraction, assignment = _pipeline_stage_states(
            status, (tiling, packaging, extraction, assignment))

    return [
        {"key": "tiling", "title": "1. Tiling", "state": tiling[0], "summary": tiling[1]},
        {"key": "packaging", "title": "2. Packaging (.h5)", "state": packaging[0], "summary": packaging[1]},
        {"key": "extraction", "title": "3. Feature extraction", "state": extraction[0], "summary": extraction[1]},
        {"key": "assignment", "title": "4. Cluster classification", "state": assignment[0], "summary": assignment[1]},
        {"key": "registration", "title": "5. Register in the Knowledge Bank", "state": registration[0], "summary": registration[1]},
        {"key": "kb_load", "title": "6. Knowledge Bank load", "state": kb_load[0], "summary": kb_load[1]},
        {"key": "anorak", "title": "7. Growth patterns (ANORAK)", "state": anorak[0], "summary": anorak[1]},
    ]


# Stages 1-4 of a pipeline run (POST /pipeline-runs) are one Nextflow run. Their
# state comes from the stage's done marker and the head job (hpl_nf_state), and
# "done" still means what it means for a clicked-through stage: the server's own
# validator accepted the output. The one thing that changes is that no stage
# has a button of its own — the pipeline starts each when the one before it
# has verified its output.
_PIPELINE_STAGES = ("tiling", "packaging", "extraction", "assignment")
_PIPELINE_STAGE_READY = {
    "tiling": "tiling_complete",
    "packaging": "h5_ready",
    "extraction": "extraction_ready",
    "assignment": "assignment_ready",
}
_PIPELINE_STAGE_NAME = {
    "tiling": "tiling", "packaging": "packaging",
    "extraction": "feature extraction", "assignment": "classification",
}


def _pipeline_stage_verified(status: dict, stage: str) -> bool:
    pipeline_stage = ((status.get("pipeline") or {}).get("stages") or {}).get(stage) or {}
    if stage == "tiling":
        # tiling_complete is "terminal", not "succeeded"; for the pipeline the
        # stage's own verdict is what counts.
        return pipeline_stage.get("state") == "COMPLETED" and bool(status.get("tiling_complete"))
    return bool(status.get(_PIPELINE_STAGE_READY[stage]))


def _pipeline_stage_states(status: dict, computed: tuple) -> tuple:
    """(state, summary) for Stages 1-4 of a pipeline run.

    A verified stage keeps the summary the clicked-through path computes
    (slide counts, tile counts), so the two kinds of run read the same once
    finished. Everything else is said in the pipeline's terms: queued behind
    the stage before it, running, or where the run stopped.
    """
    pipeline = status["pipeline"]
    stages = pipeline.get("stages") or {}
    out = []
    previous_done = True
    for stage, (state, summary) in zip(_PIPELINE_STAGES, computed):
        slurm = (stages.get(stage) or {}).get("state")
        if _pipeline_stage_verified(status, stage):
            out.append(("done", summary))
        elif slurm == "COMPLETED":
            # The task said done, the server's validator disagrees. Never
            # papered over: this is the file-of-the-right-shape failure.
            out.append(("attention", "pipeline marked it done, but the output fails validation"))
        elif slurm == "RUNNING":
            out.append(("running", "running in the pipeline"))
        elif slurm in _SLURM_IN_FLIGHT:
            out.append(("blocked", "queued in the pipeline" if previous_done
                        else f"waits for {_PIPELINE_STAGE_NAME[_PIPELINE_STAGES[len(out) - 1]]}"))
        elif slurm is None:
            out.append(("attention", "can't reach Slurm — state unknown"))
        elif previous_done:
            out.append(("failed", f"pipeline stopped here ({slurm})"))
        else:
            out.append(("blocked", "not reached"))
        previous_done = out[-1][0] == "done"
    return tuple(out)


def _render_legacy_stage_readonly(status: dict, stage: str):
    """Stages 1-4 of a run from before the pipeline: what it did, no buttons."""
    job_key, path_key, reason_key = {
        "tiling": ("job_id", None, None),
        "packaging": ("h5_job_id", "h5_output_path", "h5_invalid_reason"),
        "extraction": ("extraction_job_id", "extraction_output_path", "extraction_invalid_reason"),
        "assignment": ("assignment_job_id", "assignment_output_path", "assignment_invalid_reason"),
    }[stage]
    if status.get(job_key):
        st.caption(f"Slurm job(s): `{status[job_key]}`")
    if stage == "tiling" and status.get("succeeded") is not None:
        st.caption(f"{status['succeeded']:,} of {status.get('total_slides') or 0:,} slides have tiles.")
    if path_key and status.get(path_key):
        st.caption("Output:")
        st.code(status[path_key], language=None)
    if reason_key and status.get(reason_key):
        st.caption(f"Not usable: {status[reason_key]}")
    st.caption("Read-only: Stages 1-4 now run as one pipeline. Run the pipeline for "
               "this dataset to carry it on; it reuses whatever this run produced.")


def _render_pipeline_overview(status: dict, submission_id: str, key_prefix: str):
    """The head job, in one line, above the steps."""
    pipeline = status["pipeline"]
    head = ", ".join(pipeline.get("head_job_ids") or []) or "not submitted yet"
    state = pipeline.get("head_state") or "unknown"
    if pipeline.get("finished"):
        st.success("Nextflow pipeline finished: Stages 1-4 are verified. "
                   "Register and load into the Knowledge Bank below.")
    elif state in _SLURM_IN_FLIGHT:
        st.info(f"Nextflow pipeline {state.lower()} — head job {head}. It submits a "
                f"job per slide and per shard itself, so squeue shows many more.")
    else:
        st.warning(f"Nextflow pipeline stopped ({state}) — head job {head}.")
    selection = pipeline.get("selection") or {}
    if selection.get("scope") == "subset":
        st.caption(f"Random subset: {selection.get('slides')} of {selection.get('pool')} "
                   f"slides, seed {selection.get('seed')}. A test run — the full "
                   f"directory has not been processed.")
    settings = pipeline.get("settings") or {}
    if settings:
        st.caption(
            f"Checkpoint {settings.get('checkpoint')} · {settings.get('extraction_shards')} "
            f"GPU shard(s) on {settings.get('gpu_gres')} · reference "
            f"{settings.get('reference')} · vote {settings.get('vote')} · "
            f"{settings.get('assignment_shards')} assignment shard(s) on "
            f"{settings.get('device')}"
        )
    st.caption(f"Run directory (config, stage markers, nextflow.log, report): "
               f"`{pipeline.get('out_dir')}`")


def _render_pipeline_resume(status: dict, submission_id: str, key_prefix: str):
    """Why it stopped, and the one button that continues it."""
    pipeline = status["pipeline"]
    if pipeline.get("stop_reason"):
        st.caption("Why it stopped (the supervisor's stop marker):")
        st.code(pipeline["stop_reason"], language=None)
    if pipeline.get("log_tail"):
        with st.expander("End of nextflow.log", expanded=True):
            st.code(pipeline["log_tail"], language=None)
    if not pipeline.get("resumable"):
        return
    st.caption("Resuming re-runs only what did not finish, with this run's own "
               "recorded settings; every stage is verified again before it counts.")
    columns = st.columns(2)
    chain = columns[0].number_input(
        "Head jobs", min_value=1, max_value=10, value=3, step=1,
        key=f"{key_prefix}pipeline_resume_chain_{submission_id}")
    time_limit = columns[1].text_input(
        "Head job walltime (optional)", value="",
        key=f"{key_prefix}pipeline_resume_time_{submission_id}")
    already = bool((pipeline.get("settings") or {}).get("allow_incomplete"))
    allow_incomplete = st.checkbox(
        "Package without slides that fail to tile", value=already,
        disabled=already,
        key=f"{key_prefix}pipeline_resume_incomplete_{submission_id}",
        help="For a run that stopped because some slides cannot be tiled: they "
             "are left out of the .h5 and named on the tiling step. Everything "
             "already tiled is kept.",
    )
    if st.button("Resume pipeline", type="primary",
                 key=f"{key_prefix}pipeline_resume_{submission_id}"):
        try:
            result = client.resume_pipeline_run(
                submission_id, chain=int(chain), time_limit=time_limit.strip() or None,
                allow_incomplete=True if allow_incomplete and not already else None)
        except requests.exceptions.HTTPError as e:
            st.error(f"Refused: {_http_detail(e)}")
        except Exception as e:
            st.error(f"Resume failed: {e}")
        else:
            st.success(f"Resumed — head job {result.get('nf_job_id')}.")


def _render_pipeline_stage_step(status: dict, submission_id: str, key_prefix: str,
                                stage: str, state: str):
    """Stages 1-4 of a pipeline run: what the stage verified, where its output
    is, and — at the stage the run stopped — why, and Resume."""
    pipeline = status["pipeline"]
    info = (pipeline.get("stages") or {}).get(stage) or {}
    done = info.get("done") or {}

    if stage == "tiling":
        total = status.get("total_slides")
        if done:
            st.caption(f"{done.get('succeeded', 0):,} of {done.get('slides', total) or 0:,} "
                       f"slides have tiles (every CSV checked against its summary).")
        excluded = done.get("excluded_slides") or []
        if excluded:
            st.warning(
                f"{done.get('excluded', len(excluded))} slide(s) failed to tile and were "
                f"left out of the .h5 (this run allows it). Their TILE task logs say why."
            )
            if st.checkbox(f"Show the {len(excluded)} left-out slide(s)",
                           key=f"{key_prefix}pipeline_excluded_{submission_id}"):
                st.code("\n".join(excluded), language=None)
        zero_tile = status.get("zero_tile_slides") or done.get("zero_tile_slides") or []
        if zero_tile:
            st.warning(f"{len(zero_tile)} slide(s) ran but saved zero tiles (no tissue above "
                       f"the minimum). Re-running won't change that.")
            if st.checkbox(f"Show the {len(zero_tile)} zero-tile slide ID(s)",
                           key=f"{key_prefix}pipeline_zero_tile_{submission_id}"):
                st.code("\n".join(zero_tile), language=None)
    else:
        path_key = {"packaging": "h5_output_path", "extraction": "extraction_output_path",
                    "assignment": "assignment_output_path"}[stage]
        if status.get(path_key):
            st.caption("Output:")
            st.code(status[path_key], language=None)
        count = done.get("tiles") or done.get("embeddings") or done.get("assignments")
        if count:
            st.caption(f"Verified by the pipeline: {count:,} rows.")
        reason = status.get({"packaging": "h5_invalid_reason",
                             "extraction": "extraction_invalid_reason",
                             "assignment": "assignment_invalid_reason"}[stage])
        if reason:
            st.error(f"The server's check rejects this output: {reason}")
        if stage == "assignment" and status.get("assignment_vote"):
            st.caption(f"Vote: {status['assignment_vote']}")
        if stage == "assignment" and _pipeline_stage_verified(status, stage):
            # Worth running before the KB load, exactly as for a clicked run.
            _render_cohort_shift(submission_id, key_prefix)

    if state == "failed":
        _render_pipeline_resume(status, submission_id, key_prefix)


def _render_tiling_step(status: dict, submission_id: str, key_prefix: str):
    tiling_job_ids = [j for j in (status.get("job_id") or "").split(",") if j]
    if tiling_job_ids:
        st.caption(
            f"Slurm job{'s' if len(tiling_job_ids) > 1 else ''} "
            f"({len(tiling_job_ids)} batch{'es' if len(tiling_job_ids) > 1 else ''}):"
        )
        st.code("\n".join(tiling_job_ids), language=None)

    counts = status.get("slurm_state_counts") or {}
    if counts:
        st.caption("Task states: " + ", ".join(f"{n:,}x {s}" for s, n in sorted(counts.items())))

    total = status.get("total_slides")
    succeeded = status.get("succeeded")
    zero_tile = status.get("zero_tile_slides") or []
    not_attempted = status.get("not_yet_attempted") or 0

    if status.get("tiling_complete") and succeeded is not None:
        if not_attempted:
            # Only these are fixable by resubmitting — resume diffs the
            # manifest against which slides have usable tile metadata, so it
            # can only ever find slides that never ran or whose metadata is
            # corrupt, never the zero-tile ones below.
            st.warning(
                f"{not_attempted} of {total} slides produced no usable output. "
                f"Resume them before packaging, or the .h5 will be missing them."
            )
        if zero_tile:
            # These did run and complete — they just found no tissue above the
            # threshold. Resume is a no-op for them.
            st.warning(
                f"{len(zero_tile)} of {total} slides ran but saved zero tiles (no tissue "
                f"above the minimum-tissue threshold). Resubmitting won't change this — "
                f"check the masking / min-tissue settings if they should have tissue."
            )
            if st.checkbox(
                f"Show the {len(zero_tile)} zero-tile slide ID(s)",
                key=f"{key_prefix}zero_tile_{submission_id}",
            ):
                st.code("\n".join(zero_tile), language=None)
        if not not_attempted and not zero_tile:
            st.success(f"All {succeeded} slides have tiles.")

    if st.button("Resume missing slides", key=f"{key_prefix}dataset_job_resume_{submission_id}"):
        try:
            _report_resume_result(client.resume_dataset_job(submission_id))
        except requests.exceptions.HTTPError as e:
            st.error(f"Resume failed: {_error_detail(e)[1]}")
        except Exception as e:
            st.error(f"Resume failed: {e}")


@st.fragment(run_every="15s")
def _render_test_packaging_progress(submission_id: str, key_prefix: str, current_params):
    """Live view of the recorded test packaging job.

    A fragment on its own timer for the same reason _render_packaging_live_progress
    is one: without it this block only redrew when something else happened to
    rerun the page, so a job that was progressing perfectly well looked frozen.
    """
    try:
        status = client.get_dataset_job_status(submission_id)
    except Exception as e:
        st.caption(f"Couldn't read test packaging status: {e}")
        return

    params = status.get("test_h5_params") or {}
    scope_label = ("everything tiled on disk" if params.get("scope") == "tiled"
                   else "this run's manifest")

    # Survives a reload, unlike the session_state comparison below it: the
    # recorded params are on the row, so "this job isn't what's in the form" can
    # still be said after the form state is gone.
    if current_params is not None and params:
        recorded = (
            "Random N" if params.get("sample_size") else "Specific slides",
            params.get("scope") or "run",
            params.get("sample_size"),
            tuple(params.get("slide_names") or ()),
            "" if params.get("random_seed") is None else str(params.get("random_seed")),
        )
        # Seed is excluded from the comparison: the server picks a fresh one for
        # every unseeded run, so including it would mark every job stale the
        # moment it was submitted.
        if recorded[:4] != tuple(current_params)[:4]:
            st.warning(
                "The job below is from an earlier setup, not the selection "
                "currently in the form. Click **Start test packaging** to run "
                "the current one."
            )

    st.caption("Test packaging job:")
    st.code(status["test_h5_job_id"], language=None)

    if status.get("test_h5_ready"):
        st.success("Test packaging finished — .h5 is readable and ready to use.")
        path = status.get("test_h5_output_path")
        st.code(path, language=None)
        detail = []
        if params.get("sample_size"):
            detail.append(f"{params['sample_size']} slides drawn from {scope_label}")
        if params.get("pool_size"):
            detail.append(f"pool was {params['pool_size']}")
        if params.get("random_seed") is not None:
            detail.append(f"seed {params['random_seed']}")
        if detail:
            st.caption(" · ".join(detail))
        # Hands the path straight to the test-extraction form instead of asking
        # for it to be copied across. Only set when empty, so a path typed by
        # hand is never overwritten.
        ext_key = f"{key_prefix}test_ext_h5_{submission_id}"
        if path and not st.session_state.get(ext_key):
            st.session_state[ext_key] = path
            st.caption(
                "Filled in as the **Test .h5 path** in feature extraction's "
                "\"Test on a subset\" step — add a checkpoint there to run it."
            )
        return

    state = status.get("test_h5_slurm_state")
    if status.get("test_h5_invalid_reason"):
        # Terminal and specific: Slurm may say COMPLETED while the file is
        # unreadable, and saying so beats an indefinite "waiting".
        st.error(f"Test .h5 is not usable: {status['test_h5_invalid_reason']}")
    elif status.get("test_writing_now"):
        st.info(f"Packaging in progress ({state or 'no Slurm record yet'}) — "
                f"the .h5 is being written to right now.")
    elif state in (None, ""):
        # Distinguished because they mean opposite things: no accounting row yet
        # for a just-submitted job, versus a job Slurm has forgotten.
        st.info("Queued — no Slurm accounting record yet.")
    else:
        st.info(f"State: {state}")


def _render_test_packaging(submission_id: str, key_prefix: str, status: dict | None = None):
    status = status or {}
    st.caption(
        "Package a handful of slides into a separate .h5 to check the pipeline "
        "before committing to a multi-hour full run. Tracked separately — it "
        "neither blocks nor is blocked by the real packaging job."
    )
    # Said explicitly because it is the one thing about this step that is not
    # guessable and caused real confusion: a finished test run does not advance
    # stage 2 or 3. Only "Full dataset" packaging sets the h5_ready that stage 3
    # gates on.
    st.caption(
        ":grey[A test run does **not** advance the pipeline — stage 3 still "
        "waits on **Full dataset** packaging. Use it to validate a checkpoint, "
        "not to produce the dataset's .h5.]"
    )
    # The pool the sample is drawn from. A subset run's manifest caps every
    # test sample at that run's size however large a number is typed below, so
    # this has to be selectable or the count silently means something else.
    pool_choice = st.radio(
        "Draw from",
        ["This run's slides", "Everything tiled on disk"],
        key=f"{key_prefix}test_pkg_scope_{submission_id}",
        horizontal=True,
        help=(
            "This run's slides is limited to the manifest fixed when it was "
            "submitted. Everything tiled on disk covers the whole raw directory, "
            "whichever run (or resume, or hand-run job) produced the tiles."
        ),
    )
    scope = "run" if pool_choice == "This run's slides" else "tiled"

    mode = st.radio(
        "Pick slides",
        ["Random N", "Specific slides"],
        key=f"{key_prefix}test_pkg_mode_{submission_id}",
        horizontal=True,
    )
    sample_size = None
    slide_names = None
    seed_input = ""
    if mode == "Random N":
        sample_size = st.number_input(
            "How many slides", min_value=1, value=3,
            key=f"{key_prefix}test_pkg_n_{submission_id}",
        )
        seed_input = st.text_input(
            "Random seed (optional)",
            key=f"{key_prefix}test_pkg_seed_{submission_id}",
            help=(
                "Leave blank for a fresh draw each time. Paste a seed reported "
                "by an earlier run to package that exact set of slides again — "
                "though an identical seed and count counts as the same attempt "
                "and will be refused as a duplicate."
            ),
        )
    else:
        raw = st.text_area(
            "Slide IDs / filenames (one per line)",
            key=f"{key_prefix}test_pkg_names_{submission_id}",
        )
        slide_names = [line.strip() for line in raw.splitlines() if line.strip()]

    # What the form is asking for *right now*, so a tracked job from an earlier
    # setup can be recognised as stale below. scope and seed are part of that:
    # changing either changes which slides get packaged, so a job submitted
    # under the previous value is exactly as stale as one with a different count.
    current_params = (
        mode,
        scope,
        int(sample_size) if sample_size else None,
        tuple(slide_names or ()),
        seed_input.strip(),
    )

    job_key = f"{key_prefix}test_pkg_job_{submission_id}"
    if st.button("Start test packaging", key=f"{key_prefix}test_pkg_submit_{submission_id}"):
        seed = None
        if mode == "Specific slides" and not slide_names:
            st.error("Enter at least one slide.")
        elif seed_input.strip() and not seed_input.strip().lstrip("-").isdigit():
            st.error("Random seed must be a whole number, or left blank.")
        else:
            if seed_input.strip():
                seed = int(seed_input.strip())
            try:
                result = client.start_test_packaging(
                    submission_id,
                    sample_size=int(sample_size) if sample_size else None,
                    slide_names=slide_names,
                    random_seed=seed,
                    scope=scope,
                )
                st.session_state[job_key] = {
                    "job_id": result.get("h5_job_id"),
                    "output_path": result.get("h5_output_path"),
                    "params": current_params,
                }
                st.success(f"Test packaging queued (job {result.get('h5_job_id')}).")
                # What was actually drawn, and from how many. The count in the
                # form only means something alongside the pool it came from.
                drawn = result.get("pool_size")
                if drawn is not None:
                    pool_label = (
                        "everything tiled on disk"
                        if result.get("scope") == "tiled"
                        else "this run's manifest"
                    )
                    st.caption(f"Drawn from {drawn} slides ({pool_label}).")
                if result.get("random_seed") is not None:
                    st.caption(
                        f"Random seed {result['random_seed']} — keep this to "
                        f"repackage the same slides."
                    )
            except requests.exceptions.HTTPError as e:
                detail, message = _error_detail(e)
                st.error(f"Test packaging failed: {message}")
                # The server rejects an over-large sample against the pool it
                # measured; offering the wider pool here saves re-reading the
                # message and hunting for the radio button that fixes it.
                if detail and detail.get("error") == "sample_larger_than_pool" \
                        and detail.get("scope") == "run":
                    st.info(
                        "Switch \"Draw from\" to \"Everything tiled on disk\" to "
                        "sample beyond this run's own slides."
                    )
            except Exception as e:
                st.error(f"Test packaging failed: {e}")

    # Server-recorded test job (migrate_dataset_runs_test_packaging.sql) takes
    # precedence over session_state, which is discarded on every reload — that
    # is what made a running test job disappear from this panel while its Slurm
    # job carried on and its .h5 kept being written.
    if status.get("test_h5_job_id"):
        _render_test_packaging_progress(submission_id, key_prefix, current_params)
        return

    # Fallback for a server that predates the migration: the job id is only in
    # session state, so it is lost on reload and the path is shown as the only
    # durable record of it.
    tracked = st.session_state.get(job_key)
    if tracked and tracked.get("job_id"):
        # Widget keys make every field above persist for the whole session, so
        # editing the slide count does not disturb the job recorded on the last
        # click — the panel below kept reporting that older run's id, path and
        # Slurm state as though it were the current selection's. Nothing on
        # screen distinguished the two, which is what made packaging look like
        # it was stuck re-running a previous 30-slide setup and ignoring the
        # numbers actually in the form. Compare and say so explicitly.
        stale = tracked.get("params") is not None and tuple(tracked["params"]) != current_params
        if stale:
            prior_n = tracked["params"][1]
            prior_desc = (
                f"{prior_n} random slide(s)" if prior_n
                else f"{len(tracked['params'][2])} named slide(s)"
            )
            st.warning(
                f"The job below is from an earlier setup ({prior_desc}), not the "
                f"selection currently in the form. Click **Start test packaging** "
                f"to run the current one."
            )
        st.caption("Earlier test packaging job:" if stale else "Test packaging job:")
        st.code(tracked["job_id"], language=None)
        try:
            test_status = client.get_test_packaging_status(
                submission_id, tracked["job_id"], tracked["output_path"]
            )
            if test_status.get("ready"):
                st.success("Test .h5 ready:")
                st.code(test_status.get("output_path"), language=None)
                st.caption("Use this path in the feature-extraction test below.")
            else:
                st.info(f"State: {test_status.get('slurm_state') or 'waiting'}")
        except Exception as e:
            st.caption(f"Couldn't read test status: {e}")


def _full_packaging_scope_caption(status: dict) -> str:
    """What the "Full dataset" option will actually package, in slides.

    The old caption ("Package every tiled slide in this run into one .h5")
    named no number and no scope, so there was nothing on screen to contradict
    the assumption that it meant the whole raw directory.
    """
    total = status.get("total_slides")
    if total is None:
        return "Package every tiled slide in this run into one .h5."
    # int() because total_slides comes back off a numeric DB column as a float,
    # which formats as "30.0 slides".
    scope = "in this run's subset manifest" if status.get("is_subset") else "in this run"
    return f"Package all {int(total):,} slides {scope} into one .h5."


@st.fragment(run_every="30s")
def _render_packaging_live_progress(submission_id: str, key_prefix: str):
    """Live view of a packaging job in flight: tiles written, bytes on disk,
    how long since the last write.

    A fragment on its own 30s timer rather than part of the 10s status poll.
    exact=False is what makes that affordable — it estimates tiles written from
    the .partial's size (one stat) instead of counting a completed-tiles
    checkpoint that reaches hundreds of MB on a run this size. The estimate is
    labelled as one; the exact count is still used for the resume decision,
    where it's fetched once rather than on a timer.
    """
    try:
        progress = client.get_packaging_progress(submission_id, exact=False)
    except Exception as e:
        st.caption(f"Couldn't read packaging progress: {e}")
        return

    done = progress.get("tiles_done")
    total = progress.get("tiles_total")
    percent = progress.get("percent")
    estimated = progress.get("tiles_done_is_estimate")
    approx = "~" if estimated else ""

    if done is not None and total:
        st.progress(
            min((percent or 0) / 100.0, 1.0),
            text=f"{approx}{done:,} / {total:,} tiles ({percent}%)",
        )
    elif done is not None:
        st.caption(f"{approx}{done:,} tiles written (total unknown).")

    left, right = st.columns(2)
    with left:
        st.metric(".h5 written so far", _human_bytes(progress.get("partial_bytes") or 0))
        if total and progress.get("bytes_per_tile"):
            st.caption(
                f"Projected final size: "
                f"{_human_bytes(total * progress['bytes_per_tile'])}"
            )
    with right:
        since = progress.get("seconds_since_write")
        st.metric(
            "Last write",
            "just now" if since is None or since < 60 else f"{int(since // 60)} min ago",
        )
        skipped = progress.get("skipped_tiles")
        if skipped:
            st.caption(f"{skipped:,} tiles skipped (unreadable JPEGs).")

    if estimated:
        st.caption(
            "Tile count is estimated from the file's size — every tile occupies "
            "exactly the same number of bytes, so it's close, but the exact figure "
            "comes from the checkpoint and is only read when deciding a resume."
        )
    if not progress.get("writing_now"):
        st.warning(
            "Nothing has been written for a while — the job may have been killed. "
            "Slurm state: "
            f"{progress.get('slurm_state') or 'unavailable'}."
        )
    st.caption(f"Output: `{progress.get('output_path')}`")


def _render_run_scoped_packaging(status: dict, submission_id: str, key_prefix: str, covered: str):
    """Packaging limited to this run's own manifest — the original behaviour,
    now the secondary option behind packaging everything that's tiled."""
    if status.get("h5_ready"):
        st.success("Already packaged:")
        st.code(status.get("h5_output_path"), language=None)
        st.caption(
            "Repackaging would rebuild this .h5 from the same slides. Use the "
            "full-coverage option above if you want more of them."
        )
        return
    st.caption(_full_packaging_scope_caption(status))
    _render_start_packaging_button(
        submission_id, key_prefix,
        button_label=f"Package this run's {covered} slides",
    )


def _submit_tiled_scope_packaging(
    submission_id: str, allow_incomplete: bool, resume: bool | None = False,
):
    """Package every slide with tiles on disk, regardless of which run made
    them (scope="tiled" server-side).

    allow_incomplete is passed through as the user's answer to the disk check
    they were just shown: the server refuses scope="tiled" outright when any
    slide in the directory is untiled, unless told the gap is intentional.
    """
    try:
        result = client.start_packaging_job(
            submission_id, allow_incomplete=allow_incomplete, scope="tiled",
            resume=resume,
        )
        st.success(f"Packaging queued (job {result.get('h5_job_id')}).")
        st.code(result.get("h5_output_path"), language=None)
        st.rerun()
    except requests.exceptions.HTTPError as e:
        detail, message = _error_detail(e)
        if detail and detail.get("error") == "tiling_incomplete":
            st.error(detail.get("message", message))
        else:
            st.error(f"Failed to start packaging: {message}")
    except Exception as e:
        st.error(f"Failed to start packaging: {e}")


def _render_full_dataset_run(status: dict, submission_id: str, key_prefix: str):
    """"Full dataset" for a run that was submitted as a subset.

    Packaging can only ever cover slides this run tiled, so honouring the label
    means starting a *new* run over every slide in the raw directory rather
    than re-packaging the subset. Reusing the same dataset_name is what keeps
    that cheap: submit_mask_tile_slurm.py skips any slide that already has a
    non-empty _tile_metadata.csv and _tiling_summary.json under
    tile_dir/<dataset_name>/<slide_id>/, so only the untiled remainder is
    actually processed.
    """
    raw_dir = status.get("raw_dir") or ""
    # Falls back to the directory's own name, which is exactly what the server
    # defaults to when dataset_name is omitted — so the tile-skip still lines
    # up even against a backend too old to return the field.
    dataset_name = status.get("dataset_name") or Path(raw_dir).name
    total = status.get("total_slides")
    covered = f"{int(total):,}" if total is not None else "some"

    st.caption(f"Directory: `{raw_dir}` · dataset folder `{dataset_name}`")

    # Ask disk, not the manifest. The run's manifest is frozen at 30 slides,
    # but the dataset folder may already hold tiles for the whole directory
    # from earlier runs — in which case packaging everything needs no tiling at
    # all, and telling the user to start a tiling run (as this step used to,
    # unconditionally) is simply wrong.
    coverage = None
    with st.spinner("Checking which slides have tiles on disk…"):
        try:
            coverage = client.get_tiled_coverage(submission_id)
        except Exception as e:
            st.caption(f"Couldn't check tile coverage on disk: {e}")

    if coverage is None:
        # No disk answer available (usually a backend too old to have the
        # endpoint). Don't guess in either direction.
        st.warning(
            "Couldn't read tile coverage from disk, so there's no way to tell here "
            "how much of the directory is tiled. Package this run's "
            f"{covered} slides below, or check the dataset folder directly."
        )
        _render_run_scoped_packaging(status, submission_id, key_prefix, covered)
        return

    in_dir = coverage.get("slides_in_directory") or 0
    tiled = coverage.get("slides_tiled") or 0
    untiled = coverage.get("slides_untiled") or 0

    if untiled == 0 and tiled:
        st.success(
            f"All **{tiled:,} slides** in this directory are tiled on disk. "
            f"No tiling needed — package them all now."
        )
    else:
        st.warning(
            f"**{tiled:,} of {in_dir:,}** slides in this directory have tiles on disk. "
            f"The remaining **{untiled:,}** need tiling before they can be packaged."
        )
        if coverage.get("untiled_sample"):
            with st.expander(f"Show untiled slides ({untiled:,})"):
                st.code("\n".join(coverage["untiled_sample"]), language=None)
                if untiled > len(coverage["untiled_sample"]):
                    st.caption(
                        f"Showing the first {len(coverage['untiled_sample'])} of {untiled:,}."
                    )

    if tiled:
        st.caption(
            f"Packages all {tiled:,} tiled slides into `{dataset_name}` — a separate "
            f"output from this run's own subset .h5, which is left untouched."
        )
        # An earlier full-coverage attempt may have been interrupted, leaving a
        # checkpoint at this very target — which is exactly the case that must
        # not resume silently. Ask before offering the button.
        prior = None
        try:
            prior = client.get_packaging_progress(submission_id)
        except Exception:
            pass
        resume_choice = False
        if prior and prior.get("resumable"):
            prior_done = prior.get("tiles_done") or 0
            st.warning(
                f"An unfinished attempt at this output is already on disk — "
                f"{prior_done:,} tiles "
                f"({_human_bytes(prior.get('partial_bytes') or 0)}). Choose what "
                f"happens to it; nothing is resumed automatically."
            )
            st.caption(f"`{prior.get('output_path')}`")
            pick = st.radio(
                "That attempt",
                [
                    f"Resume — keep the {prior_done:,} tiles already written",
                    "Start fresh — discard them and repackage everything",
                ],
                key=f"{key_prefix}pkg_tiled_resume_{submission_id}",
                index=0,
            )
            resume_choice = pick.startswith("Resume")
            if not resume_choice:
                fresh_key = f"{key_prefix}pkg_tiled_fresh_confirm_{submission_id}"
                st.error(
                    f"This deletes the checkpoint and the "
                    f"{_human_bytes(prior.get('partial_bytes') or 0)} `.partial`, and "
                    f"re-decodes all {prior_done:,} tiles. It cannot be undone."
                )
                st.checkbox(
                    f"Yes, discard {prior_done:,} tiles and start over", key=fresh_key,
                )
                if not st.session_state.get(fresh_key):
                    st.button(
                        f"Package all {tiled:,} tiled slides",
                        key=f"{key_prefix}pkg_tiled_scope_{submission_id}",
                        type="primary", disabled=True,
                    )
                    return
        if st.button(
            f"Resume packaging ({(prior or {}).get('tiles_done') or 0:,} done)"
            if resume_choice else f"Package all {tiled:,} tiled slides",
            key=f"{key_prefix}pkg_tiled_scope_{submission_id}",
            type="primary",
        ):
            _submit_tiled_scope_packaging(
                submission_id, allow_incomplete=bool(untiled), resume=resume_choice,
            )
        if untiled:
            st.caption(
                f"This packages only the {tiled:,} that are ready; the {untiled:,} "
                f"untiled slides are left out of the .h5."
            )

    if untiled == 0:
        with st.expander(f"Or just (re)package this run's {covered} slides"):
            _render_run_scoped_packaging(status, submission_id, key_prefix, covered)
        return

    st.divider()
    st.caption(f"**Tile the {untiled:,} missing slides** — starts a new tiling run:")

    # Inherited from the run rather than retyped. Tiling the remainder of a
    # dataset at different settings than the part already on disk produces one
    # dataset with two tile geometries, and a slider defaulting to 30.0 made
    # that the easy mistake to make. None means the run predates the
    # tiling_params column, which is the only case still worth asking about.
    recorded_params = status.get("tiling_params")
    if recorded_params:
        st.caption(
            "Reusing this run's own tiling settings, so the new slides match the "
            "ones already tiled:"
        )
        st.code(
            "\n".join(f"{k} = {v}" for k, v in sorted(recorded_params.items())),
            language=None,
        )
        min_tissue = None
    else:
        st.warning(
            "This run has no recorded tiling settings (it predates them being "
            "saved), so they can't be inherited. Set the minimum tissue % to "
            "whatever the original run used — a different value here leaves the "
            "dataset tiled two different ways."
        )
        min_tissue = st.slider(
            "Minimum tissue % per tile (for the slides not yet tiled)",
            min_value=0.0, max_value=100.0, value=30.0, step=5.0,
            key=f"{key_prefix}full_run_min_tissue_{submission_id}",
        )

    confirm_key = f"{key_prefix}full_run_confirm_{submission_id}"
    st.checkbox(
        "Yes — queue tiling for the rest of the directory", key=confirm_key,
        help="Submits a new tiling run on the HPC. It does not touch this run or its .h5, "
             "and it does not package anything.",
    )
    if st.button(
        "Tile the rest of the directory (new run)",
        key=f"{key_prefix}full_run_submit_{submission_id}",
        disabled=not st.session_state.get(confirm_key),
    ):
        try:
            result = client.submit_dataset_job(
                dataset_path=raw_dir,
                # Only one of these is ever sent: the recorded block when the
                # run has one (so nothing is re-derived), or a hand-set
                # min_tissue when it doesn't. Sending min_tissue alongside
                # tiling_params would override the recorded value with it.
                min_tissue=None if recorded_params else float(min_tissue),
                tiling_params=recorded_params,
                # The point of the whole branch: no sample_size, no
                # slide_names, so the server records is_subset=False and builds
                # a manifest of every slide it discovers in raw_dir.
                sample_size=None,
                slide_names=None,
                partition=status.get("partition") or None,
                notify_email=status.get("notify_email") or None,
                dataset_name=dataset_name or None,
            )
            st.success(
                f"Tiling queued for the whole directory — submission "
                f"{result['submission_id']}. Find it under 'Recent dataset jobs'; "
                f"**package it from that run's step 2 once its tiling finishes.** "
                f"This run and its .h5 are untouched."
            )
        except requests.exceptions.HTTPError as e:
            st.error(f"Couldn't start the full run: {_error_detail(e)[1]}")
        except Exception as e:
            st.error(f"Couldn't start the full run: {e}")

    with st.expander(f"Or just (re)package this run's {covered} slides"):
        _render_run_scoped_packaging(status, submission_id, key_prefix, covered)


def _render_packaging_step(status: dict, submission_id: str, key_prefix: str, state: str):
    # The mode selector is rendered before the "blocked" check, not after it.
    # Blocking only applies to the full-dataset run, which genuinely does need
    # every tiling task terminal — but test packaging is explicitly designed to
    # be independent of that (it submits with allow_incomplete=True precisely
    # so a handful of slides can be checked while the rest of the dataset is
    # still tiling). Returning early on "blocked" hid the whole selector, so
    # neither option was reachable and the step looked like it had no controls
    # at all — the single most common state to catch it in, since that is most
    # of a long run's lifetime.
    # Default to whichever mode this run actually has activity in — not
    # always "Full dataset" — so reopening this step lands you on your own
    # job instead of requiring you to remember to click over to "Test on a
    # subset" to find it. Only affects the widget's FIRST render for this
    # key; once st.radio has session state, a later rerun of this exact
    # widget always uses that instead of `index`, so an explicit switch by
    # the user is never overridden by this.
    default_mode = (
        1 if status.get("test_h5_job_id") and not status.get("h5_job_id") else 0
    )
    mode = st.radio(
        "Run",
        ["Full dataset", "Test on a subset"],
        index=default_mode,
        key=f"{key_prefix}pkg_mode_{submission_id}",
        horizontal=True,
    )
    if mode == "Test on a subset":
        _render_test_packaging(submission_id, key_prefix, status)
        return

    if status.get("is_subset"):
        # A subset run cannot package the whole directory, and no button on
        # this step could make it: the remaining slides were never tiled under
        # this run, so there are no tiles of theirs to package. Offering "Full
        # dataset" as a *packaging* action here was therefore mislabelled — it
        # silently meant "all N slides of this subset", so clicking it re-ran
        # the same N and overwrote the .h5 already sitting there. The only
        # thing that can honour the label is a new run covering every slide in
        # the directory, so that is what it now does.
        _render_full_dataset_run(status, submission_id, key_prefix)
        return

    if state == "blocked":
        st.caption(
            "Waiting on tiling. Full-dataset packaging can start once every "
            "tiling task has reached a terminal state — or use "
            "'Test on a subset' above to check a few slides now."
        )
        return

    if status.get("h5_ready"):
        st.success("Packaged .h5 ready:")
        st.code(status.get("h5_output_path"), language=None)
        return

    h5_job_id = status.get("h5_job_id")
    if not h5_job_id:
        st.caption(_full_packaging_scope_caption(status))
        if status.get("slurm_unreachable"):
            st.warning(
                "The server can't reach Slurm (`sacct`/`squeue`), so tiling's state "
                "couldn't be verified — starting packaging is allowed, but check "
                "tiling really has finished first. The submitted job carries its own "
                "Slurm dependency as a backstop."
            )
        _render_start_packaging_button(submission_id, key_prefix)
        return

    st.caption("Packaging job:")
    st.code(h5_job_id, language=None)
    h5_state = status.get("h5_slurm_state")
    if h5_state in _SLURM_IN_FLIGHT or status.get("h5_packaging_active"):
        if h5_state in _SLURM_IN_FLIGHT:
            st.info(f"Running (Slurm state: {h5_state}).")
        else:
            # No usable Slurm state, but the .partial is being written — see
            # _packaging_write_activity. Saying so beats the old silence, which
            # was indistinguishable from the job having never started.
            st.info(
                "Running — the .h5 is being written to right now "
                "(Slurm state unavailable, judged from the output file)."
            )
        _render_packaging_live_progress(submission_id, key_prefix)
        return

    if status.get("h5_invalid_reason"):
        st.error(f"The .h5 exists but is not usable: {status['h5_invalid_reason']}")
    else:
        st.warning(f"Previous attempt ended as: {h5_state or 'no Slurm record'}")

    # Only fetched here — for a run that is actually interrupted — because it
    # counts the lines of a checkpoint that can be hundreds of MB. Putting it
    # in the 10s status poll would make every open tab pay for it.
    progress = None
    try:
        progress = client.get_packaging_progress(submission_id)
    except Exception as e:
        st.caption(f"Couldn't read checkpoint progress: {e}")

    if progress and progress.get("resumable"):
        _render_resume_choice(progress, submission_id, key_prefix)
    else:
        if progress is not None and not progress.get("resumable"):
            st.caption("No resumable checkpoint on disk — this will start from the beginning.")
        _render_start_packaging_button(
            submission_id, key_prefix, button_label="Retry packaging (.h5)",
            # Nothing to resume, so say so explicitly rather than leaving the
            # server to infer it from an absent checkpoint.
            resume=False,
        )


def _render_resume_choice(progress: dict, submission_id: str, key_prefix: str):
    """Make continuing a previous attempt an explicit, confirmed choice.

    Resuming used to just happen: a checkpoint on disk meant the next
    submission silently continued it, and the button only changed its label.
    That's the wrong default for a decision with real consequences in both
    directions — resuming keeps whatever an earlier attempt wrote (including,
    say, tiles decoded from JPEGs that have since been re-made), while starting
    fresh throws away hours of work and terabytes of .partial. Neither belongs
    in a label; both are now selected deliberately, and the destructive one
    needs a second confirmation.
    """
    done = progress.get("tiles_done") or 0
    total_tiles = progress.get("tiles_total")
    percent = progress.get("percent")
    partial = _human_bytes(progress.get("partial_bytes") or 0)

    st.warning(
        f"**An unfinished attempt is on disk.** {done:,} tiles were already written "
        f"({partial}). Choose what to do with it — nothing is resumed automatically."
    )
    if total_tiles:
        st.progress(
            min((percent or 0) / 100.0, 1.0),
            text=f"{done:,} / {total_tiles:,} tiles ({percent}%)",
        )

    choice = st.radio(
        "This attempt",
        [
            f"Resume — keep the {done:,} tiles already written",
            "Start fresh — discard them and repackage everything",
        ],
        key=f"{key_prefix}pkg_resume_choice_{submission_id}",
        index=0,
    )
    resume = choice.startswith("Resume")

    if resume:
        st.caption(
            f"Packaging continues from tile {done:,}. Tiles already written are not "
            f"re-decoded, so this is much faster — but it also keeps whatever that "
            f"attempt wrote. Start fresh instead if the tiles on disk have changed "
            f"since it ran."
        )
        _render_start_packaging_button(
            submission_id, key_prefix,
            button_label=f"Resume packaging ({done:,} tiles already done)",
            resume=True,
        )
        return

    confirm_key = f"{key_prefix}pkg_fresh_confirm_{submission_id}"
    st.error(
        f"Starting fresh deletes the checkpoint and the {partial} `.partial`, and "
        f"re-decodes all {done:,} tiles already done. This cannot be undone."
    )
    st.checkbox(
        f"Yes, discard {done:,} tiles and repackage from scratch",
        key=confirm_key,
    )
    _render_start_packaging_button(
        submission_id, key_prefix,
        button_label="Discard and repackage from scratch",
        resume=False,
        disabled=not st.session_state.get(confirm_key),
    )


def _render_test_extraction(submission_id: str, key_prefix: str):
    st.caption(
        "Validate a checkpoint against a small .h5 (typically one from test "
        "packaging above) before running it over the full dataset."
    )
    h5_path = st.text_input(
        "Test .h5 path",
        key=f"{key_prefix}test_ext_h5_{submission_id}",
        placeholder="/path/to/hdf5_..._test_sample_....h5",
    )
    checkpoint = st.text_input(
        "Model checkpoint path",
        key=f"{key_prefix}test_ext_ckpt_{submission_id}",
        placeholder="/path/to/BarlowTwins_3.ckt",
    )
    job_key = f"{key_prefix}test_ext_job_{submission_id}"
    if st.button("Start test extraction", key=f"{key_prefix}test_ext_submit_{submission_id}"):
        if not h5_path.strip() or not checkpoint.strip():
            st.error("Both a .h5 path and a checkpoint are required.")
        else:
            try:
                result = client.start_test_feature_extraction(
                    submission_id, h5_path.strip(), checkpoint.strip()
                )
                st.session_state[job_key] = {
                    "job_id": result.get("extraction_job_id"),
                    "output_path": result.get("expected_output_path"),
                }
                st.success(f"Test extraction queued (job {result.get('extraction_job_id')}).")
            except requests.exceptions.HTTPError as e:
                st.error(f"Test extraction failed: {_error_detail(e)[1]}")
            except Exception as e:
                st.error(f"Test extraction failed: {e}")

    tracked = st.session_state.get(job_key)
    if tracked and tracked.get("job_id"):
        st.caption("Test extraction job:")
        st.code(tracked["job_id"], language=None)
        try:
            test_status = client.get_test_feature_extraction_status(
                submission_id, tracked["job_id"], tracked["output_path"]
            )
            if test_status.get("ready"):
                st.success("Test features ready:")
                st.code(test_status.get("output_path"), language=None)
            else:
                st.info(f"State: {test_status.get('slurm_state') or 'waiting'}")
        except Exception as e:
            st.caption(f"Couldn't read test status: {e}")


def _render_extraction_step(status: dict, submission_id: str, key_prefix: str, state: str):
    mode = st.radio(
        "Run",
        ["Full dataset", "Test on a sample .h5"],
        key=f"{key_prefix}ext_mode_{submission_id}",
        horizontal=True,
    )
    if mode == "Test on a sample .h5":
        _render_test_extraction(submission_id, key_prefix)
        return

    if state == "blocked":
        st.caption(
            "Waiting on packaging. The server requires the .h5 to be complete "
            "and readable before extraction can start."
        )
        return

    if status.get("extraction_ready"):
        st.success("Features ready:")
        st.code(status.get("extraction_output_path"), language=None)
        return

    extraction_job_id = status.get("extraction_job_id")
    if not extraction_job_id:
        st.caption("Run the packaged .h5 through the model:")
        _render_extract_features_form(submission_id, key_prefix)
        return

    st.caption("Feature extraction job:")
    st.code(extraction_job_id, language=None)
    ext_state = status.get("extraction_slurm_state")
    if ext_state in _SLURM_IN_FLIGHT:
        st.info(f"Running (Slurm state: {ext_state}).")
        return
    st.warning(f"This attempt ended as: {ext_state or 'no Slurm record'}")
    if status.get("extraction_invalid_reason"):
        st.error(
            f"An output file exists but is not usable: "
            f"{status['extraction_invalid_reason']}"
        )
        st.caption(
            "Retrying clears that file first — the encoder skips its work when an "
            "output is already in place, so it has to go before a rerun can succeed."
        )
    st.caption("Retry with the same or a different checkpoint:")
    _render_extract_features_form(
        submission_id, key_prefix, button_label="Retry feature extraction"
    )


def _render_assignment_step(status: dict, submission_id: str, key_prefix: str, state: str):
    """Stage 4: assign HPL cluster IDs to this run's embeddings.

    Cheap to redo compared with the stages before it — it reads embeddings, not
    images, and runs on CPU in minutes — so this offers a plain retry rather
    than the stale-output ceremony extraction needs.
    """
    if status.get("assignment_ready"):
        st.success("Cluster assignments ready:")
        st.code(status.get("assignment_output_path"), language=None)
        if status.get("assignment_reference"):
            st.caption(f"Reference: {status['assignment_reference']}")
        # Which vote produced it. Surfaced here because the CSV cannot carry it
        # — Stage 5 finds its cluster column by elimination, so an extra column
        # there would break the load. Absent for runs assigned before this was
        # recorded, and for deployments without
        # migrate_dataset_runs_assignment_vote.sql applied.
        if status.get("assignment_vote"):
            st.caption(f"Vote: {status['assignment_vote']}")
        # Offered here rather than in Stage 5 because it decides whether to go on
        # to Stage 5 at all. The load can be perfectly clean while the cluster
        # IDs mean nothing for this cohort.
        _render_cohort_shift(submission_id, key_prefix)
        return

    if state == "blocked":
        # Blocked only means *this run's* tracked extraction has not produced a
        # usable output. Assigning an existing projections file by path does not
        # depend on that at all, and returning here would have made embeddings
        # that already exist on disk unreachable from the UI — the exact case of a
        # subset encoded before the run was being tracked.
        _render_assign_clusters_form(
            status, submission_id, key_prefix,
            button_label="Start cluster classification", full_available=False,
        )
        return

    assignment_job_id = status.get("assignment_job_id")
    if assignment_job_id:
        st.caption("Cluster assignment job:")
        st.code(assignment_job_id, language=None)
        asg_state = status.get("assignment_slurm_state")
        if asg_state in _SLURM_IN_FLIGHT:
            st.info(f"Running (Slurm state: {asg_state}).")
            return
        st.warning(f"This attempt ended as: {asg_state or 'no Slurm record'}")
        if status.get("assignment_invalid_reason"):
            st.error(f"An output exists but is not usable: {status['assignment_invalid_reason']}")

    _render_assign_clusters_form(
        status, submission_id, key_prefix,
        button_label="Retry cluster assignment" if assignment_job_id else "Start cluster assignment",
    )


def _render_anorak_step(status: dict, submission_id: str, key_prefix: str, state: str):
    """Stage 7: ANORAK growth-pattern segmentation and IASLC grading.

    The one stage that runs as a Nextflow pipeline rather than a Slurm job of
    its own, which changes what there is to show. The job id here is a *head
    process* — it submits a job per slide per stage itself — so its Slurm state
    answers "is the pipeline alive" and nothing about how far through it is.
    Progress lives in the run's own output directory, which is why that path is
    shown whether the run is finished or still going.
    """
    if status.get("anorak_ready"):
        st.success("Growth pattern grading complete:")
        st.code(status.get("anorak_grades_csv"), language=None)
        scope = status.get("anorak_scope")
        slides = status.get("anorak_slides")
        if scope == "subset":
            # The seed is shown, not buried in the run record: a subset result
            # that disagrees with a later full run is explained by which slides
            # it saw, and that is the only way to ask for them again.
            st.caption(
                f"Random subset: {slides:,} of {status.get('anorak_sample_size') or slides} "
                f"requested, seed {status.get('anorak_seed')}. "
                f"This is a test run — the full cohort has not been graded."
            )
        elif slides:
            st.caption(f"Full slide list: {slides:,} slides.")
        if status.get("anorak_out_dir"):
            st.caption("Per-slide masks, proportions and the Nextflow report:")
            st.code(status["anorak_out_dir"], language=None)
        # overwrite, and only here: it means "replace a finished table", which
        # the server refuses to do unasked. It never gets past a live head job.
        _render_anorak_form(status, submission_id, key_prefix,
                            button_label="Run again", expanded=False, overwrite=True)
        return

    if status.get("anorak_error"):
        st.error(status["anorak_error"])

    job_id = status.get("anorak_job_id")
    if job_id:
        st.caption("Nextflow head job:")
        st.code(job_id, language=None)
        anorak_state = status.get("anorak_slurm_state")
        if status.get("anorak_in_flight") or anorak_state in _SLURM_IN_FLIGHT:
            st.info(
                f"Pipeline running (head job state: {anorak_state}). It submits "
                f"one job per slide per stage, so `squeue` shows many more jobs "
                f"than this one."
            )
            if status.get("anorak_out_dir"):
                st.caption("Live progress — Nextflow's own trace and report:")
                st.code(f"{status['anorak_out_dir']}/pipeline_info", language=None)
            return
        if status.get("anorak_submit_blocked"):
            # The server's refusal, shown instead of a form it would refuse:
            # with Slurm unreachable the head job may still be running, and
            # a second one would rewrite its slide list underneath it.
            st.warning(status["anorak_submit_blocked"])
            return
        st.warning(f"This attempt ended as: {anorak_state or 'no Slurm record'}")
        if status.get("anorak_invalid_reason"):
            st.error(f"An output exists but is not a finished grading table: "
                     f"{status['anorak_invalid_reason']}")
        # Written by the head job's supervisor when the run ended for good —
        # the reason, and why no standby took over. A bare FAILED was all this
        # used to say, with the reason in a file nobody was pointed at.
        if status.get("anorak_stop_reason"):
            st.caption("Why it stopped (the supervisor's stop marker):")
            st.code(status["anorak_stop_reason"], language=None)
        if status.get("anorak_out_dir"):
            st.caption("The head job's log is the first place to look:")
            st.code(f"{status['anorak_out_dir']}/nextflow.log", language=None)

    _render_anorak_form(
        status, submission_id, key_prefix,
        button_label="Retry ANORAK" if job_id else "Run ANORAK",
    )


def _render_anorak_run_step(status: dict, submission_id: str, key_prefix: str):
    """An ANORAK run started on its own (Run ANORAK): what it graded, where its
    outputs are, and — if it stopped short — why, and Resume."""
    if status.get("anorak_tumour_verified") is False:
        st.warning("Every slide in the directory was graded; tumour status was not "
                   "checked, so non-tumour slides are in the grading table too.")
    if status.get("anorak_scope") == "subset":
        st.caption(f"Test run: {status.get('anorak_slides')} slides sampled at random, "
                   f"seed {status.get('anorak_seed')}.")
    elif status.get("anorak_slides"):
        st.caption(f"{status['anorak_slides']:,} slides.")
    if status.get("anorak_error"):
        st.error(status["anorak_error"])
    if status.get("anorak_ready"):
        st.success("Growth-pattern grading complete:")
        st.code(status.get("anorak_grades_csv"), language=None)
    elif status.get("anorak_in_flight"):
        st.info(f"ANORAK running (head job {status.get('anorak_job_id')}, "
                f"{status.get('anorak_slurm_state')}).")
    elif status.get("anorak_job_id"):
        st.warning(f"ANORAK stopped ({status.get('anorak_slurm_state') or 'no Slurm record'}).")
        if status.get("anorak_stop_reason"):
            st.caption("Why it stopped (the supervisor's stop marker):")
            st.code(status["anorak_stop_reason"], language=None)
        if status.get("anorak_invalid_reason"):
            st.caption(f"Output not usable: {status['anorak_invalid_reason']}")
        if not status.get("anorak_submit_blocked") and st.button(
                "Resume ANORAK", type="primary", key=f"{key_prefix}anorak_resume_{submission_id}"):
            try:
                result = client.resume_anorak_run(submission_id)
            except requests.exceptions.HTTPError as e:
                st.error(f"Refused: {_http_detail(e)}")
            except Exception as e:
                st.error(f"Resume failed: {e}")
            else:
                st.success(f"Resumed — head job {result.get('anorak_job_id')}.")
    if status.get("anorak_out_dir"):
        st.caption("Run directory (masks, proportions, nextflow.log, report):")
        st.code(status["anorak_out_dir"], language=None)


def _render_anorak_form(status: dict, submission_id: str, key_prefix: str,
                        button_label: str = "Run ANORAK", expanded: bool = True,
                        overwrite: bool = False):
    """The slide list, the scope, and the one button.

    Resubmitting resumes by default. Nextflow caches on task inputs, so a retry
    after a fixed container or a raised time limit re-runs only what actually
    failed — which for a cohort this size is the difference between an hour and
    a week.
    """
    with st.expander("Run settings", expanded=expanded):
        st.caption(
            "ANORAK segments growth patterns in lung adenocarcinoma, so it runs "
            "only on slides that carry tumour. Point it at the output of "
            "`select_tumour_slides.py --out`, which writes tumour slides only, "
            "optionally filtered by `filter_slides_by_tile_count.py` for a floor "
            "on how much malignant tissue a verdict rests on. The run refuses the "
            "list before queueing anything if a row has a blank sample or is not "
            "marked as tumour."
        )
        # Deliberately NOT pre-filled with anorak_slide_list. That field holds
        # the run's own copy of the list, inside its output directory, so
        # pre-filling it fed a re-run its own output — and a second submission
        # then truncated the file the first run's head job was reading, which
        # surfaced as "Missing 'header' in CSV file" with nothing pointing back
        # here. The previous path is shown below instead, to copy if wanted.
        slides_csv = st.text_input(
            "Tumour-slide list (.csv)",
            value="",
            key=f"{key_prefix}_anorak_csv",
            help="Absolute path on the HPC filesystem. Needs a slide_id column "
                 "and a samples column naming each slide's tumour, filled in on "
                 "every row: grades are pooled by sample, and a blank one used "
                 "to pool unrelated slides into one tumour. An is_tumour column, "
                 "if present, must be true on every row. This is the source list "
                 "— the run writes its own copy beside its outputs.",
        )
        if status.get("anorak_slide_list"):
            st.caption("The previous attempt ran on this list — a copy, inside "
                       "the run's own output directory. Point at the source it "
                       "came from, not at this:")
            st.code(status["anorak_slide_list"], language=None)

        # "Test" and "production" are the same pipeline over a different number
        # of slides, deliberately: a separate test mode would be evidence about
        # the test mode. The only thing that changes is how many slides the
        # list is narrowed to.
        scope_label = st.radio(
            "Scope",
            ["Full slide list (production)", "Random subset (test)"],
            key=f"{key_prefix}_anorak_scope",
            horizontal=True,
            help="A subset runs every stage exactly as the full cohort does, "
                 "on fewer slides — so a subset that works is evidence the "
                 "full run will.",
        )
        subset = scope_label.startswith("Random subset")

        sample_size = None
        seed = None
        if subset:
            columns = st.columns(2)
            sample_size = columns[0].number_input(
                "Slides to sample", min_value=1, value=10, step=1,
                key=f"{key_prefix}_anorak_n",
                help="Sampled at random across the list rather than taken from "
                     "the top — the first N slides of a cohort are usually one "
                     "or two patients, sharing a scanner, a batch and a stain run.",
            )
            seed_text = columns[1].text_input(
                "Seed (optional)", value="",
                key=f"{key_prefix}_anorak_seed",
                help="Leave blank and one is chosen and recorded, so the sample "
                     "can be asked for again either way. On a retry of a subset "
                     "run, blank repeats the previous sample — the server "
                     "refuses if it cannot.",
            )
            if status.get("anorak_scope") == "subset" and status.get("anorak_seed") is not None:
                columns[1].caption(f"Previous attempt: seed {status['anorak_seed']}.")
            seed = int(seed_text) if seed_text.strip().isdigit() else None

        resume = st.checkbox(
            "Continue the cached run", value=True,
            key=f"{key_prefix}_anorak_resume",
            help="Nextflow re-runs only the tasks whose inputs changed. Uncheck "
                 "to start from scratch — which for a full cohort is days.",
        )

        # Capped by the partition's MaxTime, and a submission over that ceiling
        # is rejected by sbatch rather than trimmed to fit — so it is here
        # rather than only in the server's environment. Blank uses the
        # deployment default (ANORAK_HEAD_TIME_LIMIT).
        time_limit = st.text_input(
            "Head job walltime (optional)", value="",
            key=f"{key_prefix}_anorak_time",
            help="Slurm format, e.g. 2-00:00:00 or 48:00:00. Leave blank for "
                 "the server's default. Must be within the partition's MaxTime "
                 "(sinfo -o '%P %l'); the head job has to outlive every job it "
                 "submits, so pick the largest allowed.",
        )

        # Standbys resume the run before them, so without resume there is
        # nothing for one to continue — the server refuses the combination.
        chain = st.number_input(
            "Head jobs", min_value=1, value=2 if resume else 1, step=1,
            key=f"{key_prefix}_anorak_chain_{int(resume)}",
            disabled=not resume,
            help="The first head job plus standbys. A standby starts only if the "
                 "one before it ended without finishing — it reached its walltime "
                 "or ran out of watchdog restarts — and resumes it. A real failure "
                 "or a scancel stops the whole chain. Needs 'Continue the cached "
                 "run'; without it one head job is submitted.",
        )

        if st.button(button_label, key=f"{key_prefix}_anorak_go", type="primary"):
            if not slides_csv.strip():
                st.error("A slide list is required.")
                return
            try:
                result = client.start_anorak(
                    submission_id,
                    slides_csv=slides_csv.strip(),
                    scope="subset" if subset else "full",
                    sample_size=int(sample_size) if subset else None,
                    seed=seed,
                    resume=resume,
                    overwrite=overwrite,
                    time_limit=time_limit.strip() or None,
                    chain=int(chain) if resume else 1,
                )
            except requests.exceptions.HTTPError as e:
                st.error(_http_detail(e))
                return
            except Exception as e:
                st.error(f"Submission failed: {e}")
                return

            selection = result.get("selection") or {}
            if selection.get("scope") == "subset":
                st.success(
                    f"Submitted: {selection['slides']} slides sampled at random "
                    f"from {selection.get('pool')}, seed {selection.get('seed')} "
                    f"(job {result.get('anorak_job_id')})."
                )
            else:
                st.success(
                    f"Submitted: {selection.get('slides')} slides "
                    f"(job {result.get('anorak_job_id')})."
                )
            st.caption("The list this run was given, kept beside its outputs:")
            st.code(result.get("slide_list"), language=None)


def _http_detail(error) -> str:
    """FastAPI's `detail`, not the JSON envelope around it.

    The convention elsewhere in this file is to show e.response.text, which for a
    long multi-line message renders as {"detail":"...\n..."} with the newlines
    escaped — unreadable exactly when the message matters most.
    """
    response = getattr(error, "response", None)
    if response is None:
        return str(error)
    try:
        payload = response.json()
    except Exception:
        return response.text
    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    return detail if isinstance(detail, str) else str(detail)


_SHIFT_STYLE = {
    "consistent": ("", st.success),
    "notice": ("", st.warning),
    "alarm": ("", st.error),
}


def _render_cohort_shift(submission_id: str, key_prefix: str) -> None:
    """Is this cohort's tissue in the reference at all?

    Not run automatically. It is cheap, but it is a distinct question from "did
    Stage 4 work", and a result nobody asked for is a result nobody reads. The
    button states the question so the answer means something when it appears.
    """
    with st.expander("Is this cohort represented in the reference?", expanded=False):
        st.caption(
            "k-NN gives every tile its nearest cluster however far away that "
            "cluster is, and reports nothing. So a cohort from a different "
            "scanner or stain still produces a complete assignment and per-slide "
            "proportions that differ — differences that read as biology. This "
            "compares this dataset's distance-to-reference against the "
            "reference's own, which is the cheapest way to tell the two apart."
        )
        # State the prerequisite before the button, not after. The reference
        # profile is a one-off job reused by every dataset, so "not built yet" is
        # a setup step — turning it into a 400 that renders as a JSON envelope
        # with a command mangled inside is how a check nobody can run gets built.
        try:
            readiness = client.cohort_shift_readiness(submission_id)
        except Exception:
            readiness = None

        if readiness and not readiness.get("has_profile"):
            st.warning(
                f"No reference baseline for the '{readiness.get('vote_preset')}' "
                f"vote yet. It is a one-off leave-one-out over the reference "
                f"(~20 minutes) and is then reused by every dataset. Run this "
                f"once on the cluster:"
            )
            st.code(readiness.get("build_profile_command", ""), language="bash")
            st.caption(f"It will be written to {readiness.get('profile_path')}, "
                       f"where this page looks for it.")
            return
        if readiness and not readiness.get("has_assignments"):
            st.info("No assignment CSV for this run yet — finish Stage 4 first.")
            return

        if not st.button("Check", key=f"{key_prefix}shift_{submission_id}"):
            return
        try:
            result = client.check_cohort_shift(submission_id)
        except requests.exceptions.HTTPError as e:
            st.error(_http_detail(e))
            return

        icon, box = _SHIFT_STYLE.get(result.get("level"), ("", st.info))
        box(f"{result.get('level', '?').upper()} — {result.get('verdict', '')}")

        left, middle, right = st.columns(3)
        left.metric(
            "Tiles beyond the reference's 99th percentile",
            f"{result.get('beyond_envelope', 0) * 100:.1f}%",
            delta=f"{result.get('novelty_ratio', float('nan')):.1f}x expected",
            delta_color="inverse",
        )
        middle.metric(
            "Median tile sits at reference percentile",
            f"{result.get('median_percentile', float('nan')):.0f}th",
        )
        spread = result.get("slide_spread")
        right.metric(
            "Spread across slides",
            "n/a" if spread is None else f"{spread * 100:.0f} pts",
            help="Wide means some slides carry it — look at those rather than "
                 "rejecting the cohort. Narrow means it is the cohort itself.",
        )

        levels = result.get("levels") or []
        if levels and result.get("cohort_distance"):
            st.caption("Distance to the reference, by quantile")
            st.dataframe(
                {
                    "quantile": [f"{x * 100:g}%" for x in levels],
                    "reference": result.get("reference_distance", []),
                    "this cohort": result.get("cohort_distance", []),
                },
                hide_index=True, width="stretch",
            )

        rows = result.get("per_slide") or []
        if rows:
            st.caption(
                f"Worst {len(rows)} of {result.get('n_slides', len(rows))} slides "
                f"by share beyond the envelope"
            )
            st.dataframe(rows, hide_index=True, width="stretch")

        # Which baseline this was measured against. A profile built under a
        # different vote is not a valid comparison, and the server picks by
        # preset — so showing both is how a mismatch becomes visible.
        st.caption(f"Vote: {result.get('recorded_vote') or result.get('vote_preset')}")
        st.caption(f"Reference profile: {result.get('profile_vote')}")


def _render_vote_picker(submission_id: str, key_prefix: str) -> tuple[str, dict]:
    """Which vote Stage 4 should use. Returns (preset name, overrides).

    A preset list rather than seven number inputs. The vote decides every
    cluster ID in the output, and a half-applied configuration — distance
    weighting on but the exponent left at 1, an adaptive margin with no wider k
    — produces a complete, well-formed CSV that is simply not the thing that was
    measured. Presets make that unreachable by accident, and the server refuses
    the combinations that are inert anyway.

    Fetched from the server rather than listed here: the numbers and the
    accuracy beside them live in submit_cluster_assignment.py, and a second copy
    in the UI is a copy that drifts.
    """
    try:
        served = client.vote_presets()
    except Exception as e:
        # The picker is not worth blocking the stage over. Without it the server
        # applies its own default, which is the tuned configuration.
        st.caption(f"Could not load vote presets ({e}); the server's default will be used.")
        return "", {}

    presets = served.get("presets") or {}
    if not presets:
        return "", {}

    # Best accuracy first, so the recommended one is the one already selected.
    names = sorted(presets, key=lambda n: -(presets[n].get("accuracy") or 0))
    default = served.get("default")
    index = names.index(default) if default in names else 0

    chosen = st.radio(
        "Vote",
        names,
        index=index,
        horizontal=True,
        format_func=lambda n: presets[n].get("label") or n,
        key=f"{key_prefix}assign_vote_{submission_id}",
        help="How the k-NN vote is weighted. This changes the cluster ID of "
             "every tile, so two runs under different votes are not "
             "comparable and must not be mixed in the Knowledge Bank.",
    )
    spec = presets[chosen]
    st.caption(spec.get("summary", ""))
    with st.expander("What this means", expanded=False):
        st.write(spec.get("why", ""))
        accuracy = spec.get("accuracy")
        if accuracy:
            st.caption(
                f"{accuracy * 100:.2f}% leave-one-out on the production "
                f"reference at 200,000 tiles (seed 0). That measures recovery of "
                f"the reference's own Leiden labels — not agreement with the "
                f"original TCGA transfer, which is a separate check that can "
                f"move the other way."
            )

    overrides: dict = {}
    with st.expander("Override individual settings", expanded=False):
        st.caption(
            "Starts from the preset above and changes only what you touch. For "
            "comparing configurations — anything other than a preset as-is is "
            "not a measured setting."
        )
        flags = spec.get("flags") or {}
        for field, label in (
            ("distance_power", "Distance exponent"),
            ("adaptive_margin", "Re-vote below this margin (0 = off)"),
            ("adaptive_k", "Re-vote at this k (0 = off)"),
        ):
            current = flags.get(field)
            if current is None:
                continue
            entered = st.text_input(
                f"{label} — preset uses {current:g}",
                value="",
                key=f"{key_prefix}assign_vote_{field}_{submission_id}",
                placeholder=f"{current:g}",
            )
            if not entered.strip():
                continue
            try:
                value = float(entered)
            except ValueError:
                st.error(f"{label}: '{entered}' is not a number — ignoring it.")
                continue
            overrides[field] = int(value) if field == "adaptive_k" else value
        if overrides:
            st.warning(
                "Overridden: " + ", ".join(f"{k}={v:g}" for k, v in overrides.items())
                + ". This is no longer the measured configuration."
            )
    return chosen, overrides


def _render_assign_clusters_form(status: dict, submission_id: str, key_prefix: str,
                                 button_label: str = "Start cluster assignment",
                                 full_available: bool = True):
    """Reference + backend inputs for Stage 4.

    The reference is left blank by default on purpose: it is a deployment-level
    setting the server already knows, not a per-run choice, so the common case
    is one click. It is exposed at all because comparing two references is a
    real thing to want, and because a run assigned against the wrong one looks
    completely healthy.
    """
    # Both modes are always offered, matching Stages 2 and 3 — a stage that
    # silently drops one of its options reads as a different stage. What varies
    # is whether "Full dataset" can actually run: it needs this run's tracked
    # extraction output, which full_available reports. Rather than a dead button,
    # choosing it without that says why and points at the other mode.
    mode = st.radio(
        "Run",
        ["Full dataset", "Test on a sample .h5"],
        horizontal=True,
        index=0 if full_available else 1,
        key=f"{key_prefix}assign_mode_{submission_id}",
    )

    # The vote, chosen by name and shown outside the Options expander. It
    # changes every cluster ID in the output, so it is not an option — it is the
    # second thing about this run worth knowing, after which reference.
    preset_name, overrides = _render_vote_picker(submission_id, key_prefix)

    with st.expander("Options", expanded=False):
        reference = st.text_input(
            "Reference .npz (blank = the server's configured reference)",
            key=f"{key_prefix}assign_reference_{submission_id}",
            placeholder="hpc_reference_leiden_2p5_fold2.npz",
            help="The .npz built by build_hpc_reference.py — NOT the Leiden .h5ad "
                 "it is built from. Leave blank unless you are deliberately "
                 "comparing two references.",
        )

    projections = ""
    if mode == "Test on a sample .h5":
        projections = st.text_input(
            "Projections .h5 to assign",
            key=f"{key_prefix}assign_test_h5_{submission_id}",
            placeholder="/path/to/results/.../hdf5_DS_he_train.h5",
            help="Typically what a test feature extraction wrote. Results are not "
                 "recorded against this run.",
        )

    if mode == "Full dataset" and not full_available:
        st.info(
            "This run has no finished feature extraction, so there is no tracked "
            "projections file to assign. Either finish Stage 3, or switch to "
            "\"Test on a sample .h5\" above and give the path to a projections .h5 "
            "you already have."
        )
        return

    if not st.button(button_label, key=f"{key_prefix}assign_start_{submission_id}"):
        return

    ref = reference.strip() or None
    try:
        if mode == "Test on a sample .h5":
            if not projections.strip():
                st.error("Enter the projections .h5 to assign.")
                return
            result = client.start_test_cluster_assignment(
                submission_id, projections.strip(), reference=ref,
                vote_preset=preset_name, vote_overrides=overrides,
            )
            st.success("Test cluster assignment queued (not recorded against this run).")
        else:
            result = client.start_cluster_assignment(
                submission_id, reference=ref, overwrite=True,
                vote_preset=preset_name, vote_overrides=overrides,
            )
            st.success("Cluster assignment job queued.")
        # Echo back the vote the server actually resolved, not the one this form
        # thinks it asked for. An override that was dropped, or a preset that
        # resolved to something else, is worth seeing now rather than inferring
        # from cluster IDs later.
        if result.get("vote"):
            st.caption(f"Vote: {result['vote']}")
        st.rerun()
    except requests.exceptions.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        st.error(f"Failed to start cluster assignment: {detail}")
    except Exception as e:
        st.error(f"Failed to start cluster assignment: {e}")


def _render_kb_job_state(status: dict, prefix: str, label: str) -> bool:
    """Show a Slurm-backed KB write's state. True if one is in flight.

    Stages 5 and 6 can run either in the server or as a Slurm job. The job is
    the durable one — it outlives this page and the server — so its state has to
    be visible here, or a queued write looks exactly like nothing having
    happened, which is the confusion this whole feature exists to end.
    """
    job_id = status.get(f"{prefix}_job_id")
    if not job_id or status.get(f"{prefix}_done"):
        return False

    job_state = status.get(f"{prefix}_slurm_state")
    st.caption(f"{label} job:")
    st.code(job_id, language=None)
    if job_state in _SLURM_IN_FLIGHT:
        st.info(
            f"Running on Slurm (state: {job_state}). This continues whether or "
            f"not this page is open, and whether or not the tile server is "
            f"running. The step turns green when the job records that it "
            f"committed."
        )
        if status.get(f"{prefix}_log_path"):
            st.caption(f"Log: `{status[f'{prefix}_log_path']}`")
        return True

    # Not in flight and not done: it ended without committing. The job's own
    # error is worth more than the Slurm state, since every refusal this stage
    # makes is a sentence rather than an exit code.
    st.warning(f"The last {label.lower()} job ended as: "
               f"{job_state or 'no Slurm record'}, without recording a commit.")
    if status.get(f"{prefix}_error"):
        st.error(status[f"{prefix}_error"])
    if status.get(f"{prefix}_log_path"):
        st.caption(f"Log: `{status[f'{prefix}_log_path']}` — a refusal inside the "
                   f"job is printed there in full.")
    return False


def _render_write_mode(submission_id: str, key_prefix: str, name: str) -> str:
    """Where the write runs. Returns "slurm" or "server"."""
    choice = st.radio(
        "Run the write",
        ["On Slurm (survives closing this page)", "In the server (waits here)"],
        horizontal=True,
        key=f"{key_prefix}{name}_write_mode_{submission_id}",
        help="Slurm is the durable one: the job keeps writing if you close the "
             "browser or the tile server dies, and records the outcome on the "
             "run itself. Running it in the server keeps the numbers in front of "
             "you, but a killed server takes the write with it. Both run exactly "
             "the same code with the same guards.",
    )
    return "slurm" if choice.startswith("On Slurm") else "server"


def _render_registration_step(status: dict, submission_id: str, key_prefix: str, state: str):
    """Stage 5: create this cohort's identity rows in the Knowledge Bank.

    This step exists because Stage 6 could not work without it and said so only
    obliquely. load_hpc_assignments.py exclusively UPDATEs tile_registry, so a
    cohort that has never been in the KB has nothing to update and the load
    refuses at a 0% match rate — a number that reads like a naming bug and is
    in fact a missing step.

    Same preview-then-commit shape as Stage 6, and for the same reason: it
    writes to the shared Knowledge Bank, and its failure mode is a registration
    that succeeds against the wrong cohort.
    """
    if status.get("registration_done"):
        rows = status.get("registration_rows") or {}
        dataset_id = status.get("registration_dataset_id")
        st.success(
            "Registered in the Knowledge Bank"
            + (f" as `{dataset_id}`" if dataset_id else "")
        )
        if isinstance(rows, dict) and rows:
            st.caption(" · ".join(f"{t}: {n:,}" for t, n in rows.items()))
        if status.get("registration_at"):
            st.caption(f"Last registered: {status['registration_at']}")

    if _render_kb_job_state(status, "registration", "Registration"):
        # A queued or running job owns this stage; offering the form under it
        # would invite a second write against the same cohort.
        return

    if state == "blocked":
        st.info(
            "Registration reads tile identity out of the packaged .h5, so it "
            "needs Stage 2 to have finished. It does not need Stages 3 or 4 — "
            "as soon as the .h5 is ready this can run, and Stage 6 will be "
            "waiting only on the assignment."
        )
        return

    default_id = (status.get("registration_dataset_id")
                  or (status.get("dataset_name") or "").upper())
    dataset_id = st.text_input(
        "Knowledge Bank cohort (dataset_id)",
        value=default_id,
        key=f"{key_prefix}reg_dataset_id_{submission_id}",
        help="Every row this writes is scoped to this key, and a --replace only "
             "ever touches its own. It defaults to the run's dataset_name but is "
             "a different thing: dataset_name is the folder of slides on scratch, "
             "this is the cohort the KB groups by. A second, fuller run of the "
             "same cohort has a different folder and the same key.",
    )
    # Mandatory, and asked for even when the run recorded it, because Stage 5's
    # whole coordinate half is read out of <tile_dir>/<this name>/. It is NOT
    # dataset_id above: that is the cohort key the KB groups by, this is a
    # directory on disk. Conflating the two is what made the refusal below read
    # as "I already told you the dataset name".
    recorded_dataset_name = (status.get("dataset_name") or "").strip()
    try:
        tile_folders = client.get_tile_dataset_names()
    except Exception:
        # Server unreachable or the route missing — fall back to typing it
        # rather than blocking the step on a convenience lookup.
        tile_folders = []

    tile_folder_help = (
        "The folder under processed_tiles that Stage 1 wrote this run's "
        "per-slide _tile_metadata.csv files into. Every tile's x/y comes from "
        "there, so a wrong name does not fail — it registers tiles with no "
        "coordinates."
    )
    _TYPE_IT = "Other — type it"
    if tile_folders:
        # A picker rather than free text wherever possible: the names come off
        # disk, so a typo cannot silently select a folder that does not exist.
        options = [*tile_folders, _TYPE_IT]
        index = tile_folders.index(recorded_dataset_name) \
            if recorded_dataset_name in tile_folders else 0
        choice = st.selectbox(
            "Tile folder (dataset_name)",
            options,
            index=index,
            key=f"{key_prefix}reg_tile_folder_{submission_id}",
            help=tile_folder_help,
        )
        tile_dataset_name = "" if choice == _TYPE_IT else choice
        if choice == _TYPE_IT:
            tile_dataset_name = st.text_input(
                "Tile folder name",
                value=recorded_dataset_name,
                key=f"{key_prefix}reg_tile_folder_other_{submission_id}",
                placeholder="TCGA",
            )
    else:
        tile_dataset_name = st.text_input(
            "Tile folder (dataset_name)",
            value=recorded_dataset_name,
            key=f"{key_prefix}reg_tile_folder_other_{submission_id}",
            placeholder="TCGA",
            help=tile_folder_help,
        )

    if not recorded_dataset_name:
        st.caption(
            ":grey[This run recorded no tile folder — it predates that column, or "
            "was tiled outside the submit flow — so the choice above is the only "
            "thing that says where its Stage 1 metadata is.]"
        )
    elif tile_dataset_name.strip() and tile_dataset_name.strip() != recorded_dataset_name:
        st.warning(
            f"This run recorded its tiles under `{recorded_dataset_name}`, not "
            f"`{tile_dataset_name.strip()}`. Registration reads the folder "
            f"selected above — check it before previewing."
        )

    registration_scope = st.radio(
        "Registration scope",
        ["Full packaged dataset", "Subset"],
        horizontal=True,
        key=f"{key_prefix}reg_scope_{submission_id}",
        help="Register every slide in the packaged .h5, or only selected slides from it.",
    )

    registration_slide_names = None

    if registration_scope == "Subset":
        subset_text = st.text_area(
            "Slides to register (one per line)",
            key=f"{key_prefix}reg_subset_slides_{submission_id}",
            placeholder="SLIDE-001\nSLIDE-002.svs\nSLIDE-003",
            help=(
                "Enter slide IDs or filenames that already belong to the packaged .h5. "
                "Only those slides and their tiles will be registered."
            ),
        )

        registration_slide_names = [
            line.strip()
            for line in subset_text.splitlines()
            if line.strip()
        ]

    # Registration normally takes these four off the run record. A run that
    # predates a column — or work done on the cluster before this pipeline
    # existed — has no dataset_name and sometimes no packaged .h5, and until now
    # the only thing this step could say was "register it with the CLI instead".
    #
    # Pre-filled from the run and left blank where it holds nothing, so the
    # common case is untouched. Blank means "use the run's value", so clearing a
    # box does not silently override with an empty string.
    with st.expander(
        "Where the data is — normally taken from the run record",
        expanded=not status.get("dataset_name"),
    ):
        st.caption(
            "The tile folder itself is chosen above; these are the other three "
            "paths registration reads. Blank uses the run's own value."
        )
        ov_tile_dir = st.text_input(
            "Tile root",
            value=status.get("tile_dir") or "",
            key=f"{key_prefix}reg_ov_tiledir_{submission_id}",
            help="The directory the tile folder sits in — the --tile-dir Stage 1 "
                 "was given.",
        )
        ov_h5 = st.text_input(
            "Packaged .h5",
            value=status.get("h5_output_path") or "",
            key=f"{key_prefix}reg_ov_h5_{submission_id}",
            help="Tile identity and each tile's row position are read out of this "
                 "file. It must be the .h5 the assignment CSV was produced from, or "
                 "image_index points at the wrong tiles.",
        )
        ov_raw_dir = st.text_input(
            "Raw slide directory",
            value=status.get("raw_dir") or "",
            key=f"{key_prefix}reg_ov_rawdir_{submission_id}",
            help="Searched recursively for each slide's file. Without it "
                 "wsi_registry is not written and the cohort's slides will not open "
                 "in the viewer.",
        )

    # One set of path arguments for all three calls — preview, commit and the
    # Slurm submit — so they cannot read different folders. The tile folder is
    # sent only when it differs from the run's own, so the preview's "from"
    # column says "supplied" only for a real override.
    chosen_folder = tile_dataset_name.strip()
    _overrides = dict(
        tile_dataset_name=chosen_folder if chosen_folder != recorded_dataset_name else None,
        tile_dir=ov_tile_dir.strip() or None,
        h5_path=ov_h5.strip() or None,
        raw_dir=ov_raw_dir.strip() or None,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        slide_metadata = st.checkbox(
            "Also read slide headers (wsi_metadata)",
            value=False,
            key=f"{key_prefix}reg_slide_meta_{submission_id}",
            help="Opens every slide file to record mpp, objective power and level "
                 "dimensions. Minutes, not seconds, on a large cohort. Nothing "
                 "reads wsi_metadata today — it is where the numbers would live "
                 "that let the viewer stop assuming every cohort was scanned at "
                 "0.252 mpp.",
        )
    with col_b:
        write_dataset_config = st.checkbox(
            "Write dataset_config",
            value=True,
            key=f"{key_prefix}reg_dataset_config_{submission_id}",
            help="Records this cohort's target_mpp and tile size from the run's own "
                 "tiling_params. Skipped automatically for a run that predates that "
                 "column, since the alternative is asserting a geometry nobody "
                 "recorded.",
        )

    replace = st.checkbox(
        "Replace this cohort's existing rows",
        value=False,
        key=f"{key_prefix}reg_replace_{submission_id}",
        help="Required to re-register a dataset_id that already has rows. Only ever "
             "deletes WHERE dataset_id = the key above; a tile or slide claimed by a "
             "different cohort is refused outright, not reassigned.",
    )

    preview_key = f"{key_prefix}reg_preview_{submission_id}"
    if st.button(
        "Preview registration",
        key=f"{key_prefix}reg_preview_btn_{submission_id}",
    ):
        if not dataset_id.strip():
            st.error("Enter a dataset_id — every row written is scoped to it.")

        elif not tile_dataset_name.strip():
            st.error(
                "Choose the tile folder (dataset_name) — Stage 1's per-slide "
                "metadata is read from processed_tiles/<that folder>/, and it is "
                "not the same thing as the dataset_id above."
            )

        elif registration_scope == "Subset" and not registration_slide_names:
            st.error(
                "Enter at least one slide ID or filename for subset registration."
            )

        else:
            try:
                # Spinner because this runs the whole scan inside the request —
                # the .h5, a metadata CSV per slide, and the collision queries.
                # On a real cohort that is minutes, and an unexplained frozen
                # page is what a 30s read timeout used to look like.
                with st.spinner("Scanning the .h5, Stage 1 metadata and the "
                                "Knowledge Bank — minutes on a large cohort."):
                    st.session_state[preview_key] = client.preview_registration(
                        submission_id,
                        dataset_id=dataset_id.strip(),
                        scope="subset" if registration_scope == "Subset" else "full",
                        slide_names=registration_slide_names,
                        slide_metadata=slide_metadata,
                        write_dataset_config=write_dataset_config,
                        **_overrides,
                        replace=replace,
                    )
            except Exception as e:  # noqa: BLE001 — surfaced, not swallowed
                st.session_state.pop(preview_key, None)
                st.error(f"Preview failed: {e}")

    report = st.session_state.get(preview_key)
    if not report:
        st.caption("Preview first — this writes to the shared Knowledge Bank, so "
                   "it never commits without showing you the numbers.")
        return

    resolved, sources = report.get("resolved") or {}, report.get("sources") or {}
    if resolved:
        st.caption("Reading from — an override is a chance to register the wrong "
                   "directory, so this is what will actually be opened:")
        st.table([{"input": k, "value": v or "—", "from": sources.get(k, "")}
                  for k, v in resolved.items()])

    cols = st.columns(4)
    cols[0].metric("Slides", f"{report.get('slides', 0):,}")
    cols[1].metric("Tiles in .h5", f"{report.get('tiles_in_h5', 0):,}")
    cols[2].metric("With coordinates", f"{report.get('tiles_with_coordinates', 0):,}")
    cols[3].metric("Slides registered", f"{report.get('slides_registered', 0):,}")

    # The folder the numbers above came out of. "With coordinates" is only
    # interpretable alongside it: a zero there means either Stage 1 never ran or
    # this name points somewhere else, and nothing else on screen separates them.
    if report.get("tile_dataset_name"):
        st.caption(f"Tile metadata read from tile folder `{report['tile_dataset_name']}`.")

    # A correction to tile identity, so it is stated rather than left to be
    # inferred from the fact that the numbers came out right.
    renamed = report.get("tile_names_normalized") or {}
    if any(renamed.values()):
        st.info(
            f"`.jpeg` was appended to {renamed.get('h5', 0):,} tile name(s) from "
            f"the packaged .h5 and {renamed.get('coordinates', 0):,} from Stage 1's "
            f"metadata, so they match the `18_15.jpeg` form the Knowledge Bank "
            f"joins on. The rows written here are correct; the files on disk still "
            f"hold the short form, which `migrate_tile_names.py` fixes there."
        )

    if not report.get("slides_registered"):
        st.warning(
            "No wsi_registry rows would be written — the run's raw slide "
            "directory could not be read. The tiles would register and Stage 6 "
            "would load, and the viewer would still 404 on every slide in this "
            "cohort, because it resolves slide paths from wsi_registry alone."
        )

    for field, label in (
        ("ambiguous_slides", "slide id(s) match more than one file — neither is registered"),
        ("slides_without_files", "slide(s) have no raw file; their tiles register, the slide will not open"),
        ("unreadable_slides", "slide(s) could not be opened for metadata"),
        ("conflicting_samples", "slide(s) carry more than one sample_id in the .h5"),
        ("missing_slides", "slide(s) have no usable Stage 1 metadata; no coordinates for their tiles"),
    ):
        items = report.get(field) or []
        if items:
            with st.expander(f"{len(items):,} {label}", expanded=False):
                for item in items[:200]:
                    st.text(item)
                if len(items) > 200:
                    st.caption(f"...and {len(items) - 200:,} more")

    if report.get("missing_tables"):
        st.error(
            f"This database has no {', '.join(report['missing_tables'])}. Run "
            f"`psql ... -f backend/migrate_kb_base_tables.sql` first — eight of "
            f"the Knowledge Bank's tables had no CREATE TABLE in git until that "
            f"file existed."
        )
    if report.get("missing_run_tracking_columns"):
        st.error(
            "`slurm_dataset_runs` is missing "
            f"{', '.join(report['missing_run_tracking_columns'])} in the "
            "**production** database. Run tracking lives there whichever "
            "Knowledge Bank you write to, so registering would succeed and then "
            "fail recording it — leaving the rows in place and this run saying "
            "it never registered. Run `psql ... -f backend/migrate_all.sql` "
            "against production first."
        )
    if report.get("would_refuse_collision"):
        st.error(
            "Refusing: some of these tiles or slides already belong to a "
            "different dataset_id. Two cohorts cannot claim the same tile, and "
            "overwriting would repoint the viewer at another cohort's files. "
            "This needs investigating, not overwriting."
        )
    if report.get("needs_replace"):
        st.warning(
            "This dataset_id already has rows. Tick “Replace this cohort's "
            "existing rows” and preview again to overwrite them."
        )

    blocked = bool(report.get("would_refuse_collision")
                   or report.get("needs_replace")
                   or report.get("missing_tables")
                   or report.get("missing_run_tracking_columns"))
    write_mode = _render_write_mode(submission_id, key_prefix, "reg")
    if st.button(
        "Register subset in the Knowledge Bank"
        if registration_scope == "Subset"
        else "Register full dataset in the Knowledge Bank",
        key=f"{key_prefix}reg_commit_btn_{submission_id}",
        type="primary",
        disabled=blocked,
    ):
        if not tile_dataset_name.strip():
            st.error(
                "Choose the tile folder (dataset_name) — Stage 1's per-slide "
                "metadata is read from processed_tiles/<that folder>/."
            )
            return

        if registration_scope == "Subset" and not registration_slide_names:
            st.error(
                "Enter at least one slide ID or filename for subset registration."
            )
            return

        kwargs = dict(
            dataset_id=dataset_id.strip(),
            **_overrides,
            scope="subset" if registration_scope == "Subset" else "full",
            slide_names=registration_slide_names,
            slide_metadata=slide_metadata,
            write_dataset_config=write_dataset_config,
            replace=replace,
        )
        try:
            if write_mode == "slurm":
                result = client.submit_registration(submission_id, **kwargs)
                st.success(
                    f"Queued as Slurm job {result.get('registration_job_id')}. "
                    f"It writes to {result.get('database')} whether or not this "
                    f"page stays open."
                )
                if result.get("registration_log_path"):
                    st.caption(f"Log: `{result['registration_log_path']}`")
            else:
                with st.spinner("Writing this cohort's identity rows — minutes on "
                                "a large cohort. Don't reload the page."):
                    result = client.commit_registration(submission_id, **kwargs)
                written = result.get("written") or {}
                st.success("Registered: " + ", ".join(
                    f"{t} +{n:,}" for t, n in written.items()))
            st.session_state.pop(preview_key, None)
            st.rerun()
        except requests.exceptions.HTTPError as e:
            st.error(f"Registration refused: {_error_detail(e)[1]}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Registration refused: {e}")


def _render_kb_load_step(status: dict, submission_id: str, key_prefix: str, state: str):
    """Stage 6: load this run's cluster-assignment CSV into the Knowledge Bank.

    Mirrors load_hpc_assignments.py's own shape — a dry-run preview, then an
    explicit commit — rather than a single button. This is the one stage that
    mutates the shared KB every other view (slide viewer, chatbot, HPC panels)
    reads from, so it never auto-commits: a preview has to be pulled up first,
    and the numbers it shows are exactly what /kb-load would act on.
    """
    if status.get("kb_load_done"):
        rows = status.get("kb_load_rows")
        reference = status.get("kb_load_reference")
        st.success(
            "Loaded into the Knowledge Bank"
            + (f": {rows:,} tiles" if rows is not None else "")
            + (f" (reference `{reference}`)" if reference else "")
        )
        if status.get("kb_load_at"):
            st.caption(f"Last loaded: {status['kb_load_at']}")
        st.caption("If Stage 4 has been re-run since, preview below and reload.")

    if _render_kb_job_state(status, "kb_load", "Knowledge Bank load"):
        return

    # "blocked" only means *this run's* tracked assignment (Stage 4's "Full
    # dataset" mode) has not produced a usable output. Loading from an
    # explicit path does not depend on that at all — it is how output from
    # Stage 4's "Test on a sample .h5" mode gets into the KB, since that mode
    # deliberately never records a path against any run (same reasoning as
    # Stage 4 itself still offering its test form while "blocked").
    use_manual_path = state == "blocked"
    if not use_manual_path:
        source = st.radio(
            "Source",
            ["This run's tracked assignment", "A specific CSV path"],
            horizontal=True,
            key=f"{key_prefix}kb_load_source_{submission_id}",
            help="Use \"A specific CSV path\" for output from Stage 4's \"Test on "
                 "a sample .h5\" mode — that mode never records a path against "
                 "this run, so there's nothing tracked to load from automatically.",
        )
        use_manual_path = source != "This run's tracked assignment"

    manual_csv_path = ""
    if use_manual_path:
        if state == "blocked":
            st.info(
                "This run has no tracked assignment yet. Load from a specific CSV "
                "instead — typically output from Stage 4's \"Test on a sample .h5\" "
                "mode — or finish Stage 4's \"Full dataset\" option first."
            )
        manual_csv_path = st.text_input(
            "Assignment CSV path",
            key=f"{key_prefix}kb_load_csv_path_{submission_id}",
            placeholder="/path/to/DS_hpc_assignments.csv",
            help="This still writes to the Knowledge Bank for real — it just isn't "
                 "recorded against this run's own KB-load status, since the CSV "
                 "may not be this run's tracked output.",
        )

    min_margin = st.number_input(
        "Exclude tiles below this vote_margin from the aggregates",
        min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        key=f"{key_prefix}kb_load_min_margin_{submission_id}",
        help="Leave-one-out validation against the reference: margin below 0.1 was "
             "57% correct, 0.1-0.25 was 76%, 0.25+ was 92%+. tile_registry keeps every "
             "tile's own hpc_id and margin regardless of this — it only changes what "
             "counts toward the per-slide composition the chatbot and HPC panels read. "
             "0 (default) excludes nothing.",
    )

    preview_key = f"{key_prefix}kb_load_preview_{submission_id}"
    if st.button("Preview Knowledge Bank load", key=f"{key_prefix}kb_load_preview_btn_{submission_id}"):
        if use_manual_path and not manual_csv_path.strip():
            st.error("Enter the assignment CSV path.")
        else:
            try:
                with st.spinner("Reading the assignment CSV and matching it "
                                "against tile_registry — minutes on a large cohort."):
                    st.session_state[preview_key] = client.preview_kb_load(
                        submission_id, min_margin=min_margin,
                        csv_path=manual_csv_path.strip() or None,
                    )
            except requests.exceptions.HTTPError as e:
                st.error(f"Preview failed: {_error_detail(e)[1]}")
                st.session_state.pop(preview_key, None)
            except Exception as e:
                st.error(f"Preview failed: {e}")
                st.session_state.pop(preview_key, None)

    report = st.session_state.get(preview_key)
    if not report:
        return

    st.write(
        f"**{report['rows']:,}** rows in the CSV · cluster column "
        f"`{report['cluster_column']}` · reference `{report['reference']}`"
    )
    st.write(
        f"Matched **{report['matched']:,}/{report['rows']:,}** "
        f"({report['match_rate'] * 100:.1f}%) tiles in `tile_registry`"
    )
    if report.get("tile_names_normalized"):
        st.info(
            f"`.jpeg` was appended to {report['tile_names_normalized']:,} tile "
            f"name(s) from this CSV so they match the Knowledge Bank's "
            f"`18_15.jpeg` form — the match rate above depends on that "
            f"correction. The CSV on disk still holds the short form."
        )
    # Only surfaced when some of it disagrees. A CSV carrying a slide_tile
    # column that matches the rebuilt key is the normal, healthy case and does
    # not need a notice; a partial disagreement means the CSV's own columns
    # disagree with each other, which the match rate alone would not explain.
    if report.get("slide_tile_supplied") and report.get("slide_tile_disagreed"):
        disagreed = report["slide_tile_disagreed"]
        example = report.get("slide_tile_example") or ("", "")
        if disagreed == report["rows"]:
            st.info(
                f"This CSV carries its own `slide_tile` column and every value "
                f"differs from the join key rebuilt from `slides` + `tiles` "
                f"(`{example[0]}` → `{example[1]}`) — the usual sign of a column "
                f"written before the tile names were normalised. The rebuilt key "
                f"is what joins, so this is expected and harmless."
            )
        else:
            st.warning(
                f"This CSV carries its own `slide_tile` column and "
                f"{disagreed:,} of {report['rows']:,} values differ from the key "
                f"rebuilt from `slides` + `tiles` (`{example[0]}` → "
                f"`{example[1]}`). A *partial* disagreement means the CSV's own "
                f"columns disagree with each other — worth checking before "
                f"loading, since the rebuilt key is what joins."
            )
    if report["unmatched"]:
        st.warning(f"{report['unmatched']:,} unmatched, e.g. {report['unmatched_examples']}")
    if report["overwriting"]:
        note = f"Will overwrite {report['overwriting']:,} tile(s) that already carry a cluster ID"
        if report["overwriting_other_reference"]:
            note += f" ({report['overwriting_other_reference']:,} from a different reference)"
        st.info(note)
    if report.get("unwritable_cluster_ids"):
        st.error(
            f"{report['unwritable_cluster_ids']:,} cluster ID(s) are not whole "
            f"numbers and `tile_registry.hpc_id` is "
            f"`{report.get('cluster_column_type')}`: "
            f"{report['unwritable_cluster_examples']}. The load will refuse — "
            f"the assignment CSV's cluster column has to be fixed. Shown here "
            f"because the preview stages only the join key, so this would "
            f"otherwise surface as a failed UPDATE after every row was copied."
        )
    if report["unknown_clusters"]:
        st.warning(
            f"{len(report['unknown_clusters'])} cluster ID(s) have no `hpc_dictionary` "
            f"row: {report['unknown_clusters'][:10]}. Those tiles would show a cluster "
            f"with no pattern or malignancy annotation unless allowed below."
        )
    if report["low_margin"]:
        st.caption(f"{report['low_margin']:,} tile(s) have vote_margin below 0.1")
    tradeoff = report.get("margin_tradeoff")
    if tradeoff:
        with st.expander(
            "What a confidence threshold would cost and buy on this cohort",
            expanded=report["min_margin"] == 0,
        ):
            st.caption(
                "Every tile's `vote_margin` is the winning cluster's share of "
                "the weighted k-NN vote minus the runner-up's — 1.0 means all "
                "25 neighbours agreed, 0.0 a dead tie. The threshold below "
                "decides which tiles count toward the per-slide composition the "
                "chatbot and HPC panels read. **`tile_registry` keeps every "
                "tile either way**, with its own cluster and margin."
            )
            st.table([
                {
                    "min_margin": f"{row['min_margin']:.2f}",
                    "tiles kept": f"{row['tiles_kept']:,}",
                    "% kept": f"{row['share_kept'] * 100:.1f}%",
                    "expected accuracy": f"{row['expected_accuracy'] * 100:.2f}%",
                }
                for row in tradeoff
            ])
            st.caption(
                ":grey[Expected accuracy is this cohort's own margin "
                "distribution weighted by leave-one-out accuracy measured on the "
                "reference (61.7% below 0.10, 81.2% to 0.25, 95.5% to 0.50, "
                "99.5% to 0.75, 100% above). That mapping was measured inside "
                "LATTICeA, so applying it here assumes a margin means the same "
                "thing on this cohort's scanner and stain — the one thing no "
                "accuracy number can settle without labels. Treat it as an "
                "estimate, and read `cohort_shift` alongside it.]"
            )

    if report["min_margin"] > 0:
        st.info(
            f"At the {report['min_margin']} threshold above, "
            f"**{report['excluded_from_aggregates']:,}** tile(s) would be excluded "
            f"from the per-slide aggregates (tile_registry keeps them regardless)."
        )

    if report["would_refuse_low_match"]:
        st.error(
            f"Match rate {report['match_rate'] * 100:.1f}% is below the required "
            f"{report['min_match_rate'] * 100:.0f}% — committing would be refused "
            f"the same way the CLI refuses it."
        )

    with st.expander("Commit options", expanded=False):
        cancer_type = st.text_input(
            "Cancer type (optional — fills hpl_profile_summary.cancer_type)",
            key=f"{key_prefix}kb_load_cancer_{submission_id}",
            placeholder="LUAD",
            help="Left blank leaves it unset rather than guessing, same as the CLI.",
        )
        allow_unknown = st.checkbox(
            "Load cluster IDs with no hpc_dictionary row anyway",
            value=False,
            key=f"{key_prefix}kb_load_allow_unknown_{submission_id}",
            disabled=not report["unknown_clusters"],
        )
        skip_profiles = st.checkbox(
            "Skip refreshing the per-slide aggregates (tile_registry only)",
            value=False,
            key=f"{key_prefix}kb_load_skip_profiles_{submission_id}",
            help="Leaves hpl_profile_proportion/summary disagreeing with the new "
                 "tile_registry values. Off unless you have a specific reason.",
        )

    write_mode = _render_write_mode(submission_id, key_prefix, "kb_load")
    if st.button(
        "Commit to Knowledge Bank",
        key=f"{key_prefix}kb_load_commit_{submission_id}",
        disabled=report["would_refuse_low_match"],
    ):
        kwargs = dict(
            cancer_type=cancer_type.strip() or None,
            allow_unknown_clusters=allow_unknown,
            skip_profiles=skip_profiles,
            min_margin=min_margin,
            csv_path=manual_csv_path.strip() or None,
        )
        try:
            if write_mode == "slurm":
                result = client.submit_kb_load(submission_id, **kwargs)
                st.success(
                    f"Queued as Slurm job {result.get('kb_load_job_id')}. It "
                    f"writes to {result.get('database')} whether or not this page "
                    f"stays open."
                )
                if result.get("kb_load_log_path"):
                    st.caption(f"Log: `{result['kb_load_log_path']}`")
                st.session_state.pop(preview_key, None)
                st.rerun()
            with st.spinner("Writing tile_registry and refreshing the per-slide "
                            "aggregates — minutes on a large cohort. Don't reload "
                            "the page."):
                result = client.commit_kb_load(submission_id, **kwargs)
            msg = f"Committed {result['updated_rows']:,} tile(s) to the Knowledge Bank."
            if result.get("excluded_from_aggregates"):
                msg += (f" {result['excluded_from_aggregates']:,} tile(s) below "
                        f"margin {result['min_margin']} were excluded from the aggregates.")
            if not result.get("recorded_on_run", True):
                msg += (" Not recorded against this run's own KB-load status, since "
                        "the CSV wasn't this run's tracked output.")
            st.success(msg)
            st.session_state.pop(preview_key, None)
            st.rerun()
        except requests.exceptions.HTTPError as e:
            st.error(f"Load failed: {_error_detail(e)[1]}")
        except Exception as e:
            st.error(f"Load failed: {e}")


def _render_job_progress(job: dict, key_prefix: str = ""):
    """Render one run as a three-step pipeline: every stage listed with its
    own state icon and summary, each expandable for its actions.

    Shared by 'this path already has a run' (inline) and the 'Recent dataset
    jobs' list — key_prefix keeps widget keys from colliding when the same job
    is rendered in both places on one page.
    """
    submission_id = job.get("submission_id")
    label_id = _job_label(job)
    total_slides = job.get("total_slides")
    st.caption(
        f"Submitted: {job.get('submitted_at', '')} · "
        f"{int(total_slides):,} slides" if total_slides is not None
        else f"Submitted: {job.get('submitted_at', '')} · discovering slides"
    )
    try:
        status = client.get_dataset_job_status(submission_id)
    except Exception as e:
        st.warning(f"Could not load status: {e}")
        return

    stage = status.get("status")

    display_stage = _dataset_job_display_stage(status)
    state_key = f"{key_prefix}dataset_job_last_stage_{submission_id}"
    previous_stage = st.session_state.get(state_key)
    if previous_stage is not None and previous_stage != display_stage:
        st.toast(f"Job {label_id}: {previous_stage} → {display_stage}")
    st.session_state[state_key] = display_stage

    if stage == "error":
        st.error(f"Run failed: {status.get('error')}")
    elif status.get("error"):
        # A non-fatal note (e.g. some sbatch batches failed but others ran).
        st.warning(status["error"])
    if stage == "cancelled":
        st.info("This run was cancelled. Stages already finished on disk are still usable.")

    steps = _pipeline_steps(status)
    renderers = {
        "tiling": lambda s: _render_tiling_step(status, submission_id, key_prefix),
        "packaging": lambda s: _render_packaging_step(status, submission_id, key_prefix, s["state"]),
        "extraction": lambda s: _render_extraction_step(status, submission_id, key_prefix, s["state"]),
        "assignment": lambda s: _render_assignment_step(status, submission_id, key_prefix, s["state"]),
        "registration": lambda s: _render_registration_step(status, submission_id, key_prefix, s["state"]),
        "kb_load": lambda s: _render_kb_load_step(status, submission_id, key_prefix, s["state"]),
        "anorak": lambda s: _render_anorak_step(status, submission_id, key_prefix, s["state"]),
    }
    if status.get("run_kind") == "anorak":
        # An ANORAK run on its own: Stages 1-6 were never part of it.
        steps = [s for s in steps if s["key"] == "anorak"]
        renderers["anorak"] = lambda s: _render_anorak_run_step(status, submission_id, key_prefix)
    elif status.get("pipeline"):
        # A pipeline run: Stages 1-4 have no buttons of their own — the
        # pipeline starts each once the one before it has verified its output.
        _render_pipeline_overview(status, submission_id, key_prefix)
        for stage in _PIPELINE_STAGES:
            renderers[stage] = (lambda stage: lambda s: _render_pipeline_stage_step(
                status, submission_id, key_prefix, stage, s["state"]))(stage)
    elif not _is_upload_run(status):
        # A run from before the pipeline: its Stages 1-4 are history. Shown
        # read-only — a new pipeline run carries the dataset on and reuses
        # whatever these produced. Upload runs keep their Stage 3/4 buttons:
        # an uploaded slide does not go through the pipeline.
        for stage in _PIPELINE_STAGES:
            renderers[stage] = (lambda stage: lambda s: _render_legacy_stage_readonly(
                status, stage))(stage)
    for step in steps:
        icon = _STEP_ICON.get(step["state"], "")
        # Auto-open whichever step is waiting on the user, so the next action
        # is visible without hunting for it; finished and blocked steps stay
        # collapsed but still show their state in the label.
        # "blocked" is expanded too, which it deliberately wasn't before. That
        # rule assumed a blocked step has nothing in it but a "waiting on the
        # previous stage" line, and packaging is no longer such a step — its
        # subset/test option is usable during tiling. Leaving it collapsed put
        # the only controls people were looking for behind an accordion row
        # reading "⚪ 2. Packaging (.h5) — waiting on tiling", which reads as
        # "nothing to do here yet" rather than "click to open".
        with st.expander(
            f"{icon} {step['title']} — {step['summary']}",
            expanded=step["state"] in ("action", "attention", "failed", "blocked"),
        ):
            renderers[step["key"]](step)

    # Collapsed by default: the three steps above answer "what can I do next",
    # which is the common question. This answers "what has already been tried",
    # and it costs an sacct call, so it is opened deliberately.
    with st.expander("Slurm job history", expanded=False):
        _render_job_history(submission_id, key_prefix)

    finished_anorak = status.get("run_kind") == "anorak" and not status.get("anorak_in_flight")
    if stage not in ("error", "cancelled") and not finished_anorak:
        if st.button("Stop run", key=f"{key_prefix}dataset_job_cancel_{submission_id}"):
            try:
                cancel_result = client.cancel_dataset_job(submission_id)
                if cancel_result.get("cancelled_job_ids"):
                    st.success("Cancelled: " + ", ".join(cancel_result["cancelled_job_ids"]))
                else:
                    st.info("Marked cancelled (no Slurm jobs had been queued yet).")
            except requests.exceptions.HTTPError as e:
                st.error(f"Stop failed: {_error_detail(e)[1]}")
            except Exception as e:
                st.error(f"Stop failed: {e}")


with st.sidebar:
    st.subheader("v25 NL pipeline")
    if llm_enabled():
        st.success(f"Ollama: **on** (`{os.getenv('OLLAMA_MODEL', 'deepseek-r1:1.5b')}`)")
    else:
        st.warning("Ollama disabled (`LLM_ENABLED=0`)")
    if llm_planner_enabled():
        st.caption("Planner: **NLP + regex + LLM**")
    else:
        st.caption("Planner: **NLP + regex** (LLM planner off)")

    render_wsi_upload_panel()
    render_dataset_job_panel()


# OpenSeadragon pyramid viewer helper
# ---------------------------------------------------------------------------
def render_openseadragon_viewer(slide_id: str, viewer_key: str = "main", height: int = 780,
                                overlay_tiles=None, tile_index=None, tile_size_native: float = 0.0):
    """Render an OpenSeadragon pyramid viewer for a slide with optional SVG tile overlays.

    tile_index (see build_osd_tile_index) adds hover and click: the cell under
    the pointer is outlined and named, and a click pins that highlight. The
    lookup is floor(x / tile_size_native) against a Map, not a scan, because
    the tiles sit on a regular lattice — x_native = col * pitch, which is what
    tile_coordinates records.

    The selection lives inside the iframe and does not come back to Streamlit:
    components.html is one-way, so a click here cannot set session state the
    way the click-inspector's streamlit_image_coordinates can. That is why the
    readout is drawn in the iframe rather than as a widget beside it — Click
    tile inspector mode remains the way to pull a tile into the page itself.
    """
    slide_id = (slide_id or "").strip().upper()
    safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", f"{slide_id}_{viewer_key}")
    # Three things this URL has to get right, none of which the browser will
    # forgive:
    #
    #  * the browser's address for the server, not this process's (see
    #    TILE_SERVER_BROWSER_URL);
    #  * the slide id percent-encoded. Ours contain spaces and colons
    #    ("BB232000 A1 -1 - 2023-08-29 20.07.22"), and this string is pasted
    #    straight into a JS string literal and an st.caption — a raw '#' in an
    #    id would truncate the request at the fragment and a raw space is not a
    #    URL character at all;
    #  * kb_target, because the viewer fetches this itself and therefore sends
    #    none of the client's parameters. Without it the server resolves the
    #    slide against production's wsi_registry and 404s on every test-KB
    #    cohort, while the rest of the page — all server-side calls — reads test
    #    perfectly well.
    #
    # The query has to sit on the ".dzi" URL rather than be dropped because
    # OpenSeadragon 4.1.1 carries it onto the tile requests too, and only in
    # that form: DziTileSource matches /\.(dzi|xml|js)\?/ and appends whatever
    # it finds to every ..._files/{level}/{col}_{row}.jpeg it builds. Put the
    # target anywhere else and the metadata resolves on test while all 996
    # tiles 404 against production.
    slide_path = quote(slide_id, safe="")
    target = quote(client.kb_target, safe="")
    dzi_url = f"{TILE_SERVER_BROWSER_URL}/dzi/{slide_path}.dzi?kb_target={target}"

    overlay_tiles = overlay_tiles or []
    overlay_json = json.dumps(overlay_tiles)
    tile_index_json = json.dumps(tile_index or [])
    pitch_json = json.dumps(float(tile_size_native or 0))

    st.caption(f"OpenSeadragon DZI source: {dzi_url}")

    components.html(
        f"""
        <div id="osd_wrap_{safe_key}" style="position:relative;width:100%;height:{height}px;">
            <div id="osd_{safe_key}" style="
                width:100%;
                height:{height}px;
                background:#111827;
                border:1px solid #374151;
                border-radius:10px;
                overflow:hidden;
            "></div>

            <svg id="osd_overlay_{safe_key}" style="
                position:absolute;
                left:0;
                top:0;
                width:100%;
                height:100%;
                pointer-events:none;
                z-index:5;
            "><g id="osd_grid_{safe_key}"></g><g id="osd_highlight_{safe_key}"></g></svg>

            <!-- The hover readout, inside the viewer so the answer appears
                 where the pointer already is. Hidden until the pointer is
                 actually over a tile: an empty box parked in the corner reads
                 as a broken widget. -->
            <div id="osd_hud_{safe_key}" style="
                position:absolute;
                left:10px;
                top:10px;
                z-index:6;
                display:none;
                pointer-events:none;
                padding:6px 10px;
                border-radius:8px;
                background:rgba(17,24,39,0.82);
                border:1px solid rgba(255,255,255,0.28);
                color:#f9fafb;
                font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                font-size:12px;
                line-height:1.35;
                max-width:60%;
                overflow-wrap:anywhere;
            "></div>
        </div>

        <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/openseadragon.min.js"></script>
        <script>
            const overlayTiles = {overlay_json};
            // [col, row, slide_tile, hpc_id] per tile — see build_osd_tile_index.
            const tileIndexRows = {tile_index_json};
            const tilePitch = {pitch_json};

            // Keyed "<col>_<row>". One Map get per mouse move, rather than a
            // scan over every tile on the slide, because the tiles are a
            // regular lattice: x_native = col * pitch.
            const tileByCell = new Map();
            tileIndexRows.forEach(function(t) {{
                tileByCell.set(t[0] + "_" + t[1], {{ col: t[0], row: t[1], slideTile: t[2], hpcId: t[3] }});
            }});

            const viewer = OpenSeadragon({{
                id: "osd_{safe_key}",
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
                tileSources: "{dzi_url}",
                showNavigator: true,
                navigatorPosition: "BOTTOM_RIGHT",
                animationTime: 0.4,
                blendTime: 0.1,
                constrainDuringPan: true,
                visibilityRatio: 0.8,
                minZoomImageRatio: 0.8,
                maxZoomPixelRatio: 3.0,
                showRotationControl: false,
                gestureSettingsMouse: {{
                    // Off, where it used to be on: a single click now pins the
                    // tile under the pointer, and the two cannot share the
                    // gesture. Double click still zooms, and so does the wheel.
                    clickToZoom: false,
                    dblClickToZoom: true,
                    dragToPan: true,
                    scrollToZoom: true
                }}
            }});

            const overlaySvg = document.getElementById("osd_overlay_{safe_key}");
            const gridGroup = document.getElementById("osd_grid_{safe_key}");
            const highlightGroup = document.getElementById("osd_highlight_{safe_key}");
            const hud = document.getElementById("osd_hud_{safe_key}");

            let hoveredTile = null;
            let pinnedTile = null;

            function sizeOverlay() {{
                const container = viewer.container.getBoundingClientRect();
                overlaySvg.setAttribute("viewBox", `0 0 ${{container.width}} ${{container.height}}`);
                return container;
            }}

            // How heavy a grid outline is, given the tile's current width in
            // screen pixels: a constant fraction of the cell, which is what
            // the click inspector draws (its 2-unit stroke on a thumbnail
            // viewBox is ~4.2% of a tile at any display scale). The rule here
            // used to be min(4, w/35), which agrees with that only while a
            // tile is under ~140 px — past there the cap holds while the cell
            // keeps growing, so at 600 px the outline was 0.67% of the cell.
            // The floor keeps it visible zoomed out; the ceiling stops it
            // eating the tile it frames. Mirrors gridStrokeWidth() in
            // frontend/src/components/viewer/overlayBuilders.js.
            const GRID_STROKE_FRACTION = 0.042;
            const GRID_STROKE_MIN = 1.5;
            const GRID_STROKE_MAX = 9;
            // Drawn under the colour, slightly wider, so the outline holds its
            // edge on pale H&E as well as on the dark background. The HPC
            // colours themselves are untouched — they are what the legend is
            // keyed on.
            const GRID_HALO_COLOR = "rgba(17, 24, 39, 0.72)";
            const GRID_HALO_EXTRA = 2.5;

            function gridStrokeWidth(w) {{
                if (!(w > 0)) {{ return GRID_STROKE_MIN; }}
                return Math.max(GRID_STROKE_MIN, Math.min(GRID_STROKE_MAX, w * GRID_STROKE_FRACTION));
            }}

            function makeRect(box, stroke, strokeWidth, fill, opacity) {{
                const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                r.setAttribute("x", box.x);
                r.setAttribute("y", box.y);
                r.setAttribute("width", Math.max(box.w, 1));
                r.setAttribute("height", Math.max(box.h, 1));
                r.setAttribute("fill", fill || "none");
                r.setAttribute("stroke", stroke);
                r.setAttribute("stroke-width", strokeWidth);
                r.setAttribute("opacity", opacity || "0.95");
                return r;
            }}

            function appendRect(group, t, container) {{
                const rectVp = viewer.viewport.imageToViewportRectangle(
                    Number(t.x),
                    Number(t.y),
                    Number(t.w),
                    Number(t.h)
                );

                const p1 = viewer.viewport.pixelFromPoint(rectVp.getTopLeft(), true);
                const p2 = viewer.viewport.pixelFromPoint(rectVp.getBottomRight(), true);

                const x = Math.min(p1.x, p2.x);
                const y = Math.min(p1.y, p2.y);
                const w = Math.abs(p2.x - p1.x);
                const h = Math.abs(p2.y - p1.y);

                if (x + w < 0 || y + h < 0 || x > container.width || y > container.height) {{
                    return;
                }}

                const box = {{ x: x, y: y, w: w, h: h }};
                // A record carrying `color` is a grid outline; one carrying
                // `fill` and its own stroke_width is a heatmap or risk cell,
                // whose hairline edge belongs to a colour scale and is left
                // exactly as it was.
                const isGridOutline = !t.stroke_width && !!t.color;
                const strokeWidth = t.stroke_width || gridStrokeWidth(w);

                if (isGridOutline) {{
                    group.appendChild(makeRect(
                        box, GRID_HALO_COLOR, strokeWidth + GRID_HALO_EXTRA, null, "1.0"));
                }}

                group.appendChild(makeRect(
                    box, t.stroke || t.color || "yellow", strokeWidth, t.fill, t.opacity));
            }}

            // The hovered and pinned cells only. Its own group, and its own
            // redraw, because this runs on every mouse move — rebuilding the
            // grid's several thousand rects to move one outline would make
            // pointing at a tile cost more than panning does.
            function drawHighlight() {{
                if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) {{
                    return;
                }}
                const container = sizeOverlay();
                highlightGroup.innerHTML = "";
                if (!(tilePitch > 0)) {{
                    return;
                }}

                if (pinnedTile) {{
                    const px = pinnedTile.col * tilePitch;
                    const py = pinnedTile.row * tilePitch;
                    const pad = tilePitch * 0.06;
                    appendRect(highlightGroup, {{
                        x: px - pad, y: py - pad, w: tilePitch + 2 * pad, h: tilePitch + 2 * pad,
                        fill: "none", stroke: "yellow", stroke_width: 6, opacity: "1.0"
                    }}, container);
                    appendRect(highlightGroup, {{
                        x: px, y: py, w: tilePitch, h: tilePitch,
                        fill: "none", stroke: "lime", stroke_width: 3, opacity: "1.0"
                    }}, container);
                }}

                if (hoveredTile) {{
                    appendRect(highlightGroup, {{
                        x: hoveredTile.col * tilePitch,
                        y: hoveredTile.row * tilePitch,
                        w: tilePitch,
                        h: tilePitch,
                        fill: "rgba(255,255,255,0.18)",
                        stroke: "#ffffff",
                        stroke_width: 2,
                        opacity: "1.0"
                    }}, container);
                }}
            }}

            function describe(tile, pinned) {{
                const label = tile.hpcId === null || tile.hpcId === undefined
                    ? "no HPC label" : ("HPC " + tile.hpcId);
                return '<div style="font-weight:700;">' + tile.slideTile + "</div>"
                     + '<div style="opacity:0.88;">' + label + (pinned ? " · pinned" : "") + "</div>";
            }}

            function drawHud() {{
                const tile = hoveredTile || pinnedTile;
                if (!tile) {{
                    hud.style.display = "none";
                    return;
                }}
                hud.innerHTML = describe(tile, !hoveredTile);
                hud.style.display = "block";
            }}

            // The tile under a point in the viewer's own pixel space, or null
            // where the slide has no tile (background, or tissue below the
            // threshold that Stage 1 skipped).
            function tileAtPixel(px, py) {{
                if (!(tilePitch > 0) || tileByCell.size === 0) {{ return null; }}
                if (!viewer.world || viewer.world.getItemCount() === 0) {{ return null; }}
                const vp = viewer.viewport.pointFromPixel(new OpenSeadragon.Point(px, py), true);
                const img = viewer.viewport.viewportToImageCoordinates(vp);
                if (img.x < 0 || img.y < 0) {{ return null; }}
                const col = Math.floor(img.x / tilePitch);
                const row = Math.floor(img.y / tilePitch);
                return tileByCell.get(col + "_" + row) || null;
            }}

            viewer.container.addEventListener("mousemove", function(event) {{
                const bounds = viewer.container.getBoundingClientRect();
                const tile = tileAtPixel(event.clientX - bounds.left, event.clientY - bounds.top);
                if (tile === hoveredTile) {{ return; }}  // same cell — nothing to redraw
                hoveredTile = tile;
                drawHighlight();
                drawHud();
            }});

            viewer.container.addEventListener("mouseleave", function() {{
                if (hoveredTile === null) {{ return; }}
                hoveredTile = null;
                drawHighlight();
                drawHud();
            }});

            // canvas-click rather than a DOM click: OpenSeadragon sets
            // event.quick false for a press that turned into a drag, which is
            // the difference between picking a tile and finishing a pan on top
            // of one. Clicking the pinned tile again unpins it.
            viewer.addHandler("canvas-click", function(event) {{
                if (!event.quick) {{ return; }}
                const tile = tileAtPixel(event.position.x, event.position.y);
                if (!tile) {{ return; }}
                pinnedTile = (pinnedTile && pinnedTile.col === tile.col && pinnedTile.row === tile.row)
                    ? null : tile;
                drawHighlight();
                drawHud();
            }});

            function drawTileOverlay() {{
                if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) {{
                    return;
                }}

                const container = sizeOverlay();
                gridGroup.innerHTML = "";

                overlayTiles.forEach(function(t) {{
                    appendRect(gridGroup, t, container);
                }});

                // The highlight sits in its own group but still has to follow
                // the viewport, so a pan or zoom redraws both.
                drawHighlight();
            }}

            viewer.addHandler("open", drawTileOverlay);
            viewer.addHandler("animation", drawTileOverlay);
            viewer.addHandler("animation-finish", drawTileOverlay);
            viewer.addHandler("resize", drawTileOverlay);
            viewer.addHandler("zoom", drawTileOverlay);
            viewer.addHandler("pan", drawTileOverlay);

            viewer.addHandler("open-failed", function(event) {{
                console.error("OpenSeadragon open failed", event);
                const el = document.getElementById("osd_{safe_key}");
                el.innerHTML = '<div style="color:#fecaca;padding:16px;font-family:Arial;">OpenSeadragon failed to open the DZI source. Try opening this in browser: {dzi_url}</div>';
            }});

            viewer.addHandler("tile-load-failed", function(event) {{
                console.error("OpenSeadragon tile load failed", event);
            }});
        </script>
        """,
        height=height + 20,
    )


# ---------------------------------------------------------------------------
# Fallback: fetch tile region directly from SVS if H5 tile endpoint fails
def fetch_tile_region_from_svs(slide_id: str, x_native, y_native, tile_size_native: int):
    """Fetch the selected tile directly from the SVS using native coordinates.

    This is a fallback for cases where the H5-backed `/tile_image/{slide_tile}`
    endpoint fails. It uses the existing tile server `/slide/{slide_id}/region`
    endpoint and should be sharper because it comes from the original WSI.
    """
    url = f"{TILE_SERVER_URL}/slide/{str(slide_id).strip().upper()}/region"
    params = {
        "x": int(float(x_native)),
        "y": int(float(y_native)),
        "w": int(tile_size_native),
        "h": int(tile_size_native),
        "level": 0,
        "quality": 95,
        # This one call builds its own request instead of going through
        # TileServerClient, so it is the one call that does not get kb_target
        # attached for it. Omitted, the fallback resolves the slide against
        # production and the tile a test cohort falls back to is either a 404 or
        # — for an id present in both banks — the wrong slide's pixels.
        "kb_target": client.kb_target,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local DB engine — still needed for chat query pipeline (Phase 1)
# ---------------------------------------------------------------------------
# DB_USER = "vpandya"
# DB_PASS = ""
# DB_HOST = "127.0.0.1"
# DB_PORT = "5433"
# DB_NAME = "hpl_kb"

# engine = create_engine(
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
#     pool_pre_ping=True,
# )



# Local Streamlit connects through the SSH tunnel on the Mac.
# Keep DB_HOST/DB_PORT fixed here to avoid old shell variables overriding the tunnel.
DB_USER = os.getenv("HPL_DB_USER", "vpandya")
DB_PASS = os.getenv("HPL_DB_PASS", "")
DB_HOST = "127.0.0.1"
DB_PORT = "5433"
# The chatbot and three viewer helpers query Postgres directly rather than
# through the API, so the selection above has to reach this engine too —
# otherwise the pipeline writes to test and every answer still comes from
# production.
DB_NAME = KB_DATABASES[kb_target]

# For local Streamlit + SSH tunnel, the app should connect to the local
# forwarded port, usually 127.0.0.1:5433.
# Do not silently prefer DATABASE_URL/HPL_DB_URL because old shell values can
# point to localhost:5432 and cause confusing connection refused errors.
# Built by backend/db_url.py rather than formatted here: a password containing
# '@', '/' or ':' re-splits an interpolated URL into a different host and
# database, and the branch on DB_PASS is unnecessary because an empty password
# is dropped, which is what lets libpq fall through to ~/.pgpass.
DATABASE_URL = database_url(DB_NAME, user=DB_USER, password=DB_PASS,
                            host=DB_HOST, port=DB_PORT)

# str() on a URL masks the password, so this is what any message may show.
SAFE_DATABASE_URL = safe_text(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)
st.sidebar.caption(f"Streamlit DB URL: {SAFE_DATABASE_URL}")

@st.cache_data(show_spinner=False, ttl=300)
def load_hpc_titles() -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        hpc_id,
                        hpc_title
                    FROM hpc_dictionary
                    ORDER BY hpc_id
                    """
                ),
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["hpc_id", "hpc_title"])

    if df.empty:
        return pd.DataFrame(columns=["hpc_id", "hpc_title"])

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
    df["hpc_title"] = df["hpc_title"].fillna("").astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_hpc_title_map() -> dict[int, str]:
    df = load_hpc_titles()
    if df.empty:
        return {}
    out = {}
    for _, row in df.dropna(subset=["hpc_id"]).iterrows():
        hid = int(row["hpc_id"])
        title = str(row.get("hpc_title") or "").strip()
        if title:
            out[hid] = title
    return out

@st.cache_data(show_spinner=False, ttl=300)
def load_survival_coefficients(p_threshold=0.05):
    """
    Load Cox survival coefficients for HPCs.

    Returns:
        dict:
        {
            hpc_id: {
                "coef": float,
                "expcoef": float,
                "p": float,
                "ci_low": float,
                "ci_high": float
            }
        }
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT
                        hpc_id,
                        coef,
                        expcoef,
                        p,
                        expcoef_lower_95,
                        expcoef_upper_95
                    FROM hpc_survival_analysis
                    WHERE hpc_id IS NOT NULL
                      AND coef IS NOT NULL
                    """
                ),
                conn,
            )
    except Exception as e:
        st.warning(
            "Could not load survival coefficients. Check that the PostgreSQL database or SSH tunnel is running. "
            f"Current DATABASE_URL: {SAFE_DATABASE_URL}. Error: {e}"
        )
        return {}

    if df.empty:
        return {}

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce")
    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["expcoef"] = pd.to_numeric(df["expcoef"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    df["expcoef_lower_95"] = pd.to_numeric(df["expcoef_lower_95"], errors="coerce")
    df["expcoef_upper_95"] = pd.to_numeric(df["expcoef_upper_95"], errors="coerce")

    df = df.dropna(subset=["hpc_id", "coef"])

    if p_threshold is not None:
        df = df[df["p"] < float(p_threshold)]

    survival_map = {}

    for _, row in df.iterrows():
        hid = int(row["hpc_id"])

        survival_map[hid] = {
            "coef": float(row["coef"]),
            "expcoef": float(row["expcoef"]) if not pd.isna(row["expcoef"]) else np.nan,
            "p": float(row["p"]) if not pd.isna(row["p"]) else np.nan,
            "ci_low": float(row["expcoef_lower_95"]) if not pd.isna(row["expcoef_lower_95"]) else np.nan,
            "ci_high": float(row["expcoef_upper_95"]) if not pd.isna(row["expcoef_upper_95"]) else np.nan,
        }

    return survival_map


def add_survival_risk_score(tile_df, survival_map):
    """
    Compute smooth survival risk score per tile.

    Uses only:
    1. HPCs present in hpc_survival_analysis
    2. HPC probability columns present in this WSI
    3. Continuous p_hpc_* probabilities, not class labels

    Formula:
        tile_risk_score = sum(p_hpc_i * coef_i)
    """
    df = tile_df.copy()

    prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]

    available_hpcs_in_wsi = set()
    for col in prob_cols:
        try:
            hid = int(str(col).replace("p_hpc_", ""))
            available_hpcs_in_wsi.add(hid)
        except Exception:
            continue

    survival_hpcs = set(survival_map.keys())

    usable_hpcs = sorted(survival_hpcs & available_hpcs_in_wsi)

    risk = np.zeros(len(df), dtype=float)

    contribution_cols = []

    for hid in usable_hpcs:
        col = f"p_hpc_{hid}"
        coef = survival_map[hid]["coef"]

        probs = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        contribution_col = f"risk_contrib_hpc_{hid}"
        df[contribution_col] = probs * coef
        contribution_cols.append(contribution_col)

        risk += probs * coef

    df["survival_risk_score"] = risk

    if len(risk) == 0:
        max_abs = 0
    else:
        max_abs = float(np.nanmax(np.abs(risk)))

    if max_abs == 0 or np.isnan(max_abs):
        df["survival_risk_norm"] = 0.0
    else:
        df["survival_risk_norm"] = df["survival_risk_score"] / max_abs

    df["survival_risk_abs"] = np.abs(df["survival_risk_norm"])

    return df, usable_hpcs


def risk_to_rgba(score_norm, alpha_min=20, alpha_max=170):
    """
    Convert normalized survival score into overlay colour.

    Positive score:
        red, poor survival associated

    Negative score:
        blue, protective survival associated

    Near zero:
        almost transparent
    """
    try:
        score_norm = float(score_norm)
    except Exception:
        return (255, 255, 255, 0)

    score_norm = max(-1.0, min(1.0, score_norm))

    intensity = abs(score_norm)

    if intensity < 0.02:
        return (255, 255, 255, 0)

    alpha = int(alpha_min + (alpha_max - alpha_min) * intensity)

    if score_norm > 0:
        return (255, 0, 0, alpha)

    if score_norm < 0:
        return (0, 80, 255, alpha)

    return (255, 255, 255, 0)



def hpc_label(hpc_id, max_title_chars: int = 80) -> str:
    if hpc_id is None or pd.isna(hpc_id):
        return "HPC unknown"

    hid = int(hpc_id)
    title = load_hpc_title_map().get(hid, "")

    if not title:
        return f"HPC {hid}"

    if len(title) > max_title_chars:
        title = title[: max_title_chars - 1] + "…"

    return f"HPC {hid}: {title}"



# ---------------------------------------------------------------------------
# Slide registry — fetched once from the tile server
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_slide_list() -> list[str]:
    try:
        return client.list_slides()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tile metadata — fetched from server, cached per slide
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading tile metadata…", ttl=300)
def load_tile_coords_for_slide(slide_id: str) -> pd.DataFrame:
    slide_id = (slide_id or "").strip().upper()
    if not slide_id:
        return pd.DataFrame()
    df = client.get_tiles_meta(slide_id)
    if df.empty:
        return df
    # Normalise columns the same way v21 did
    df.columns = df.columns.astype(str).str.strip()
    if "slide_tile" in df.columns:
        df["slide_tile"] = df["slide_tile"].astype(str).str.strip().str.upper()
    if "slides" in df.columns:
        df["slides"] = df["slides"].astype(str).str.strip().str.upper()
    if "hpc_id" in df.columns:
        df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Adjacency — fetched from server, cached
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_adjacency_for_slide(slide_id: str):
    slide_id = (slide_id or "").strip().upper()
    data = client.get_adjacency(slide_id)
    pair_edge_counts = {}
    for k, v in data.get("pair_edge_counts", {}).items():
        a, b = k.split("_")
        pair_edge_counts[(int(a), int(b))] = v
    tile_neighbor_pairs = {}
    for k, v in data.get("tile_neighbor_pairs", {}).items():
        a, b = k.split("_")
        tile_neighbor_pairs[(int(a), int(b))] = {
            "a_touch": set(v.get("a_touch", [])),
            "b_touch": set(v.get("b_touch", [])),
        }
    return pair_edge_counts, tile_neighbor_pairs


# ---------------------------------------------------------------------------
# HPC → slides from KB only (same tables as handle_slide: proportion + summary)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def slides_for_hpc_from_kb(hpc_id: int):
    hid = int(hpc_id)
    try:
        with engine.connect() as conn:
            base = pd.read_sql(
                text(
                    """
                    SELECT UPPER(TRIM(hp.slides)) AS slide_id,
                           hp.proportion,
                           hp.samples
                    FROM hpl_profile_proportion hp
                    WHERE hp.hpc_id = :hid
                      AND hp.slides IS NOT NULL
                    ORDER BY hp.proportion DESC NULLS LAST
                    """
                ),
                conn,
                params={"hid": hid},
            )

        if base is None or base.empty:
            return {"ok": True, "df": pd.DataFrame(columns=["slide_id", "proportion", "samples", "kb_summary"]), "error": None}

        base["slide_id"] = base["slide_id"].astype(str).str.strip().str.upper()
        base = base.drop_duplicates(subset=["slide_id"], keep="first")
        slides = sorted(base["slide_id"].unique().tolist())

        previews = {}
        with engine.connect() as conn:
            ss = text(
                """
                SELECT * FROM hpl_profile_summary
                WHERE UPPER(TRIM(slides)) IN :slides
                """
            ).bindparams(bindparam("slides", expanding=True))
            s_df = pd.read_sql(ss, conn, params={"slides": slides})

        if s_df is not None and not s_df.empty and "slides" in s_df.columns:
            s_df = s_df.copy()
            s_df["_sk"] = s_df["slides"].astype(str).str.strip().str.upper()
            for _, srow in s_df.iterrows():
                d = srow.to_dict()
                sk = str(d.get("_sk") or d.get("slides", "")).strip().upper()
                d.pop("_sk", None)
                d.pop("slides", None)
                if sk:
                    previews[sk] = _kb_summary_preview(d)

        base["kb_summary"] = base["slide_id"].map(lambda x: previews.get(str(x).strip().upper(), ""))
        return {"ok": True, "df": base.reset_index(drop=True), "error": None}

    except Exception as e:
        return {"ok": False, "df": None, "error": str(e)}


@st.cache_data(show_spinner=False, ttl=300)
def load_valid_hpc_ids() -> set[int]:
    df = load_hpc_titles()
    if df.empty:
        return set()
    return {int(x) for x in df["hpc_id"].dropna().astype(int).tolist()}


_VIEWER_CTX = None  # reserved


def _kb_summary_preview(row_dict: dict, max_chars: int = 520) -> str:
    """Turn a summary row into a compact line (v21-style KB fields, truncated for the table)."""
    parts = []
    n = 0
    for k, v in row_dict.items():
        lk = str(k).lower()
        if lk == "slides" or v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        chunk = f"{k}={v}"
        if len(chunk) > 120:
            chunk = chunk[:117] + "…"
        if n + len(chunk) > max_chars:
            parts.append("…")
            break
        parts.append(chunk)
        n += len(chunk) + 2
    return " · ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Color helpers (unchanged from v21)
# ---------------------------------------------------------------------------

def color_for_hpc(hpc_id):
    if hpc_id is None or pd.isna(hpc_id):
        return (160, 160, 160)
    h = int(hashlib.md5(str(int(hpc_id)).encode()).hexdigest(), 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def color_for_inflammation(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)
    mapping = {"none-sparse": (80, 200, 120), "mild-moderate": (255, 200, 0), "marked": (255, 80, 80)}
    return mapping.get(str(label).strip().lower(), (160, 160, 160))


def color_for_necrosis(label):
    if label is None or pd.isna(label) or str(label).strip() == "":
        return (160, 160, 160)
    s = str(label).strip().lower()
    if s == "none": return (60, 200, 120)
    if s == "some": return (255, 165, 0)
    if s == "universal": return (200, 40, 40)
    return (160, 160, 160)


def color_for_malignant(flag):
    # Grey means "this row does not say" — a NULL or a spelling
    # backend/malignancy.py does not recognise. Deliberately not folded into
    # the non-malignant green: a dictionary row that needs attention should be
    # visible in the viewer rather than shown as benign.
    flag = malignant_flag(flag)
    if flag is None:
        return (160, 160, 160)
    return (255, 80, 80) if flag else (80, 200, 120)


def color_for_adjacency_group(group_name):
    if group_name == "a_touch": return (80, 200, 120)
    if group_name == "b_touch": return (255, 165, 0)
    return (160, 160, 160)


def rgb_to_css(rgb):
    r, g, b = map(int, rgb)
    return f"rgb({r}, {g}, {b})"


# ---------------------------------------------------------------------------
# Intent detection helpers (unchanged from v21)
# ---------------------------------------------------------------------------

def detect_adjacency_intent(q: str):
    q0 = (q or "").lower()
    keywords = ["beside", "besides", "next to", "adjacent", "touching", "near", "around",
                 "cooccur", "co-occur", "co occur", "cooccurrence", "co-occurrence", "co occurrence"]
    if not any(k in q0 for k in keywords):
        return None
    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    if len(hpcs) >= 2:
        a, b = int(hpcs[0]), int(hpcs[1])
        if a != b:
            return a, b
    return None


def detect_single_hpc_adjacency_intent(q: str):
    q0 = (q or "").lower()
    adj_words = ["beside", "besides", "next to", "adjacent", "touching", "near", "around", "neighbors"]
    if not any(w in q0 for w in adj_words):
        return None
    hpcs = re.findall(r"hpc\s*([0-9]+)", q0)
    return int(hpcs[0]) if len(hpcs) == 1 else None


def parse_hpc_id(q: str):
    if not q: return None
    m = re.search(r"hpc\s*([0-9]+)", q.lower())
    return int(m.group(1)) if m else None


def parse_inflammation(q: str):
    if not q: return None
    q0 = q.lower()
    if "marked" in q0: return "marked"
    if "mild" in q0 or "moderate" in q0: return "mild-moderate"
    if "non" in q0 and "sparse" in q0: return "none-sparse"
    return None


def parse_necrosis(q: str):
    if not q: return None
    q0 = q.lower()
    if "universal" in q0: return "universal"
    if "some" in q0: return "some"
    if "none" in q0: return "none"
    return None


def parse_highlight_mode(q: str):
    q0 = (q or "").lower()
    if "heatmap" in q0: return "Heatmap"
    if "inflammation" in q0: return "Inflammation"
    if "necrosis" in q0: return "Necrosis"
    if "malignant" in q0: return "Malignant"
    if "adjacent" in q0 or "beside" in q0 or "cooccur" in q0: return "Adjacency"
    if "hpc" in q0 or "cluster" in q0: return "HPC clusters"
    return None


def get_viewer_ctx() -> ViewerContext:
    return ViewerContext(
        load_tile_coords_for_slide=load_tile_coords_for_slide,
        load_adjacency_for_slide=load_adjacency_for_slide,
        slides_for_hpc_from_kb=slides_for_hpc_from_kb,
        detect_entity_patterns=detect_entity_patterns,
        detect_adjacency_intent=detect_adjacency_intent,
        detect_single_hpc_adjacency_intent=detect_single_hpc_adjacency_intent,
        parse_hpc_id=parse_hpc_id,
        parse_inflammation=parse_inflammation,
        parse_necrosis=parse_necrosis,
        parse_highlight_mode=parse_highlight_mode,
    )


# ---------------------------------------------------------------------------
# apply_chat_query_to_wsi_state — legacy; prefer apply_plan_to_session (v25)
# ---------------------------------------------------------------------------

def apply_chat_query_to_wsi_state(prompt: str, slide_id: str):
    det = detect_entity_patterns(prompt)
    if det and det.get("slide"):
        slide_id = str(det["slide"]).strip().upper()
        st.session_state.active_slide = slide_id
    if not prompt:
        return
    slide_id = str(slide_id).strip().upper()

    st.session_state.query_inflammation = None
    st.session_state.query_necrosis = None
    st.session_state.adj_hpc_a = None
    st.session_state.adj_hpc_b = None
    st.session_state.adj_tile_sets = None

    df = load_tile_coords_for_slide(slide_id)
    if df.empty:
        return

    pair = detect_adjacency_intent(prompt)
    single = detect_single_hpc_adjacency_intent(prompt)

    if pair or single is not None:
        needed = {"slide_tile", "x_native", "y_native", "hpc_id"}
        if needed - set(df.columns):
            return

    if pair:
        a, b = pair
        st.session_state.highlight_mode = "Adjacency"
        st.session_state.adj_hpc_a = a
        st.session_state.adj_hpc_b = b
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        p = (a, b) if a < b else (b, a)
        st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
        return

    if single is not None:
        st.session_state.highlight_mode = "Adjacency"
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        candidates = [((a, b), cnt) for (a, b), cnt in pair_edge_counts.items() if a == single or b == single]
        if candidates:
            (a_sel, b_sel), _ = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            st.session_state.adj_hpc_a = a_sel
            st.session_state.adj_hpc_b = b_sel
            p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
            st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(p, {"a_touch": set(), "b_touch": set()})
        else:
            st.session_state.adj_tile_sets = {"a_touch": set(), "b_touch": set()}
        return

    hpc_id = parse_hpc_id(prompt)
    infl = parse_inflammation(prompt)
    nec = parse_necrosis(prompt)
    mode = parse_highlight_mode(prompt)
    st.session_state.query_inflammation = infl
    st.session_state.query_necrosis = nec

    q0 = (prompt or "").lower()
    if "heatmap" in q0 and hpc_id is not None:
        st.session_state.highlight_mode = "Heatmap"
        st.session_state.heat_hpc = int(hpc_id)
        st.session_state.setdefault("heat_min", 0.0)
        st.session_state.setdefault("heat_alpha", 0.6)
        st.session_state.selected_hpc = None
        return

    if hpc_id is not None:
        st.session_state.selected_hpc = hpc_id
        st.session_state.highlight_mode = mode or "HPC clusters"
    elif mode:
        st.session_state.highlight_mode = mode


# ---------------------------------------------------------------------------
# HPC annotation renderer (calls tile server API)
# ---------------------------------------------------------------------------

def render_hpc_annotation(hpc_id):
    try:
        h = int(hpc_id)
    except (TypeError, ValueError):
        st.warning(f"Invalid HPC ID: {hpc_id}")
        return

    try:
        info = client.get_hpc_info(h)
    except Exception as e:
        st.warning(f"Could not fetch HPC {h} info: {e}")
        return

    hpc_title = str(info.get("hpc_title") or load_hpc_title_map().get(h, "") or "").strip()
    title_text = hpc_title if hpc_title else "No title available"

    st.markdown(
        f"""
        <div style="
            background:#111827;
            border:1px solid #374151;
            border-left:6px solid {rgb_to_css(color_for_hpc(h))};
            border-radius:10px;
            padding:12px 14px;
            margin-top:8px;
            margin-bottom:12px;
            color:#f3f4f6;
            line-height:1.45;
        ">
            <div style="font-size:15px;font-weight:800;margin-bottom:5px;">HPC {h}</div>
            <div style="font-size:16px;font-weight:700;color:#ffffff;">{html.escape(title_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_fields = [
        ("malignant", "Malignant"),
        ("inflammation", "Inflammation"),
        ("necrosis", "Necrosis"),
        ("cluster_homogeneity", "Cluster homogeneity"),
    ]

    rows = []
    for key, label in summary_fields:
        value = info.get(key)
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            rows.append({"Field": label, "Value": str(value)})

    mal = info.get("malignant_details") or {}
    non = info.get("non_malignant_details") or {}
    detail_source = mal if mal else non

    for key, value in detail_source.items():
        if key in ("id", "hpc_id"):
            continue
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        rows.append({"Field": str(key).replace("_", " ").title(), "Value": str(value)})

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No additional phenotype details found.")


# ---------------------------------------------------------------------------
# Tile info panel
# ---------------------------------------------------------------------------

def render_tile_info(tile_row, heat_hpc=None):
    st.subheader("Tile info")
    info = {
            "slide_tile": tile_row.get("slide_tile"),
            "tiles": tile_row.get("tiles"),
            "hpc_id": tile_row.get("hpc_id"),
            "hpc_title": tile_row.get("hpc_title"),
            "inflammation": tile_row.get("inflammation"),
            "necrosis": tile_row.get("necrosis"),
            "malignant": tile_row.get("malignant"),
        }
    def _clean_value(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except Exception:
            pass
        return str(v)

    info_df = pd.DataFrame(
        [{"Field": k, "Value": _clean_value(v)} for k, v in info.items()]
    )
    st.dataframe(info_df, width="stretch", hide_index=True)

    if heat_hpc is not None:
        col = f"p_hpc_{int(heat_hpc)}"
        if col in tile_row.index:
            p = tile_row.get(col)
            if p is not None and not pd.isna(p):
                st.metric(f"Heatmap probability for HPC {heat_hpc}", f"{float(p):.4f}")


# ===========================================================================
# show_wsi — THE MAIN WSI VIEWER
#
# Key difference from v21: thumbnail comes from tile server (HTTP JPEG),
# NOT from openslide over sshfs. Tile images on click also come from server.
# ===========================================================================

def show_wsi(slide_id, ui_suffix=""):
    """WSI viewer. ``ui_suffix`` disambiguates Streamlit keys when two viewers are open."""

    def _k(base: str) -> str:
        return f"{base}__{ui_suffix}" if ui_suffix else base

    slide_id = (slide_id or "").strip().upper()

    coords_df = load_tile_coords_for_slide(slide_id)
    if coords_df is None or coords_df.empty:
        st.warning(f"No tile coordinates found for slide {slide_id}")
        return

    for key, default in [
        (_k("selected_tile"), None),
        (_k("query_malignant"), None),
        ("adj_hpc_a", None),
        ("adj_hpc_b", None),
        ("adj_tile_sets", None),
        ("query_inflammation", None),
        ("query_necrosis", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ------------------------------------------------------------------
    # 1. Fetch thumbnail from tile server (cached locally after first hit)
    # ------------------------------------------------------------------
    try:
        info = client.get_slide_info(slide_id)
        w0 = info["level_dimensions"][0]["width"]
        h0 = info["level_dimensions"][0]["height"]
        tile_size_native = int(info["tile_size_native"])
    except Exception as e:
        st.error(f"Cannot reach tile server for slide info: {e}")
        return

    thumb_width = 3000
    try:
        base_region = client.get_thumbnail(slide_id, max_width=thumb_width)
    except Exception as e:
        st.error(f"Cannot fetch thumbnail from tile server: {e}")
        return

    downsample = w0 / base_region.size[0]

    # ------------------------------------------------------------------
    # 2. Build working dataframe
    # ------------------------------------------------------------------
    df = coords_df.copy()
    base_cols = ["tiles", "slides", "x_native", "y_native", "h5_index", "hpc_id",
                 "inflammation", "necrosis", "malignant", "slide_tile"]
    prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]
    cols_to_keep = [c for c in base_cols if c in df.columns] + prob_cols
    df = df[cols_to_keep].copy()

    if df.empty:
        st.warning(f"No tile data for slide {slide_id}")
        return

    # ------------------------------------------------------------------
    # 3. Grid overlay controls
    # ------------------------------------------------------------------
    if _k("highlight_mode") not in st.session_state:
        st.session_state[_k("highlight_mode")] = "HPC clusters"

    viewer_mode = st.radio(
        "Viewer mode",
        ["Pyramid zoom", "Click tile inspector"],
        horizontal=True,
        key=_k("viewer_mode"),
    )

    show_grid = st.checkbox(
        "Show tile grid overlays",
        value=True,
        key=_k("grid-toggle"),
    )

    st.radio(
        "Highlight mode",
        options=["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency", "Heatmap", "Survival Risk Heatmap"],
        key=_k("highlight_mode"),
        horizontal=True,
    )

    # Heatmap controls
    if st.session_state.get(_k("highlight_mode")) == "Heatmap":
        hpc_prob_cols = [c for c in df.columns if str(c).startswith("p_hpc_")]
        available_prob_hpcs = set()
        for col in hpc_prob_cols:
            try:
                available_prob_hpcs.add(int(str(col).replace("p_hpc_", "")))
            except Exception:
                continue

        if "hpc_id" in df.columns:
            present_hpcs = set(
                pd.to_numeric(df["hpc_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        else:
            present_hpcs = set()

        hpc_ids_heat = sorted(present_hpcs & available_prob_hpcs)
        st.write("DEBUG unique HPCs on slide:", len(present_hpcs))
        st.write(sorted(list(present_hpcs))[:100])
        st.write("DEBUG heatmap dropdown HPCs:", len(hpc_ids_heat))


        if not hpc_ids_heat:
            st.warning("No heatmap cluster options found for this WSI. This slide may not have matching hpc_id values and p_hpc_* probability columns.")
        else:
            current_heat_hpc = st.session_state.get(_k("heat_hpc"))
            if current_heat_hpc not in hpc_ids_heat:
                st.session_state[_k("heat_hpc")] = hpc_ids_heat[0]

            title_map = load_hpc_title_map()
            heat_options = {
                f"HPC {hid}" + (f": {title_map.get(hid)}" if title_map.get(hid) else ""): hid
                for hid in hpc_ids_heat
            }

            selected_heat_label = st.selectbox(
                "Heatmap HPC clusters present in this WSI",
                options=list(heat_options.keys()),
                index=list(heat_options.values()).index(st.session_state[_k("heat_hpc")]),
                key=_k("heat_hpc_label"),
            )
            st.session_state[_k("heat_hpc")] = heat_options[selected_heat_label]
            st.caption(f"Showing heatmap options for {len(hpc_ids_heat)} HPC clusters present on this WSI.")
            st.slider("Heat intensity", 0.1, 1.0, 0.6, 0.05, key=_k("heat_alpha"))

    df["hpc_id"] = pd.to_numeric(df["hpc_id"], errors="coerce").astype("Int64")

    used_survival_hpcs = []

    if st.session_state.get(_k("highlight_mode")) == "Survival Risk Heatmap":
        survival_map = load_survival_coefficients(p_threshold=0.05)

        df, used_survival_hpcs = add_survival_risk_score(
            df,
            survival_map,
        )

        st.session_state[_k("selected_hpc")] = None
        selected_hpc = None

        if not used_survival_hpcs:
            st.warning(
                "No survival-linked HPC probability columns found for this slide. "
                "This slide may not contain HPCs that also have survival data."
            )
        else:
            st.caption(
                f"Survival risk heatmap uses {len(used_survival_hpcs)} HPCs that are both present in this WSI and available in survival analysis."
            )

    if st.session_state.get(_k("highlight_mode")) != "Survival Risk Heatmap":
        st.session_state[_k("survival_risk_filter")] = None

    hpc_titles_df = load_hpc_titles()

    if not hpc_titles_df.empty and "hpc_id" in df.columns:
        df = df.merge(hpc_titles_df, on="hpc_id", how="left")
    else:
        df["hpc_title"] = ""

    # ------------------------------------------------------------------
    # 4. Draw overlay
    # ------------------------------------------------------------------
    highlight_mode = st.session_state.get(_k("highlight_mode"), "HPC clusters")
    selected_hpc = st.session_state.get(_k("selected_hpc"))

    if selected_hpc is not None and highlight_mode == "HPC clusters":
        grid_df = df[df["hpc_id"] == selected_hpc]
    else:
        grid_df = df

    base_rgba = base_region.convert("RGBA")
    overlay = base_rgba.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    heat_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    heat_draw = ImageDraw.Draw(heat_layer, "RGBA")

    # HPC filter panel
    if _k("selected_hpc") not in st.session_state:
        st.session_state[_k("selected_hpc")] = None
    hpc_list = sorted(df["hpc_id"].dropna().astype(int).unique().tolist())

    # Adjacency controls
    if highlight_mode == "Adjacency":
        pair_edge_counts, tile_has_neighbor_pair = load_adjacency_for_slide(slide_id)
        with st.expander("Adjacency and cooccurrence controls", expanded=False):
            top_pairs = sorted(pair_edge_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            if top_pairs:
                options = [f"HPC {a} ↔ HPC {b} ({cnt} edges)" for (a, b), cnt in top_pairs]
                selected_pair = st.selectbox(
                    "Pick a top cooccurring pair", options, key=_k("adj_top_pair_select")
                )
                idx = options.index(selected_pair)
                (a_sel, b_sel), _ = top_pairs[idx]
                if st.button("Highlight selected top pair", key=_k("adj_top_pair_btn")):
                    p = (a_sel, b_sel) if a_sel < b_sel else (b_sel, a_sel)
                    st.session_state.adj_hpc_a = a_sel
                    st.session_state.adj_hpc_b = b_sel
                    st.session_state.adj_tile_sets = tile_has_neighbor_pair.get(
                        p, {"a_touch": set(), "b_touch": set()}
                    )
                    st.rerun()
            else:
                st.info("No adjacent cross HPC pairs found on this slide.")


    selected_hpc = st.session_state.get(_k("selected_hpc"))
    if selected_hpc is not None:
                selected_hpc_title = load_hpc_title_map().get(int(selected_hpc), "")

                if selected_hpc_title:
                    st.markdown(
                        f"""
                        <div style="
                            background:#111827;
                            border:1px solid #374151;
                            border-left:6px solid {rgb_to_css(color_for_hpc(selected_hpc))};
                            border-radius:10px;
                            padding:12px 14px;
                            margin-top:10px;
                            margin-bottom:10px;
                            color:#f3f4f6;
                            font-size:15px;
                            line-height:1.5;
                        ">
                            <span style="font-weight:700;">HPC {selected_hpc}</span><br>
                            <span style="opacity:0.92;">{html.escape(selected_hpc_title)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("HPC Biological Interpretation", expanded=True):
                    render_hpc_annotation(selected_hpc)

    # Filters
    infl_f = st.session_state.get("query_inflammation")
    nec_f = st.session_state.get("query_necrosis")
    mal_f = st.session_state.get(_k("query_malignant"))
    filtered_df = df.copy()

    if st.session_state.get(_k("selected_hpc")) is not None:
        filtered_df = filtered_df[filtered_df["hpc_id"] == st.session_state[_k("selected_hpc")]]
    if infl_f is not None:
        filtered_df = filtered_df[filtered_df["inflammation"].astype(str).str.lower().str.strip() == infl_f]
    if nec_f is not None:
        filtered_df = filtered_df[filtered_df["necrosis"].astype(str).str.lower().str.strip() == nec_f]
    if mal_f is not None and "malignant" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["malignant"].map(describe_malignant) == mal_f]
    st.caption(f"Matched tiles: {len(filtered_df)}")

    # ------------------------------------------------------------------
    # 4a. Heatmap overlay (vectorized, same as v21)
    # ------------------------------------------------------------------
    if highlight_mode == "Heatmap":
        h = int(st.session_state.get(_k("heat_hpc"), 0))
        col_name = f"p_hpc_{h}"
        if col_name in df.columns:
            _draw_heatmap(
                heat_draw,
                df,
                col_name,
                downsample,
                float(st.session_state.get(_k("heat_alpha"), 0.6)),
                tile_size_native,
            )


    if highlight_mode == "Survival Risk Heatmap":
            _draw_survival_risk_heatmap(
                heat_draw,
                df,
                downsample,
                tile_size_native,
                alpha=140,
                risk_filter=st.session_state.get(_k("survival_risk_filter")),
            )
    # ------------------------------------------------------------------
    # 4b. Grid outlines (vectorized where possible)
    # ------------------------------------------------------------------
    if show_grid and highlight_mode not in ("Heatmap", "Survival Risk Heatmap"):
        _draw_grid(
            draw,
            grid_df if selected_hpc is not None and highlight_mode not in ("Adjacency", "Heatmap", "Survival Risk Heatmap") else df,
            downsample,
            highlight_mode,
            infl_f,
            nec_f,
            mal_f,
            tile_size_native,
        )


    # Selected tile highlight
    sel = st.session_state.get(_k("selected_tile"))
    if sel and "x_native" in sel and "y_native" in sel:
        sx = int(sel["x_native"] / downsample)
        sy = int(sel["y_native"] / downsample)
        ts = int(tile_size_native / downsample)
        draw.rectangle([sx - 3, sy - 3, sx + ts + 3, sy + ts + 3], outline="yellow", width=15)
        draw.rectangle([sx, sy, sx + ts, sy + ts], outline="lime", width=15)

    if highlight_mode in ("Heatmap", "Survival Risk Heatmap"):
        overlay = Image.alpha_composite(overlay, heat_layer)

    overlay_np = np.asarray(overlay.convert("RGB"), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 5. Viewer + sticky legend (same viewport column)
    # ------------------------------------------------------------------
    # st.subheader(f"Slide Viewer — {slide_id}")
    col_img, col_leg = st.columns([5, 1])
    with col_leg:
        _hh = int(st.session_state.get(_k("heat_hpc"), 0)) if highlight_mode == "Heatmap" else None
        if highlight_mode == "Survival Risk Heatmap" and used_survival_hpcs:
            st.markdown("**Legend**")
            st.caption("Red = higher risk-associated signal. Blue = protective-associated signal. Transparent = near neutral.")
            st.markdown(
                """
                <div style="background:#f1f5f9;color:#0f172a;border:1px solid #64748b;border-radius:10px;padding:10px;font-size:12px;line-height:1.35;">
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(255,0,0,0.75);border:1px solid #0f172a;"></span><span>Risk-associated contribution</span></div>
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(0,80,255,0.75);border:1px solid #0f172a;"></span><span>Protective-associated contribution</span></div>
                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;"><span style="width:14px;height:14px;background:rgba(255,255,255,0.15);border:1px solid #0f172a;"></span><span>Near neutral</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if highlight_mode == "Survival Risk Heatmap":
            st.markdown("**Survival Risk Legend**")
            st.caption("Red = risk-associated. Blue = protective. Transparent = neutral.")

            if used_survival_hpcs:
                st.caption(f"Using survival HPCs: {used_survival_hpcs}")
            else:
                st.warning("No survival-linked HPCs found.")

        elif highlight_mode == "HPC clusters":
            st.markdown("**Legend**")
            st.caption("Click an HPC to isolate it.")

            if st.button("Show all", key=_k("legend_show_all_hpcs"), use_container_width=True):
                st.session_state[_k("selected_hpc")] = None
                st.rerun()

            n_tiles = max(len(df), 1)
            hpc_counts = pd.to_numeric(df["hpc_id"], errors="coerce").dropna().astype(int).value_counts()

            for hid, count in hpc_counts.head(14).items():
                hid = int(hid)
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_hpc(hid))

                st.markdown(
                    f"""
                    <div style="
                        width:100%;
                        height:4px;
                        background:{css_color};
                        border-radius:4px 4px 0 0;
                        margin-top:6px;
                        margin-bottom:-6px;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"HPC {hid} · {pct:.1f}%",
                    key=_k(f"legend_hpc_btn_{hid}"),
                    use_container_width=True,
                ):
                    st.session_state[_k("selected_hpc")] = hid
                    st.rerun()

        elif highlight_mode == "Inflammation":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all inflammation", key=_k("legend_all_inflammation"), use_container_width=True):
                st.session_state.query_inflammation = None
                st.rerun()

            n_tiles = max(len(df), 1)
            infl_counts = (
                df["inflammation"]
                .fillna("Missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": "missing"})
                .value_counts()
            )

            for label, count in infl_counts.items():
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_inflammation(label))
                pretty = str(label).title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_infl_btn_{label}"),
                    use_container_width=True,
                ):
                    st.session_state.query_inflammation = label
                    st.rerun()

        elif highlight_mode == "Necrosis":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all necrosis", key=_k("legend_all_necrosis"), use_container_width=True):
                st.session_state.query_necrosis = None
                st.rerun()

            n_tiles = max(len(df), 1)
            nec_counts = (
                df["necrosis"]
                .fillna("Missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"": "missing"})
                .value_counts()
            )

            for label, count in nec_counts.items():
                pct = 100.0 * float(count) / n_tiles
                css_color = rgb_to_css(color_for_necrosis(label))
                pretty = str(label).title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_nec_btn_{label}"),
                    use_container_width=True,
                ):
                    st.session_state.query_necrosis = label
                    st.rerun()

        elif highlight_mode == "Malignant":
            st.markdown("**Legend**")
            st.caption("Click a category to isolate it.")

            if st.button("Show all malignant", key=_k("legend_all_malignant"), use_container_width=True):
                st.session_state[_k("query_malignant")] = None
                st.rerun()

            n_tiles = max(len(df), 1)
            mal_counts = (
                df["malignant"]
                .fillna("missing")
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
            )

            for label, count in mal_counts.items():
                pct = 100.0 * float(count) / n_tiles
                label_key = describe_malignant(label)

                css_color = rgb_to_css(color_for_malignant(label))
                pretty = label_key.replace("-", " ").title()

                st.markdown(
                    f'<div style="width:100%;height:4px;background:{css_color};border-radius:4px 4px 0 0;margin-top:6px;margin-bottom:-6px;"></div>',
                    unsafe_allow_html=True,
                )

                if st.button(
                    f"{pretty} · {pct:.1f}%",
                    key=_k(f"legend_mal_btn_{label_key}"),
                    use_container_width=True,
                ):
                    st.session_state[_k("query_malignant")] = label_key
                    st.rerun()

        else:
            st.markdown(build_legend_panel_html(highlight_mode, df, _hh), unsafe_allow_html=True)

    with col_img:
        if st.session_state.get(_k("viewer_mode"), "Pyramid zoom") == "Pyramid zoom":
            # st.error(f"DEBUG highlight_mode = {highlight_mode}")
            st.caption("OpenSeadragon pyramid viewer with optional tile-grid overlay.")

            if show_grid:
                if highlight_mode == "Survival Risk Heatmap":
                    osd_overlay_records = build_survival_osd_overlay_records(
                        df=df,
                        tile_size_native=tile_size_native,
                    )
                else:
                    osd_overlay_records = build_osd_overlay_records(
                        df=grid_df if selected_hpc is not None and highlight_mode not in ("Adjacency", "Heatmap", "Survival Risk Heatmap") else df,
                        highlight_mode=highlight_mode,
                        infl_f=infl_f,
                        nec_f=nec_f,
                        mal_f=mal_f,
                        tile_size_native=tile_size_native,
                        heat_hpc=st.session_state.get(_k("heat_hpc")),
                        heat_alpha=st.session_state.get(_k("heat_alpha"), 0.6),
                    )
            else:
                osd_overlay_records = []

            if highlight_mode == "Survival Risk Heatmap":
                st.caption(f"Survival OSD overlay records: {len(osd_overlay_records)}")

            render_openseadragon_viewer(
                slide_id,
                viewer_key=f"{_k('osd')}_{highlight_mode}_{len(osd_overlay_records)}_survival_v2_{st.session_state.get(_k('heat_hpc'), 'none')}_{st.session_state.get(_k('heat_alpha'), 0.6)}",
                height=780,
                overlay_tiles=osd_overlay_records,
                # Built from df, not from the overlay records: the overlay is
                # filtered by the legend and capped at 6,000, and neither has
                # anything to do with "which tile is under my pointer".
                tile_index=build_osd_tile_index(df, tile_size_native),
                tile_size_native=tile_size_native,
            )
            return

        st.write("Click a tile to view it.")
        click = streamlit_image_coordinates(overlay_np, key=_k("wsi-click-coords"))
        if not click:
            st.info("Click anywhere on the slide to select a tile.")
            return

        cx, cy = click["x"], click["y"]
        native_x = cx * downsample
        native_y = cy * downsample

        tol = tile_size_native * 0.1
        mask = (
            (df["x_native"].astype(float) - tol <= native_x)
            & (native_x <= df["x_native"].astype(float) + tile_size_native + tol)
            & (df["y_native"].astype(float) - tol <= native_y)
            & (native_y <= df["y_native"].astype(float) + tile_size_native + tol)
        )
        matches = df[mask]
        if matches.empty:
            st.warning("Clicked area does not match any tile.")
            return
        tile_row = matches.iloc[0]

        new_selected = {
            "slide_tile": str(tile_row.get("slide_tile", "")),
            "x_native": float(tile_row["x_native"]),
            "y_native": float(tile_row["y_native"]),
        }
        prev = st.session_state.get(_k("selected_tile"))
        same_tile = isinstance(prev, dict) and prev.get("slide_tile") == new_selected["slide_tile"] and float(
            prev.get("x_native", -1)
        ) == new_selected["x_native"] and float(prev.get("y_native", -1)) == new_selected["y_native"]
        if not same_tile:
            st.session_state[_k("selected_tile")] = new_selected
            st.rerun()

        # ------------------------------------------------------------------
        # 6. Show selected tile image — fetched from tile server
        # ------------------------------------------------------------------
        slide_tile_key = str(tile_row.get("slide_tile", ""))
        tile_name = tile_row.get("tiles", slide_tile_key)
        tile_hpc = tile_row.get("hpc_id")
        st.success(f"Tile selected: {tile_name}")

        if tile_hpc is not None and not pd.isna(tile_hpc):
            st.info(hpc_label(tile_hpc, max_title_chars=160))

        try:
            tile_img = fetch_tile_region_from_svs(
                slide_id=slide_id,
                x_native=tile_row.get("x_native"),
                y_native=tile_row.get("y_native"),
                tile_size_native=tile_size_native,
            )
            display_width = min(max(int(tile_img.width * 2), tile_img.width), 700)
            st.image(
                tile_img,
                caption=f"{tile_name} · native SVS region · {tile_img.width}×{tile_img.height}px",
                width=display_width,
            )
        except Exception as e:
            st.warning(f"Could not load native SVS tile region: {e}")

        heat_hpc = (
            st.session_state.get(_k("heat_hpc"))
            if st.session_state.get(_k("highlight_mode")) == "Heatmap"
            else None
        )
        render_tile_info(tile_row, heat_hpc=heat_hpc)

        hpc_id = tile_row.get("hpc_id")
        if hpc_id is not None and not pd.isna(hpc_id):
            render_hpc_annotation(int(hpc_id))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_heatmap(heat_draw, df, col_name, downsample, alpha_max, tile_size_native):
    p_series = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
    xs = (df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (df["y_native"].astype(float) / downsample).astype(int).to_numpy()
    ps = p_series.to_numpy(dtype=np.float32)
    valid = np.isfinite(ps)
    if not valid.any():
        return
    pmin, pmax = float(np.nanmin(ps[valid])), float(np.nanmax(ps[valid]))
    if pmax <= pmin:
        pmax = pmin + 1e-9
    epsilon = 1e-6
    denom = np.log10(pmax + epsilon) - np.log10(pmin + epsilon)
    if denom == 0: denom = 1e-9
    t_all = np.clip((np.log10(ps + epsilon) - np.log10(pmin + epsilon)) / denom, 0.0, 1.0)
    alpha_min = 0.10
    alpha_max = max(0.2, min(alpha_max, 0.95))
    a_all = alpha_min + (alpha_max - alpha_min) * t_all
    alpha_all = (255.0 * a_all).astype(np.uint8)
    ts = int(tile_size_native / downsample)
    for x, y, t, alpha in zip(xs[valid], ys[valid], t_all[valid], alpha_all[valid]):
        R = int(255 * float(t))
        G = 255
        B = int(255 * (1 - float(t)))
        heat_draw.rectangle([int(x), int(y), int(x) + ts, int(y) + ts],
                            fill=(R, G, B, int(alpha)), outline=(255, 255, 255, 40))


def _draw_survival_risk_heatmap(heat_draw, df, downsample, tile_size_native, alpha=140, risk_filter=None):
    if "survival_risk_norm" not in df.columns:
        return

    ts = int(tile_size_native / downsample)

    xs = (df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (df["y_native"].astype(float) / downsample).astype(int).to_numpy()
    scores = pd.to_numeric(df["survival_risk_norm"], errors="coerce").fillna(0).to_numpy()

    for x, y, score in zip(xs, ys, scores):
        score = float(score)

        if risk_filter == "risky" and score <= 0.02:
            continue
        if risk_filter == "protective" and score >= -0.02:
            continue
        if risk_filter == "neutral" and not (-0.02 <= score <= 0.02):
            continue

        fill = risk_to_rgba(score, alpha_max=alpha)
        heat_draw.rectangle(
            [int(x), int(y), int(x) + ts, int(y) + ts],
            fill=fill,
            outline=(255, 255, 255, 35),
        )


def _draw_grid(draw, grid_df, downsample, highlight_mode, infl_f, nec_f, mal_f, tile_size_native):
    ts = int(tile_size_native / downsample)
    # Pre-extract numpy arrays for speed
    xs = (grid_df["x_native"].astype(float) / downsample).astype(int).to_numpy()
    ys = (grid_df["y_native"].astype(float) / downsample).astype(int).to_numpy()

    for i, (x, y) in enumerate(zip(xs, ys)):
        r = grid_df.iloc[i]
        if infl_f is not None and str(r.get("inflammation", "")).strip().lower() != infl_f:
            continue
        if nec_f is not None and str(r.get("necrosis", "")).strip().lower() != nec_f:
            continue
        if mal_f is not None:
            val = r.get("malignant")

            if isinstance(val, (int, np.integer, bool, np.bool_)):
                current = "malignant" if bool(val) else "non-malignant"
            else:
                s = str(val).strip().lower()

                if s in ("true", "t", "1", "yes", "y", "malignant"):
                    current = "malignant"
                elif s in ("false", "f", "0", "no", "n", "non-malignant", "non malignant"):
                    current = "non-malignant"
                else:
                    current = "missing"

            if current != mal_f:
                continue

        if highlight_mode == "Inflammation":
            color = color_for_inflammation(r.get("inflammation"))
        elif highlight_mode == "Necrosis":
            color = color_for_necrosis(r.get("necrosis"))
        elif highlight_mode == "Malignant":
            color = color_for_malignant(r.get("malignant"))
        elif highlight_mode == "Adjacency":
            adj = st.session_state.get("adj_tile_sets")
            if adj is None: continue
            tkey = str(r.get("slide_tile"))
            if tkey in adj.get("a_touch", set()):
                color = color_for_adjacency_group("a_touch")
            elif tkey in adj.get("b_touch", set()):
                color = color_for_adjacency_group("b_touch")
            else:
                continue
        else:
            color = color_for_hpc(r.get("hpc_id"))
        draw.rectangle([x, y, x + ts, y + ts], outline=color, width=5)




def build_survival_osd_overlay_records(df, tile_size_native, max_tiles=6000):
    records = []

    if df is None or df.empty or "survival_risk_norm" not in df.columns:
        return records

    use_df = df.head(max_tiles).copy()
    scores = pd.to_numeric(use_df["survival_risk_norm"], errors="coerce").fillna(0.0)

    for idx, r in use_df.iterrows():
        score = float(scores.loc[idx])
        score = max(-1.0, min(1.0, score))
        intensity = abs(score)

        if intensity < 0.02:
            continue

        alpha = 0.25 + 0.65 * intensity

        if score > 0:
            fill = f"rgba(255, 0, 0, {alpha:.3f})"
            stroke = "rgba(180, 0, 0, 0.95)"
        else:
            fill = f"rgba(0, 80, 255, {alpha:.3f})"
            stroke = "rgba(0, 45, 180, 0.95)"

        records.append({
            "x": float(r["x_native"]),
            "y": float(r["y_native"]),
            "w": int(tile_size_native),
            "h": int(tile_size_native),
            "fill": fill,
            "stroke": stroke,
            "stroke_width": 2,
            "opacity": "1.0",
            "slide_tile": str(r.get("slide_tile", "")),
            "survival_risk_norm": score,
        })

    return records


def build_osd_tile_index(df, tile_size_native):
    """What the pyramid viewer needs to answer "which tile is under the
    pointer" — one compact row per tile: [col, row, slide_tile, hpc_id].

    Compact lists rather than dicts because this is serialised into the
    iframe's HTML: a slide with 20,000 tiles is a few hundred KB this way and
    roughly three times that with a key repeated per field.

    Not filtered and not capped, unlike build_osd_overlay_records below. What
    a tile *is* does not depend on the legend filter, and the 6,000-record cap
    exists because drawing that many SVG rects is slow — looking one up in a
    Map is not.

    col/row come straight from tile_coordinates. They are derived from
    x_native only when absent, using the same pitch the viewer hit-tests with,
    so a row that has them and a row that does not land in the same lattice.
    """
    if df is None or df.empty:
        return []

    pitch = float(tile_size_native or 0)
    rows = []
    for _, r in df.iterrows():
        col, row = r.get("col"), r.get("row")
        if col is None or pd.isna(col) or row is None or pd.isna(row):
            if pitch <= 0:
                continue
            x, y = r.get("x_native"), r.get("y_native")
            if x is None or pd.isna(x) or y is None or pd.isna(y):
                continue
            col, row = int(float(x) // pitch), int(float(y) // pitch)
        hpc = r.get("hpc_id")
        rows.append([
            int(col),
            int(row),
            str(r.get("slide_tile") or ""),
            None if hpc is None or pd.isna(hpc) else int(hpc),
        ])
    return rows


def build_osd_overlay_records(df, highlight_mode, infl_f, nec_f, mal_f, tile_size_native, heat_hpc=None, heat_alpha=0.6, max_tiles=6000):
    records = []

    if df is None or df.empty:
        return records

    use_df = df.head(max_tiles).copy()

    for _, r in use_df.iterrows():
        if infl_f is not None and str(r.get("inflammation", "")).strip().lower() != infl_f:
            continue

        if nec_f is not None and str(r.get("necrosis", "")).strip().lower() != nec_f:
            continue
        if highlight_mode == "Heatmap":
            if heat_hpc is None:
                continue
            col = f"p_hpc_{int(heat_hpc)}"

            if col not in use_df.columns:
                continue

            pval = float(pd.to_numeric(pd.Series([r.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])

            p_all = pd.to_numeric(use_df[col], errors="coerce").fillna(0.0)
            pmin, pmax = float(p_all.min()), float(p_all.max())
            if pmax <= pmin:
                pmax = pmin + 1e-9

            t = (pval - pmin) / (pmax - pmin)
            red = int(255 * t)
            green = 255
            blue = int(255 * (1 - t))

            records.append({
                "x": float(r["x_native"]),
                "y": float(r["y_native"]),
                "w": int(tile_size_native),
                "h": int(tile_size_native),
                "fill": f"rgba({red}, {green}, {blue}, 0.55)",
                "stroke": "rgba(255,255,255,0.25)",
                "stroke_width": 1,
                "opacity": "1.0",
            })
            continue
        if highlight_mode == "Survival Risk Heatmap":
            if "survival_risk_norm" not in use_df.columns:
                continue

            score = float(pd.to_numeric(pd.Series([r.get("survival_risk_norm", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            score = max(-1.0, min(1.0, score))
            intensity = abs(score)

            if intensity < 0.02:
                continue

            alpha = 0.15 + 0.60 * intensity

            if score > 0:
                fill = f"rgba(255, 0, 0, {alpha:.3f})"
            else:
                fill = f"rgba(0, 80, 255, {alpha:.3f})"

            records.append({
                "x": float(r["x_native"]),
                "y": float(r["y_native"]),
                "w": int(tile_size_native),
                "h": int(tile_size_native),
                "fill": fill,
                "stroke": "rgba(255,255,255,0.20)",
                "stroke_width": 1,
                "opacity": "1.0",
            })
            continue
        if highlight_mode == "Inflammation":
            color = color_for_inflammation(r.get("inflammation"))
        elif highlight_mode == "Necrosis":
            color = color_for_necrosis(r.get("necrosis"))
        elif highlight_mode == "Malignant":
            color = color_for_malignant(r.get("malignant"))

        elif highlight_mode == "Adjacency":
            adj = st.session_state.get("adj_tile_sets")
            if adj is None:
                continue

            tkey = str(r.get("slide_tile"))
            if tkey in adj.get("a_touch", set()):
                color = color_for_adjacency_group("a_touch")
            elif tkey in adj.get("b_touch", set()):
                color = color_for_adjacency_group("b_touch")
            else:
                continue
        else:
            color = color_for_hpc(r.get("hpc_id"))

        records.append({
            "x": float(r["x_native"]),
            "y": float(r["y_native"]),
            "w": int(tile_size_native),
            "h": int(tile_size_native),
            "color": rgb_to_css(color),
            "slide_tile": str(r.get("slide_tile", "")),
            "hpc_id": "" if pd.isna(r.get("hpc_id")) else str(int(r.get("hpc_id"))),
        })

    return records


def _legend_coverage_pairs(df: pd.DataFrame, mode: str, heat_hpc: int | None) -> list[tuple[str, float]]:
    """(label, percent) for each category; percents sum to ~100 over all tiles in ``df``."""
    n = len(df)
    if n == 0:
        return []

    if mode == "Inflammation":
        if "inflammation" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def infl_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
                return "Missing"
            return str(x).strip().lower()

        vc = df["inflammation"].map(infl_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Necrosis":
        if "necrosis" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def nec_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == "":
                return "Missing"
            return str(x).strip().lower()

        vc = df["necrosis"].map(nec_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Malignant":
        if "malignant" not in df.columns:
            return [("Missing / unknown", 100.0)]

        def mal_lab(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "Missing"
            if isinstance(x, (int, np.integer)):
                return "Malignant" if bool(x) else "Non-malignant"
            s = str(x).strip().lower()
            if s in ("true", "t", "1", "yes", "y"):
                return "Malignant"
            if s in ("false", "f", "0", "no", "n"):
                return "Non-malignant"
            return "Missing"

        vc = df["malignant"].map(mal_lab).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    if mode == "Adjacency":
        adj = st.session_state.get("adj_tile_sets")
        if adj is None or "slide_tile" not in df.columns:
            return [("(select an HPC pair)", 100.0)]
        keys = df["slide_tile"].astype(str)
        in_a = keys.isin(adj.get("a_touch", set()))
        in_b = keys.isin(adj.get("b_touch", set()))
        a = st.session_state.get("adj_hpc_a", "?")
        b = st.session_state.get("adj_hpc_b", "?")
        other = ~(in_a | in_b)
        pairs = [
            (f"HPC {a} → {b} (green)", 100.0 * float(in_a.sum()) / n),
            (f"HPC {b} → {a} (orange)", 100.0 * float(in_b.sum()) / n),
            ("Not in selected pair", 100.0 * float(other.sum()) / n),
        ]
        return pairs

    if mode == "Heatmap":
        if heat_hpc is None:
            return []
        col = f"p_hpc_{int(heat_hpc)}"
        if col not in df.columns:
            return [("(no probability column)", 100.0)]
        p = pd.to_numeric(df[col], errors="coerce")
        valid = p.dropna()
        if valid.empty:
            return [("Missing score", 100.0)]
        q1, q2 = float(valid.quantile(0.33)), float(valid.quantile(0.66))

        def bucket(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "Missing"
            xf = float(x)
            if xf <= q1:
                return "Low P"
            if xf <= q2:
                return "Mid P"
            return "High P"

        vc = p.map(bucket).value_counts()
        return [(str(k), 100.0 * float(v) / n) for k, v in vc.items()]

    # HPC clusters (default)
    if "hpc_id" not in df.columns:
        return [("(no hpc_id)", 100.0)]
    h = pd.to_numeric(df["hpc_id"], errors="coerce")
    unlabeled = int(h.isna().sum())
    pairs: list[tuple[str, float]] = []
    vc = h.dropna().astype(int).value_counts()
    for hid, c in vc.head(14).items():
        pairs.append((f"HPC {int(hid)}", 100.0 * float(c) / n))
    if unlabeled:
        pairs.append(("Unlabeled", 100.0 * float(unlabeled) / n))
    if len(vc) > 14:
        shown = int(vc.head(14).sum())
        rest = int(vc.iloc[14:].sum())
        pairs.append((f"Other HPCs ({len(vc) - 14} ids)", 100.0 * float(rest) / n))
    return pairs


def _swatch_row(items: list[tuple[str, str]]) -> str:
    """items: (hex_bg, label)"""
    parts = []
    for bg, lab in items:
        parts.append(
            f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0;color:#0f172a;">'
            f'<span style="width:14px;height:14px;background:{bg};border:1px solid #0f172a;flex-shrink:0;"></span>'
            f"<span>{html.escape(lab)}</span></div>"
        )
    return '<div style="display:flex;flex-direction:column;gap:2px;">' + "".join(parts) + "</div>"


def build_legend_panel_html(mode: str, df: pd.DataFrame, heat_hpc: int | None) -> str:
    """Single HTML block: high-contrast panel + swatches + % tile coverage."""
    outer = (
        "position:sticky;top:3.5rem;max-height:88vh;overflow-y:auto;"
        "background:#f1f5f9;color:#0f172a;border:1px solid #64748b;border-radius:10px;"
        "padding:10px 10px 12px 10px;font-size:12px;line-height:1.35;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.12);"
    )
    parts = [f'<div style="{outer}">']
    parts.append('<div style="font-weight:700;font-size:13px;color:#020617;margin-bottom:6px;">Legend</div>')

    if mode == "Inflammation":
        parts.append(
            _swatch_row(
                [
                    ("#50C878", "None-sparse"),
                    ("#FFC800", "Mild-moderate"),
                    ("#FF5050", "Marked"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Necrosis":
        parts.append(
            _swatch_row(
                [
                    ("#3CC878", "None"),
                    ("#FFA500", "Some"),
                    ("#C82828", "Universal"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Malignant":
        parts.append(
            _swatch_row(
                [
                    ("#FF5050", "Malignant"),
                    ("#50C878", "Non-malignant"),
                    ("#A0A0A0", "Missing"),
                ]
            )
        )
    elif mode == "Adjacency":
        a = st.session_state.get("adj_hpc_a", "?")
        b = st.session_state.get("adj_hpc_b", "?")
        parts.append(
            _swatch_row(
                [
                    ("#50C878", f"HPC {a} touching HPC {b}"),
                    ("#FFA500", f"HPC {b} touching HPC {a}"),
                ]
            )
        )
    elif mode == "Heatmap":
        hh = int(heat_hpc) if heat_hpc is not None else 0
        parts.append(
            f'<div style="color:#0f172a;font-size:11px;margin:4px 0;">'
            f"Colormap for <b>P(HPC {hh})</b>: warm = higher probability; "
            "alpha = confidence.</div>"
        )
    else:
        parts.append(
            '<div style="color:#0f172a;font-size:11px;margin:4px 0;">'
            "Tile outlines use a <b>stable color per HPC id</b>. "
            "Use <i>Highlight tiles by HPC cluster</i> to isolate one HPC.</div>"
        )

    pairs = _legend_coverage_pairs(df, mode, heat_hpc)
    parts.append('<hr style="border:none;border-top:1px solid #94a3b8;margin:10px 0;">')
    parts.append(
        '<div style="font-weight:600;color:#0f172a;margin-bottom:4px;">'
        "Tile coverage (% of tiles on this slide)</div>"
    )
    if not pairs:
        parts.append('<div style="color:#475569;font-size:11px;">No coverage data.</div>')
    else:
        for lab, pct in pairs:
            parts.append(
                f'<div style="display:flex;justify-content:space-between;gap:8px;margin:3px 0;'
                f'color:#0f172a;font-size:11px;"><span>{html.escape(lab)}</span>'
                f'<span style="font-weight:600;white-space:nowrap;">{pct:.1f}%</span></div>'
            )

    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about your H&E cancer slides — I'll fetch insights instantly."}
    ]
if "active_slide" not in st.session_state:
    st.session_state.active_slide = None
if "viewer_open" not in st.session_state:
    st.session_state.viewer_open = False
if "hpc_wsi_explore_id" not in st.session_state:
    st.session_state.hpc_wsi_explore_id = None
if "hpc_wsi_ranking_df" not in st.session_state:
    st.session_state.hpc_wsi_ranking_df = None


def _ensure_active_slide(prompt_text):
    """Resolve a slide from the prompt or existing session state. Does NOT
    fall back to slides[0] — callers must handle an unresolved (None) slide."""
    det = detect_entity_patterns(prompt_text or "")
    slide_from_prompt = det.get("slide") if det else None
    if slide_from_prompt:
        st.session_state.active_slide = str(slide_from_prompt).strip().upper()
    return st.session_state.active_slide


def _plan_needs_slide(plan: dict) -> bool:
    """True when the plan implies a slide-scoped viewer/highlight action."""
    ui = plan.get("ui_actions") or {}
    flags = plan.get("flags") or {}
    ents = plan.get("entities") or {}
    return bool(
        ui.get("open_slide_viewer")
        or ui.get("highlight_mode")
        or ui.get("selected_hpc") is not None
        or flags.get("malignant")
        or flags.get("non_malignant")
        or ents.get("hpc")
        or ents.get("tile")
    )


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something about HPCs...", key="main_chat_input")

rerun_needed = False
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    session_context = {
        "active_slide": st.session_state.get("active_slide"),
        "viewer_open": st.session_state.get("viewer_open"),
        "selected_hpc": st.session_state.get("selected_hpc"),
        "highlight_mode": st.session_state.get("highlight_mode"),
    }

    slide_list = load_slide_list()
    plan = build_query_plan_v25(
        prompt,
        slide_list=slide_list,
        hpc_title_map=load_hpc_title_map(),
        valid_hpc_ids=load_valid_hpc_ids(),
        session_context=session_context,
    )
    path = save_query_plan_to_file(
        plan, slide_id=plan["entities"]["slide"][0] if plan["entities"]["slide"] else None
    )

    slide_id = _ensure_active_slide(prompt)

    with st.expander("Query Plan (debug)", expanded=False):
        st.json(plan)
        st.caption(f"Saved → {path} · planner: {plan.get('planner', '?')}")

    if not slide_id and _plan_needs_slide(plan):
        examples = ", ".join(slide_list[:3]) if slide_list else "TCGA-XX-XXXX"
        final_answer = (
            "Which slide would you like me to look at? "
            f"Name a slide (e.g. {examples}) or pick one from the dropdown below, then ask again."
        )
        with st.chat_message("assistant"):
            st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.rerun()

    slide_id = apply_plan_to_session(plan, prompt, slide_id, get_viewer_ctx())

    if should_fetch_from_db(plan):
        query_for_db = plan.get("enriched_query") or prompt
        try:
            structured_answer = chat_fetch_answer_from_db(query_for_db, engine, client)
        except Exception as e:
            structured_answer = f"DB query failed: {e}"
    else:
        structured_answer = None

    try:
        final_answer = explain_answer(
            plan,
            structured_answer,
            user_query=prompt,
            history=st.session_state.messages,
        )
    except Exception:
        final_answer = structured_answer or "Hi! How can I help you?"

    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.rerun()


# ---------------------------------------------------------------------------
# Slide picker + viewer
# ---------------------------------------------------------------------------
slide_options = sorted(load_slide_list())
if not slide_options:
    st.warning("No slides available from tile server.")
    slide_id = None
else:
    current = (st.session_state.get("active_slide") or "").strip().upper()
    default_idx = slide_options.index(current) if current in slide_options else 0
    slide_id = st.selectbox("Choose a slide", slide_options, index=default_idx, key="slide_selectbox")
    st.session_state.active_slide = slide_id

if st.session_state.viewer_open:
    if st.button("Close Slide Viewer"):
        st.session_state.viewer_open = False
        rerun_needed = True
else:
    if slide_id and st.button(f"Open Slide Viewer for {slide_id}"):
        st.session_state.viewer_open = True
        rerun_needed = True

if st.session_state.get("hpc_wsi_explore_id") is not None:
    st.divider()
    _hid = int(st.session_state.hpc_wsi_explore_id)
    h1, h2 = st.columns([4, 1])
    with h1:
        st.subheader(f"HPC {_hid} — slides from KB (profile proportion)")
    with h2:
        if st.button("Clear HPC explorer", key="clear_hpc_wsi_explorer"):
            st.session_state.hpc_wsi_explore_id = None
            st.session_state.hpc_wsi_ranking_df = None
            for _k in ("hpc_dd_compare_a", "hpc_dd_compare_b"):
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()
    st.caption(
        "Same KB sources as **handle_slide**: ``hpl_profile_proportion`` (this HPC) + ``hpl_profile_summary``. "
        "Choose two slides from the dropdowns below to compare WSIs side by side."
    )
    rank_result = st.session_state.get("hpc_wsi_ranking_df")

    if rank_result is None:
        st.error("No HPC lookup has been run yet.")

    elif not rank_result["ok"]:
        st.error(f"KB lookup failed: {rank_result['error']}")

    elif rank_result["df"].empty:
        st.info("KB query succeeded, but no rows were found in `hpl_profile_proportion` for this HPC.")

    else:
        rank_df = rank_result["df"]
        _ph = "— Select slide —"
        opts_list = rank_df["slide_id"].astype(str).str.strip().str.upper().tolist()
        _choices = [_ph] + opts_list

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.selectbox("Compare — slide A", _choices, key="hpc_dd_compare_a")
        with c2:
            st.selectbox("Compare — slide B", _choices, key="hpc_dd_compare_b")
        with c3:
            if st.button("Clear A & B", key="hpc_cmp_clear_ab"):
                st.session_state.hpc_dd_compare_a = _ph
                st.session_state.hpc_dd_compare_b = _ph
                st.rerun()

        show_cols = ["slide_id", "proportion", "samples", "kb_summary"]
        show_cols = [c for c in show_cols if c in rank_df.columns]
        st.dataframe(
            rank_df[show_cols].rename(
                columns={
                    "slide_id": "Slide",
                    "proportion": "Proportion (KB)",
                    "samples": "Samples (KB)",
                    "kb_summary": "Slide summary (KB)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        left = st.session_state.get("hpc_dd_compare_a")
        right = st.session_state.get("hpc_dd_compare_b")
        if left == _ph:
            left = None
        if right == _ph:
            right = None

        if left and right and left != right:
            st.divider()
            st.subheader("Side-by-side viewers (same controls as main `show_wsi`)")
            v1, v2 = st.columns(2)
            with v1:
                st.markdown(f"### {left}")
                show_wsi(left, ui_suffix="cmpL")
            with v2:
                st.markdown(f"### {right}")
                show_wsi(right, ui_suffix="cmpR")
        elif left and right and left == right:
            st.info("Slide A and B are the same — pick **two different** slides in the dropdowns.")

if st.session_state.viewer_open and slide_id:
    show_wsi(slide_id)

if rerun_needed:
    st.rerun()
