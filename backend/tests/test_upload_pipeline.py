"""A single uploaded slide is a one-slide dataset run, all the way to the KB.

Uploading a slide used to mask it, tile it, package a .h5 and stop. Everything
after that — feature extraction, cluster classification, registration, the
Knowledge Bank load — is keyed on a slurm_dataset_runs row, and an upload had
none, so an uploaded slide could be looked at in the viewer and never carried a
single HPC label. It now gets a run, and Stages 1 and 2 record themselves
against it as they finish in-process.

Three things that go wrong quietly, which is why they are tested here rather
than left to the first upload someone tries:

  * the sentinel job id standing in for a stage that ran in this process. sacct
    rejects a whole call for one id it does not recognise, so a sentinel
    reaching it would have reported "can't reach Slurm" for every real run in
    the same listing — not just for the upload;
  * the cohort each upload is registered under. register_dataset.commit()
    scopes --replace to a dataset_id and DELETEs what it finds there, so one
    shared "UPLOADED" cohort would mean registering the second uploaded slide
    either refused outright or deleted the first one's rows;
  * the name the raw file is saved under. Registration finds each slide's file
    via slide_id_from_raw_path(), and a name it cannot recover the slide_id
    from does not fail — it registers a cohort with no wsi_registry row, and
    the viewer 404s on a slide sitting right there on disk.
"""

import inspect
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
APP = BACKEND.parent / "app" / "app_v28.py"

# Standalone mode is how this suite runs on the cluster, where nothing else has
# put backend/ on the path.
sys.path.insert(0, str(BACKEND))

import tile_server_v2_ as srv  # noqa: E402
from slide_naming import slide_id_from_raw_path  # noqa: E402
from tile_mask import run_tissue_detection  # noqa: E402
from auto_tile_from_mask import tile_slide_from_mask  # noqa: E402

_SERVER_SOURCE = (BACKEND / "tile_server_v2_.py").read_text()


class _Patched:
    """Set module attributes for the duration of a block and put them back."""

    def __init__(self, module, **attrs):
        self.module, self.attrs, self.original = module, attrs, {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.original[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self.module

    def __exit__(self, *exc):
        for name, value in self.original.items():
            setattr(self.module, name, value)
        return False


def _slurm_must_not_be_called(*args, **kwargs):
    raise AssertionError("Slurm was queried about a stage that never went to Slurm")


# --- the sentinel for a stage that ran in this process ---------------------

def test_an_in_process_stage_reads_as_completed(_tmp=None):
    """The gate every later stage gates on. Without it an uploaded slide's
    packaging has no Slurm state, and Stage 3 stays permanently blocked behind
    a .h5 that is sitting on disk."""
    with _Patched(srv, _run_slurm=_slurm_must_not_be_called,
                  _slurm_jobs_live_states=_slurm_must_not_be_called):
        assert srv._get_slurm_job_state(srv._local_job_id("packaging")) == "COMPLETED"


def test_a_real_job_id_is_still_asked_about(_tmp=None):
    """The companion: proves the shortcut above is narrow. A real id must still
    be resolved against Slurm, and an unreachable Slurm must still come back
    unknown rather than as a cheerful COMPLETED."""
    asked = []

    def _record(cmd, timeout):
        asked.append(cmd)
        return None

    with _Patched(srv, _run_slurm=_record, _slurm_jobs_live_states=lambda ids: []):
        assert srv._get_slurm_job_state("9876543") is None
    assert asked, "a real job id never reached Slurm"


def test_a_sentinel_is_kept_out_of_the_sacct_call(_tmp=None):
    """The blast radius this prevents: sacct rejects the whole call for one
    unknown id, so a single upload run would have blanked the states of every
    real run listed beside it."""
    asked = []

    def _record(cmd, timeout):
        asked.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="9876543|COMPLETED\n", stderr="")

    local = srv._local_job_id("tiling")
    with _Patched(srv, _run_slurm=_record):
        states = srv._slurm_states_by_job([local, "9876543"])

    assert states[local] == {"COMPLETED"}
    assert states["9876543"] == {"COMPLETED"}
    assert asked and local not in asked[0], f"the sentinel reached sacct: {asked}"
    assert "9876543" in asked[0], "the real id was dropped along with the sentinel"


def test_one_sentinel_counts_as_one_finished_task(_tmp=None):
    """tiling_complete is "no in-flight state present" over these counts, so an
    upload whose Stage 1 never went through sbatch has to answer here."""
    with _Patched(srv, _run_slurm=_slurm_must_not_be_called):
        counts = srv._get_slurm_array_state_counts([srv._local_job_id("tiling")])
    assert counts == {"COMPLETED": 1}
    assert not (set(counts) & srv.IN_FLIGHT_SLURM_STATES)


def test_a_real_job_with_no_accounting_row_is_not_answered_by_a_sentinel(_tmp=None):
    """Proves the sentinel's count is added to sacct's answer rather than
    standing in for it: a real job Slurm cannot account for must still fall
    through to the live queue, even when a sentinel sits beside it."""
    consulted = []

    with _Patched(
        srv,
        _run_slurm=lambda cmd, timeout: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
        _slurm_jobs_live_states=lambda ids: consulted.append(ids) or ["RUNNING"],
    ):
        counts = srv._get_slurm_array_state_counts([srv._local_job_id("tiling"), "9876543"])

    assert consulted == [["9876543"]], consulted
    assert counts == {"COMPLETED": 1, "RUNNING": 1}, counts


def test_a_sentinel_does_not_blank_the_dataset_listing(_tmp=None):
    """The listing asks squeue about every run's jobs in chunks, and squeue
    fails the whole chunk for one id the controller never had. An upload run
    sharing a chunk with fifty real ones must not be what makes them all read
    as "no record"."""
    asked = []

    def _fake_squeue(cmd, **kwargs):
        asked.append(" ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="9876543|RUNNING\n", stderr="")

    local = srv._local_job_id("tiling")
    with _Patched(subprocess, run=_fake_squeue):
        states = srv._squeue_states_by_job([local, "9876543"])

    assert states[local] == {"COMPLETED"}
    assert states["9876543"] == {"RUNNING"}
    assert asked and local not in asked[0], f"the sentinel reached squeue: {asked}"


def test_a_sentinel_is_never_handed_to_scancel(_tmp=None):
    """scancel rejects the call for an id it does not know, which would take
    the run's real jobs down with it into scancel_error."""
    real, local = srv._split_local_job_ids(["123", srv._local_job_id("tiling"), "456"])
    assert real == ["123", "456"]
    assert local == [srv._local_job_id("tiling")]


# --- one cohort per uploaded slide ----------------------------------------

def test_each_uploaded_slide_gets_its_own_cohort(_tmp=None):
    """Two uploads must not share a dataset_id. register_dataset.commit()
    refuses to re-register an occupied cohort without --replace, and --replace
    DELETEs every row under that dataset_id — so a shared cohort would mean the
    second uploaded slide could only be registered by deleting the first."""
    assert srv.upload_dataset_name("slide-a") != srv.upload_dataset_name("slide-b")
    assert srv.upload_dataset_name("slide-a") == "UPLOADED_SLIDE-A"


def test_the_cohort_name_is_also_the_tile_folder(_tmp=None):
    """Registration reads Stage 1's _tile_metadata.csv out of
    tile_dir/<the run's dataset_name>/<slide_id>/. The upload pipeline writes
    its tiles under the same name, which is what makes registering an upload
    need no special case at all."""
    source = _SERVER_SOURCE[_SERVER_SOURCE.index("def _run_postupload_pipeline"):]
    source = source[:source.index("\ndef ", 1)]
    assert "dataset_name = upload_dataset_name(slide_id)" in source
    assert "PROCESSED_TILES_DIR / dataset_name" in source
    assert "tile_dataset_name=dataset_name" in source


def test_an_upload_from_before_this_is_still_recognised_as_an_upload(_tmp=None):
    """Rows written before per-slide cohorts carry the bare "UPLOADED". Reading
    one as somebody else's dataset would turn re-uploading such a slide into a
    hard 409 about a cohort collision that does not exist."""
    assert srv._is_upload_dataset_id("UPLOADED")
    assert srv._is_upload_dataset_id(srv.upload_dataset_name("TCGA-55-7574"))


def test_a_curated_cohort_is_not_mistaken_for_an_upload(_tmp=None):
    """The half that guards a real slide: an upload must never be allowed to
    repoint a curated cohort's registry row at whatever was just uploaded."""
    for dataset_id in ("TCGA_LUAD_5x", "Radiogenomics", "", None):
        assert not srv._is_upload_dataset_id(dataset_id), dataset_id


def test_the_upload_registers_the_slide_under_that_cohort(_tmp=None):
    """The viewer needs a wsi_registry row immediately, and Stage 5 refuses to
    register a slide that already belongs to a *different* dataset_id — so the
    row written at upload time has to carry the cohort Stage 5 will use."""
    call = _SERVER_SOURCE[_SERVER_SOURCE.index("        _register_uploaded_slide("):]
    call = call[:call.index("\n\n")]
    assert "upload_dataset_name(safe_user_slide_id)" in call, call


# --- the name the raw file is saved under ---------------------------------

def test_the_saved_name_gives_the_slide_id_back(_tmp=None):
    """Registration matches raw files by slide_id_from_raw_path(). A name it
    cannot parse does not fail: the cohort registers with no wsi_registry row
    and every tile of that slide 404s in the viewer."""
    uuid4 = "0f4f2f8c-1a2b-4c3d-8e9f-abcdef012345"
    saved = Path(f"/scratch/uploads/raw/{uuid4}/TCGA-55-7574_{uuid4}_original_scan.svs")
    assert slide_id_from_raw_path(saved) == "TCGA-55-7574"


def test_the_name_without_the_uuid_does_not_parse(_tmp=None):
    """Why the line above is load-bearing rather than decorative — this is the
    shape uploads were saved under, and the whole stem comes back as the slide
    id."""
    saved = Path("/scratch/uploads/raw/0f4f2f8c/TCGA-55-7574_original_scan.svs")
    assert slide_id_from_raw_path(saved) != "TCGA-55-7574"


def test_the_endpoint_saves_under_the_parseable_name(_tmp=None):
    """Pins the two together: the test above proves the convention, this proves
    /upload-slide still writes it."""
    assert 'save_dir / f"{safe_user_slide_id}_{internal_id}_{safe_filename}"' in _SERVER_SOURCE


# --- what the run records --------------------------------------------------

def test_the_recorded_tiling_params_are_the_ones_the_upload_runs_with(_tmp=None):
    """Registration writes these into dataset_config, so a wrong pair claims a
    cohort was tessellated at a resolution it was not. Read off the two
    functions that actually own them rather than copied."""
    params = srv._upload_tiling_params()
    assert set(params) == set(srv._TILING_PARAM_NAMES), params
    assert params["min_tissue"] == srv.MIN_TISSUE_PERCENT
    tiler = inspect.signature(tile_slide_from_mask).parameters
    masker = inspect.signature(run_tissue_detection).parameters
    assert params["target_mpp"] == tiler["target_mpp"].default
    assert params["target_tile_px"] == tiler["target_tile_px"].default
    assert params["mask_max_size"] == masker["max_size"].default
    assert params["mask_saturation"] == masker["saturation_threshold"].default


def _run_upload_pipeline(tmp_path, package=None):
    """Drive _run_postupload_pipeline with the three heavy stages stubbed out,
    returning everything it recorded against the run."""
    mask_dir, tile_dir = tmp_path / "masks", tmp_path / "tiles"
    slide_id = "TCGA-55-7574"
    dataset_name = srv.upload_dataset_name(slide_id)
    recorded = []

    def _fake_mask(slide_path, output_dir, slide_id):
        out = Path(output_dir)
        return {"mask_path": str(out / f"{slide_id}_mask.png"),
                "overlay_path": str(out / f"{slide_id}_overlay.png")}

    def _fake_tile(slide_path, mask_path, output_dir, min_tissue_percent, slide_id):
        return {"output_dir": str(Path(output_dir) / slide_id), "saved_tiles": 512}

    def _fake_package(**kwargs):
        if package is not None:
            return package(**kwargs)
        return {"output_h5_path": str(tmp_path / dataset_name /
                                      f"hdf5_{dataset_name}_he_train.h5")}

    with _Patched(
        srv,
        TISSUE_MASK_DIR=mask_dir,
        PROCESSED_TILES_DIR=tile_dir,
        HPL_DATASETS_ROOT=tmp_path,
        run_tissue_detection=_fake_mask,
        tile_slide_from_mask=_fake_tile,
        package_slides_to_h5=_fake_package,
        _set_processing_status=lambda *a, **k: None,
        _update_dataset_run_best_effort=lambda sid, **fields: recorded.append(fields),
    ):
        srv._run_postupload_pipeline(slide_id, str(tmp_path / "slide.svs"), "run-1")

    merged = {}
    for fields in recorded:
        merged.update(fields)
    return merged


def test_an_upload_leaves_the_run_ready_for_feature_extraction(_tmp=None):
    """The whole point: Stage 3 gates on h5_job_id plus a recorded .h5 path,
    and an upload's Stages 1 and 2 never go near Slurm to produce either."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
    recorded = _run_upload_pipeline(tmp_path)

    assert recorded["job_id"] == srv._local_job_id("tiling")
    assert recorded["h5_job_id"] == srv._local_job_id("packaging")
    assert recorded["h5_output_path"].endswith("hdf5_UPLOADED_TCGA-55-7574_he_train.h5")
    assert recorded["status"] == "completed"
    assert recorded["total_slides"] == 1


def test_a_failed_packaging_leaves_stage_3_shut(_tmp=None):
    """Proves the recording above is evidence and not decoration. Packaging
    that raised must not leave a sentinel behind: the sentinel reads as
    COMPLETED everywhere, so Stage 3 would queue a GPU job against a .h5 that
    was never written."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))

    def _explode(**kwargs):
        raise RuntimeError("no space left on device")

    recorded = _run_upload_pipeline(tmp_path, package=_explode)

    assert "h5_job_id" not in recorded
    assert "h5_output_path" not in recorded
    # Stage 1 still finished, and saying so is what keeps the retry on
    # packaging rather than sending the user back to re-tile.
    assert recorded["job_id"] == srv._local_job_id("tiling")
    assert "packaging failed" in recorded["error"]


def test_a_slide_with_no_tissue_stops_without_claiming_a_h5(_tmp=None):
    """Tissue below the threshold is an expected outcome for one ad-hoc slide,
    not a failure — but it must not look like a run with something to extract
    features from."""
    tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))

    def _no_tiles(slide_path, mask_path, output_dir, min_tissue_percent, slide_id):
        return {"output_dir": str(Path(output_dir) / slide_id), "saved_tiles": 0}

    recorded = []
    with _Patched(
        srv,
        TISSUE_MASK_DIR=tmp_path / "masks",
        PROCESSED_TILES_DIR=tmp_path / "tiles",
        run_tissue_detection=lambda slide_path, output_dir, slide_id: {
            "mask_path": str(Path(output_dir) / "m.png"),
            "overlay_path": str(Path(output_dir) / "o.png"),
        },
        tile_slide_from_mask=_no_tiles,
        package_slides_to_h5=lambda **k: (_ for _ in ()).throw(
            AssertionError("packaging ran for a slide with no tiles")),
        _set_processing_status=lambda *a, **k: None,
        _update_dataset_run_best_effort=lambda sid, **fields: recorded.append(fields),
    ):
        srv._run_postupload_pipeline("TCGA-55-7574", str(tmp_path / "slide.svs"), "run-1")

    merged = {}
    for fields in recorded:
        merged.update(fields)
    assert "h5_job_id" not in merged
    assert "tissue" in merged["error"]


# --- the UI's route into the run ------------------------------------------

def test_the_processing_status_names_the_run(_tmp=None):
    """The upload response carries submission_id too, but it is gone the moment
    the page reloads — this endpoint is what the UI polls, so the run has to be
    reachable from it or the pipeline view is only available to the tab that
    started the upload."""
    payload = srv.slide_processing_status("TCGA-55-7574")
    assert "submission_id" in payload
    assert payload["dataset_name"] == srv.upload_dataset_name("TCGA-55-7574")
    assert "status" in payload


def test_the_upload_response_carries_the_run(_tmp=None):
    upload = _SERVER_SOURCE[_SERVER_SOURCE.index("async def upload_slide("):]
    upload = upload[:upload.index("\n@app.get")]
    assert '"submission_id": submission_id' in upload
    assert "_start_upload_run(safe_user_slide_id, save_path)" in upload


def _pipeline_steps():
    """app_v28._pipeline_steps, exec'd with stubs — same technique
    test_pipeline_steps.py uses, since importing that module pulls in
    streamlit."""
    source = APP.read_text()
    start = source.index("def _pipeline_steps")
    end = source.index("\ndef ", start)
    namespace = {
        "_SLURM_IN_FLIGHT": srv.IN_FLIGHT_SLURM_STATES,
        "_test_packaging_note": lambda status: "",
        "_human_bytes": lambda n: f"{n} B",
        "re": re,
    }
    exec(compile(source[start:end], "probe", "exec"), namespace)
    return namespace["_pipeline_steps"]


def test_the_stepper_offers_feature_extraction_for_an_uploaded_slide(_tmp=None):
    """End of the chain, in the terms the user actually sees: Stages 1 and 2
    read as done and Stage 3 is the next thing to click."""
    steps = {s["key"]: s for s in _pipeline_steps()({
        "status": "completed",
        "total_slides": 1, "succeeded": 1, "tiling_complete": True,
        "h5_job_id": srv._local_job_id("packaging"),
        "h5_slurm_state": "COMPLETED",
        "h5_ready": True,
    })}
    assert steps["tiling"]["state"] == "done", steps["tiling"]
    assert steps["packaging"]["state"] == "done", steps["packaging"]
    assert steps["extraction"]["state"] == "action", steps["extraction"]


def test_the_upload_panel_renders_that_stepper(_tmp=None):
    """A run nothing renders is a run nobody can advance — the failure this
    whole change is about."""
    app_source = APP.read_text()
    assert "_render_upload_pipeline(processing_slide_id" in app_source
    panel = app_source[app_source.index("def _render_upload_pipeline"):]
    panel = panel[:panel.index("\ndef ", 1)]
    assert "_render_job_progress(" in panel
    assert 'status_payload.get("submission_id")' in panel


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_upload_test_"))
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
