"""Stage 7's server-side state: when a submission is refused, and when a
grading table counts as this run's.

The pipeline itself is tested in test_anorak_chain.py. What is tested here is
the tile server's bookkeeping around it, which failed without any error:

  * the in-flight guard. The UI sent overwrite=True on every click, trusting
    "the server's" in-flight check, and the server ran that check only when
    overwrite was *not* set. So a second head job could be queued over a live
    one — rewriting slide_list.csv under it — and the row then tracked the new
    job while the original ran unobserved. Unreachable Slurm and CONFIGURING
    both opened the same door;
  * a false "complete". The status used the generic _job_output_ready, whose
    Slurm-unreachable shortcut trusts "file present, no .partial". ANORAK
    writes no .partial, so an earlier attempt's table read as done while the
    new run was still going;
  * a Retry of a random subset with the seed left blank drew a different
    sample into the same output directory.

Everything goes through the real endpoint functions, with Slurm, the database
and sbatch stubbed at the module attributes the server calls.
"""

import contextlib
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from fastapi import HTTPException  # noqa: E402

import tile_server_v2_ as srv  # noqa: E402


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


class _FakeSubmit:
    """Stands in for submit_anorak_job, recording whether it was reached —
    which is the point, since the real one rewrites slide_list.csv first."""

    def __init__(self, job_id="5555"):
        self.calls = []
        self.job_id = job_id

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        out_dir = Path(kwargs["out_dir"])
        return {
            "anorak_job_id": self.job_id, "chain_job_ids": [],
            "out_dir": str(out_dir), "slides_csv": str(out_dir / "slide_list.csv"),
            "grades_csv": str(out_dir / "anorak_tumour_grades.csv"),
            "selection": {"scope": kwargs["scope"], "slides": 1,
                          "sample_size": kwargs.get("sample_size"),
                          "seed": kwargs.get("seed")},
        }


def _grades(path: Path, *, age_hours: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("sample,slides,predominant_pattern,iaslc_grade\n"
                    "T1,S1,acinar,2\n")
    # The list the table was graded from, so a validator that checks the one
    # against the other has something consistent to find.
    slide_list = path.parent / "slide_list.csv"
    if not slide_list.exists():
        slide_list.write_text("slide_id,samples\nS1,T1\n")
    if age_hours:
        stamp = time.time() - age_hours * 3600
        os.utime(path, (stamp, stamp))
    return path


def _setup(tmp_path: Path, **row_overrides):
    """A run row whose previous ANORAK attempt lives in tmp_path/out, plus a
    real source slide list and raw directory, so only Slurm and sbatch are
    fake."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    source = tmp_path / "tumour_slides.csv"
    source.write_text("slide_id\n" + "".join(f"S{i}\n" for i in range(20)))
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    row = {
        "submission_id": "sub1", "raw_dir": str(raw), "dataset_name": "cohort",
        "notify_email": None,
        "anorak_job_id": "1111",
        "anorak_submitted_at": datetime.now(timezone.utc) - timedelta(hours=1),
        "anorak_out_dir": str(out),
        "anorak_slide_list": str(out / "slide_list.csv"),
        "anorak_scope": "full", "anorak_sample_size": None, "anorak_seed": None,
        "anorak_slides": 20, "anorak_error": None,
    }
    row.update(row_overrides)
    return row, source, out


def _endpoint(row, states, submit, updates=None):
    """The patches every call to start_anorak_job needs. `states` maps job id
    to what _get_slurm_job_state answers; an id it lacks reads as unreachable."""
    updates = updates if updates is not None else []
    return _Patched(
        srv,
        _slurm_submission_lock=contextlib.nullcontext,
        _get_dataset_run_row=lambda sid: dict(row),
        _get_slurm_job_state=lambda job_id: states.get(job_id),
        submit_anorak_job=submit,
        _update_dataset_run_best_effort=lambda sid, **f: updates.append(f),
        _record_run_job=lambda *a, **k: None,
    )


def _refused(call) -> HTTPException:
    try:
        call()
    except HTTPException as e:
        return e
    raise AssertionError("the submission was accepted")


# --- finding 1: the in-flight guard -----------------------------------------

def test_overwrite_does_not_get_past_a_running_head_job(tmp_path):
    row, source, _ = _setup(tmp_path)
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "RUNNING"}, submit):
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source), overwrite=True)))
    assert err.status_code == 400 and "already running" in err.detail
    assert submit.calls == [], "submit_anorak_job was reached and rewrote slide_list.csv"


def test_a_configuring_head_job_is_in_flight(tmp_path):
    """squeue's first word for a job it is starting. The UI's own set lacked
    it, which is how the form was reachable at all."""
    row, source, _ = _setup(tmp_path)
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "CONFIGURING"}, submit):
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source), overwrite=True)))
    assert err.status_code == 400
    assert submit.calls == []


def test_unreachable_slurm_refuses_rather_than_guesses(tmp_path):
    row, source, _ = _setup(tmp_path)
    submit = _FakeSubmit()
    for overwrite in (False, True):
        with _endpoint(row, {}, submit):  # every id unknown: sacct/squeue down
            err = _refused(lambda: srv.start_anorak_job(
                "sub1", srv.AnorakRequest(slides_csv=str(source), overwrite=overwrite)))
        assert err.status_code == 503, (overwrite, err.detail)
    assert submit.calls == []


def test_a_pending_chain_standby_counts_as_in_flight(tmp_path):
    """The head timed out and its --chain successor is queued behind it: the
    run is not over, and the recorded id list says so."""
    row, source, _ = _setup(tmp_path, anorak_job_id="1111,1112")
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "TIMEOUT", "1112": "PENDING"}, submit):
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source))))
    assert err.status_code == 400
    assert submit.calls == []


def test_overwrite_is_what_replaces_a_finished_table_and_a_failure_retries_without_it(tmp_path):
    row, source, out = _setup(tmp_path)
    _grades(out / "anorak_tumour_grades.csv")
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "COMPLETED"}, submit):
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source))))
        assert "overwrite" in err.detail and submit.calls == []
        srv.start_anorak_job("sub1", srv.AnorakRequest(slides_csv=str(source), overwrite=True))
    assert len(submit.calls) == 1
    with _endpoint(row, {"1111": "FAILED"}, submit):
        srv.start_anorak_job("sub1", srv.AnorakRequest(slides_csv=str(source)))
    assert len(submit.calls) == 2


def test_the_ui_no_longer_hands_the_check_to_the_server_by_overwriting(_tmp=None):
    """Both forms sent overwrite on every click. Only "Run again" may."""
    app = (BACKEND.parent / "app" / "app_v28.py").read_text()
    jsx = (BACKEND.parent / "frontend" / "src" / "components" / "pipeline"
           / "stages" / "AnorakStage.jsx").read_text()
    assert "overwrite=True,  # the in-flight check is the server's" not in app
    assert "overwrite: true," not in jsx
    for source in (app, (BACKEND.parent / "frontend" / "src" / "components"
                         / "pipeline" / "utils.js").read_text()):
        assert '"CONFIGURING"' in source


# --- finding 2: a false "complete" ------------------------------------------

def _status(row):
    """dataset_job_status for a run with no tiling job, so the response is the
    row's own fields plus Stage 7's block."""
    full = {k: None for k in (
        "status", "error", "raw_dir", "job_id", "total_slides", "is_subset",
        "partition", "notify_email", "manifest_path", "tile_dir", "h5_job_id",
        "h5_output_path", "test_h5_output_path", "extraction_job_id",
        "extraction_output_path", "extraction_checkpoint", "assignment_job_id",
        "assignment_output_path", "registration_at", "registration_job_id",
        "kb_load_at", "kb_load_job_id")}
    full["is_subset"] = False
    full.update(row)

    class _Result:
        def mappings(self):
            return self

        def fetchone(self):
            return full

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *a, **k):
            return _Result()

    class _Engine:
        def connect(self):
            return _Conn()

    with _Patched(srv, _get_engine=lambda: _Engine(),
                  _find_job_ids_by_name_prefix=lambda prefix: []):
        return srv.dataset_job_status("sub1")


def test_an_earlier_attempts_table_is_not_complete_while_slurm_is_unreachable(tmp_path):
    row, _, out = _setup(tmp_path)
    _grades(out / "anorak_tumour_grades.csv", age_hours=5)  # before this submission
    with _Patched(srv, _get_slurm_job_state=lambda job_id: None):
        status = _status(row)
    assert status["anorak_ready"] is False
    assert status["anorak_state_unknown"] is True
    assert status["anorak_submit_blocked"]


def test_a_table_older_than_the_submission_is_not_accepted_once_aged_out(tmp_path):
    row, _, out = _setup(tmp_path)
    grades = _grades(out / "anorak_tumour_grades.csv", age_hours=5)
    with _Patched(srv, _get_slurm_job_state=lambda job_id: ""):
        assert _status(row)["anorak_ready"] is False
        # ...and a table written after it is, which is what "" exists to allow.
        os.utime(grades, None)
        assert _status(row)["anorak_ready"] is True


def test_the_server_publishes_its_own_in_flight_verdict(tmp_path):
    row, _, _ = _setup(tmp_path)
    with _Patched(srv, _get_slurm_job_state=lambda job_id: "CONFIGURING"):
        status = _status(row)
    assert status["anorak_in_flight"] is True and status["anorak_submit_blocked"]


# --- finding 3: anorak_error is written -------------------------------------

def test_a_failed_sbatch_is_recorded_on_the_row_and_reported(tmp_path):
    row, source, _ = _setup(tmp_path, anorak_job_id=None)
    updates = []

    def refused(**kwargs):
        raise RuntimeError("sbatch failed (exit 1): invalid partition")

    with _endpoint(row, {}, refused, updates):
        err = _refused(lambda: srv.start_anorak_job(
            "sub1", srv.AnorakRequest(slides_csv=str(source))))
    assert err.status_code == 500
    assert any("invalid partition" in (u.get("anorak_error") or "") for u in updates)

    with _Patched(srv, _get_slurm_job_state=lambda job_id: None):
        status = _status({**row, "anorak_error": updates[-1]["anorak_error"]})
    assert "invalid partition" in status["anorak_error"]


def test_a_successful_submission_clears_the_previous_error(tmp_path):
    row, source, _ = _setup(tmp_path, anorak_job_id=None, anorak_error="old")
    updates = []
    with _endpoint(row, {}, _FakeSubmit(), updates):
        srv.start_anorak_job("sub1", srv.AnorakRequest(slides_csv=str(source)))
    assert updates[-1]["anorak_error"] is None


# --- finding 4: a retry repeats its sample ----------------------------------

def _previous_subset(tmp_path, seed=42, sample_size=5):
    """A run whose last attempt drew `sample_size` of the 20 source slides with
    `seed`, recorded exactly as submit_anorak_job records it."""
    row, source, out = _setup(tmp_path, anorak_scope="subset",
                              anorak_sample_size=sample_size, anorak_seed=seed)
    chosen, _ = srv._anorak_select_slide_rows(
        srv._anorak_read_slide_csv(source), scope="subset", sample_size=sample_size, seed=seed)
    chosen.to_csv(out / "slide_list.csv", index=False)
    return row, source


def test_a_blank_seed_on_retry_reuses_the_recorded_one(tmp_path):
    row, source = _previous_subset(tmp_path)
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "FAILED"}, submit):
        srv.start_anorak_job("sub1", srv.AnorakRequest(
            slides_csv=str(source), scope="subset", sample_size=5, seed=None))
    assert submit.calls[0]["seed"] == 42


def test_a_blank_seed_that_cannot_repeat_the_sample_is_refused(tmp_path):
    row, source = _previous_subset(tmp_path)
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "FAILED"}, submit):
        err = _refused(lambda: srv.start_anorak_job("sub1", srv.AnorakRequest(
            slides_csv=str(source), scope="subset", sample_size=6, seed=None)))
        assert "seed" in err.detail and "42" in err.detail
        # A seed typed on purpose is a deliberate new sample, and goes through.
        srv.start_anorak_job("sub1", srv.AnorakRequest(
            slides_csv=str(source), scope="subset", sample_size=6, seed=7))
    assert [c["seed"] for c in submit.calls] == [7]



# --- chain, the stop marker, and what the form says -------------------------

def test_the_chain_reaches_the_submitter(tmp_path):
    """The UI had no way to ask for a standby head job, so a run launched from
    it that reached its walltime ended with nothing to take over."""
    row, source, _ = _setup(tmp_path)
    submit = _FakeSubmit()
    with _endpoint(row, {"1111": "FAILED"}, submit):
        srv.start_anorak_job("sub1", srv.AnorakRequest(slides_csv=str(source), chain=3))
        srv.start_anorak_job("sub1", srv.AnorakRequest(slides_csv=str(source)))
    assert [c.get("chain") for c in submit.calls] == [3, 1]


def test_why_a_run_stopped_is_in_the_status(tmp_path):
    """nf_supervise.stop holds the reason a run ended for good; the stage used
    to show a bare FAILED."""
    row, _, out = _setup(tmp_path)
    with _Patched(srv, _get_slurm_job_state=lambda job_id: "FAILED"):
        assert _status(row)["anorak_stop_reason"] is None
        (out / srv.SUPERVISOR_STOP_MARKER).write_text(
            "nextflow exited 1 on its own\njob: 1111 on node7\n")
        reason = _status(row)["anorak_stop_reason"]
    assert reason.startswith("nextflow exited 1 on its own")


def test_the_streamlit_form_says_a_sample_is_required(_tmp=None):
    app = (BACKEND.parent / "app" / "app_v28.py").read_text()
    assert "if you want grades aggregated per tumour" not in app
    assert "naming each slide's tumour" in app
    assert "chain=int(chain) if resume else 1" in app
    assert 'status.get("anorak_stop_reason")' in app

# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_server_test_"))
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
