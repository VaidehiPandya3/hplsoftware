#!/usr/bin/env python3
"""Tests for --chain in submit_anorak_nf.

The cohort takes weeks and the partition's MaxTime is two days, so the head
job has to be replaced about a dozen times. A chain does that with Slurm
dependencies instead of a person: N head jobs submitted at once, each waiting
on the one before it.

What these tests are for is the ways that can be wrong while looking right —
a chain whose successors all depend on the FIRST job (so they start together
the moment it ends), one whose successors do not resume (so each restarts the
cohort into the same directory), a partial chain reported as a whole one, and
a chain that cannot stop: every standby waits on afternotok, so without the
supervisor's stop marker one deterministic failure starts them all in turn.

Runs under pytest and standalone.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import submit_anorak_nf as sub  # noqa: E402

# Module attributes a test swaps out, and what to put back. These tests used to
# assign sub.LOG_DIR and sub._run_sbatch_with_retry and leave them, so every
# later test in the same process — any module — submitted through a fake and
# logged into a deleted tmp dir. pytest calls teardown_function after each test;
# the standalone runners call it too.
_saved: list[tuple[object, str, object]] = []


def _patch(owner, name, value):
    _saved.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


def teardown_function(function=None):
    while _saved:
        owner, name, value = _saved.pop()
        setattr(owner, name, value)


_REAL = {"LOG_DIR": sub.LOG_DIR, "_run_sbatch_with_retry": sub._run_sbatch_with_retry}


class FakeSbatch:
    """Stands in for _run_sbatch_with_retry, handing out job ids in order."""

    def __init__(self, first_id=9000, refuse_on=None):
        self.calls = []
        self.next_id = first_id
        self.refuse_on = refuse_on          # 1-based call number to refuse

    def __call__(self, command, *args, **kwargs):
        self.calls.append(list(command))
        if self.refuse_on == len(self.calls):
            raise subprocess.CalledProcessError(
                1, command, output="", stderr="sbatch: error: QOSMaxSubmitJobPerUserLimit")
        self.next_id += 1
        return SimpleNamespace(stdout=f"Submitted batch job {self.next_id}\n",
                               stderr="", returncode=0)

    def dependency_of(self, index):
        """The job id the `index`-th call (0-based) was told to wait for."""
        for argument in self.calls[index]:
            if argument.startswith("--dependency="):
                return argument.split(":", 1)[1]
        return None


def _cohort(tmp_path, monkeypatch=None):
    """Enough of a real cohort for submit_anorak_job to accept it."""
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "S1.ndpi").write_bytes(b"")
    slides = tmp_path / "slides.csv"
    slides.write_text("slide_id,samples,is_tumour\nS1,T1,True\n", encoding="utf-8")
    pipeline = tmp_path / "anorak-nf"; pipeline.mkdir()
    (pipeline / "main.nf").write_text("// stub\n", encoding="utf-8")
    (pipeline / "tools").mkdir()
    (pipeline / "tools" / "nf_supervise.sh").write_text("# stub\n", encoding="utf-8")
    anorak = tmp_path / "AIgrading"; (anorak / "models").mkdir(parents=True)
    _patch(sub, "LOG_DIR", tmp_path / "logs")     # never write into the repo
    return dict(slides_csv=slides, raw_dir=raw, out_dir=tmp_path / "out",
                pipeline_dir=pipeline, anorak_dir=anorak)


def _submit(tmp_path, fake, **kwargs):
    _patch(sub, "_run_sbatch_with_retry", fake)
    return sub.submit_anorak_job(**_cohort(tmp_path), **kwargs)


# --- the guards ----------------------------------------------------------

def test_a_chain_without_resume_is_refused(tmp_path):
    """Each successor would start the cohort again in the same directory."""
    try:
        _submit(tmp_path, FakeSbatch(), chain=4, resume=False)
    except ValueError as refusal:
        assert "resume" in str(refusal)
    else:
        raise AssertionError("a chain without resume was accepted")


def test_a_chain_of_zero_is_refused(tmp_path):
    try:
        _submit(tmp_path, FakeSbatch(), chain=0)
    except ValueError as refusal:
        assert "at least 1" in str(refusal)
    else:
        raise AssertionError("--chain 0 was accepted")


# --- the default is unchanged --------------------------------------------

def test_no_chain_submits_one_job_with_no_dependency(tmp_path):
    fake = FakeSbatch()
    info = _submit(tmp_path, fake)
    assert len(fake.calls) == 1
    assert not any(a.startswith("--dependency") for a in fake.calls[0])
    assert info["chain_job_ids"] == []


# --- the chain itself ----------------------------------------------------

def test_each_successor_waits_on_the_one_before_it(tmp_path):
    """Not on the first job — that would start them all at once."""
    fake = FakeSbatch(first_id=9000)
    info = _submit(tmp_path, fake, chain=4)

    assert len(fake.calls) == 4
    ids = [info["anorak_job_id"], *info["chain_job_ids"]]
    assert ids == ["9001", "9002", "9003", "9004"]
    for position in range(1, 4):
        assert fake.dependency_of(position) == ids[position - 1], (
            f"job {position + 1} waits on {fake.dependency_of(position)}, "
            f"not on its predecessor {ids[position - 1]}"
        )


def test_successors_run_only_if_the_previous_did_not_finish(tmp_path):
    fake = FakeSbatch()
    _submit(tmp_path, fake, chain=3)
    for call in fake.calls[1:]:
        assert any(a.startswith("--dependency=afternotok:") for a in call)
        # Without this, a chain whose first job succeeds leaves the rest
        # pending forever on a dependency that can never be satisfied.
        assert "--kill-on-invalid-dep=yes" in call


def test_every_job_in_the_chain_resumes_into_the_same_work_directory(tmp_path):
    fake = FakeSbatch()
    _submit(tmp_path, fake, chain=3)
    wraps = [call[call.index("--wrap") + 1] for call in fake.calls]
    assert len(set(wraps)) == 1, "the chain's jobs do not run the same command"
    assert "-resume" in wraps[0]


# --- a chain that could not be completed ---------------------------------

def test_a_refused_successor_leaves_a_shorter_chain_and_says_so(tmp_path):
    """The jobs already queued still run; the caller is told it is short."""
    fake = FakeSbatch(refuse_on=3)
    info = _submit(tmp_path, fake, chain=5)

    assert info["anorak_job_id"] == "9001"
    assert info["chain_job_ids"] == ["9002"]
    assert "2 of 5" in info["chain_error"]
    assert "QOSMaxSubmitJobPerUserLimit" in info["chain_error"]


def test_no_successors_when_the_first_job_reported_no_id(tmp_path):
    """There is nothing for them to depend on, so submitting them anyway
    would produce a chain of jobs that start immediately and concurrently."""
    class Silent(FakeSbatch):
        def __call__(self, command, *args, **kwargs):
            self.calls.append(list(command))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

    fake = Silent()
    info = _submit(tmp_path, fake, chain=4)
    assert len(fake.calls) == 1
    assert info["chain_job_ids"] == []
    assert "nothing for them to depend on" in info["chain_error"]


# --- stopping the chain ---------------------------------------------------

def _wrap_argv(call):
    import shlex
    return shlex.split(call[call.index("--wrap") + 1].split("exec ", 1)[1])


def test_every_head_job_is_given_the_runs_stop_marker(tmp_path):
    """The marker is how a successor learns not to run; a chain whose jobs
    were not told where it is cascades exactly as before."""
    fake = FakeSbatch()
    cohort = _cohort(tmp_path)
    _patch(sub, "_run_sbatch_with_retry", fake)
    sub.submit_anorak_job(**cohort, chain=3)
    for call in fake.calls:
        argv = _wrap_argv(call)
        before = argv[:argv.index("--")]
        marker = before[before.index("--stop-marker") + 1]
        assert marker == str(cohort["out_dir"].resolve() / sub.SUPERVISOR_STOP_MARKER)


def test_the_supervisors_shutdown_fits_inside_slurms_warning(tmp_path):
    """--signal=B:TERM@N in the sbatch line and the lead the supervisor budgets
    its shutdown against are one number in two places. They are compared here,
    from the commands actually submitted, because a supervisor still stopping
    Nextflow when the limit arrives is SIGKILLed with the cleanup undone."""
    fake = FakeSbatch()
    _submit(tmp_path, fake)
    call = fake.calls[0]
    lead = int(next(a for a in call if a.startswith("--signal=B:TERM@")).rsplit("@", 1)[1])
    argv = _wrap_argv(call)
    assert int(argv[argv.index("--signal-lead-seconds") + 1]) == lead
    assert lead == sub.HEAD_SIGNAL_SECONDS


def test_a_fresh_submission_clears_the_stop_marker_and_says_what_it_said(tmp_path):
    cohort = _cohort(tmp_path)
    marker = cohort["out_dir"] / sub.SUPERVISOR_STOP_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("nextflow exited with status 1 on its own\n", encoding="utf-8")
    _patch(sub, "_run_sbatch_with_retry", FakeSbatch())
    info = sub.submit_anorak_job(**cohort)
    assert not marker.exists(), "the new head job would find it and not run"
    assert "status 1" in info["cleared_stop_marker"]


def test_a_dry_run_leaves_the_stop_marker_alone(tmp_path):
    cohort = _cohort(tmp_path)
    marker = cohort["out_dir"] / sub.SUPERVISOR_STOP_MARKER
    marker.parent.mkdir(parents=True)
    marker.write_text("stopped\n", encoding="utf-8")
    _patch(sub, "_run_sbatch_with_retry", FakeSbatch())
    sub.submit_anorak_job(**cohort, dry_run=True)
    assert marker.exists()


# --- options the submitter owns ------------------------------------------

def test_extra_args_may_not_repeat_an_option_the_submitter_sets(tmp_path):
    """Nextflow refuses a repeated launcher option at launch ("Can only specify
    option -profile once", checked against 26.04.6), which is a head job dying
    in a second; a repeated --outdir is worse, since the last one wins."""
    for extra in (["-profile", "test"], ["-w", "/x"], ["-work-dir", "/x"],
                  ["-resume"], ["-ansi-log", "true"], ["-log", "/x.log"],
                  ["--outdir", "/elsewhere"], ["--outdir=/elsewhere"],
                  ["--sample_column", "patient"]):
        try:
            sub.build_nextflow_command(
                pipeline_dir=tmp_path, slides_csv=tmp_path / "s.csv",
                raw_dir=tmp_path, anorak_dir=tmp_path, out_dir=tmp_path,
                work_dir=tmp_path / "work", extra_args=extra)
        except ValueError as refusal:
            assert extra[0] in str(refusal)
        else:
            raise AssertionError(f"extra_args {extra} accepted")
    # And what does not clash still passes through.
    command = sub.build_nextflow_command(
        pipeline_dir=tmp_path, slides_csv=tmp_path / "s.csv", raw_dir=tmp_path,
        anorak_dir=tmp_path, out_dir=tmp_path, work_dir=tmp_path / "work",
        extra_args=["-with-trace", "--tile_size", "512"])
    assert command[-3:] == ["-with-trace", "--tile_size", "512"]


# --- hygiene --------------------------------------------------------------

def test_zz_no_test_left_the_module_patched():
    """Named to run last (definition order under pytest, sorted standalone)."""
    assert sub.LOG_DIR == _REAL["LOG_DIR"], sub.LOG_DIR
    assert sub._run_sbatch_with_retry is _REAL["_run_sbatch_with_retry"]


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_chain_test_"))
        try:
            fn(tmp_path) if fn.__code__.co_argcount else fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            teardown_function(fn)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
