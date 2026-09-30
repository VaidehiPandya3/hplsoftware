#!/usr/bin/env python3
"""Tests for the head job's Nextflow watchdog, anorak-nf/tools/nf_supervise.sh.

On 2026-09-23 Nextflow's task monitor blocked forever on one CephFS read of a
.exitcode file, and the run sat idle for five hours with nothing failing. The
supervisor restarts Nextflow with -resume when that thread goes silent.

These run the real script against a stand-in `nextflow` that can stall, stay
healthy, fail, or ignore TERM. What they guard against is a watchdog that looks
right and is not: one that kills a healthy run (so the "fix" restarts the
cohort every half hour), one that kills a run still resolving its slide list,
one that restarts without -resume (redoing the cohort), one that swallows
Nextflow's exit status or Slurm's TERM, one that cancels another run's jobs,
one whose TERM handling outlives Slurm's warning, and one that lets a --chain
successor start after a failure it can only repeat.

The stand-in is only as useful as it is strict. Job 1250456 died on a double
-resume that an accepting fake had let through, and the watchdog armed on
Nextflow's log rotation, which a fake that appended to its log never did. So it
does what Nextflow 26.04.6 was seen to do: rotates -log to .1 once the JVM is
up, refuses a repeated launcher option, and exits 1 on TERM. And the tests of
anything the submitter sets run the submitter's own command, not one written
here.

Runs under pytest and standalone.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import submit_anorak_nf as sub  # noqa: E402
from test_anorak_chain import (  # noqa: E402,F401
    _REAL, FakeSbatch, _cohort, _patch, teardown_function,
)

SUPERVISOR = HERE.parent.parent / "anorak-nf" / "tools" / "nf_supervise.sh"

# Stands in for `nextflow`. Which behaviour each invocation gets is MODE_<n>
# (1-based), falling back to MODE_DEFAULT. A "beat" is a line in the format the
# real task monitor writes, which is all the supervisor looks at.
FAKE_NEXTFLOW = r"""#!/usr/bin/env bash
n=$(( $(cat "$STATE/n" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STATE/n"
echo "$*" >> "$STATE/calls"
# Real Nextflow refuses any repeated launcher option before doing anything; a
# stand-in that accepted a second -resume is how that reached the cluster.
seen=" "
prev=
for arg in "$@"; do
  case "$arg" in
    -[a-z]*)
      case "$seen" in *" $arg "*)
        echo "Can only specify option ${arg#-} once." >&2; exit 1 ;;
      esac
      seen="$seen$arg " ;;
  esac
  [ "$prev" = "-log" ] && LOG=$arg
  prev=$arg
done
# Nextflow rotates its -log once the JVM is up (nextflow.log -> .1), so the
# first thing an attempt does to its log is make it shorter.
sleep 0.3
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.1"
: > "$LOG"
mode_var="MODE_$n"
mode=${!mode_var:-$MODE_DEFAULT}
beat() { echo "Sep-23 07:10:25.400 [Task monitor] DEBUG n.processor.TaskPollingMonitor - beat" >> "$LOG"; }
submitter() { echo "Sep-23 07:12:29.607 [Task submitter] DEBUG n.processor.TaskPollingMonitor - queue" >> "$LOG"; }
case "$mode" in
  stall)    beat; sleep 0.3; beat; exec sleep 60 ;;
  stall_submitter_alive)
            beat; sleep 0.3; beat
            while :; do submitter; sleep 0.2; done ;;
  ok)       beat; exit 0 ;;
  healthy)  for i in $(seq 1 20); do beat; sleep 0.2; done; exit 0 ;;
  silent)   sleep 4; exit 0 ;;
  fail)     beat; exit 3 ;;
  deaf)     trap '' TERM; beat; sleep 0.3; beat; while :; do sleep 0.2; done ;;
  mark_term)
            trap 'echo got-term > "$STATE/term"; exit 1' TERM
            beat; while :; do sleep 0.1; done ;;
  finish_on_term)
            trap 'exit 0' TERM
            beat; while :; do sleep 0.1; done ;;
esac
"""


class Run:
    """One supervisor invocation in a scratch directory."""

    def __init__(self, tmp_path: Path, *, max_restarts=2, stall=2, real=False,
                 lead=33, **modes):
        self.tmp = tmp_path
        self.state = tmp_path / "state"; self.state.mkdir()
        self.out = tmp_path / "out"; self.out.mkdir()
        self.log = self.out / "nextflow.log"
        self.work = self.out / "work"; self.work.mkdir()
        self.marker = self.out / sub.SUPERVISOR_STOP_MARKER
        self.fake = tmp_path / "nextflow"
        self.fake.write_text(FAKE_NEXTFLOW, encoding="utf-8")
        self.bin = tmp_path / "bin"; self.bin.mkdir()
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("SLURM_")}
        self.env.update({"STATE": str(self.state), "LOG": str(self.log),
                         "MODE_DEFAULT": "ok",
                         "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}"})
        self.env.update({key: value for key, value in modes.items()})
        timings = ["--stall-seconds", str(stall), "--poll-seconds", "0.2",
                   "--grace-seconds", "2", "--max-restarts", str(max_restarts),
                   "--signal-lead-seconds", str(lead)]
        if not real:
            self.command = ["bash", str(SUPERVISOR), "--log", str(self.log),
                            "--work-dir", str(self.work), *timings,
                            "--", "bash", str(self.fake), "run", "pipeline"]
            return
        # The head job's own command, from the submitter's own builders, with
        # only the timings shortened (a later option wins in the supervisor)
        # and `nextflow` pointed at the stand-in.
        nextflow = sub.build_nextflow_command(
            pipeline_dir=tmp_path, slides_csv=self.out / "slide_list.csv",
            raw_dir=tmp_path, anorak_dir=tmp_path, out_dir=self.out,
            work_dir=self.work)
        command = sub.build_supervised_command(
            supervisor=SUPERVISOR, out_dir=self.out, work_dir=self.work,
            nextflow_command=nextflow)
        split = command.index("--")
        assert command[split + 1] == "nextflow"
        self.command = [*command[:split], *timings, "--",
                        "bash", str(self.fake), *command[split + 2:]]

    def tool(self, name, script):
        path = self.bin / name
        path.write_text("#!/usr/bin/env bash\n" + script, encoding="utf-8")
        path.chmod(0o755)

    def start(self):
        return subprocess.Popen(self.command, env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def run(self, timeout=60):
        proc = self.start()
        out, err = proc.communicate(timeout=timeout)
        self.stderr = err
        return proc.returncode

    def terminate(self, *, timeout=60):
        """Start, TERM once the stand-in is up, and return (status, seconds
        from the TERM to the supervisor's exit). Give these runs a long
        stall limit: on a loaded machine the watchdog can otherwise fire
        first, and the test then measures a restart instead of the TERM."""
        proc = self.start()
        assert _wait_for(lambda: "[Task monitor]" in _read(self.log)), \
            "fake nextflow never started"
        sent = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        try:
            _, self.stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, self.stderr = proc.communicate()
            raise AssertionError(f"supervisor still running {timeout}s after TERM")
        return proc.returncode, time.monotonic() - sent

    @property
    def calls(self):
        path = self.state / "calls"
        return path.read_text().splitlines() if path.exists() else []


def _read(path):
    try:
        return path.read_text()
    except OSError:
        return ""


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# --- the stall it exists for ---------------------------------------------

def test_a_stalled_monitor_is_restarted_with_resume(tmp_path):
    run = Run(tmp_path, MODE_1="stall", MODE_2="ok")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 2
    assert "-resume" not in run.calls[0].split()
    assert run.calls[1].split()[-1] == "-resume", run.calls[1]
    assert "stuck" in run.stderr


def test_a_live_submitter_does_not_hide_a_dead_monitor(tmp_path):
    """The 2026-09-23 log kept growing the whole time, from the submitter's
    five-minute dumps. A watchdog on log size or mtime would have seen a live
    run; only the monitor thread's own lines tell the two apart."""
    run = Run(tmp_path, MODE_1="stall_submitter_alive", MODE_2="ok")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 2


def test_resume_is_not_added_twice(tmp_path):
    run = Run(tmp_path, MODE_1="stall", MODE_2="stall", MODE_3="ok")
    assert run.run() == 0, run.stderr
    assert [c.split().count("-resume") for c in run.calls] == [0, 1, 1]


def test_a_command_that_already_resumes_is_restarted_as_is(tmp_path):
    """Run 1250456: the submitter passes -resume by default, the supervisor
    appended a second, and Nextflow refused it — the restart meant to rescue
    a stalled run ended the job in one second."""
    run = Run(tmp_path, MODE_1="stall", MODE_2="ok")
    run.command.append("-resume")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 2
    assert [c.split().count("-resume") for c in run.calls] == [1, 1]


def test_the_submitters_own_command_survives_a_restart(tmp_path):
    """The command the head job really runs, not a hand-written one: every
    other test here built its own, which is why none of them carried -resume."""
    run = Run(tmp_path, real=True, MODE_1="stall", MODE_2="ok")
    assert run.run() == 0, run.stderr
    assert [c.split().count("-resume") for c in run.calls] == [1, 1]


def test_a_nextflow_that_ignores_term_is_killed(tmp_path):
    run = Run(tmp_path, MODE_1="deaf", MODE_2="ok")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 2
    assert "sending KILL" in run.stderr


def test_restarts_run_out_and_the_job_exits_non_zero(tmp_path):
    """Non-zero so a --chain successor takes over, rather than the head job
    restarting forever against a filesystem that keeps failing."""
    run = Run(tmp_path, max_restarts=1, MODE_DEFAULT="stall")
    assert run.run() == 75, run.stderr
    assert len(run.calls) == 2
    assert "giving up" in run.stderr


# --- what it must not do -------------------------------------------------

def test_a_healthy_run_longer_than_the_stall_limit_is_left_alone(tmp_path):
    """Beats for 4 s against a 2 s limit: a watchdog timing the whole run
    rather than the gaps between beats would kill this."""
    run = Run(tmp_path, MODE_1="healthy")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 1


def test_silence_before_the_first_beat_is_not_a_stall(tmp_path):
    """Before the monitor starts, Nextflow is resolving the slide list, which
    logs nothing from that thread and can take minutes on a large raw_dir."""
    run = Run(tmp_path, MODE_1="silent")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 1


def test_a_rotated_log_does_not_arm_the_watchdog(tmp_path):
    """Every restart and every chain successor starts with the previous
    attempt's log, full of beats, and Nextflow rotates it away once the JVM is
    up. The count falls from 50 to 0; a watchdog arming on "changed" armed on
    that, and killed a run still walking raw_dir as if it had stalled."""
    run = Run(tmp_path, MODE_1="silent")
    run.log.write_text("x [Task monitor] old\n" * 50, encoding="utf-8")
    assert run.run() == 0, run.stderr
    assert len(run.calls) == 1, run.stderr


def test_nextflows_exit_status_is_passed_through(tmp_path):
    run = Run(tmp_path, MODE_1="fail")
    assert run.run() == 3, run.stderr
    assert len(run.calls) == 1, "a failed run was restarted as if it had stalled"


def test_slurms_term_reaches_nextflow(tmp_path):
    """sbatch --signal=B:TERM is what turns the walltime into a clean stop, and
    it reaches the supervisor, not Nextflow. Swallowed, Nextflow is SIGKILLed
    at the limit with its jobs still running."""
    run = Run(tmp_path, stall=600, MODE_1="mark_term")
    status, _ = run.terminate()
    assert status == 143, run.stderr
    assert (run.state / "term").read_text().strip() == "got-term"


def test_term_stops_a_wedged_nextflow_inside_slurms_warning(tmp_path):
    """The supervisor used to `wait` on Nextflow after forwarding TERM, with no
    limit — so a Nextflow stuck in the kernel, the state this script exists
    for, held the job until Slurm's KILL at the limit and the leftover jobs
    were never cancelled. The squeue here never answers either, and the whole
    shutdown must still end inside the --signal lead."""
    run = Run(tmp_path, stall=600, MODE_1="deaf", lead=33, SLURM_JOB_ID="42")
    run.tool("squeue", "sleep 120\n")
    run.tool("scancel", "exit 0\n")
    status, took = run.terminate()
    assert took < 33, f"{took:.1f}s after TERM, beyond the 33s lead: {run.stderr}"
    assert "sending KILL" in run.stderr
    assert "squeue failed" in run.stderr
    assert status == 143


def test_term_cancels_this_runs_jobs(tmp_path):
    run = Run(tmp_path, stall=600, MODE_1="mark_term")
    run.tool("squeue", f"echo '101 {run.work}/ab/cdef'\n"
                       f"echo '102 /somewhere/else/work/ef/0123'\n")
    run.tool("scancel", f"echo \"$*\" >> '{run.state}/scancel'\n")
    run.terminate()
    assert _read(run.state / "scancel").split() == ["101"], run.stderr


def test_a_run_that_finished_as_term_arrived_is_not_failed(tmp_path):
    """143 would read as FAILED and start a --chain successor on a finished
    cohort."""
    run = Run(tmp_path, stall=600, MODE_1="finish_on_term")
    status, _ = run.terminate()
    assert status == 0, run.stderr


def test_a_lead_too_short_to_stop_nextflow_in_is_refused(tmp_path):
    run = Run(tmp_path, lead=20)
    assert run.run() == 64
    assert run.calls == []


# --- when the chain must stop --------------------------------------------

def test_a_scancel_writes_the_stop_marker(tmp_path):
    """A manual scancel is a TERM an hour before the limit. afternotok starts
    the next head job for it, and without the marker that job resumes the
    cohort someone had just stopped."""
    end = int(time.time()) + 3600
    run = Run(tmp_path, stall=600, real=True, MODE_1="mark_term", SLURM_JOB_END_TIME=str(end))
    status, _ = run.terminate()
    assert status == 143, run.stderr
    assert "scancel" in _read(run.marker), run.stderr


def test_the_time_limit_warning_leaves_the_chain_running(tmp_path):
    end = int(time.time()) + 100
    run = Run(tmp_path, stall=600, real=True, MODE_1="mark_term", SLURM_JOB_END_TIME=str(end))
    status, _ = run.terminate()
    assert status == 143, run.stderr
    assert not run.marker.exists(), _read(run.marker)


def test_the_walltime_is_read_live_from_squeue(tmp_path):
    """squeue first: `scontrol update TimeLimit` does not reach the job's
    environment, so SLURM_JOB_END_TIME can be stale. Here it says an hour, and
    squeue says 1:40 — the limit."""
    end = int(time.time()) + 3600
    for left, marked in (("1:40", False), ("1-00:00:00", True)):
        sub_tmp = tmp_path / left.replace(":", "_")
        sub_tmp.mkdir()
        run = Run(sub_tmp, stall=600, real=True, MODE_1="mark_term", SLURM_JOB_ID="42",
                  SLURM_JOB_END_TIME=str(end))
        run.tool("squeue", f'case "$*" in *%L*) echo "{left}" ;; esac\n')
        run.terminate()
        assert run.marker.exists() is marked, (left, run.stderr)


def test_a_term_whose_timing_cannot_be_read_is_final(tmp_path):
    """Stopped loudly costs a resubmission; resumed unasked costs whatever the
    successor then does."""
    run = Run(tmp_path, stall=600, real=True, MODE_1="mark_term")
    run.terminate()
    assert "unknown" in _read(run.marker), run.stderr


def test_a_pipeline_failure_stops_the_chain(tmp_path):
    """A failure Nextflow reports itself — a config error, a slide that fails
    every time, "Unable to acquire lock" — is one every successor repeats, each
    in seconds, each starting the next."""
    run = Run(tmp_path, real=True, MODE_1="fail")
    assert run.run() == 3, run.stderr
    assert "status 3" in _read(run.marker)

    successor = run.run()
    assert successor == 0, "a successor that fails wakes the next one"
    assert len(run.calls) == 1, "the successor ran nextflow despite the marker"
    assert "stop marker" in run.stderr


def test_running_out_of_restarts_leaves_the_chain_running(tmp_path):
    """Repeated stalls are the filesystem, and a fresh head job on another node
    may well not see them."""
    run = Run(tmp_path, real=True, max_restarts=1, MODE_DEFAULT="stall")
    assert run.run() == 75, run.stderr
    assert not run.marker.exists(), _read(run.marker)


def test_a_squeue_failure_is_reported(tmp_path):
    """It went to /dev/null, and a controller that timed out looked exactly
    like a run with nothing left in the queue."""
    run = Run(tmp_path, MODE_1="stall", MODE_2="ok")
    run.tool("squeue", "echo 'slurm_load_jobs error: Socket timed out' >&2; exit 1\n")
    assert run.run() == 0, run.stderr
    assert "Socket timed out" in run.stderr


# --- cleaning up after a stall -------------------------------------------

def test_only_this_runs_jobs_are_cancelled(tmp_path):
    run = Run(tmp_path, MODE_1="stall", MODE_2="ok")
    other = tmp_path / "work2"          # shares a string prefix, not a directory
    (run.bin / "squeue").write_text(
        "#!/usr/bin/env bash\n"
        f"echo '101 {run.work}/ab/cdef'\n"
        f"echo '102 /somewhere/else/work/ef/0123'\n"
        f"echo '103 {other}/12/3456'\n"
        f"echo '104 {run.work}/9f/8e7d'\n", encoding="utf-8")
    (run.bin / "scancel").write_text(
        f"#!/usr/bin/env bash\necho \"$*\" >> '{run.state}/scancel'\n", encoding="utf-8")
    for tool in ("squeue", "scancel"):
        (run.bin / tool).chmod(0o755)

    assert run.run() == 0, run.stderr
    assert (run.state / "scancel").read_text().split() == ["101", "104"]


# --- the submitter's side ------------------------------------------------

def test_the_head_job_runs_nextflow_under_the_supervisor(tmp_path):
    fake = FakeSbatch()
    _patch(sub, "_run_sbatch_with_retry", fake)
    cohort = _cohort(tmp_path)
    sub.submit_anorak_job(**cohort)
    wrap = fake.calls[0][fake.calls[0].index("--wrap") + 1]

    assert wrap.startswith("exec bash "), wrap
    assert str(cohort["pipeline_dir"].resolve() / "tools" / "nf_supervise.sh") in wrap
    before, after = wrap.split(" -- ", 1)
    assert after.startswith("nextflow "), "nextflow is not what gets supervised"
    assert f"--stall-seconds {sub.WATCHDOG_STALL_SECONDS}" in before
    assert "--log " in before and "--work-dir " in before


def test_a_pipeline_copy_without_the_supervisor_is_refused(tmp_path):
    fake = FakeSbatch()
    _patch(sub, "_run_sbatch_with_retry", fake)
    cohort = _cohort(tmp_path)
    (cohort["pipeline_dir"] / "tools" / "nf_supervise.sh").unlink()
    try:
        sub.submit_anorak_job(**cohort)
    except ValueError as refusal:
        assert "nf_supervise.sh" in str(refusal)
    else:
        raise AssertionError("submitted a head job with no watchdog")
    assert fake.calls == []


def test_the_stall_limit_outlasts_several_heartbeats():
    """The monitor logs every dumpInterval; a limit near it would kill healthy
    runs between two summaries."""
    config = (HERE.parent.parent / "anorak-nf" / "conf" / "beatson.config").read_text()
    assert re.search(r"dumpInterval\s*=\s*'5 min'", config), "dumpInterval moved"
    assert sub.WATCHDOG_STALL_SECONDS >= 3 * 5 * 60


def test_zz_no_test_left_the_module_patched():
    """Named to run last (definition order under pytest, sorted standalone)."""
    assert sub.LOG_DIR == _REAL["LOG_DIR"], sub.LOG_DIR
    assert sub._run_sbatch_with_retry is _REAL["_run_sbatch_with_retry"]


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_watchdog_test_"))
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
