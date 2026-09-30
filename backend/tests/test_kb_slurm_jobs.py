"""Stages 5 and 6 as Slurm jobs, so a write outlives the server.

Both stages used to run inside the HTTP request. Closing the UI was never the
problem — FastAPI runs a sync endpoint in a threadpool and uvicorn does not
cancel it on disconnect — but a killed server took an hours-long write with it
and left nothing on the run to say whether it had committed.

The thing most worth testing here is not that a job gets submitted. It is the
refusal that stops one being submitted into a configuration where it cannot
possibly work: a compute node reaching Postgres. The server itself reaches the
database over a unix socket or localhost quite happily, and neither of those
means anything on another machine, so a job built from the server's own settings
would queue, start, and fail to connect — minutes or hours later, in a log.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import submit_kb_write as kb  # noqa: E402


class _FakeSbatch:
    """Stands in for _run_sbatch_with_retry, capturing the argv it was given."""

    def __init__(self, stdout="Submitted batch job 987654"):
        self.stdout = stdout
        self.argv = None

    def __call__(self, argv, *a, **k):
        self.argv = argv
        return subprocess.CompletedProcess(argv, 0, stdout=self.stdout, stderr="")


def _fixture(tmp_path: Path) -> dict:
    h5 = tmp_path / "hdf5_Radiogenomics_he_train.h5"
    h5.write_bytes(b"not a real .h5, but a real file")
    tile_dir = tmp_path / "processed_tiles"
    (tile_dir / "Radiogenomics").mkdir(parents=True)
    csv = tmp_path / "Radiogenomics_hpc_assignments.csv"
    csv.write_text("samples,slides,tiles\n")
    return {"h5": h5, "tile_dir": tile_dir, "csv": csv}


def _wrap(argv) -> str:
    """The --wrap payload: everything the job will actually run."""
    return argv[argv.index("--wrap") + 1]


# --- the refusal the whole feature rests on -------------------------------

def test_a_socket_or_localhost_db_host_is_refused_not_submitted(_tmp=None):
    """Each of these reaches the database perfectly from the server and not at
    all from a compute node, which is why none of them can be a default."""
    for env in ({}, {"DB_HOST": "127.0.0.1"}, {"DB_HOST": "localhost"},
                {"DB_HOST": "/mnt/cephfs-lts/.../socket"},
                {kb.JOB_DB_HOST_ENV: "localhost"},
                {kb.JOB_DB_HOST_ENV: "/mnt/cephfs-lts/.../socket"}):
        try:
            kb.resolve_job_db_host(env)
        except ValueError as e:
            assert kb.JOB_DB_HOST_ENV in str(e), str(e)
        else:
            raise AssertionError(f"{env} was accepted as a job DB host")


def test_a_real_hostname_is_accepted_from_either_variable(_tmp=None):
    assert kb.resolve_job_db_host({kb.JOB_DB_HOST_ENV: "hpc-login-01"}) == "hpc-login-01"
    assert kb.resolve_job_db_host({"DB_HOST": "db.internal"}) == "db.internal"
    # The explicit variable wins, because DB_HOST is what the server needs and
    # is often exactly the value a job cannot use.
    assert kb.resolve_job_db_host(
        {"DB_HOST": "127.0.0.1", kb.JOB_DB_HOST_ENV: "hpc-login-01"}) == "hpc-login-01"


def test_nothing_is_submitted_when_the_host_is_unusable(tmp_path):
    """The refusal has to happen before sbatch, or the operator gets a job id
    for a job that cannot work."""
    f = _fixture(tmp_path)
    fake = _FakeSbatch()
    old_env, old_sbatch = dict(os.environ), kb._run_sbatch_with_retry
    try:
        os.environ.pop(kb.JOB_DB_HOST_ENV, None)
        os.environ["DB_HOST"] = "127.0.0.1"
        kb._run_sbatch_with_retry = fake
        try:
            kb.submit_registration_job(
                submission_id="S1", h5=f["h5"], tile_dir=f["tile_dir"],
                tile_dataset_name="Radiogenomics", dataset_id="RADIOGENOMICS",
                db_name="hpl_kb", run_db_name="hpl_kb")
        except ValueError:
            pass
        else:
            raise AssertionError("submitted despite an unusable DB host")
        assert fake.argv is None, "sbatch was called anyway"
    finally:
        os.environ.clear(); os.environ.update(old_env)
        kb._run_sbatch_with_retry = old_sbatch


# --- what the job is actually told to do ----------------------------------

def _submit(tmp_path, which, **overrides):
    f = _fixture(tmp_path)
    fake = _FakeSbatch()
    old_env, old_sbatch = dict(os.environ), kb._run_sbatch_with_retry
    try:
        os.environ[kb.JOB_DB_HOST_ENV] = "hpc-login-01"
        os.environ["DB_PASS"] = "hunter2"
        kb._run_sbatch_with_retry = fake
        if which == "registration":
            args = dict(submission_id="S1", h5=f["h5"], tile_dir=f["tile_dir"],
                        tile_dataset_name="Radiogenomics",
                        dataset_id="RADIOGENOMICS", db_name="hpl_kb",
                        run_db_name="hpl_kb")
            args.update(overrides)
            info = kb.submit_registration_job(**args)
        else:
            args = dict(submission_id="S1", csv_path=f["csv"], db_name="hpl_kb",
                        run_db_name="hpl_kb")
            args.update(overrides)
            info = kb.submit_kb_load_job(**args)
        return info, fake.argv
    finally:
        os.environ.clear(); os.environ.update(old_env)
        kb._run_sbatch_with_retry = old_sbatch


def test_the_registration_job_runs_the_cli_with_commit_and_records_the_run(tmp_path):
    info, argv = _submit(tmp_path, "registration")

    assert info["registration_job_id"] == "987654"
    command = _wrap(argv)
    assert "register_dataset.py" in command
    assert "--commit" in command
    # Without these the job writes the KB and the run still says "not done",
    # which is the state that made a killed server unreadable in the first place.
    assert "--record-run S1" in command
    assert "--record-run-db hpl_kb" in command
    assert info["registration_log_path"].endswith("hpl_register_987654.out")


def test_the_kb_load_job_runs_the_cli_with_commit(tmp_path):
    info, argv = _submit(tmp_path, "kb_load", min_margin=0.25,
                         allow_unknown_clusters=True)

    assert info["kb_load_job_id"] == "987654"
    command = _wrap(argv)
    assert "load_hpc_assignments.py" in command
    assert "--commit" in command
    assert "--min-margin 0.25" in command
    assert "--allow-unknown-clusters" in command


def test_the_job_is_pinned_to_one_knowledge_bank(tmp_path):
    """DB_NAME is exported into the job rather than inherited. A write aimed at
    the test KB must not land in production because the server happened to be
    started with the production default."""
    _info, argv = _submit(tmp_path, "kb_load", db_name="hpl_kb_test")
    command = _wrap(argv)

    assert "export DB_NAME=hpl_kb_test" in command
    assert "export DB_HOST=hpc-login-01" in command
    # Run tracking is separate and stays in production.
    assert "--record-run-db hpl_kb" in command


def test_the_database_password_never_reaches_the_command_line(tmp_path):
    """sbatch exports the submitting environment, so DB_PASS travels in the
    environment. Anything in argv shows up in `squeue -o %o` for every user on
    the cluster."""
    _info, argv = _submit(tmp_path, "registration")

    assert "hunter2" not in " ".join(argv)
    assert "DB_PASS" not in " ".join(argv)


def test_a_missing_input_is_refused_before_sbatch(tmp_path):
    """Same posture as every other submitter here: refuse at submit time, where
    the message gets read, rather than in a log an hour later."""
    fake = _FakeSbatch()
    old_env, old_sbatch = dict(os.environ), kb._run_sbatch_with_retry
    try:
        os.environ[kb.JOB_DB_HOST_ENV] = "hpc-login-01"
        kb._run_sbatch_with_retry = fake
        try:
            kb.submit_kb_load_job(submission_id="S1",
                                  csv_path=tmp_path / "nope.csv",
                                  db_name="hpl_kb", run_db_name="hpl_kb")
        except FileNotFoundError as e:
            assert "nope.csv" in str(e)
        else:
            raise AssertionError("a missing CSV was submitted")
        assert fake.argv is None
    finally:
        os.environ.clear(); os.environ.update(old_env)
        kb._run_sbatch_with_retry = old_sbatch


def test_a_missing_tile_folder_is_refused_before_sbatch(tmp_path):
    f = _fixture(tmp_path)
    fake = _FakeSbatch()
    old_env, old_sbatch = dict(os.environ), kb._run_sbatch_with_retry
    try:
        os.environ[kb.JOB_DB_HOST_ENV] = "hpc-login-01"
        kb._run_sbatch_with_retry = fake
        try:
            kb.submit_registration_job(
                submission_id="S1", h5=f["h5"], tile_dir=f["tile_dir"],
                tile_dataset_name="Nonexistent", dataset_id="X",
                db_name="hpl_kb", run_db_name="hpl_kb")
        except FileNotFoundError as e:
            assert "Nonexistent" in str(e)
        else:
            raise AssertionError("a missing tile folder was submitted")
        assert fake.argv is None
    finally:
        os.environ.clear(); os.environ.update(old_env)
        kb._run_sbatch_with_retry = old_sbatch


# --- the CLIs the jobs run ------------------------------------------------

def test_both_clis_accept_the_flags_the_jobs_pass(_tmp=None):
    """The submitter and the scripts it invokes are two files that have to agree
    on an argument list. They cannot be checked by import alone — argparse only
    complains at run time, inside the job."""
    for script in ("register_dataset.py", "load_hpc_assignments.py"):
        helptext = subprocess.run(
            [sys.executable, str(BACKEND / script), "--help"],
            capture_output=True, text=True, timeout=120).stdout
        for flag in ("--record-run", "--record-run-db", "--record-kb-target"):
            assert flag in helptext, f"{script} has no {flag}"


def test_the_migration_adds_every_column_the_server_writes(_tmp=None):
    """A column the endpoint writes but the migration never adds is an UPDATE
    that fails on a deployment that ran the migrations in good faith."""
    sql = (BACKEND / "migrate_dataset_runs_kb_slurm.sql").read_text()
    for column in ("registration_job_id", "registration_submitted_at",
                   "registration_log_path", "registration_error",
                   "kb_load_job_id", "kb_load_submitted_at",
                   "kb_load_log_path", "kb_load_error"):
        assert f"ADD COLUMN IF NOT EXISTS {column} " in sql, column
    assert "migrate_dataset_runs_kb_slurm.sql" in \
        (BACKEND / "migrate_all.sql").read_text(), \
        "the migration exists but migrate_all.sql does not run it"


def test_the_endpoints_exist_and_report_the_job_state(_tmp=None):
    import tile_server_v2_ as srv

    paths = {r.path for r in srv.app.routes}
    assert "/dataset-jobs/{submission_id}/register-submit" in paths
    assert "/dataset-jobs/{submission_id}/kb-load-submit" in paths
    # The connectivity probe is an endpoint rather than a runbook paragraph
    # because the answer is cluster configuration the server cannot infer.
    assert "/kb-job-db-check" in paths


# --- recording the outcome on a database that is behind ------------------
#
# The Radiogenomics registration committed 18,485,499 tiles and then failed to
# record it, because migrate_dataset_runs_kb_slurm.sql had not been applied and
# one new column (registration_error) was missing from the UPDATE. The run said
# "not done" over a write that had happened — the single most misleading state
# this stage can be in, and the reason the record now writes what it can.


def _runs_table(tmp_path: Path, columns: str):
    from sqlalchemy import create_engine, text
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE slurm_dataset_runs ({columns})"))
        conn.execute(text("INSERT INTO slurm_dataset_runs (submission_id) "
                          "VALUES ('d0010a4f')"))
    return engine


_PRE_MIGRATION = ("submission_id TEXT PRIMARY KEY, registration_done BOOLEAN, "
                  "registration_at TIMESTAMP, registration_dataset_id TEXT, "
                  "registration_raw_dir TEXT, registration_rows TEXT")


def test_a_missing_column_does_not_cost_the_whole_record(tmp_path):
    from datetime import datetime, timezone
    from sqlalchemy import text

    import run_record
    engine = _runs_table(tmp_path, _PRE_MIGRATION)
    original = run_record.run_engine
    run_record.run_engine = lambda name: engine
    try:
        recorded = run_record.record_run(
            "hpl_kb", "d0010a4f",
            registration_done=True,
            registration_at=datetime.now(timezone.utc),
            registration_dataset_id="RADIOGENOMICS",
            registration_rows='{"tile_registry": 18485499}',
            registration_error=None,          # only this column is missing
        )
    finally:
        run_record.run_engine = original

    assert recorded is True
    with engine.connect() as conn:
        done, dataset_id = conn.execute(text(
            "SELECT registration_done, registration_dataset_id "
            "FROM slurm_dataset_runs")).fetchone()
    assert done, "the stage must read as done — the rows are in the KB"
    assert dataset_id == "RADIOGENOMICS"


def test_it_reports_false_when_nothing_could_be_recorded(tmp_path):
    """The guard has to be able to fail: a table with none of these columns is a
    database nobody has migrated, and claiming success there would put the run
    back to looking committed when nothing said so."""
    import run_record
    engine = _runs_table(tmp_path, "submission_id TEXT PRIMARY KEY")
    original = run_record.run_engine
    run_record.run_engine = lambda name: engine
    try:
        assert run_record.record_run("hpl_kb", "d0010a4f",
                                     registration_done=True) is False
    finally:
        run_record.run_engine = original


def test_recording_never_raises_into_the_jobs_exit_status(tmp_path):
    """Reached only after a committed Knowledge Bank transaction. Failing the
    job here would report a write that happened as one that did not."""
    import run_record
    original = run_record.run_engine

    def _explode(_name):
        raise RuntimeError("database is on fire")

    run_record.run_engine = _explode
    try:
        assert run_record.record_run("hpl_kb", "d0010a4f",
                                     registration_done=True) is False
    finally:
        run_record.run_engine = original


# --- the probe that says whether any of this can work ----------------------

def _probe_stdout(tcp, connected, error=None):
    """What the srun'd probe prints back."""
    import json
    return "container packages: ok\nHPL_DB_PROBE " + json.dumps(
        {"tcp": tcp, "connected": connected, "error": error}) + "\n"


def _fake_srun(stdout, returncode=0):
    def run(argv, capture_output=True, text=True, timeout=None):
        run.argv = argv
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout,
                                           stderr="")
    return run


def test_a_socket_that_opens_but_cannot_authenticate_is_not_usable(_tmp=None):
    """The state that cost job 1243329 five minutes and an hour of reading.

    TCP succeeded, so the old check — socket.create_connection and nothing
    more — reported reachable and said go. The job then failed on fe_sendauth,
    because the server reaches Postgres over a unix socket where pg_hba says
    peer and a compute node arrives over TCP where it says scram.
    """
    original = kb.subprocess.run
    kb.subprocess.run = _fake_srun(_probe_stdout(
        True, False,
        'OperationalError: connection to server at "hpc-login-01" '
        '(172.21.234.110), port 5432 failed: fe_sendauth: no password supplied'))
    try:
        result = kb.check_db_from_compute_node(host="hpc-login-01",
                                               database="hpl_kb_test")
    finally:
        kb.subprocess.run = original

    assert result["reachable"] is True, "the socket did open"
    assert result["usable"] is False, "and the job still could not connect"
    advice = kb.probe_advice(result)
    assert "DB_PASS" in advice and "PGPASSFILE" in advice, advice


def test_a_working_database_is_usable_and_needs_no_advice(_tmp=None):
    original = kb.subprocess.run
    kb.subprocess.run = _fake_srun(_probe_stdout(True, True))
    try:
        result = kb.check_db_from_compute_node(host="hpc-login-01",
                                               database="hpl_kb_test")
    finally:
        kb.subprocess.run = original
    assert result["usable"] is True
    assert kb.probe_advice(result) == ""


def test_nothing_listening_is_reported_as_the_different_problem_it_is(_tmp=None):
    """Different fix: listen_addresses or a firewall, not a password."""
    original = kb.subprocess.run
    kb.subprocess.run = _fake_srun(_probe_stdout(
        False, False, "ConnectionRefusedError: [Errno 111] Connection refused"))
    try:
        result = kb.check_db_from_compute_node(host="hpc-login-01")
    finally:
        kb.subprocess.run = original
    assert (result["reachable"], result["usable"]) == (False, False)
    assert "listen_addresses" in kb.probe_advice(result)


def test_a_missing_database_is_not_read_as_an_auth_problem(_tmp=None):
    """hpl_kb_test not existing on that server looks identical from the server."""
    original = kb.subprocess.run
    kb.subprocess.run = _fake_srun(_probe_stdout(
        True, False, 'OperationalError: FATAL: database "hpl_kb_test" does not exist'))
    try:
        result = kb.check_db_from_compute_node(host="hpc-login-01",
                                               database="hpl_kb_test")
    finally:
        kb.subprocess.run = original
    advice = kb.probe_advice(result)
    assert "not a database" in advice and "DB_PASS" not in advice, advice


def test_the_probe_asks_about_the_database_it_was_given(_tmp=None):
    """A probe against the wrong database answers a question nobody asked."""
    original = kb.subprocess.run
    fake = _fake_srun(_probe_stdout(True, True))
    kb.subprocess.run = fake
    try:
        kb.check_db_from_compute_node(host="hpc-login-01", database="hpl_kb_test")
    finally:
        kb.subprocess.run = original
    probe_source = fake.argv[-1]
    assert "'hpl_kb_test'" in probe_source
    assert "'hpc-login-01'" in probe_source
    # The job's own credentials, not this process's: sbatch passes the
    # submitting environment through, and DB_PASS empty must stay empty so
    # libpq still reads a passfile.
    assert "os.getenv('DB_PASS') or None" in probe_source


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_kb_slurm_test_"))
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
