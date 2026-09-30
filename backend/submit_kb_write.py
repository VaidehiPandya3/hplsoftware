#!/usr/bin/env python3
"""Submit Stages 5 and 6 to Slurm instead of running them inside the request.

Both stages used to run in-process in the tile server. That survived the UI
being closed — FastAPI runs a sync endpoint in a threadpool and uvicorn does not
cancel it when the client disconnects — but not the server process going away,
which is what actually happened: a killed server took an hours-long write with
it, and the run record said nothing about whether it had committed.

These submitters hand the same two CLIs (register_dataset.py and
load_hpc_assignments.py) to Slurm, so the write outlives both the browser and
the server. Nothing about what gets written changes: the job runs the identical
functions the endpoints called, with the identical guards.

    from submit_kb_write import submit_registration_job
    info = submit_registration_job(submission_id=..., h5=..., ...)
    info["registration_job_id"]   # poll it like any other stage

THE ONE THING THAT CAN MAKE THIS IMPOSSIBLE is reaching Postgres from a compute
node. The server's default DB_HOST is 127.0.0.1, which on a compute node means
*that compute node*, and a unix socket path (what psql uses here) does not work
across machines at all. Either would produce a job that queues, starts, and
fails to connect — minutes or hours later, with the failure buried in a log. So
the host is resolved and refused at submit time instead; see resolve_job_db_host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submit_mask_tile_slurm import _run_sbatch_with_retry  # noqa: E402

#: Where a job's stdout/stderr land, matching every other stage.
LOG_DIR = Path(__file__).resolve().parent / "slurm_logs"

#: Both stages are pandas-and-Postgres work on one core. Registration is the
#: heavier of the two: it holds the .h5 identity frame and every slide's Stage 1
#: metadata in memory at once. These are generous rather than measured — the
#: cost of asking for too much here is queue time, and the cost of too little is
#: an OOM kill several hours in.
REGISTRATION_MEMORY = "64G"
REGISTRATION_TIME_LIMIT = "12:00:00"
KB_LOAD_MEMORY = "48G"
KB_LOAD_TIME_LIMIT = "12:00:00"

#: Stage 6 is no longer one core's worth of work. It reads the assignment CSV on
#: pandas' threaded pyarrow reader and COPYs it into its scratch table on
#: several connections at once (see kb_stage), so the cores are used rather than
#: held. Four rather than more: past a handful of COPY writers the limit is
#: PostgreSQL's relation extension lock, not this side.
KB_LOAD_CPUS = 4
DEFAULT_PARTITION = os.getenv("HPL_KB_PARTITION") or os.getenv("HPL_MERGE_PARTITION")

#: Set this to the hostname a compute node can reach Postgres on. It exists
#: because there is no safe default: see resolve_job_db_host.
JOB_DB_HOST_ENV = "HPL_JOB_DB_HOST"

_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def resolve_job_db_host(env: dict | None = None) -> str:
    """The hostname a Slurm job should connect to Postgres on.

    Refuses rather than guessing, because every wrong answer here fails the same
    expensive way — the job queues, starts, and cannot connect:

      * a unix socket path (DB_HOST="/mnt/.../socket") is not reachable off the
        machine running Postgres, whatever filesystem it sits on;
      * "localhost"/"127.0.0.1" resolves on the compute node to the compute
        node, which is not where the database is.

    Neither is detectable from the server, which reaches the database perfectly
    well over exactly those. So the operator states the host once, in
    HPL_JOB_DB_HOST, and a submission without it is refused at submit time —
    where the message can be read — rather than in a log an hour later.
    """
    env = os.environ if env is None else env
    explicit = (env.get(JOB_DB_HOST_ENV) or "").strip()
    if explicit:
        if explicit.startswith("/"):
            raise ValueError(
                f"{JOB_DB_HOST_ENV}={explicit!r} is a unix socket path. A socket "
                f"is local to the machine running Postgres — a compute node "
                f"cannot use it even on shared storage. Set {JOB_DB_HOST_ENV} to "
                f"a hostname the cluster can resolve, e.g. the login node's."
            )
        if explicit.lower() in _LOCAL_HOSTS:
            raise ValueError(
                f"{JOB_DB_HOST_ENV}={explicit!r} means 'this machine', which on a "
                f"compute node is the compute node, not the database host. Set it "
                f"to a resolvable hostname."
            )
        return explicit

    host = (env.get("DB_HOST") or "").strip()
    if host.startswith("/") or host.lower() in _LOCAL_HOSTS:
        raise ValueError(
            f"Cannot submit this stage to Slurm: DB_HOST is {host or 'unset'!r}, "
            f"which a compute node cannot use to reach Postgres — a socket path "
            f"is local to the database's own machine, and localhost on a compute "
            f"node is that compute node.\n\n"
            f"Set {JOB_DB_HOST_ENV} to the hostname the database listens on (and "
            f"make sure Postgres accepts TCP connections from compute nodes), "
            f"then resubmit. To check it first:\n\n"
            f"    srun -t 5 -n1 python3 -c \"import socket; "
            f"socket.create_connection(('<host>', {env.get('DB_PORT', '5432')}), 5)\""
        )
    return host


def _python() -> str:
    """The interpreter the server itself is running under.

    Used rather than a `conda activate` line because it needs no login-shell
    setup to be correct: whatever environment the server was started in is
    exactly the one with sqlalchemy, h5py, pandas and openslide already working.
    These stages need no container — unlike Stages 3 and 4, they are plain
    Python against Postgres.
    """
    return sys.executable


def _submit(argv: list[str], job_id_key: str) -> dict:
    """Run sbatch, returning {job_id_key: <id>, sbatch_command, sbatch_stdout}."""
    result = _run_sbatch_with_retry(argv)
    stdout = (result.stdout or "").strip()
    info = {"sbatch_command": argv, "sbatch_stdout": stdout, job_id_key: None}
    match = re.search(r"Submitted batch job (\d+)", stdout)
    if match:
        info[job_id_key] = match.group(1)
    return info


def _sbatch_argv(*, job_name: str, log_stem: str, memory: str, time_limit: str,
                 partition: str | None, notify_email: str | None,
                 command: str, cpus: int = 2) -> list[str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    backend_dir = Path(__file__).resolve().parent
    return [
        "sbatch",
        f"--job-name={job_name}",
        *([f"--partition={partition}"] if partition else []),
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--output={LOG_DIR}/{log_stem}_%j.out",
        f"--error={LOG_DIR}/{log_stem}_%j.err",
        f"--chdir={backend_dir}",
        *([f"--mail-user={notify_email}", "--mail-type=END,FAIL"]
          if notify_email else []),
        # sbatch exports the submitting environment by default, which is how
        # DB_USER/DB_PASS reach the job. Deliberately not passed as arguments:
        # anything in argv is visible in `squeue -o %o` to every user on the
        # cluster.
        "--wrap", f"bash -lc {shlex.quote(command)}",
    ]


def _log_path(log_stem: str, job_id: str | None) -> str | None:
    return str(LOG_DIR / f"{log_stem}_{job_id}.out") if job_id else None


def _env_prefix(db_host: str, db_name: str | None) -> str:
    """Environment the job needs, as shell assignments.

    DB_HOST is overridden because the server's own value is what a compute node
    cannot use. DB_NAME is set explicitly rather than inherited so a write aimed
    at the test Knowledge Bank cannot land in production because the server
    happened to be started with the production default.
    """
    parts = [f"export DB_HOST={shlex.quote(db_host)};"]
    if db_name:
        parts.append(f"export DB_NAME={shlex.quote(db_name)};")
    return " ".join(parts)


def submit_registration_job(
    *,
    submission_id: str,
    h5: Path,
    tile_dir: Path,
    tile_dataset_name: str,
    dataset_id: str,
    db_name: str,
    run_db_name: str,
    raw_dir: Path | None = None,
    slide_metadata: bool = False,
    target_mpp: float | None = None,
    tile_size_5x_px: int | None = None,
    scope: str = "full",
    slide_names: list[str] | None = None,
    replace: bool = False,
    partition: str | None = None,
    notify_email: str | None = None,
    memory: str = REGISTRATION_MEMORY,
    time_limit: str = REGISTRATION_TIME_LIMIT,
) -> dict:
    """Queue Stage 5 as a Slurm job. Returns registration_job_id + log path."""
    db_host = resolve_job_db_host()
    if not h5.is_file():
        raise FileNotFoundError(f"No such packaged .h5: {h5}")
    if not (tile_dir / tile_dataset_name).is_dir():
        raise FileNotFoundError(
            f"No tile folder {tile_dataset_name!r} under {tile_dir}")

    script = Path(__file__).resolve().parent / "register_dataset.py"
    args = [
        "--h5", str(h5),
        "--tile-dir", str(tile_dir),
        "--tile-dataset-name", tile_dataset_name,
        "--dataset-id", dataset_id,
        "--scope", scope,
        "--commit",
        "--record-run", submission_id,
        "--record-run-db", run_db_name,
    ]
    if raw_dir:
        args += ["--raw-dir", str(raw_dir)]
    if slide_metadata:
        args.append("--slide-metadata")
    if target_mpp is not None:
        args += ["--target-mpp", str(target_mpp)]
    if tile_size_5x_px is not None:
        args += ["--tile-size-5x", str(tile_size_5x_px)]
    if replace:
        args.append("--replace")
    for name in slide_names or []:
        args += ["--slide-name", name]

    command = (
        "set -euo pipefail; "
        + _env_prefix(db_host, db_name)
        + " echo '=== Registration (Stage 5) ==='; "
        + f"{shlex.quote(_python())} {shlex.quote(str(script))} "
        + " ".join(shlex.quote(a) for a in args)
    )
    log_stem = "hpl_register"
    info = _submit(_sbatch_argv(
        job_name=f"hpl_register_{submission_id}", log_stem=log_stem,
        memory=memory, time_limit=time_limit, partition=partition,
        notify_email=notify_email, command=command,
    ), "registration_job_id")
    info["registration_log_path"] = _log_path(log_stem, info["registration_job_id"])
    info["db_host"] = db_host
    info["database"] = db_name
    return info


def submit_kb_load_job(
    *,
    submission_id: str,
    csv_path: Path,
    db_name: str,
    run_db_name: str,
    cancer_type: str | None = None,
    allow_unknown_clusters: bool = False,
    skip_profiles: bool = False,
    min_margin: float = 0.0,
    partition: str | None = None,
    notify_email: str | None = None,
    memory: str = KB_LOAD_MEMORY,
    time_limit: str = KB_LOAD_TIME_LIMIT,
) -> dict:
    """Queue Stage 6 as a Slurm job. Returns kb_load_job_id + log path."""
    db_host = resolve_job_db_host()
    if not csv_path.is_file():
        raise FileNotFoundError(f"No such assignments CSV: {csv_path}")

    script = Path(__file__).resolve().parent / "load_hpc_assignments.py"
    args = ["--csv", str(csv_path), "--commit",
            "--record-run", submission_id, "--record-run-db", run_db_name]
    if cancer_type:
        args += ["--cancer-type", cancer_type]
    if allow_unknown_clusters:
        args.append("--allow-unknown-clusters")
    if skip_profiles:
        args.append("--skip-profiles")
    if min_margin:
        args += ["--min-margin", str(min_margin)]

    command = (
        "set -euo pipefail; "
        + _env_prefix(db_host, db_name)
        + " echo '=== Knowledge Bank load (Stage 6) ==='; "
        + f"{shlex.quote(_python())} {shlex.quote(str(script))} "
        + " ".join(shlex.quote(a) for a in args)
    )
    log_stem = "hpl_kb_load"
    info = _submit(_sbatch_argv(
        job_name=f"hpl_kb_load_{submission_id}", log_stem=log_stem,
        memory=memory, time_limit=time_limit, partition=partition,
        notify_email=notify_email, command=command, cpus=KB_LOAD_CPUS,
    ), "kb_load_job_id")
    info["kb_load_log_path"] = _log_path(log_stem, info["kb_load_job_id"])
    info["db_host"] = db_host
    info["database"] = db_name
    return info


def check_db_from_compute_node(host: str | None = None, port: str | None = None,
                              database: str | None = None,
                              time_limit: str = "5") -> dict:
    """Can a compute node actually *use* Postgres? One srun, from where the job runs.

    This used to open a TCP socket and stop there, which is a weaker question
    than the one being asked and passes in the exact state that made it worth
    asking. Job 1243329 is the example: DB_HOST resolved, the socket opened,
    and the job then died on `fe_sendauth: no password supplied` five minutes
    later, because the server authenticates over a unix socket where pg_hba
    says peer and a compute node arrives over TCP where it says scram. A
    reachability check that cannot see that is a check that says yes and costs
    an hour.

    So the probe opens the socket *and* completes a real connection, as the job
    will, with the job's own DB_USER/DB_PASS. The two results are reported
    separately because they have different fixes: no TCP is listen_addresses or
    a firewall, TCP without a connection is pg_hba, the role's password, or a
    database that does not exist on that server.
    """
    host = host or resolve_job_db_host()
    port = port or os.getenv("DB_PORT", "5432")
    database = database or os.getenv("DB_NAME", "hpl_kb")

    # Deliberately not importing db_url inside the probe: srun's cwd need not be
    # backend/, and psycopg2's own keywords are what libpq reads anyway. An
    # empty DB_PASS is passed as None so ~/.pgpass and $PGPASSWORD still apply,
    # which is how a job authenticates when the password lives in a file.
    probe = (
        "import json, os, socket\n"
        f"host, port, db = {host!r}, {int(port)}, {database!r}\n"
        "out = {'tcp': False, 'connected': False, 'error': None}\n"
        "try:\n"
        "    socket.create_connection((host, port), 5).close()\n"
        "    out['tcp'] = True\n"
        "except Exception as e:\n"
        "    out['error'] = f'{type(e).__name__}: {e}'\n"
        "if out['tcp']:\n"
        "    try:\n"
        "        import psycopg2\n"
        "        psycopg2.connect(host=host, port=port, dbname=db,\n"
        "                         user=os.getenv('DB_USER', 'vpandya'),\n"
        "                         password=os.getenv('DB_PASS') or None,\n"
        "                         connect_timeout=5).close()\n"
        "        out['connected'] = True\n"
        "    except Exception as e:\n"
        "        out['error'] = f'{type(e).__name__}: {e}'\n"
        "print('HPL_DB_PROBE ' + json.dumps(out))\n"
    )
    argv = ["srun", f"--time=00:00:{int(time_limit):02d}", "-n1",
            _python(), "-c", probe]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300)

    parsed = {"tcp": False, "connected": False, "error": None}
    for line in (result.stdout or "").splitlines():
        if line.startswith("HPL_DB_PROBE "):
            try:
                parsed.update(json.loads(line[len("HPL_DB_PROBE "):]))
            except ValueError:
                pass

    return {
        "host": host, "port": port, "database": database,
        # Kept as the TCP answer it always was, so nothing reading it changes
        # meaning; "usable" is the one to gate a submission on.
        "reachable": parsed["tcp"],
        "usable": parsed["connected"],
        "error": parsed["error"],
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "command": argv,
    }


def probe_advice(result: dict) -> str:
    """What to do about each way the probe can come out short.

    Kept next to the probe because psycopg2's error names the symptom and the
    fix is cluster configuration in every case — and because the four cases
    have four different fixes that look alike from the server.
    """
    if result.get("usable"):
        return ""
    if not result.get("reachable"):
        return (f"Nothing accepted a TCP connection on {result['host']}:"
                f"{result['port']} from a compute node. Postgres is probably "
                f"listening only on localhost or a unix socket — check "
                f"listen_addresses in postgresql.conf — or a firewall is in "
                f"the way. Stages 5 and 6 have to run in the server until this "
                f"changes.")
    error = (result.get("error") or "").lower()
    if "no password supplied" in error or "authentication failed" in error:
        return (f"Postgres answered and refused the connection: the compute "
                f"node arrives over TCP, where pg_hba.conf asks for a password, "
                f"while the server itself connects over a socket where it does "
                f"not. Give the job a password — export DB_PASS in the shell "
                f"the server is started from (sbatch passes the submitting "
                f"environment to the job), or put it in a passfile and export "
                f"PGPASSFILE, which is read by libpq only when DB_PASS is "
                f"empty. If the role has no password yet, set one with "
                f"\\password in psql.")
    if "does not exist" in error:
        return (f"Postgres answered, but {result['database']!r} is not a "
                f"database on that server. Check DB_NAME, and that the test "
                f"Knowledge Bank has been created.")
    if "no pg_hba.conf entry" in error:
        return (f"Postgres answered and has no pg_hba.conf rule covering the "
                f"compute nodes' subnet, so no password can help. A host rule "
                f"for that range is needed.")
    return (f"Postgres answered but the connection did not complete: "
            f"{result.get('error')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-db", action="store_true",
                        help="Submit a one-second srun that tries to open a TCP "
                             "connection to Postgres from a compute node, and "
                             "report whether it worked. Run this once before "
                             "relying on Slurm-backed Stages 5 and 6.")
    parser.add_argument("--database", default=None,
                        help="Database to connect to in the probe. Defaults to "
                             "$DB_NAME, then hpl_kb — pass hpl_kb_test to check "
                             "the test Knowledge Bank.")
    parser.add_argument("--host", default=None,
                        help=f"Host to probe. Defaults to ${JOB_DB_HOST_ENV}, "
                             f"then $DB_HOST.")
    args = parser.parse_args()

    if not args.check_db:
        parser.error("Nothing to do — this module is imported by the server. "
                     "Pass --check-db to test connectivity.")

    try:
        result = check_db_from_compute_node(args.host, database=args.database)
    except ValueError as e:
        print(f"Not checked: {e}")
        return 1
    print(f"host        {result['host']}:{result['port']}")
    print(f"database    {result['database']}")
    print(f"tcp         {result['reachable']}")
    print(f"connected   {result['usable']}")
    if not result["usable"]:
        if result["error"]:
            print(f"error       {result['error']}")
        if result["stderr"]:
            print(f"stderr      {result['stderr'][:500]}")
        print(f"\n{probe_advice(result)}")
    return 0 if result["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
