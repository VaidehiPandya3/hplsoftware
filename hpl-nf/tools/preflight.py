#!/usr/bin/env python3
"""Everything that can be checked before the HPL pipeline queues anything.

    python tools/preflight.py [--checkpoint <.ckt>] [--reference <.npz>]
                              [--python <interpreter>] [--chain N] [--no-compute-node]

Run it once per cluster, and again after changing a container, a config or the
cluster — the same role anorak-nf/tools/preflight.sh plays for ANORAK. The
server makes most of these refusals at every submission too (resolve_run in
backend/submit_hpl_nf.py, which this calls); what only this adds is what a
submission cannot see from the login node:

   1. nextflow is here and new enough for the manifest
   2. the effective config (-profile beatson) parses, and pins what the
      watchdog relies on (dumpInterval) and what CephFS needs (exitReadTimeout)
   3. tools/nf_supervise.sh is ANORAK's watchdog, byte for byte
   4. the interpreter the tasks run under imports what the tiler and packager
      need — openslide, h5py, pandas — and backend/ imports cleanly with it
   5. Stage 3/4 runtime: HPL repo, NGC image, container extras, GPU type,
      search device, reference, checkpoint, walltimes against partitions
   6. a compute node can run sbatch (the head job submits every task)
   7. executor.queueSize plus the head chain fits under MaxSubmitJobs
   8. the results root is writable

Exit status: 0 every check ran and passed; 1 at least one failed; 2 nothing
failed but some were skipped. 2 is not a pass — a skipped check is a question
nobody answered.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
REPO = PIPELINE.parent
BACKEND = Path(os.getenv("HPL_BACKEND_DIR", str(REPO / "backend")))
sys.path.insert(0, str(BACKEND))


class Report:
    def __init__(self):
        self.passed = self.failed = self.skipped = 0

    def step(self, title: str) -> None:
        print(f"\n{title}")

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  ok    {msg}")

    def bad(self, msg: str) -> None:
        self.failed += 1
        print(f"  FAIL  {msg}")

    def skip(self, msg: str) -> None:
        self.skipped += 1
        print(f"  --    {msg}")

    def attempt(self, label: str, fn) -> object:
        try:
            result = fn()
        except SystemExit as e:        # resolve_vote/vote_flags refuse this way
            self.bad(f"{label}: {e}")
            return None
        except Exception as e:  # noqa: BLE001 - every refusal is a finding
            self.bad(f"{label}: {e}")
            return None
        self.ok(label if result in (None, True) else f"{label}: {result}")
        return result


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_nextflow(r: Report) -> dict:
    r.step("1. nextflow")
    if not shutil.which("nextflow"):
        r.bad("nextflow is not on PATH — load it (HPL_NEXTFLOW_PRELUDE is run before it in the head job)")
        return {}
    version = _run(["nextflow", "-version"])
    found = re.search(r"version (\d+)\.(\d+)", (version.stdout if version else "") or "")
    if not found:
        r.skip("could not read nextflow's version")
    elif (int(found.group(1)), int(found.group(2))) >= (23, 4):
        r.ok(f"nextflow {found.group(1)}.{found.group(2)} (manifest needs >= 23.04)")
    else:
        r.bad(f"nextflow {found.group(1)}.{found.group(2)} is older than the manifest's 23.04")

    r.step("2. effective config (-profile beatson)")
    flat = _run(["nextflow", "config", str(PIPELINE), "-profile", "beatson", "-flat"])
    if flat is None or flat.returncode != 0:
        r.bad(f"`nextflow config -profile beatson` failed: {(flat.stderr if flat else '')[-400:]}")
        return {}
    values = {}
    for line in flat.stdout.splitlines():
        key, _, value = line.partition(" = ")
        values[key.strip()] = value.strip().strip("'\"")
    for key, want in (("executor.dumpInterval", "5 min"), ("executor.exitReadTimeout", "15 min")):
        (r.ok if values.get(key) == want else r.bad)(f"{key} = {values.get(key)!r} (want {want!r})")
    return values


def check_watchdog(r: Report) -> None:
    r.step("3. watchdog")
    import submit_hpl_nf

    supervisor = PIPELINE / "tools" / "nf_supervise.sh"
    if not supervisor.is_file():
        r.bad(f"{supervisor} is missing — copy the whole hpl-nf directory again")
        return
    drift = submit_hpl_nf.supervisor_drift(supervisor)
    (r.bad(drift) if drift else r.ok(f"{supervisor.name} is ANORAK's watchdog"))


def check_interpreter(r: Report, python: str) -> None:
    r.step(f"4. task interpreter ({python})")
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); "
             "import openslide, h5py, numpy, pandas, PIL; "
             "import submit_mask_tile_slurm, make_hpl_hdf5, stage_outputs, hpl_nf_state; "
             "print(openslide.__version__, h5py.__version__)")
    result = _run([python, "-c", probe, str(BACKEND)])
    if result is None:
        r.bad(f"{python} could not be run")
    elif result.returncode != 0:
        r.bad(f"imports failed: {(result.stderr or '').strip().splitlines()[-1:]}")
    else:
        r.ok(f"openslide / h5py {result.stdout.strip()} and backend/ import")


def check_runtime(r: Report, checkpoint: str | None, reference: str | None) -> None:
    r.step("5. Stage 3 and 4 runtime")
    import submit_cluster_assignment as sca
    import submit_feature_extraction as sfe
    import submit_hpl_nf

    r.attempt(f"HPL repo {sfe.HPL_REPO_DIR}", lambda: sfe._check_hpl_repo_dir(sfe.HPL_REPO_DIR))
    r.attempt(f"NGC image {sfe.SINGULARITY_IMAGE}",
              lambda: sfe._check_singularity_image(sfe.SINGULARITY_IMAGE, sfe.SINGULARITY_BIN))
    r.attempt(f"container extras {sfe.CONTAINER_EXTRAS}",
              lambda: sfe._check_container_extras(sfe.CONTAINER_EXTRAS, sfe.SINGULARITY_IMAGE,
                                                  sfe.SINGULARITY_BIN))
    gres = r.attempt("GPU type", lambda: " ".join(sfe.select_gpu_gres(sfe.GPU_PARTITION)))
    device = r.attempt("search device", lambda: " — ".join(sca.resolve_device("auto")))
    if device and device.startswith("gpu"):
        r.attempt(f"GPU faiss extras {sfe.CONTAINER_EXTRAS_GPU}",
                  lambda: sfe._check_container_extras(sfe.CONTAINER_EXTRAS_GPU, sfe.SINGULARITY_IMAGE,
                                                      sfe.SINGULARITY_BIN))
    del gres
    r.attempt("reference", lambda: "{reference_path} ({reference_rows:,} rows, {n_clusters} clusters)"
              .format(**sca.check_reference(sca._reference_path(Path(reference) if reference else None))))
    if checkpoint:
        r.attempt(f"checkpoint {checkpoint}", lambda: sfe._check_checkpoint(checkpoint))
    else:
        r.skip("checkpoint not given (--checkpoint) — the server checks it at submission")
    for step, partition in (("extract", sfe.GPU_PARTITION), ("assign", sca.MERGE_PARTITION)):
        limit = submit_hpl_nf.TIME_LIMITS[step]
        r.attempt(f"{step} walltime {limit} on {partition}",
                  lambda p=partition, t=limit: sca.check_time_limit(p, t))


def check_compute_node(r: Report, enabled: bool) -> None:
    r.step("6. a compute node can run sbatch")
    if not enabled:
        r.skip("--no-compute-node given")
        return
    if not shutil.which("srun"):
        r.skip("srun is not on PATH here")
        return
    import submit_hpl_nf
    result = submit_hpl_nf.check_submit_from_compute_node()
    (r.ok("sbatch answers from a compute node") if result.get("ok")
     else r.bad(f"it does not: {result.get('reason')}"))


def check_submit_limit(r: Report, values: dict, chain: int) -> None:
    r.step("7. submit limit")
    queue = values.get("executor.queueSize")
    if not queue:
        r.skip("executor.queueSize unknown (the config did not parse)")
        return
    if not shutil.which("sacctmgr"):
        r.skip("sacctmgr not available here")
        return
    user = os.environ.get("USER", "")
    assoc = _run(["sacctmgr", "-n", "-P", "show", "assoc", f"where", f"user={user}",
                  "format=Account,Partition,MaxSubmitJobs"])
    limits = [int(f) for line in (assoc.stdout if assoc else "").splitlines()
              for f in line.split("|")[2:3] if f.strip().isdigit()]
    need = int(queue) + chain
    if not limits:
        r.skip(f"no MaxSubmitJobs set for {user} — the run keeps up to {need} jobs submitted")
    elif need <= min(limits):
        r.ok(f"MaxSubmitJobs {min(limits)} >= {need} (queueSize {queue} + {chain} head jobs)")
    else:
        r.bad(f"MaxSubmitJobs is {min(limits)}, below the {need} jobs this run keeps submitted — "
              f"lower executor.queueSize in conf/beatson.config, or the chain")


def check_results_root(r: Report) -> None:
    r.step("8. results root")
    root = Path(os.getenv("HPL_NF_RESULTS_ROOT", "/hpc-home/home/users/vpandya/long-term-scratch/hpl-nf"))
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".preflight.{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
        r.ok(f"{root} is writable")
    except OSError as e:
        r.bad(f"{root} is not writable ({e}) — set HPL_NF_RESULTS_ROOT")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint")
    parser.add_argument("--reference")
    parser.add_argument("--python", default=sys.executable,
                        help="the interpreter the tile server runs under (tasks use it)")
    parser.add_argument("--chain", type=int, default=3)
    parser.add_argument("--no-compute-node", action="store_true",
                        help="skip the srun probe (e.g. on a laptop)")
    args = parser.parse_args(argv)

    r = Report()
    values = check_nextflow(r)
    check_watchdog(r)
    check_interpreter(r, args.python)
    check_runtime(r, args.checkpoint, args.reference)
    check_compute_node(r, not args.no_compute_node)
    check_submit_limit(r, values, args.chain)
    check_results_root(r)

    print(f"\n{r.passed} passed, {r.failed} failed, {r.skipped} skipped")
    if r.failed:
        return 1
    return 2 if r.skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
