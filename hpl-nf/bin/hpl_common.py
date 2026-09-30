#!/usr/bin/env python3
"""Shared helpers for the HPL process wrappers in this directory.

Every wrapper is a thin shell around code that already runs Stages 1-4 as
separate Slurm jobs from backend/ — the tiler, make_hpl_hdf5, and the
extraction and assignment command builders — so the container invocations,
bind rules and --cleanenv workarounds exist once, not twice. That code is
imported from the backend/ directory named in run_config.json, which is why
load_config() is the first thing every wrapper calls: it puts that directory
on sys.path, and nothing from backend/ is imported before it has.

What the wrappers add is only what running those stages as Nextflow tasks
needs:

  * **Everything comes from run_config.json**, which backend/submit_hpl_nf.py
    resolves once, at submission, from the server's own settings. Nothing is
    read from the environment the task happens to inherit: the chain is
    server -> head job -> task, and any link may have lost a variable.

  * **A stage is marked done only after the server's own gate passes.** The
    last task of each stage runs the validator /status uses (stage_outputs.py,
    validate_extraction_output), bound to the row count of the stage before,
    and writes stages/<stage>.done.json only if it passes. Exit status alone
    never marks a stage.

  * **Every step is safe to run twice.** A cache miss re-runs a task, so each
    first asks whether its output already exists and validates, and does
    nothing if so. submit_hpl_nf.refuse_foreign_outputs() is what makes
    "exists and validates" mean "this run made it".

  * **Shards get a fixed row range, not an array index.** Each shard is its own
    task, so there is no SLURM_ARRAY_TASK_ID; the builders take the range
    directly (row_range=) and pass it through --cleanenv by the SINGULARITYENV_
    route the array path uses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: A refusal: this step's input or output is wrong, and a retry would read the
#: same bytes. nextflow.config finishes the run on this code and retries every
#: other, the way anorak-nf does (anorak_common.REFUSAL_EXIT_CODE; 65 is
#: sysexits.h's EX_DATAERR). A `raise SystemExit("...")` exits 1 — the same code
#: as an uncaught exception — so the config could not tell a wrong input from a
#: node that dropped a CephFS read.
REFUSAL_EXIT_CODE = 65

_REQUIRED_KEYS = ("out_dir", "backend_dir", "manifest", "tiling", "packaging",
                  "extraction", "assignment")


def refuse(message: str) -> None:
    """Stop this task with a readable reason; Nextflow shows stderr's tail."""
    print(f"REFUSED: {message}", file=sys.stderr, flush=True)
    raise SystemExit(REFUSAL_EXIT_CODE)


def load_config(path: Path) -> dict:
    """Read run_config.json and put its backend/ directory on sys.path."""
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        refuse(f"Cannot read the run config {path}: {e}")
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    if missing:
        refuse(f"{path} has no {', '.join(missing)} — it was not written by "
               f"submit_hpl_nf.py, or by an older version of it.")
    backend = Path(config["backend_dir"])
    if not (backend / "hpl_nf_state.py").is_file():
        refuse(f"{backend} is not this repository's backend/ (no hpl_nf_state.py). "
               f"The cluster's copy predates this pipeline — copy backend/ again.")
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    return config


def out_dir(config: dict) -> Path:
    return Path(config["out_dir"])


def state():
    """backend/hpl_nf_state.py: the stage markers the server reads."""
    import hpl_nf_state
    return hpl_nf_state


def run_shell(command: str) -> None:
    """Run a generated container command, streaming its output into the task
    log. Non-zero is fatal: the task exits with the command's own status so
    Nextflow's retry policy sees the real code (137 for an OOM kill, say)."""
    print(f"+ {command[:400]}{'...' if len(command) > 400 else ''}", flush=True)
    result = subprocess.run(["bash", "-c", command])
    if result.returncode != 0:
        print(f"FATAL: command exited {result.returncode}", file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)


def read_manifest(config: dict) -> list[Path]:
    return [Path(line.strip())
            for line in Path(config["manifest"]).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def packaged_manifest_path(config: dict) -> Path:
    """The slides packaging reads, written by the tiling gate: the run's
    manifest less any slide it was allowed to leave out (allow_incomplete)."""
    return out_dir(config) / "manifest.packaged.txt"


def write_ranges(path: Path, ranges: list[str]) -> None:
    Path(path).write_text("".join(f"{r}\n" for r in ranges), encoding="utf-8")


def parse_range(text: str) -> tuple[int, int] | None:
    """'all' -> None (the whole file, no shard), 'lo hi' -> (lo, hi)."""
    text = text.strip()
    if text == "all":
        return None
    lo, hi = text.split()
    return int(lo), int(hi)
