#!/usr/bin/env python3
"""Time Stage 4 on a slice of the real data, so the shape of the full run is
chosen from measurement rather than from arithmetic.

Why this exists: every estimate of this stage has been wrong so far, in both
directions. FLOP counting said the k-NN search would be ~99% of the time; the
first real run's CPU accounting put it nearer 30%, which moves the answer from
"buy a GPU" to "overlap the I/O". A thread count that looked correct in the
submitter ran on one core inside the container. The only reliable number is a
measured one, and the cheapest way to measure is to run the real code over a
few hundred thousand rows.

It runs `assign_hpc_clusters.py` as a subprocess — the same script the Slurm job
runs, with the same flags — rather than importing pieces of it. A benchmark that
reimplements the thing it measures eventually measures the reimplementation.

What comes back per configuration: tiles/s, the phase split the assigner prints
itself, and the extrapolated wall clock for the whole cohort at N shards. The
extrapolation is deliberately shown per shard count, because shards multiply
every phase while threads and a GPU only touch the search.

    # One configuration, 200k rows, using the shared query mean:
    python benchmark_assignment.py \\
        --projections-h5 <results>/.../hdf5_DS_he_train.h5 \\
        --query-mean /path/to/DS_hpc_assignments.query_mean.npy \\
        --rows 200000

    # Sweep threads and devices, and project onto 32 shards:
    python benchmark_assignment.py --projections-h5 ... --query-mean ... \\
        --threads 1 2 8 --device cpu gpu --shards 32

THE QUERY MEAN IS NOT OPTIONAL for a slice. `--centering query` centres on the
mean of *every* query, so a slice that computed its own mean would time a
different computation from the one the real run does, and would produce
different cluster IDs from the same input. assign_hpc_clusters.py refuses that
outright; this passes `--query-mean` through, and says so if it is missing.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ASSIGN_SCRIPT = Path(__file__).resolve().with_name("assign_hpc_clusters.py")

#: Enough rows to be past index construction and cache warm-up without being a
#: job in its own right. At the production reference's measured ~49 tiles/s per
#: core this is a couple of minutes single-threaded.
DEFAULT_ROWS = 100_000

# The assigner prints both the elapsed time (one decimal) and its own computed
# rate. Read the rate, not the division: a sub-second run prints "0.0s", and
# dividing by that produced 2e12 tiles/s the first time this ran.
_RATE_RE = re.compile(
    r"Assigned\s*:\s*([\d,]+)\s+tiles in ([\d.]+)s\s*\(([\d,]+)\s*tiles/s\)")

#: Below this, the measurement is noise: index construction, the first chunk's
#: cache misses and process startup all land inside it.
_MIN_TIMED_SECONDS = 5.0
_PHASE_RE = re.compile(r"^Time spent:\s*(.+)$", re.MULTILINE)
_THREADS_RE = re.compile(r"^Threads\s*:\s*(\d+)\s*\((.+)\)$", re.MULTILINE)
_BACKEND_RE = re.compile(r"^Backend\s*:\s*(\S+?),", re.MULTILINE)


def _format_hours(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def run_one(*, projections_h5: Path, reference: Path | None, query_mean: Path | None,
            rows: int, threads: int, device: str, batch_size: int,
            chunk_size: int, out_dir: Path, extra_args: list[str] | None = None) -> dict:
    """One timed run of the real assigner over rows [0, rows).

    Returns what it printed about itself, plus the wall clock this process
    observed — which includes reference loading and index construction, unlike
    the assigner's own figure. Both are reported: the first is what a shard
    actually costs, the second is what scales with tiles.
    """
    out_csv = out_dir / f"bench_{device}_{threads}t.csv"
    command = [
        sys.executable, str(ASSIGN_SCRIPT),
        "--h5", str(projections_h5),
        "--out", str(out_csv),
        "--row-start", "0", "--row-stop", str(rows),
        "--batch-size", str(batch_size),
        "--chunk-size", str(chunk_size),
        "--device", device,
        "--progress", str(max(rows // 4, 1)),
        *(extra_args or []),
    ]
    if reference is not None:
        command += ["--reference", str(reference)]
    if query_mean is not None:
        command += ["--query-mean", str(query_mean)]

    # The thread count reaches the assigner the same way the Slurm job delivers
    # it: through OMP_NUM_THREADS, capped by the process's affinity mask on the
    # far side. Setting it here rather than passing a flag keeps the benchmark
    # measuring the real mechanism, including its cap.
    env = {**os.environ, "OMP_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}

    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - started

    record = {"device": device, "threads_requested": threads, "wall_seconds": wall,
              "rows": rows, "ok": result.returncode == 0,
              "stdout": result.stdout, "stderr": result.stderr,
              "tiles_per_second": None, "assign_seconds": None,
              "phases": None, "threads_used": None, "backend": None,
              "reliable": False, "rows_assigned": None}
    if result.returncode != 0:
        return record

    match = _RATE_RE.search(result.stdout)
    if match:
        record["rows_assigned"] = int(match.group(1).replace(",", ""))
        record["assign_seconds"] = float(match.group(2))
        record["tiles_per_second"] = float(match.group(3).replace(",", ""))
        record["reliable"] = record["assign_seconds"] >= _MIN_TIMED_SECONDS
    phase = _PHASE_RE.search(result.stdout)
    if phase:
        record["phases"] = phase.group(1).strip()
    threads_line = _THREADS_RE.search(result.stdout)
    if threads_line:
        record["threads_used"] = int(threads_line.group(1))
        record["threads_note"] = threads_line.group(2)
    backend = _BACKEND_RE.search(result.stdout)
    if backend:
        record["backend"] = backend.group(1)
    return record


def total_rows(projections_h5: Path, rep_key: str = "z_latent") -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from assign_hpc_clusters import query_row_count
    return query_row_count(projections_h5, rep_key)


def report(records: list[dict], cohort_rows: int, shard_counts: list[int]) -> None:
    print(f"\n{'device':>7} {'threads':>8} {'used':>5} {'tiles/s':>10} "
          f"{'per-shard phases':<44}")
    print("-" * 80)
    for r in records:
        if not r["ok"]:
            reason = (r["stderr"] or "").strip().splitlines()
            print(f"{r['device']:>7} {r['threads_requested']:>8} {'-':>5} "
                  f"{'FAILED':>10} {reason[-1][:44] if reason else '':<44}")
            continue
        flag = "" if r.get("reliable") else "  (too fast to time)"
        print(f"{r['device']:>7} {r['threads_requested']:>8} "
              f"{r['threads_used'] or '?':>5} "
              f"{r['tiles_per_second'] or 0:>10,.0f} "
              f"{(r['phases'] or '')[:44]:<44}{flag}")

    good = [r for r in records if r["ok"] and r["tiles_per_second"]]
    if not good:
        print("\nNothing measured — see the errors above.")
        return
    if not any(r.get("reliable") for r in good):
        fastest = max(r["assign_seconds"] or 0 for r in good)
        print(f"\nEvery configuration finished in under {_MIN_TIMED_SECONDS:.0f}s "
              f"(fastest slice took {fastest:.1f}s), which is index construction "
              f"and cache warm-up rather than throughput. Raise --rows by "
              f"roughly {max(int(_MIN_TIMED_SECONDS / max(fastest, 0.1)), 5)}x "
              f"and run it again; the projections below would be extrapolated "
              f"from noise.")
        return

    print(f"\nProjected wall clock for {cohort_rows:,} tiles "
          f"(search-only scaling; shards divide every phase):")
    header = "  " + " ".join(f"{f'{n} shards':>12}" for n in shard_counts)
    print(f"{'config':>18}{header}")
    for r in good:
        per_shard = r["tiles_per_second"]
        cells = " ".join(
            f"{_format_hours(cohort_rows / (per_shard * n)):>12}"
            for n in shard_counts)
        label = f"{r['device']}/{r['threads_used'] or r['threads_requested']}t"
        print(f"{label:>18}  {cells}")

    print("\nRead the phase split before choosing. Shards divide every phase; "
          "threads and a GPU only touch the search, so a run where search is a "
          "third of the time has a hard ceiling of about 1.5x from either.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projections-h5", type=Path, required=True,
                    help="The real projections .h5 from feature extraction.")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Reference .npz. Defaults to the configured one.")
    ap.add_argument("--query-mean", type=Path, default=None,
                    help="Shared query mean .npy. Required for --centering "
                         "query (the default), because a slice must not centre "
                         "on its own mean — see the module docstring. Compute "
                         "one with `assign_hpc_clusters.py --precompute-mean`.")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"Tiles to time, from the start of the file "
                         f"(default {DEFAULT_ROWS:,}).")
    ap.add_argument("--threads", type=int, nargs="+", default=[2],
                    help="Thread counts to sweep (default: 2).")
    ap.add_argument("--device", nargs="+", default=["auto"],
                    choices=["auto", "cpu", "gpu"],
                    help="Devices to sweep (default: auto).")
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 8, 32, 64],
                    help="Shard counts to project onto.")
    ap.add_argument("--batch-size", type=int, default=16_384)
    ap.add_argument("--chunk-size", type=int, default=32_768)
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Where the throwaway CSVs go. Defaults to a temporary "
                         "directory that is removed afterwards.")
    ap.add_argument("--keep-output", action="store_true",
                    help="Leave the benchmark CSVs in place for inspection.")
    args = ap.parse_args()

    if not args.projections_h5.is_file():
        print(f"No such projections .h5: {args.projections_h5}", file=sys.stderr)
        return 1
    if args.query_mean is None:
        print(
            "Refusing to benchmark a slice without --query-mean.\n\n"
            "--centering query centres on the mean of every query, so a slice "
            "that computed its own mean would time a different computation from "
            "the real run and produce different cluster IDs from the same "
            "input. Compute the mean once:\n\n"
            f"    python {ASSIGN_SCRIPT.name} --reference <ref.npz> \\\n"
            f"        --h5 {args.projections_h5} \\\n"
            f"        --precompute-mean /path/to/query_mean.npy\n\n"
            "then pass it with --query-mean. (Or pass --centering reference / "
            "none through, which are shard-independent by construction.)",
            file=sys.stderr)
        return 1

    cohort = total_rows(args.projections_h5)
    rows = min(args.rows, cohort)
    print(f"Projections : {args.projections_h5}")
    print(f"  cohort      {cohort:,} tiles")
    print(f"  timing      {rows:,} tiles per configuration "
          f"({rows / cohort * 100:.2f}% of it)")

    out_dir = args.tmp_dir or Path(tempfile.mkdtemp(prefix="hpl_bench_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for device in args.device:
            for threads in args.threads:
                print(f"\n=== {device}, {threads} thread(s) ===", flush=True)
                record = run_one(
                    projections_h5=args.projections_h5, reference=args.reference,
                    query_mean=args.query_mean, rows=rows, threads=threads,
                    device=device, batch_size=args.batch_size,
                    chunk_size=args.chunk_size, out_dir=out_dir)
                records.append(record)
                if record["ok"]:
                    print(f"    {record['tiles_per_second']:,.0f} tiles/s"
                          f"{'' if record['reliable'] else ' (too fast to time)'} "
                          f"(backend {record['backend']}, "
                          f"{record['threads_used']} thread(s) used, "
                          f"{record['wall_seconds']:.0f}s wall including setup)")
                else:
                    print("    FAILED:", (record["stderr"] or "").strip()[-400:],
                          file=sys.stderr)
        report(records, cohort, args.shards)
    finally:
        if not args.keep_output and args.tmp_dir is None:
            shutil.rmtree(out_dir, ignore_errors=True)
    return 0 if any(r["ok"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
