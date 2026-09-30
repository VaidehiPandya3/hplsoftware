#!/usr/bin/env python3
"""Stage 4: plan the assignment shards (and the shared query mean), run one, and verify the CSV.

Called by hpl-nf/main.nf as `python <bin>/hpl_assign.py <mode> --config <run_config.json>`.
See hpl_common.py for what every wrapper here promises.
"""

from __future__ import annotations

import argparse
import json  # noqa: F401 - used by some modes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpl_common import (  # noqa: E402,F401
    load_config, out_dir, parse_range, read_manifest, refuse, run_shell, state,
    write_ranges,
)


def _validators():
    """stage_outputs.py, from backend/ — importable only after load_config()."""
    import stage_outputs
    return stage_outputs


def _assignment_paths(config: dict) -> tuple[Path, Path, Path]:
    assignment = config["assignment"]
    out_csv = Path(assignment["out_csv"])
    return (Path(config["extraction"]["output_path"]), out_csv,
            out_csv.with_name(f"{out_csv.stem}.query_mean.npy"))


def step_assign_plan(config: dict, ranges_out: Path) -> None:
    """Check the reference and the input, and compute the shared query mean.

    The mean is computed here, once, before any shard runs, for the reason
    CLAUDE.md gives: --centering query subtracts the mean over *all* tiles, so
    a shard computing its own would project into a slightly different space
    and emit well-formed, wrong cluster IDs.
    """
    import numpy as np

    from assign_hpc_clusters import compute_query_mean
    import submit_cluster_assignment as sca

    run_dir = out_dir(config)
    assignment = config["assignment"]
    state().mark_started(run_dir, "assignment")
    projections, out_csv, mean_path = _assignment_paths(config)

    reference_info = sca.check_reference(Path(assignment["reference"]))
    rows = sca.check_projections(projections, assignment["rep_key"])
    ok, _ = _validators().validate_assignment_csv(out_csv, expected_rows=rows)
    if ok:
        print(f"assignment: {out_csv} already complete — nothing to assign")
        write_ranges(ranges_out, ["skip"])
        return

    sca._check_singularity_image(Path(assignment["singularity_image"]),
                                 assignment["singularity_bin"])
    sca._check_container_extras(Path(assignment["extras_dir"]),
                                Path(assignment["singularity_image"]),
                                assignment["singularity_bin"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    shards = int(assignment["shards"])
    if shards <= 1:
        write_ranges(ranges_out, ["all"])
        return

    mean = compute_query_mean(projections, assignment["rep_key"], 32_768, rows)
    # Checked for the same reasons submit_cluster_assignment_job checks a
    # reused mean: every shard centres on this one file.
    expected = int(reference_info["embedding_dims"])
    if mean.shape != (expected,) or not np.isfinite(mean).all():
        refuse(f"The query mean came out {mean.shape} (finite: "
                         f"{bool(np.isfinite(mean).all())}); the reference takes "
                         f"{expected}-dimensional embeddings.")
    tmp = mean_path.with_name(mean_path.name + ".partial.npy")
    np.save(tmp, mean)
    tmp.replace(mean_path)
    print(f"assignment: query mean over {rows:,} tiles -> {mean_path}")
    write_ranges(ranges_out, [f"{lo} {hi}" for lo, hi in sca.shard_ranges(rows, shards)])


def step_assign(config: dict, row_range: str, threads: int) -> None:
    import submit_cluster_assignment as sca

    assignment = config["assignment"]
    projections, out_csv, mean_path = _assignment_paths(config)
    parsed = parse_range(row_range)

    # A part that already validates is an attempt that finished and whose exit
    # code Nextflow could not read; keep it rather than recompute. Anything
    # short is left alone: the assigner resumes from its own chunk checkpoints
    # (CLAUDE.md, "Stage 4 resumes"), which is cheaper than starting over.
    if parsed is None:
        target, expected = out_csv, sca.check_projections(projections, assignment["rep_key"])
    else:
        target = out_csv.with_name(f"{out_csv.stem}.rows{parsed[0]}-{parsed[1]}{out_csv.suffix}")
        expected = parsed[1] - parsed[0]
    if _validators().validate_assignment_csv(target, expected_rows=expected)[0]:
        print(f"assignment: {target.name} already complete ({expected:,} rows) — keeping it")
        return
    command = sca._build_assignment_command(
        singularity_bin=assignment["singularity_bin"],
        singularity_image=Path(assignment["singularity_image"]),
        extras_dir=Path(assignment["extras_dir"]),
        assign_script=Path(config["backend_dir"]) / sca.ASSIGN_SCRIPT,
        reference=Path(assignment["reference"]),
        projections_h5=projections,
        out_csv=out_csv,
        rep_key=assignment["rep_key"],
        k=assignment["k"],
        batch_size=int(assignment["batch_size"]),
        validate_against=None,
        query_mean=mean_path if parsed is not None else None,
        vote=list(assignment["vote_flags"]),
        # The task's own cpus, baked into the command: --cleanenv means the
        # container cannot read SLURM_CPUS_PER_TASK, and Nextflow sets
        # --cpus-per-task from the same number, so the two cannot disagree.
        threads=threads,
        device=assignment["device"],
        row_range=parsed,
    )
    run_shell(command)


def step_assign_finish(config: dict) -> None:
    from merge_assignment_shards import merge_assignment_shards
    import submit_cluster_assignment as sca

    run_dir = out_dir(config)
    assignment = config["assignment"]
    projections, out_csv, _ = _assignment_paths(config)
    rows = sca.check_projections(projections, assignment["rep_key"])
    ok, reason = _validators().validate_assignment_csv(out_csv, expected_rows=rows)
    if not ok and int(assignment["shards"]) > 1:
        if out_csv.exists():
            out_csv.unlink()
        merge_assignment_shards(out_csv, expected_rows=rows, cleanup=True)
        ok, reason = _validators().validate_assignment_csv(out_csv, expected_rows=rows)
    if not ok:
        refuse(f"assignment finished but {out_csv} is not usable: {reason}")
    state().mark_done(run_dir, "assignment", {
        "out_csv": str(out_csv), "assignments": rows,
        "reference": assignment["reference"], "vote": assignment.get("vote"),
    })
    print(f"assignment: {rows:,} tiles classified -> {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--config', type=Path, required=True)
    plan.add_argument('--ranges-out', dest='ranges_out', type=Path, required=True)
    shard = sub.add_parser('shard')
    shard.add_argument('--config', type=Path, required=True)
    shard.add_argument('--range', dest='row_range', required=True)
    shard.add_argument('--threads', type=int, required=True)
    finish = sub.add_parser('finish')
    finish.add_argument('--config', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode == 'plan':
        step_assign_plan(config, args.ranges_out)
    if args.mode == 'shard':
        step_assign(config, args.row_range, args.threads)
    if args.mode == 'finish':
        step_assign_finish(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
