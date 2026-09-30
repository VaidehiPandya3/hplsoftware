#!/usr/bin/env python3
"""Stage 3: plan the encoder shards, run one, and verify the merged embeddings.

Called by hpl-nf/main.nf as `python <bin>/hpl_extract.py <mode> --config <run_config.json>`.
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


def _extraction_paths(config: dict) -> tuple[Path, Path, int | None]:
    h5_path = Path(config["packaging"]["h5_path"])
    output = Path(config["extraction"]["output_path"])
    return h5_path, output, _validators().packaged_h5_rows(h5_path)


def step_extract_plan(config: dict, ranges_out: Path) -> None:
    """Decide the shards, clear stale leftovers, and check the runtime is there.

    Clearing matters more here than anywhere: the encoder treats an existing
    output as done and then crashes on an unbound local (see
    validate_extraction_output), so one leftover makes every retry fail in
    seconds. The same rule submit_feature_extraction_job applies.
    """
    import submit_feature_extraction as sfe

    run_dir = out_dir(config)
    extraction = config["extraction"]
    state().mark_started(run_dir, "extraction")
    h5_path, output, rows = _extraction_paths(config)
    if rows is None:
        refuse(f"Cannot read the packaged .h5's tile count: {h5_path}")

    if output.exists():
        ok, reason = sfe.validate_extraction_output(output, expected_rows=rows)
        if ok:
            print(f"extraction: {output} already complete — nothing to encode")
            write_ranges(ranges_out, ["skip"])
            return
        output.unlink()
        print(f"extraction: cleared stale output {output} ({reason})")

    # The submit-time checks, run here because this is the last moment before
    # a GPU is requested — the same refusals the per-stage submitter makes.
    sfe._check_hpl_repo_dir(Path(extraction["hpl_repo_dir"]))
    sfe._check_checkpoint(extraction["checkpoint"])
    sfe._check_singularity_image(Path(extraction["singularity_image"]),
                                 extraction["singularity_bin"])
    sfe._check_container_extras(Path(extraction["extras_dir"]),
                                Path(extraction["singularity_image"]),
                                extraction["singularity_bin"])

    shards = int(extraction["shards"])
    if shards <= 1:
        write_ranges(ranges_out, ["all"])
        return
    bounds = sfe.shard_ranges(rows, shards)
    for lo, hi in bounds:
        part = sfe.shard_output_path(output, lo, hi)
        if part.exists():
            part.unlink()
            print(f"extraction: cleared stale shard part {part.name}")
    write_ranges(ranges_out, [f"{lo} {hi}" for lo, hi in bounds])


def step_extract(config: dict, row_range: str) -> None:
    import submit_feature_extraction as sfe

    extraction = config["extraction"]
    h5_path, output, rows = _extraction_paths(config)
    parsed = parse_range(row_range)

    # This task's own target, cleared or accepted before the encoder sees it.
    # The plan step cleared leftovers once, before the first attempt — but
    # errorStrategy retries this task, and an attempt killed partway (OOM,
    # preemption, node loss) leaves a half-written file at exactly this path.
    # The encoder treats any existing output as finished and then crashes on
    # an unbound local, so without this every retry failed in seconds. A target
    # that already validates is kept: that is an attempt that finished and whose
    # exit code Nextflow could not read (Integer.MAX_VALUE on CephFS).
    target, expected = ((output, rows) if parsed is None
                        else (sfe.shard_output_path(output, *parsed), parsed[1] - parsed[0]))
    if target.exists():
        ok, reason = sfe.validate_extraction_output(target, expected_rows=expected)
        if ok:
            print(f"extraction: {target.name} already complete ({expected:,} rows) — keeping it")
            return
        target.unlink()
        print(f"extraction: cleared {target.name} left by an earlier attempt ({reason})")
    command = sfe._build_extraction_command(
        singularity_bin=extraction["singularity_bin"],
        singularity_image=Path(extraction["singularity_image"]),
        hpl_repo_dir=Path(extraction["hpl_repo_dir"]),
        real_hdf5_path=h5_path,
        checkpoint=extraction["checkpoint"],
        dataset_name=extraction["dataset_name"],
        model=extraction["model"],
        marker=extraction["marker"],
        z_dim=int(extraction["z_dim"]),
        img_size=int(extraction["img_size"]),
        batch_size=int(extraction["batch_size"]),
        extras_dir=Path(extraction["extras_dir"]),
        row_range=parsed,
    )
    run_shell(command)


def step_extract_finish(config: dict) -> None:
    import submit_feature_extraction as sfe
    from merge_projection_shards import merge_projection_shards

    run_dir = out_dir(config)
    h5_path, output, rows = _extraction_paths(config)
    ok, reason = (sfe.validate_extraction_output(output, expected_rows=rows)
                  if output.exists() else (False, "not written"))
    if not ok and int(config["extraction"]["shards"]) > 1:
        if output.exists():
            output.unlink()
        # Checked against the packaged input's tile count rather than the sum
        # of the parts: a missing final shard leaves no gap to detect.
        merge_projection_shards(output, expected_rows=rows, cleanup=True)
        ok, reason = sfe.validate_extraction_output(output, expected_rows=rows)
    if not ok:
        refuse(f"extraction finished but {output} is not usable: {reason}")
    state().mark_done(run_dir, "extraction", {
        "output_path": str(output), "embeddings": rows,
        "checkpoint": config["extraction"]["checkpoint"],
    })
    print(f"extraction: {rows:,} embeddings in {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--config', type=Path, required=True)
    plan.add_argument('--ranges-out', dest='ranges_out', type=Path, required=True)
    shard = sub.add_parser('shard')
    shard.add_argument('--config', type=Path, required=True)
    shard.add_argument('--range', dest='row_range', required=True)
    finish = sub.add_parser('finish')
    finish.add_argument('--config', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode == 'plan':
        step_extract_plan(config, args.ranges_out)
    if args.mode == 'shard':
        step_extract(config, args.row_range)
    if args.mode == 'finish':
        step_extract_finish(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
