#!/usr/bin/env python3
"""Stage 2: package the cohort's tiles into one .h5, and verify it.

Called by hpl-nf/main.nf as `python <bin>/hpl_package.py <mode> --config <run_config.json>`.
See hpl_common.py for what every wrapper here promises.
"""

from __future__ import annotations

import argparse
import json  # noqa: F401 - used by some modes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hpl_common import (  # noqa: E402,F401
    load_config, out_dir, packaged_manifest_path, parse_range, read_manifest, refuse,
    run_shell, state,
    write_ranges,
)


def _validators():
    """stage_outputs.py, from backend/ — importable only after load_config()."""
    import stage_outputs
    return stage_outputs


def step_package(config: dict, processes: int) -> None:
    from make_hpl_hdf5 import package_to_h5

    run_dir = out_dir(config)
    packaging = config["packaging"]
    h5_path = Path(packaging["h5_path"])
    state().mark_started(run_dir, "packaging")

    manifest = packaged_manifest_path(config)
    if not manifest.is_file():
        refuse(f"{manifest} does not exist — the tiling gate writes it, so packaging "
               f"is running without a verified tiling stage.")

    partial = h5_path.with_name(h5_path.name + ".partial")
    ok, _ = _validators().validate_packaged_h5(h5_path) if h5_path.is_file() else (False, "")
    if ok and not partial.exists():
        print(f"packaging: {h5_path} already complete — not repackaging")
    else:
        # resume=None continues a checkpoint this run's own earlier attempt left,
        # which is the point of retrying a packaging task that timed out.
        info = package_to_h5(
            manifest_path=manifest,
            tile_dir=Path(config["tile_dir"]),
            tile_dataset_name=config["dataset_name"],
            output_root=Path(packaging["output_root"]),
            dataset_name=packaging["h5_dataset_name"],
            marker=packaging["marker"],
            split=packaging["split"],
            tile_size=packaging["tile_size"],
            n_processes=processes,
            threads_per_process=packaging["threads_per_process"],
            resume=None,
        )
        if Path(info["output_h5_path"]) != h5_path:
            refuse(
                f"make_hpl_hdf5 wrote {info['output_h5_path']}, but this run "
                f"recorded {h5_path} as its .h5. Refusing to mark packaging done "
                f"against a path nothing downstream will read."
            )

    ok, reason = _validators().validate_packaged_h5(h5_path)
    if not ok:
        refuse(f"packaging finished but {h5_path} is not usable: {reason}")
    rows = _validators().packaged_h5_rows(h5_path)
    state().mark_done(run_dir, "packaging", {"h5_path": str(h5_path), "tiles": rows})
    print(f"packaging: {rows:,} tiles in {h5_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    package = sub.add_parser('package')
    package.add_argument('--config', type=Path, required=True)
    package.add_argument('--processes', type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode == 'package':
        step_package(config, args.processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
