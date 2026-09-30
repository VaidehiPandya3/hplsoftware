#!/usr/bin/env python3
"""Stage 1: tile one slide, or check every slide in the manifest is tiled.

Called by hpl-nf/main.nf as `python <bin>/hpl_tile.py <mode> --config <run_config.json>`.
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


def step_tile(config: dict, slide: str) -> None:
    from submit_mask_tile_slurm import tile_one_slide

    state().mark_started(out_dir(config), "tiling")
    result = tile_one_slide(
        Path(slide),
        mask_dir=Path(config["mask_dir"]),
        tile_dir=Path(config["tile_dir"]),
        dataset_name=config["dataset_name"],
        **config["tiling"],
    )
    print(json.dumps(result), flush=True)


def step_tiling_gate(config: dict) -> None:
    """Every slide in the manifest tiled, checked the strict way.

    Strict meaning the metadata CSV's row count is read against the summary,
    the check _tiled_coverage skips for speed. This runs once per run, after
    the last slide, so it can afford the parse — and it is the one place a
    slide whose CSV stopped short would otherwise slip into packaging.
    """
    from submit_mask_tile_slurm import slide_output_paths, tiling_output_complete

    run_dir = out_dir(config)
    slides = read_manifest(config)
    succeeded, zero_tile, incomplete = [], [], []
    for slide in slides:
        paths = slide_output_paths(slide, Path(config["mask_dir"]),
                                   Path(config["tile_dir"]), config["dataset_name"])
        if not tiling_output_complete(paths["metadata"], paths["summary"],
                                      paths["slide_tile_dir"], verify_row_count=True):
            incomplete.append(slide.name)
            continue
        saved = json.loads(paths["summary"].read_text(encoding="utf-8")).get("saved_tiles", 0)
        (succeeded if saved else zero_tile).append(paths["slide_tile_dir"].name)

    allow_incomplete = bool(config.get("allow_incomplete"))
    if incomplete and not allow_incomplete:
        refuse(
            f"{len(incomplete)} of {len(slides)} slides are not completely tiled: "
            f"{', '.join(incomplete[:10])}{' ...' if len(incomplete) > 10 else ''}. "
            f"Packaging now would build an .h5 missing them. Their TILE tasks' "
            f"logs say why; if these slides cannot be read, resume with "
            f"'Package without slides that fail to tile'."
        )
    if not succeeded:
        refuse(
            f"None of the {len(slides)} slides saved a tile — every one ran and "
            f"found no tissue above min_tissue={config['tiling'].get('min_tissue')}%. "
            f"There is nothing to package; lower the threshold or check the masks."
        )
    # What packaging reads: every slide whose tiling is complete — all of them,
    # unless the run was explicitly allowed to leave the failures out, in which
    # case they are named in the marker and shown on the tiling step rather
    # than disappearing from the .h5 with nothing to say so.
    complete = [s for s in slides if s.name not in set(incomplete)]
    packaged = packaged_manifest_path(config)
    tmp = packaged.with_name(packaged.name + ".tmp")
    tmp.write_text("".join(f"{s}\n" for s in complete), encoding="utf-8")
    tmp.replace(packaged)
    state().mark_done(run_dir, "tiling", {
        "slides": len(slides),
        "succeeded": len(succeeded),
        "zero_tile": len(zero_tile),
        "zero_tile_slides": zero_tile[:200],
        "excluded": len(incomplete),
        "excluded_slides": incomplete[:200],
        "packaged_manifest": str(packaged),
    })
    print(f"tiling: {len(succeeded)} slides with tiles, {len(zero_tile)} with none"
          + (f", {len(incomplete)} left out (failed to tile)" if incomplete else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='mode', required=True)
    tile = sub.add_parser('tile')
    tile.add_argument('--config', type=Path, required=True)
    tile.add_argument('--slide', required=True)
    gate = sub.add_parser('gate')
    gate.add_argument('--config', type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mode == 'tile':
        step_tile(config, args.slide)
    if args.mode == 'gate':
        step_tiling_gate(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
