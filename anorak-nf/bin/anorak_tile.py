#!/usr/bin/env python3
"""Tile one slide into upstream's cws layout, then verify it is whole.

Wraps `generating_tile/save_cws.single_file_run` unchanged. Two things are
added around it, both because upstream reports success by returning:

  - the expected tile count, recomputed from the slide's own header by the
    same arithmetic as `cws_generator.generate_cws`, and checked against what
    landed on disk. Upstream has no resume and writes tiles in order, so an
    interrupted slide leaves a prefix — a directory that looks exactly like a
    finished one until something counts.

  - a refusal when the slide reports no microns-per-pixel. Upstream catches
    that, warns, and falls back to objective-power scaling, which tiles this
    slide at a different resolution to every other slide in the cohort with
    nothing in the output to say so.

Note `--output-mpp` is upstream's flag and is NOT the output resolution:
working `cws_objective_value = 20·(obj/40)·(in_mpp/out_mpp)` through the
rescale gives an effective output of exactly `2 × out_mpp` for any scanner, so
the 0.22 default yields 0.44 um/px (x20) — the paper's inference resolution.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import count_tiles, refuse  # noqa: E402

#: Written by upstream after the last tile, in `cws_all_process` order. Their
#: presence means the run reached the end; the tile count means it skipped
#: nothing on the way. Both are needed — a job killed between the final tile
#: and `param.p` passes one and fails the other.
COMPLETION_MARKERS = ("param.p", "Ss1.jpg", "FinalScan.ini")


def slide_scale(slide, slide_path: Path) -> tuple[float, float]:
    import openslide

    try:
        objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    except (KeyError, TypeError, ValueError):
        refuse(
            f"{slide_path.name} does not report both objective power and "
            f"microns-per-pixel. Upstream would fall back to objective-power "
            f"scaling and tile this slide at a resolution the rest of the "
            f"cohort does not share."
        )
    if objective <= 0 or mpp <= 0:
        refuse(f"{slide_path.name}: objective {objective}, mpp {mpp}; both must be > 0.")
    return objective, mpp


def expected_tile_count(slide_path: Path, output_mpp: float) -> int:
    """Mirror of `cws_generator.generate_cws`'s grid arithmetic.

    Opens the slide at its real path, not the staged symlink, and must: a
    .mrxs is an index file whose pixels are in a companion directory of the
    same stem beside it, and Nextflow stages only the file. Through the link
    openslide looks for that directory in the task's work directory, finds
    nothing, and fails as an unsupported format. Upstream is already handed
    the resolved parent below, so this is the same file it tiles.
    """
    import openslide

    with openslide.OpenSlide(str(slide_path.resolve())) as slide:
        objective, mpp = slide_scale(slide, slide_path)
        slide_w, slide_h = slide.level_dimensions[0]

    cws_objective_value = 20 * (objective / 40) * (mpp / output_mpp)
    side = 2000 * (objective / cws_objective_value)
    return (int(math.ceil((slide_h - side) / side + 1))
            * int(math.ceil((slide_w - side) / side + 1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--anorak-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("cws_tiling"))
    parser.add_argument("--output-mpp", type=float, default=0.22)
    args = parser.parse_args()

    sys.path.insert(0, str(args.anorak_dir / "generating_tile"))
    import save_cws  # noqa: PLC0415

    expected = expected_tile_count(args.slide, args.output_mpp)
    print(f"{args.slide.name}: expecting {expected} tiles at "
          f"{2 * args.output_mpp:.3g} um/px", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_cws.single_file_run(
        file_name=args.slide.name,
        output_dir=str(args.out_dir),
        input_dir=str(args.slide.resolve().parent),
        tif_obj=40,
        cws_objective_value=20,
        in_mpp=None,
        out_mpp=args.output_mpp,
        out_mpp_target_objective=40,
        parallel=False,
    )

    # Upstream's single_file_run dispatches on extension and returns silently
    # for one it does not recognise, so "no output directory" is a real and
    # entirely quiet outcome, not an impossible one.
    slide_dir = args.out_dir / args.slide.name
    if not slide_dir.is_dir():
        refuse(f"{args.slide.name}: no output directory. Upstream dispatches on "
               f"file type and does nothing at all for an extension it does "
               f"not know ({args.slide.suffix}).")

    produced = count_tiles(str(slide_dir))
    missing_markers = [m for m in COMPLETION_MARKERS if not (slide_dir / m).is_file()]
    if produced != expected or missing_markers:
        refuse(f"{args.slide.name}: tiling incomplete — {produced} of {expected} "
               f"tiles, missing {missing_markers or 'nothing'}.")

    print(f"{args.slide.name}: {produced} tiles, complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
