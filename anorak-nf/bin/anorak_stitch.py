#!/usr/bin/env python3
"""Stitch one slide's tile masks to Ss1 scale, then post-process.

Wraps `ss1_stich` and `ss1_final` back to back. They are one process rather
than two because the intermediate is a single PNG that nothing else reads, and
splitting them would stage a whole-slide image between tasks to no purpose.

Both are guarded the same way. `ss1_stich` writes its output every 20 tiles, so
a partial stitch is a readable PNG of the right dimensions — and its own
"already exists, skip" check would then treat that partial file as finished.
Isolating each task from previous output is what makes that unreachable here,
and the mask count is checked against the tile count before either runs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import count_masks, count_tiles, refuse, single_slide_pattern  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cws-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--slide-name", required=True)
    parser.add_argument("--anorak-dir", type=Path, required=True)
    parser.add_argument("--ss1-dir", default="ss1")
    parser.add_argument("--ss1-final-dir", default="ss1_final")
    args = parser.parse_args()

    sys.path.insert(0, str(args.anorak_dir / "inference_slide"))
    from ss1_stich import ss1_stich  # noqa: PLC0415
    from ss1_final import ss1_final  # noqa: PLC0415

    pattern = single_slide_pattern(args.cws_dir, args.slide_name)

    tiles = count_tiles(os.path.join(args.cws_dir, args.slide_name))
    masks = count_masks(os.path.join(args.mask_dir, args.slide_name))
    if tiles == 0:
        refuse(f"{args.slide_name}: no tiles staged.")
    if masks != tiles:
        refuse(f"{args.slide_name}: {masks} masks for {tiles} tiles. Stitching "
               f"a short set silently fills the gaps with background.")

    # ss1_stich reads param.p for the grid geometry; without it the placement
    # arithmetic has nothing to work from.
    param = Path(args.cws_dir) / args.slide_name / "param.p"
    if not param.is_file():
        refuse(f"{args.slide_name}: no param.p — the tiling run that produced "
               f"these tiles did not finish.")

    ss1_stich(cws_folder=args.cws_dir, annotated_dir=args.mask_dir,
              output_dir=args.ss1_dir, nfile=0, file_pattern=pattern)

    stitched = Path(args.ss1_dir) / f"{args.slide_name}_Ss1.png"
    if not stitched.is_file() or stitched.stat().st_size == 0:
        refuse(f"{args.slide_name}: stitching produced no usable {stitched.name}.")

    ss1_final(cws_folder=args.cws_dir, ss1_dir=args.ss1_dir,
              ss1_final_dir=args.ss1_final_dir, nfile=0, file_pattern=pattern)

    final = Path(args.ss1_final_dir) / f"{args.slide_name}_Ss1.png"
    if not final.is_file() or final.stat().st_size == 0:
        refuse(f"{args.slide_name}: post-processing produced no usable output.")

    print(f"{args.slide_name}: stitched and post-processed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
