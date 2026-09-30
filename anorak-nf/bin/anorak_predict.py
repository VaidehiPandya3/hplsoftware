#!/usr/bin/env python3
"""Segment growth patterns for one tiled slide (GPU step).

Wraps `inference_slide/predict_gp.generate_gp` unchanged: Reinhard colour
normalisation against the repository's own `target_gp.jpg`, then 768-pixel
patches at stride 192 through the trained model, merged and written as one
colour PNG per tile.

The slide is selected by name rather than by upstream's integer `nfile` — see
`anorak_common`. The tile count is checked afterwards because `generate_gp`
skips any tile whose PNG already exists and reports nothing when it skips them
all, so a task that did no work and a task that did all of it look identical
from the outside.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import (  # noqa: E402
    checkpoint_problem, count_masks, count_tiles, refuse, single_slide_pattern,
)


def check_one_gpu() -> None:
    """Refuse unless TensorFlow sees exactly the one card Slurm gave this task.

    Slurm tells a `--gres=gpu:1` task which card is its own through
    CUDA_VISIBLE_DEVICES, and unless Slurm's ConstrainDevices is on nothing
    else enforces it: every card on the node is openable. Nextflow
    launches the container through `env -`, passing on only what
    singularity.envWhitelist names; before that was set, every PREDICT_GP on a
    node saw every GPU, TensorFlow reserved memory on all of them and computed
    on GPU 0, and the second task to land on a node died in ResourceExhausted.
    That is a setting, not a slide, so more than one card is a refusal: every
    retry would land the same way. None visible is left to the retry, because a
    CUDA context that failed to come up does come up on another node — and
    because, left alone, generate_gp would run a whole slide on the CPU.

    Exactly one is accepted whatever CUDA_VISIBLE_DEVICES says — set to the
    assigned index, unset under a device cgroup that shows the task only its
    own card, or remapped to 0 inside one. The count is the property that
    matters; the variable is printed for the log, never compared.

    Called before predict_gp is imported, which also matters: TensorFlow reads
    CUDA_VISIBLE_DEVICES once, when it first initialises, so anything the
    upstream module does to that variable at import time comes too late to
    move this task onto another card.
    """
    import tensorflow as tf  # noqa: PLC0415 — predict_gp imports it anyway

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"CUDA_VISIBLE_DEVICES={visible!r}; TensorFlow sees {len(gpus)} GPU(s)",
          flush=True)
    if len(gpus) > 1:
        refuse(f"TensorFlow sees {len(gpus)} GPUs, and this task was given one "
               f"(CUDA_VISIBLE_DEVICES={visible!r}). The container is not "
               f"receiving Slurm's CUDA_VISIBLE_DEVICES — check "
               f"singularity.envWhitelist in conf/beatson.config. Left to run, "
               f"every task on this node computes on GPU 0.")
    if not gpus:
        print("No GPU visible to TensorFlow in a GPU task — failing so the "
              "task is retried, rather than segmenting a slide on the CPU.",
              file=sys.stderr, flush=True)
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cws-dir", required=True,
                        help="directory holding this slide's cws folder")
    parser.add_argument("--slide-name", required=True,
                        help="the cws folder's name, i.e. the slide filename")
    parser.add_argument("--anorak-dir", type=Path, required=True)
    parser.add_argument("--out-dir", default="gp_masks")
    parser.add_argument("--patch-size", type=int, default=768)
    parser.add_argument("--patch-stride", type=int, default=192)
    parser.add_argument("--n-class", type=int, default=7)
    parser.add_argument("--no-colour-norm", action="store_true",
                        help="skip Reinhard normalisation; the model was "
                             "trained on normalised input, so this is for "
                             "diagnosis only")
    args = parser.parse_args()

    # generate_gp loads the checkpoint from <anorak-dir>/models/AIgrading_anorak.h5
    # by a path derived from its own __file__, so it has to be there and a
    # missing one surfaces as a Keras error about a path nobody passed. Either
    # form load_model takes is accepted — see checkpoint_problem, which is the
    # same test main.nf applies at launch.
    checkpoint = args.anorak_dir / "models" / "AIgrading_anorak.h5"
    problem = checkpoint_problem(checkpoint)
    if problem:
        refuse(f"{problem}. The Zenodo checkpoint is at "
               f"https://zenodo.org/records/15272883 — generate_gp() builds "
               f"that path itself and cannot be pointed elsewhere.")

    check_one_gpu()

    sys.path.insert(0, str(args.anorak_dir / "inference_slide"))
    from predict_gp import generate_gp  # noqa: PLC0415

    pattern = single_slide_pattern(args.cws_dir, args.slide_name)
    expected = count_tiles(os.path.join(args.cws_dir, args.slide_name))
    if expected == 0:
        refuse(f"{args.slide_name}: no Da*.jpg in the staged cws directory.")

    generate_gp(
        datapath=args.cws_dir,
        save_dir=args.out_dir,
        file_pattern=pattern,
        nfile=0,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        nClass=args.n_class,
        color_norm=not args.no_colour_norm,
    )

    produced = count_masks(os.path.join(args.out_dir, args.slide_name))
    if produced != expected:
        refuse(f"{args.slide_name}: {produced} masks for {expected} tiles. "
               f"A stitch over a short set produces a slide-shaped mask with "
               f"holes that read as background, which is indistinguishable "
               f"from tissue the model found nothing in.")

    print(f"{args.slide_name}: {produced} masks, complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
