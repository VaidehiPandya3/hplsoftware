#!/usr/bin/env python3
"""Can this container load ANORAK's checkpoint and run a patch through it?

Run this inside the image you intend to give `process_gpu`, before anything
else. It is the one question in this pipeline with no cheap answer further
down: everything else — the slide list, the tiling, the stitching — is
plumbing that fails loudly, while a checkpoint that will not load fails after
you have built an image, filled in a config and queued a cohort.

    singularity exec --nv <image>.sif python3 check_anorak_model.py \
        --checkpoint /path/to/AIgrading/models/AIgrading_anorak.h5

Why it is in doubt. ANORAK pins tensorflow-gpu 2.2 / keras 2.4.3, and TF 2.2
ships CUDA 10.1, which supports compute capability up to 7.5 (Turing). An A100
is 8.0 and H100/H200 are 9.0, so the pinned stack cannot drive any modern card
— the same wall the HPL encoder hit, with the same answer, an NGC image
carrying a backported CUDA. But NGC's CUDA-12 TensorFlow images are TF 2.11+,
and a Keras 2.4.3 `.h5` is not guaranteed to load under them. That is the trade
this script measures: the container that can talk to the GPU is not the
container the checkpoint was written by, and only one of those problems can be
solved by choosing a different image.

Three things are checked, in order of how expensive they are to discover late:
the checkpoint loads at all; it produces the right output shape; and the GPU is
actually visible and used. A pass on the first two with no GPU is still a
useful result — it means the weights are fine and only the CUDA side is wrong.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

#: What ANORAK's segmentation head emits: one channel per class, background
#: plus the six growth patterns.
EXPECTED_CLASSES = 7

#: The window predict_gp cuts out of each 2000x2000 tile, at stride 192.
#: NOT the model's input size: `Patches.extract_patches_img_label` resizes every
#: 768 window down to 384 before the network sees it, and `merge_patches`
#: resizes the predictions back up. Both numbers are hardcoded upstream. So the
#: size to feed the model directly is the model's own input shape, which is why
#: this is only used to describe the geometry, never to build a test patch.
WINDOW_SIZE = 768
WINDOW_STRIDE = 192
#: Tiles per slide, order of magnitude, for the runtime estimate below.
TILES_PER_SLIDE = 500


def _host_has_nvidia_driver() -> bool:
    """Whether this machine has a GPU driver, independent of TensorFlow.

    Two cheap signals, either of which is enough: the device nodes the driver
    creates, and the tool it installs. Both absent means there is nothing here
    to talk to — a login node — and no conclusion about the container can be
    drawn from it.
    """
    import glob
    import shutil

    return bool(glob.glob("/dev/nvidia[0-9]*")) or shutil.which("nvidia-smi") is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="AIgrading/models/AIgrading_anorak.h5")
    parser.add_argument("--patch-size", type=int,
                        help="size of the test patch; defaults to the model's "
                             "own input shape, which is what it must be")
    parser.add_argument("--require-gpu", action="store_true",
                        help="fail if no GPU is visible, rather than reporting it")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="check the load and the shapes on a machine with "
                             "no driver at all — for a CPU-only container, e.g. "
                             "verifying a converted SavedModel")
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf

    print("=" * 64)
    print(f"TensorFlow      {tf.__version__}")
    try:
        build = tf.sysconfig.get_build_info()
        print(f"CUDA / cuDNN    {build.get('cuda_version')} / {build.get('cudnn_version')}")
    except Exception:
        print("CUDA / cuDNN    (not reported by this build)")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible    {len(gpus)}")
    for gpu in gpus:
        try:
            detail = tf.config.experimental.get_device_details(gpu)
            capability = detail.get("compute_capability")
            print(f"  {gpu.name}  {detail.get('device_name')}  "
                  f"compute capability {capability[0]}.{capability[1]}"
                  if capability else f"  {gpu.name}  {detail.get('device_name')}")
        except Exception:
            print(f"  {gpu.name}")
    print("=" * 64)

    if not gpus:
        # Named rather than left to be inferred from a slow run: a container
        # whose TensorFlow predates the card reports zero GPUs and then works
        # perfectly on CPU, which is indistinguishable from a correct setup
        # until someone notices a slide taking hours.
        #
        # But "no GPU" has two completely different causes, and the useless one
        # is far more common: being on a login node, where there is no driver
        # to find and singularity's --nv says so in a warning that scrolls past.
        # Reporting a CUDA mismatch there would send someone rebuilding an
        # image that was fine. So the host is checked first.
        if not _host_has_nvidia_driver() and not args.allow_no_gpu:
            print("\nFAIL: this host has no NVIDIA driver at all — no "
                  "/dev/nvidia* and no nvidia-smi.\n"
                  "      That is what a login node looks like, and it says "
                  "nothing about the container.\n"
                  "      Re-run it on a GPU node:\n"
                  "        srun -p gpu --gres=gpu:1 -t 20 --pty \\\n"
                  "            singularity exec --nv <image>.sif python3 "
                  "check_anorak_model.py --checkpoint <path>\n"
                  "      Pass --allow-no-gpu if you meant to check the load "
                  "and shapes on a CPU-only machine.",
                  file=sys.stderr)
            return 1
        message = ("No GPU is visible to TensorFlow, though this host has a "
                   "driver. That points at the container: this build's CUDA is "
                   "probably older than the card — TF 2.2 is CUDA 10.1 and "
                   "cannot see an A100 (8.0) or H100/H200 (9.0).")
        if args.require_gpu:
            print(f"\nFAIL: {message}", file=sys.stderr)
            return 1
        print(f"\nNote: {message}\n      Continuing on CPU — the load and shape "
              f"checks below are still worth having.\n")

    # The pipeline's own test, not a local one: this used to ask exists(),
    # which passes a directory that the GPU task's is_file() then refused.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
    from anorak_common import checkpoint_problem  # noqa: PLC0415
    problem = checkpoint_problem(args.checkpoint)
    if problem:
        print(f"FAIL: {problem}\n"
              f"      The checkpoint is at https://zenodo.org/records/15272883.",
              file=sys.stderr)
        return 1

    print(f"Loading {args.checkpoint} ...", flush=True)
    started = time.time()
    try:
        # Exactly how predict_gp.generate_gp loads it, custom_objects and all.
        # A check that loads it some other way can pass while the pipeline
        # fails, which would be worse than no check.
        from tensorflow.keras.models import load_model
        model = load_model(str(args.checkpoint), custom_objects={"tf": tf},
                           compile=False)
    except Exception as error:
        print(f"\nFAIL: the checkpoint did not load under TensorFlow "
              f"{tf.__version__}:\n  {type(error).__name__}: {error}\n\n"
              f"This is the expected failure for a Keras 2.4.3 .h5 under a "
              f"much newer Keras, and it is usually a Lambda or custom layer.\n"
              f"The way out is conversion rather than a fight: run "
              f"convert_anorak_model.py in a CPU-only TF 2.2 container (no GPU, "
              f"so CUDA 10.1 does not matter) to re-export this as a "
              f"SavedModel, which crosses TensorFlow versions far better.",
              file=sys.stderr)
        return 1
    print(f"  loaded in {time.time() - started:.1f}s")
    print(f"  input  {model.input_shape}")
    print(f"  output {model.output_shape}")

    # Taken from the model rather than assumed. Upstream's patch_size=768 is
    # the window cut from the tile, not the network's input — it is resized to
    # 384 on the way in — so a checker that fed 768 straight to the model would
    # fail on a checkpoint that is perfectly fine.
    size = args.patch_size
    if size is None:
        shape = model.input_shape
        if not (isinstance(shape, tuple) and len(shape) == 4 and shape[1]):
            print(f"\nFAIL: cannot read a patch size from input_shape "
                  f"{shape}; pass --patch-size.", file=sys.stderr)
            return 1
        size = int(shape[1])
    print(f"\nRunning one {size}x{size} patch through it "
          f"(upstream cuts {WINDOW_SIZE}x{WINDOW_SIZE} windows at stride "
          f"{WINDOW_STRIDE} and resizes them to this) ...", flush=True)
    patch = np.random.rand(1, size, size, 3).astype("float32")
    started = time.time()
    try:
        prediction = model.predict(patch, verbose=0)
    except Exception as error:
        print(f"\nFAIL: the model loaded but could not predict:\n"
              f"  {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    print(f"  output shape {prediction.shape} in {elapsed:.1f}s")
    expected = (1, size, size, EXPECTED_CLASSES)
    if prediction.shape != expected:
        print(f"\nFAIL: expected {expected}, got {prediction.shape}. The "
              f"pipeline takes an argmax over the last axis and maps it to six "
              f"growth patterns plus background, so a different class count is "
              f"a different model.", file=sys.stderr)
        return 1

    # A rough marker, not a benchmark. Patches per 2000x2000 tile is
    # ceil((2000 - window)/stride + 1) squared — 64 at upstream's 768/192 — and
    # a slide is a few hundred tiles, so the per-patch cost is what decides
    # whether a cohort is days or months.
    import math as _math
    per_dim = _math.ceil((2000 - WINDOW_SIZE) / WINDOW_STRIDE + 1)
    per_slide_hours = elapsed * per_dim ** 2 * TILES_PER_SLIDE / 3600
    print(f"\nPASS: the checkpoint loads and predicts under TensorFlow "
          f"{tf.__version__}.")
    print(f"      Very roughly {per_slide_hours:.1f} h/slide at this speed "
          f"(~{per_dim ** 2} patches x ~{TILES_PER_SLIDE} tiles), first-call "
          f"overhead included — treat it as an order of magnitude, not an "
          f"estimate. Time a real slide before sizing the cohort.")
    if not gpus:
        print(f"      On CPU. Re-run in a CUDA-12 image before drawing any "
              f"conclusion about runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
