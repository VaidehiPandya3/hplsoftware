#!/usr/bin/env python3
"""Re-export ANORAK's checkpoint as a SavedModel, from a TF 2.2 container.

Only needed if `check_anorak_model.py` fails to load the `.h5` in the image you
want to run inference in. The problem it solves is that the two requirements
pull apart: the checkpoint was written by Keras 2.4.3, and the only TensorFlow
that can drive an A100 or an H100 is far newer than that.

Conversion works because loading does not need a GPU. Run this in a **CPU-only**
TF 2.2 environment — CUDA 10.1 never comes into it — and the weights come out in
a format that crosses TensorFlow versions far better than an `.h5` with
custom_objects does.

    singularity exec <tf2.2-cpu>.sif python3 convert_anorak_model.py \
        --checkpoint /path/to/models/AIgrading_anorak.h5 \
        --out /path/to/models/AIgrading_anorak_savedmodel

Then check the result loads in the *inference* image:

    singularity exec --nv <ngc-tf2>.sif python3 check_anorak_model.py \
        --checkpoint /path/to/models/AIgrading_anorak_savedmodel

**Getting the pipeline to use it without patching upstream.** `generate_gp`
builds the path `<anorak_dir>/models/AIgrading_anorak.h5` from its own location
and cannot be pointed elsewhere. But `load_model` takes a directory as happily
as a file, so the shortest route is to put the SavedModel *at that name*:

    mv models/AIgrading_anorak.h5 models/AIgrading_anorak.h5.orig
    mv models/AIgrading_anorak_savedmodel models/AIgrading_anorak.h5

A directory called `.h5` is a lie on disk, so `--in-place` does this for you and
leaves a README beside it saying what happened — an unexplained one would cost
somebody an afternoon.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_README = """\
AIgrading_anorak.h5 is a SavedModel DIRECTORY, not an HDF5 file.

The original checkpoint is AIgrading_anorak.h5.orig. It was re-exported by
anorak-nf/tools/convert_anorak_model.py because it would not load under the
TensorFlow the inference container carries — the checkpoint is Keras 2.4.3 and
the only TensorFlow that can drive this cluster's GPUs is much newer.

The name is kept because predict_gp.generate_gp() builds this exact path from
its own __file__ and cannot be pointed anywhere else. tf.keras.models.load_model
accepts a directory, so nothing else had to change.

Converted from: {source}
By TensorFlow:  {tf_version}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        help="SavedModel directory to write "
                             "(default: <checkpoint>_savedmodel)")
    parser.add_argument("--in-place", action="store_true",
                        help="also move the result into the checkpoint's own "
                             "name, keeping the original as .orig")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow.keras.models import load_model

    print(f"TensorFlow {tf.__version__}")
    if not tf.__version__.startswith("2.2"):
        # Not refused: a 2.3 or 2.4 container may well load it too, and being
        # told "wrong version" by a script that never tried would be worse than
        # trying. But said out loud, because a conversion that runs under a
        # version the checkpoint does not load in is about to fail confusingly.
        print(f"  note: ANORAK pins 2.2. If the load below fails, this version "
              f"is the first thing to change.")

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1
    if args.checkpoint.is_dir():
        # Almost certainly a re-run of --in-place. Named directly, because the
        # load failure it would otherwise produce reads as "your container is
        # wrong" when the truth is "this is already done".
        print(f"{args.checkpoint} is a directory, so it is already a "
              f"SavedModel — this conversion has been done. The original .h5 "
              f"should be beside it as .orig.", file=sys.stderr)
        return 1

    out = args.out or args.checkpoint.with_name(args.checkpoint.stem + "_savedmodel")
    if out.exists():
        print(f"{out} already exists — remove it or choose another --out",
              file=sys.stderr)
        return 1

    print(f"Loading {args.checkpoint} ...", flush=True)
    try:
        model = load_model(str(args.checkpoint), custom_objects={"tf": tf},
                           compile=False)
    except Exception as error:
        print(f"\nThe checkpoint did not load here either:\n"
              f"  {type(error).__name__}: {error}\n\n"
              f"Conversion has to happen somewhere the .h5 opens. Try a "
              f"CPU-only container matching the pinned stack exactly: "
              f"python 3.8, tensorflow 2.2, keras 2.4.3, h5py 2.10 "
              f"(AIgrading/requirments.txt).", file=sys.stderr)
        return 1

    print(f"  input  {model.input_shape}")
    print(f"  output {model.output_shape}")
    print(f"Writing {out} ...", flush=True)
    try:
        model.save(str(out), save_format="tf")
    except (ValueError, TypeError):
        # Keras 3 dropped save_format and exports SavedModel through export().
        # The path that matters is the one above — this script is meant to run
        # in the TF 2.2 container the checkpoint came from — but the fallback
        # costs two lines and makes the conversion runnable, and testable,
        # anywhere newer.
        print("  (Keras 3 detected; exporting via model.export)")
        model.export(str(out))

    # Reloading here rather than trusting save(): this script exists precisely
    # because a model file can be written and not be readable, and a conversion
    # nobody opened is the same bet that made it necessary.
    print("Re-loading it to check it survived ...", flush=True)
    reloaded_shape = None
    try:
        reloaded_shape = load_model(str(out), compile=False).output_shape
    except Exception as keras_error:
        # Keras 3 cannot load a TF SavedModel through load_model at all, so a
        # failure here is not yet evidence the export is bad. Fall back to the
        # serving signature, which is what the SavedModel format actually
        # guarantees and what an older Keras will read.
        try:
            signature = tf.saved_model.load(str(out)).signatures[
                "serving_default"]
            output = list(signature.structured_outputs.values())[0]
            reloaded_shape = tuple(output.shape)
        except Exception:
            print(f"\nThe SavedModel was written but does not load back:\n"
                  f"  {type(keras_error).__name__}: {keras_error}",
                  file=sys.stderr)
            return 1

    # Compared on the trailing axes: a SavedModel signature can carry a
    # concrete batch dimension where the Keras model had None, and refusing
    # over that would reject a perfectly good conversion.
    if tuple(reloaded_shape)[1:] != tuple(model.output_shape)[1:]:
        print(f"\nShape changed in conversion: {model.output_shape} -> "
              f"{reloaded_shape}", file=sys.stderr)
        return 1
    print(f"  ok — {reloaded_shape}")

    if args.in_place:
        original = args.checkpoint.with_suffix(args.checkpoint.suffix + ".orig")
        if original.exists():
            print(f"\n{original} already exists; not overwriting it. Move the "
                  f"SavedModel into place by hand.", file=sys.stderr)
            return 1
        shutil.move(str(args.checkpoint), str(original))
        shutil.move(str(out), str(args.checkpoint))
        (args.checkpoint.parent / "README_MODEL_FORMAT.txt").write_text(
            _README.format(source=original, tf_version=tf.__version__),
            encoding="utf-8",
        )
        print(f"\nIn place: {args.checkpoint} is now a SavedModel directory, "
              f"original kept at {original}.")
        print(f"See {args.checkpoint.parent / 'README_MODEL_FORMAT.txt'}.")
    else:
        print(f"\nWrote {out}. Check it in the inference image with "
              f"check_anorak_model.py, then move it into the checkpoint's own "
              f"name (or re-run with --in-place).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
