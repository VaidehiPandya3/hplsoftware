#!/usr/bin/env python3
"""Shared helpers for the ANORAK process wrappers.

Every wrapper in this directory exists for one reason: upstream's functions
select their slide with `sorted(glob(cws_folder/pattern))[nfile]`, an integer
index into a directory listing. Across three separate steps that is a silent
mismatch waiting to happen — a directory that gained or lost an entry between
prediction and stitching shifts every index above it, and the stitch then
assembles one slide's masks against another slide's `param.p`, producing a
mask of the right size, the right colours and the wrong slide.

The fix is not to pass a better index. It is to make the index meaningless:
each task is given exactly one slide, and the pattern handed to upstream is
that slide's own name, escaped, so the glob resolves to exactly one entry and
`nfile=0` is correct by construction. `single_slide_pattern` below is that,
and it asserts the match is unique rather than trusting it.
"""

from __future__ import annotations

import glob as globlib
import os
import sys
from pathlib import Path

#: The exit status of every refusal in this directory, and the only one
#: nextflow.config's errorStrategy treats as final. 65 is sysexits.h's
#: EX_DATAERR: "the input data was incorrect". A refusal is a statement about
#: this slide — its header has no mpp, its tiles are short, its mask carries a
#: colour the palette does not know — and running it again reads the same
#: bytes and refuses again. Everything else a task can exit with is presumed
#: transient until it has failed maxRetries times: a CUDA context that did not
#: come up, an EIO off CephFS, a node that died, an .exitcode Nextflow could
#: not read. Status 1 cannot carry this distinction, because an uncaught
#: Python exception exits 1 too, and on this cluster most of those are the
#: transient kind.
REFUSAL_EXIT_CODE = 65

#: RGB -> pattern, read off `ss1_final.py`'s own thresholds rather than the
#: repository README, which disagrees: the README calls cribriform cyan
#: (#00ffff) while both `class_colors[1]` and `ss1_final` use green (0,255,0).
#: The code is what produced the pixels, so the code is what this follows.
#: Class 0 (black) is background and is excluded from every proportion.
PATTERN_COLOURS_RGB = {
    "cribriform": (0, 255, 0),
    "micropapillary": (255, 0, 255),
    "solid": (128, 0, 0),
    "papillary": (255, 255, 0),
    "acinar": (255, 0, 0),
    "lepidic": (0, 0, 255),
}

#: The IASLC high-grade patterns. Their combined share decides grade 3
#: regardless of which pattern predominates (paper, Methods).
HIGH_GRADE_PATTERNS = ("solid", "micropapillary", "cribriform")

#: Pattern order for output columns — the paper's own order, so a proportions
#: table can be read straight against Fig. 2b.
PATTERN_ORDER = (
    "lepidic", "papillary", "acinar", "cribriform", "micropapillary", "solid",
)


def single_slide_pattern(cws_dir: str, slide_name: str) -> str:
    """A glob pattern matching this slide's cws directory and nothing else.

    Returned rather than an index, so upstream's `[nfile]` is handed a
    one-element list. Refuses if it is not exactly one: zero means the task was
    staged wrong, more than one means the isolation this whole module exists to
    provide has failed, and both are worth stopping for.
    """
    pattern = globlib.escape(slide_name)
    matches = globlib.glob(os.path.join(globlib.escape(cws_dir), pattern))
    if len(matches) != 1:
        refuse(
            f"Expected exactly one entry named {slide_name!r} in {cws_dir}, "
            f"found {len(matches)}: {matches[:5]}. Upstream selects its slide "
            f"by index, so anything but one match means a different slide "
            f"could be processed than the one this task was given."
        )
    return pattern


def count_tiles(slide_cws_dir: str) -> int:
    """How many `Da*.jpg` a tiled slide directory holds.

    The directory is escaped because it ends in the slide's own filename, and
    glob reads `[...]` in *any* component as a character class: a slide called
    `S2 [repeat].ndpi` counted zero tiles in a directory full of them, and
    every step refused it as short.
    """
    return len(globlib.glob(os.path.join(globlib.escape(slide_cws_dir), "Da*.jpg")))


def count_masks(slide_mask_dir: str) -> int:
    """How many `Da*.png` a predicted slide directory holds (escaped, as above)."""
    return len(globlib.glob(os.path.join(globlib.escape(slide_mask_dir), "Da*.png")))


def refuse(message: str) -> None:
    """Stop this task with a readable reason on stderr.

    Nextflow reports the exit status and the tail of stderr, so a refusal has
    to say what was wrong in the last few lines or it arrives as a number.
    """
    print(f"REFUSED: {message}", file=sys.stderr, flush=True)
    raise SystemExit(REFUSAL_EXIT_CODE)


#: HDF5's format signature. The superblock sits at byte 0 or, when the file
#: carries a user block, at 512 and every doubling after it (HDF5 spec, "Format
#: Signature"); a file with the signature nowhere on that ladder is not HDF5,
#: whatever its name says — an HTML error page saved by a failed download, or a
#: transfer cut off before the first byte landed.
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def checkpoint_problem(path) -> str | None:
    """Why `path` is not something `generate_gp` can load, or None if it is.

    THE definition, which main.nf, preflight.sh (which calls this module) and
    tools/check_anorak_model.py all apply, because they used to disagree:
    anorak_predict.py asked `is_file()` while the other three asked `exists()`,
    so after `convert_anorak_model.py --in-place` turned the checkpoint into a
    SavedModel *directory*, every launch check passed and every GPU task
    refused. Upstream loads it with Keras `load_model`, which takes either form,
    so this accepts exactly those two:

      - an HDF5 file (the Zenodo download), recognised by its signature rather
        than its extension, since the SavedModel keeps the `.h5` name too;
      - a SavedModel directory, recognised by the `saved_model.pb` that
        `load_model` itself looks for.

    main.nf restates this in Groovy — it cannot import Python — and
    tools/test_checkpoint_predicate.py runs both against the same fixtures.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "saved_model.pb").is_file():
            return None
        return (f"{path} is a directory but not a SavedModel (no saved_model.pb "
                f"in it) — a conversion that did not finish?")
    if not path.is_file():
        return f"no model checkpoint at {path}"
    size = path.stat().st_size
    with path.open("rb") as handle:
        offset = 0
        while offset + len(HDF5_SIGNATURE) <= size:
            handle.seek(offset)
            if handle.read(len(HDF5_SIGNATURE)) == HDF5_SIGNATURE:
                return None
            offset = 512 if offset == 0 else offset * 2
    return (f"{path} is a file of {size} bytes with no HDF5 signature, so it is "
            f"not a Keras checkpoint — an interrupted or failed download?")


if __name__ == "__main__":
    # `anorak_common.py checkpoint <path>` — so preflight.sh can ask this
    # module rather than restate the rule in bash.
    if len(sys.argv) == 3 and sys.argv[1] == "checkpoint":
        problem = checkpoint_problem(sys.argv[2])
        print(problem or f"{sys.argv[2]}: loadable checkpoint")
        raise SystemExit(1 if problem else 0)
    print("usage: anorak_common.py checkpoint <path>", file=sys.stderr)
    raise SystemExit(2)
