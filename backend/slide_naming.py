"""Shared helper for recovering a clean slide_id from an uploaded WSI path.

tile_server_v2.upload_slide() saves raw files as
"{slide_id}_{upload_uuid}_{original_filename}", so Path(...).stem alone
pulls the upload uuid and original filename in with it. tile_mask.py and
auto_tile_from_mask.py both need the same clean slide_id (it becomes the
mask filename prefix and the output folder name), so this lives in one
place instead of being reimplemented per script.
"""

import re
from pathlib import Path

_UUID4 = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

# App uploads: "{slide_id}_{upload_uuid}_{original_filename}"
_UPLOAD_UUID_RE = re.compile(rf"^(?P<slide_id>.+?)_{_UUID4}_")
# GDC/TCGA downloads: "{barcode}.{file_uuid}.svs"
_GDC_UUID_RE = re.compile(rf"^(?P<slide_id>[^.]+)\.{_UUID4}\.")


def slide_id_from_raw_path(raw_path) -> str:
    name = Path(raw_path).name
    for pattern in (_UPLOAD_UUID_RE, _GDC_UUID_RE):
        match = pattern.match(name)
        if match:
            return match.group("slide_id")
    return Path(raw_path).stem


# GDC/TCGA downloads carry the file's own uuid between the barcode and the
# extension: "{barcode}.{file_uuid}.svs". wsi_registry has a file_uuid column
# for it, so recover it here — beside the pattern that already knows the shape
# — rather than re-deriving the same regex at the call site.
_GDC_FILE_UUID_RE = re.compile(rf"^[^.]+\.(?P<file_uuid>{_UUID4})\.")


def file_uuid_from_raw_path(raw_path):
    """The GDC file uuid in a downloaded slide's name, or None.

    None for anything not named that way — an app upload, a locally produced
    slide — because there is no uuid to report, not because one is missing.
    """
    match = _GDC_FILE_UUID_RE.match(Path(raw_path).name)
    return match.group("file_uuid") if match else None


# --- the tile_coordinates / tile_registry join key -------------------------
#
# slide_tile is "<slides>_<tiles>" upper-cased, and it is the primary key of
# both tile_coordinates and tile_registry. The two sides of the pipeline
# disagree about the tile name, which is why this is centralised:
#
#   auto_tile_from_mask.py  writes tiles as "24_10.jpeg"  (Stage 1 metadata CSV)
#   make_hpl_hdf5.py        writes tiles as "24_10"       (packaged .h5, and so
#                                                          the Stage 4 CSV too)
#   existing TCGA registry rows are        "..._18_15.JPEG"
#
# So a key built from Stage 4's CSV could never match a row registered from
# Stage 1's CSV, and neither could match the TCGA rows already loaded. Both
# sides go through here instead.
#
# The key builders below do NOT repair a tile name that is missing its
# extension, and that is deliberate: a key repaired in place would match while
# the (slides, tiles) columns it was built from still disagree with Kai's
# reference CSV, and a rule that appends ".jpeg" to anything unsuffixed would
# also "fix" a name that legitimately is not one.
#
# Repair happens one level up instead, at the boundaries that read an artifact
# off disk — normalize_tile_names() below, called by register_dataset.py and
# load_hpc_assignments.py, which append the suffix explicitly, count what they
# touched and report it. That keeps the correction visible and keeps it out of
# the key definition. The source fix is still make_hpl_hdf5.py, which is where
# the suffix was being dropped.
_TILE_SUFFIX = ".JPEG"

# What normalize_tile_names() appends. Lower case, because it is a filename —
# _TILE_SUFFIX above is upper only because the join key is upper-cased whole.
_TILE_SUFFIX_LOWER = ".jpeg"

# Only the tile part is ever inspected for an extension. A slide name may itself
# contain dots — a real one is "BB232560 A3-1 - 2023-10-11 16.41.02" — so
# testing the concatenated key would read that timestamp's ".02" as a file
# extension.
_HAS_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def make_slide_tile(slide, tile) -> str:
    """The join key for one tile: "<slides>_<tiles>", upper-cased."""
    return f"{str(slide).strip().upper()}_{str(tile).strip().upper()}"


def make_slide_tile_series(slides, tiles):
    """make_slide_tile over two pandas Series, returning a Series.

    Kept beside the scalar version and pinned to it by test, so the vectorised
    path used for millions of rows cannot drift from the definition.
    """
    return (
        slides.astype(str).str.strip().str.upper()
        + "_"
        + tiles.astype(str).str.strip().str.upper()
    )


def _as_text(value) -> str:
    """Tile names arrive as str from a CSV and as bytes from HDF5. Decoding
    matters more than it looks: str(b"18_15.jpeg") is "b'18_15.jpeg'", whose
    last character is a quote, so an extension test against it says the suffix
    is missing on a file that has it."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace").strip()
    return str(value).strip()


def tiles_missing_suffix(tiles) -> bool:
    """True if these tile names have no file extension.

    The signature of a .h5 packaged before make_hpl_hdf5.py started storing
    "18_15.jpeg" rather than "18_15". Every consumer of such a file is wrong in
    the same way — the KB join matches nothing and --validate-against merges
    zero rows — so it is worth naming rather than working around.
    """
    sample = [t for t in (_as_text(v) for v in list(tiles)[:100]) if t]
    if not sample:
        return False
    return not any(_HAS_EXTENSION_RE.search(t) for t in sample)


def tile_name_verdict(tiles) -> str:
    """"short" (none carry an extension), "done" (all do), or "mixed".

    Reads every name rather than sampling the first 100 the way
    tiles_missing_suffix() does. That sampling is why a mixed file — some names
    suffixed, some not — reads as "not missing" to the guard and passes it
    silently. Mixed is the one state that cannot be repaired: it is what a
    resume straddling the tile-name fix leaves behind, and the rows on either
    side of that boundary are indistinguishable by name, so appending a suffix
    would mislabel real tiles. Callers refuse it.

    Blank names are ignored, and all-blank counts as "done" — there is nothing
    to append to, and calling that "short" would send a caller off to migrate a
    file whose tile column is empty for an entirely different reason.
    """
    names = [t for t in (_as_text(v) for v in tiles) if t]
    if not names:
        return "done"
    with_suffix = sum(1 for n in names if _HAS_EXTENSION_RE.search(n))
    if with_suffix == 0:
        return "short"
    if with_suffix == len(names):
        return "done"
    return "mixed"


def normalize_tile_names(tiles) -> tuple[list[str], int]:
    """(names with ".jpeg" appended where absent, how many were changed).

    Only ever appends — a name that already carries any extension is returned
    untouched, so a .png or .tiff is left alone rather than turned into
    "24_10.png.jpeg". Safe because auto_tile_from_mask.py:150 hardcodes
    f"{col}_{row}.jpeg" for every tile this pipeline has ever written, which
    makes "24_10" -> "24_10.jpeg" a bijection rather than a guess.

    Does not itself refuse a mixed set — check tile_name_verdict() first. This
    returns a count so the caller can report the correction instead of making it
    silently.
    """
    out, changed = [], 0
    for value in tiles:
        name = _as_text(value)
        if name and not _HAS_EXTENSION_RE.search(name):
            name += _TILE_SUFFIX_LOWER
            changed += 1
        out.append(name)
    return out, changed
