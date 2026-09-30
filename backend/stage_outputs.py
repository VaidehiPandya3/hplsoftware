"""Is a stage's output real? Validators shared by the server and the pipeline.

These used to live in tile_server_v2_.py, which is fine for a check the server
runs and useless for one a Slurm task has to run: that module builds a FastAPI
app, opens HDF5 handles and connects to Postgres at import time. The Nextflow
pipeline (hpl-nf/, via its bin/ wrappers) marks a stage done only after the same
check the server's gate applies, so the two have to be one function — a task
that accepted a file the server then refuses leaves a run reporting a finished
stage the UI will not let anything read.

The server imports these under their old private names, so nothing there
changed behaviour by moving.
"""

from __future__ import annotations

from pathlib import Path

import h5py

# What make_hpl_hdf5.py writes, in the order it writes them.
HPL_H5_DATASETS = ("img", "samples", "slides", "tiles")

# Columns assign_hpc_clusters.py writes. The cluster column itself is named
# after the reference's groupby (e.g. 'leiden_2.5'), so it is matched by
# elimination rather than by name — hardcoding a name here would break the
# moment the reference changes resolution, which is the kind of coupling that
# makes a validator call a healthy file broken.
ASSIGNMENT_REQUIRED_COLUMNS = (
    "samples", "slides", "tiles", "vote_margin", "neighbor_distance", "hpc_reference",
)


def validate_packaged_h5(path: Path) -> tuple[bool, str]:
    """Confirm a packaged .h5 is a complete, readable dataset — not merely a
    file sitting at the right path.

    Readiness used to be inferred from existence plus a Slurm state, which
    can't detect a file that is present and non-empty but unusable: an .h5
    truncated after its header opens without complaint and only fails when
    the missing chunks are read, which previously happened for the first time
    inside feature extraction, hours into a GPU job. Everything checked here
    is cheap (metadata plus two row reads) and runs on a human-triggered
    request, not a hot path.

    Returns (ok, reason) so callers can tell the user *why* it was rejected
    rather than just refusing.
    """
    try:
        with h5py.File(path, "r") as f:
            absent = [name for name in HPL_H5_DATASETS if name not in f]
            if absent:
                return False, f"missing dataset(s) {absent}"

            rows = f["img"].shape[0]
            if rows == 0:
                return False, "contains zero tiles"

            mismatched = {
                name: f[name].shape[0]
                for name in HPL_H5_DATASETS
                if f[name].shape[0] != rows
            }
            if mismatched:
                return False, f"dataset lengths disagree with img={rows}: {mismatched}"

            # Actually touch the first and last row. HDF5 validates the
            # superblock on open, so a file truncated partway through the data
            # still opens cleanly — reading the final row is what forces the
            # missing chunk to be resolved, and is the cheapest check that
            # distinguishes "complete" from "cut short".
            f["img"][0]
            f["img"][rows - 1]
            f["slides"][rows - 1]

    except (OSError, KeyError, ValueError) as e:
        return False, f"unreadable HDF5: {e}"
    return True, ""


def packaged_h5_rows(path: Path) -> int | None:
    """Tile count of a packaged .h5, or None if it cannot be read."""
    try:
        with h5py.File(path, "r") as f:
            return int(f["img"].shape[0])
    except (OSError, KeyError, ValueError):
        return None


def validate_assignment_csv(path: Path, expected_rows: int | None = None) -> tuple[bool, str]:
    """Confirm an assignment CSV is a full set of cluster IDs, not a stub.

    Same role as validate_extraction_output plays for Stage 3: a file existing
    at the right path is not evidence the job produced anything usable. A run
    killed partway leaves a CSV with a header and some rows, which reads as
    success to anything that only checks existence.
    """
    if not path.is_file():
        return False, "no output file"
    try:
        with path.open() as fh:
            header = fh.readline().strip()
            if not header:
                return False, "the file is empty"
            columns = [c.strip() for c in header.split(",")]
            missing = [c for c in ASSIGNMENT_REQUIRED_COLUMNS if c not in columns]
            if missing:
                return False, f"missing column(s): {', '.join(missing)}"
            if len(columns) <= len(ASSIGNMENT_REQUIRED_COLUMNS):
                return False, "no cluster-ID column alongside the metadata columns"
            rows = sum(1 for _ in fh)
    except OSError as e:
        return False, f"could not be read: {e}"

    if rows == 0:
        return False, "holds a header but no assignments"
    if expected_rows is not None and rows != expected_rows:
        return False, (
            f"holds {rows:,} assignments but the projections file has "
            f"{expected_rows:,} embeddings"
        )
    return True, ""
