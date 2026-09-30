#!/usr/bin/env python3
"""Make <root>/<name> an exact hard-linked copy of one task's output directory.

This replaces `publishDir mode: 'link'` for the tiles and the Ss1 masks, which
could leave a published directory in two wrong states with nothing to say so:

  - partial. publishDir runs in the head job after the task, one file at a
    time. A head job killed part-way through a slide's few thousand tiles left
    a prefix, and on -resume `overwrite: false` saw the directory exists and
    skipped it — for good.
  - stale. A slide re-tiled because its code changed kept the previous
    tiling's links under the same name, for the same reason.

And it could not simply be `overwrite: true`: on -resume that re-publishes
every cached task, and ~20-30M hard links made by the head job is the
publish death of 2026-09-16 over again (see TILE_SLIDE in main.nf) — while the
head job's own CephFS reads are what hung on 2026-09-23. So publishing is a
task of its own, on a compute node, cached by Nextflow per producing task: a
resume with nothing re-tiled runs none of them.

What it guarantees is complete-or-absent and current. The published directory
is compared against the source by inode — a hard link *is* the same inode, so
equality means current rather than merely same-named — and if anything differs
it is rebuilt beside the old one and renamed into place. Between the two
renames the name is absent, never half-filled, and a reader never sees a mix of
two tilings.

Links are made from the source's real path, never through the staged symlink's
own directory: with `scratch = true` the task runs in node-local $TMPDIR, and a
hard link cannot cross filesystems. Outdir and the work directory must be on
one filesystem; if they are not this refuses rather than copying terabytes.
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anorak_common import refuse  # noqa: E402


def inventory(top: Path) -> tuple[set[str], dict[str, tuple[int, int]]]:
    """(directories, {file: (st_dev, st_ino)}) under `top`, relative to it.

    Files are stat()ed through any symlink, so a file staged as a link counts
    as the inode it names.
    """
    dirs: set[str] = set()
    files: dict[str, tuple[int, int]] = {}
    for current, subdirs, names in os.walk(top):
        rel = os.path.relpath(current, top)
        if rel != ".":
            dirs.add(rel)
        for name in names:
            path = os.path.join(current, name)
            st = os.stat(path)
            files[os.path.normpath(os.path.join(rel, name))] = (st.st_dev, st.st_ino)
    return dirs, files


def matches(dest: Path, expected) -> bool:
    if dest.is_symlink() or not dest.is_dir():
        return False
    return inventory(dest) == expected


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def publish(src: Path, root: Path, name: str) -> str:
    """Publish `src` as root/name; returns what was done, for the task log."""
    # The name becomes a path component under a shared directory, so one that
    # climbs out of it or into another slide's entry is refused, not joined.
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        refuse(f"{name!r} cannot be a published directory name.")
    real = src.resolve(strict=True)
    if not real.is_dir():
        refuse(f"{src} is not a directory.")
    expected = inventory(real)
    if not expected[1]:
        refuse(f"{src} holds no files; nothing that ran to completion "
               f"produces an empty output.")

    root.mkdir(parents=True, exist_ok=True)
    dest = root / name
    if matches(dest, expected):
        return f"{dest}: up to date, {len(expected[1])} files"

    # Fixed names beside dest, not unique ones, so the leftovers of an attempt
    # killed part-way are found by name rather than by listing a directory
    # with an entry per slide in the cohort — which is 7,000 entries read per
    # task, on the filesystem whose metadata reads already hung a run.
    staging = root / f".{name}.publishing"
    retired = root / f".{name}.retired"
    remove(staging)
    remove(retired)

    staging.mkdir()
    for rel in sorted(expected[0]):
        (staging / rel).mkdir(parents=True, exist_ok=True)
    for rel in sorted(expected[1]):
        try:
            os.link(os.path.realpath(real / rel), staging / rel)
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                remove(staging)
                refuse(f"{root} and the work directory ({real}) are on different "
                       f"filesystems, so a hard link between them is impossible. "
                       f"Put --outdir on the work directory's filesystem.")
            raise
    if inventory(staging) != expected:
        raise SystemExit(f"{staging} does not match {real} after linking")

    replaced = dest.exists() or dest.is_symlink()
    if replaced:
        os.rename(dest, retired)
    os.rename(staging, dest)
    remove(retired)

    # Checked again at its final name: a second attempt at this same slide,
    # still running on another node, is the one way the rename could land
    # something else here. Exit 1, not a refusal — that is transient.
    if not matches(dest, expected):
        raise SystemExit(f"{dest} changed while it was being published")
    verb = "replaced" if replaced else "published"
    return f"{dest}: {verb}, {len(expected[1])} files linked"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True,
                        help="the task output to publish (a staged symlink is fine)")
    parser.add_argument("--root", type=Path, required=True,
                        help="directory the published copy goes under")
    parser.add_argument("--name", required=True,
                        help="name of the published copy under --root")
    args = parser.parse_args()
    print(publish(args.src, args.root, args.name), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
