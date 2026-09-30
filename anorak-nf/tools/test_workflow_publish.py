#!/usr/bin/env python3
"""Published tiles and masks are complete-or-absent, current, and cost a
resume nothing.

publishDir mode 'link' with overwrite:false could leave results/cws_tiling/<id>
partial (the head job killed part-way through a slide) or stale (a re-tiled
slide kept its old links), and overwrite:true would re-link every cached
slide on every resume — ~20-30M links from the head job, the publish death of
2026-09-16. PUBLISH_SLIDE (bin/publish_tree.py) replaces it; these check that
it repairs a partial directory, replaces a stale one, links nothing when
nothing changed, and links from the task's real work directory — a hard link
is the same inode, so the inode is what is compared.

The workflow tests use test_workflow_cache's harness and skip without
nextflow; the unit tests of publish_tree need nothing. Runs under pytest and
standalone.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bin"))
from test_workflow_cache import SLIDES, Harness, have_nextflow, ran  # noqa: E402


def published(h: Harness, kind: str, slide: str) -> Path:
    return h.outdir / kind / slide


def tree(top: Path) -> dict[str, os.stat_result]:
    return {str(p.relative_to(top)): p.stat() for p in sorted(top.rglob("*")) if p.is_file()}


def tilings(h: Harness, filename: str) -> list[Path]:
    """Every TILE_SLIDE output for one slide file — real directories, not the
    symlinks the downstream tasks stage them as."""
    return [p for p in (h.root / "work").glob("*/*/cws_tiling")
            if not p.is_symlink() and (p / filename).is_dir()]


def task_output(h: Harness, filename: str) -> Path:
    found = tilings(h, filename)
    assert len(found) == 1, found
    return found[0]


def assert_mirrors(dest: Path, src: Path) -> None:
    """dest holds exactly src's files, each the same inode."""
    want, got = tree(src), tree(dest)
    assert set(got) == set(want), f"{dest}: {sorted(set(got) ^ set(want))}"
    for rel, st in want.items():
        assert (got[rel].st_dev, got[rel].st_ino) == (st.st_dev, st.st_ino), f"{rel} is not a hard link of the task's file"
        assert got[rel].st_nlink >= 2


# --- through the workflow -------------------------------------------------

def test_published_dirs_are_hard_links_of_the_work_dir_even_under_scratch(tmp_path):
    if not have_nextflow():
        return
    h = Harness(Path(tmp_path) / "h")
    scratch = h.root / "scratch.config"
    scratch.write_text("process.scratch = true\n")
    h.run("-c", str(scratch), resume=False)
    for slide, (rel, _) in SLIDES.items():
        cws = task_output(h, Path(rel).name)
        assert_mirrors(published(h, "cws_tiling", slide), cws)
        assert (published(h, "ss1_final", slide) / f"{Path(rel).name}_Ss1.png").is_file()


def test_unchanged_resume_links_nothing(tmp_path):
    if not have_nextflow():
        return
    h = Harness(Path(tmp_path) / "h")
    h.run(resume=False)
    before = {s: tree(published(h, "cws_tiling", s)) for s in SLIDES}
    assert ran(h.run()) == set()
    after = {s: tree(published(h, "cws_tiling", s)) for s in SLIDES}
    # ctime moves with the link count, so an unchanged ctime is no link made.
    assert {s: {r: (st.st_ino, st.st_ctime_ns) for r, st in t.items()} for s, t in before.items()} == \
           {s: {r: (st.st_ino, st.st_ctime_ns) for r, st in t.items()} for s, t in after.items()}


def test_a_partial_published_dir_is_repaired(tmp_path):
    if not have_nextflow():
        return
    h = Harness(Path(tmp_path) / "h")
    # What the old publishDir left behind when the head job died mid-slide:
    # a prefix of the tiles, plus an interrupted publish's staging directory.
    partial = published(h, "cws_tiling", "S3") / "S3.svs"
    partial.mkdir(parents=True)
    (partial / "Da0.jpg").write_text("from a head job killed mid-publish")
    (h.outdir / "cws_tiling" / ".S3.publishing").mkdir()
    h.run(resume=False)
    assert_mirrors(published(h, "cws_tiling", "S3"), task_output(h, "S3.svs"))
    assert not (h.outdir / "cws_tiling" / ".S3.publishing").exists()

    # Damage after the task finished is invisible to the cache, by design; a
    # new --republish re-checks every slide and re-links only the damaged one.
    (published(h, "cws_tiling", "S3") / "S3.svs" / "Da1.jpg").unlink()
    untouched = tree(published(h, "cws_tiling", "S2 (repeat)"))
    assert ran(h.run()) == set()
    assert ran(h.run("--republish", "1")) == {f"PUBLISH_SLIDE ({s})" for s in SLIDES}
    assert_mirrors(published(h, "cws_tiling", "S3"), task_output(h, "S3.svs"))
    assert {r: st.st_ctime_ns for r, st in tree(published(h, "cws_tiling", "S2 (repeat)")).items()} == \
           {r: st.st_ctime_ns for r, st in untouched.items()}


def test_a_stale_published_dir_is_replaced(tmp_path):
    if not have_nextflow():
        return
    h = Harness(Path(tmp_path) / "h")
    h.run(resume=False)
    old = tree(published(h, "cws_tiling", "S3"))
    h.touch(h.pipe / "bin" / "anorak_tile.py")   # re-tiles every slide
    h.run()
    new_cws = [p for p in tilings(h, "S3.svs")
               if {st.st_ino for st in tree(p).values()} != {st.st_ino for st in old.values()}]
    assert len(new_cws) == 1, new_cws
    assert_mirrors(published(h, "cws_tiling", "S3"), new_cws[0])


# --- publish_tree itself ------------------------------------------------------

def _source(root: Path) -> Path:
    src = root / "task" / "cws_tiling"
    (src / "slide.svs").mkdir(parents=True)
    for i in range(3):
        (src / "slide.svs" / f"Da{i}.jpg").write_text(str(i))
    return src


def test_publish_tree_rebuilds_rather_than_merging(tmp_path):
    from publish_tree import publish
    root = Path(tmp_path)
    src = _source(root)
    dest_root = root / "out"
    extra = dest_root / "S" / "slide.svs"
    extra.mkdir(parents=True)
    (extra / "Da9.jpg").write_text("a tile no current tiling has")
    assert "replaced" in publish(src, dest_root, "S")
    assert_mirrors(dest_root / "S", src)
    assert "up to date" in publish(src, dest_root, "S")


def test_publish_tree_follows_a_staged_symlink_to_the_real_files(tmp_path):
    from publish_tree import publish
    root = Path(tmp_path)
    src = _source(root)
    staged = root / "scratch" / "cws_tiling"
    staged.parent.mkdir()
    staged.symlink_to(src)
    publish(staged, root / "out", "S")
    assert_mirrors(root / "out" / "S", src)


def test_publish_tree_refuses_a_name_that_leaves_its_root(tmp_path):
    from publish_tree import publish
    src = _source(Path(tmp_path))
    for bad in ("..", "a/b", ""):
        try:
            publish(src, Path(tmp_path) / "out", bad)
        except SystemExit as stop:
            assert stop.code == 65
        else:
            raise AssertionError(f"accepted {bad!r}")


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_pub_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
