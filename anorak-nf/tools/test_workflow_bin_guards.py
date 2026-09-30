#!/usr/bin/env python3
"""The wrappers' own guards, against the inputs that used to get past them.

  - a slide filename with `[...]` in it, which glob reads as a character class
    in the directory part too, so its tiles counted zero and it was refused;
  - a .mrxs, whose pixels are in a companion directory Nextflow does not
    stage, so opening it through the staged symlink failed;
  - the GPU check, which must refuse more than one visible card and must NOT
    refuse the one it was given, whatever CUDA_VISIBLE_DEVICES says about it.

TensorFlow and openslide are replaced by stand-ins, so this runs anywhere.
Runs under pytest and standalone.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bin"))
import anorak_common  # noqa: E402

BRACKETED = "S2 [repeat] (b).ndpi"


def _tiled(root: Path, name: str = BRACKETED, n: int = 3) -> Path:
    slide = root / "cws_tiling" / name
    slide.mkdir(parents=True)
    for i in range(n):
        (slide / f"Da{i}.jpg").write_bytes(b"")
        (slide / f"Da{i}.png").write_bytes(b"")
    return slide


def test_tiles_are_counted_in_a_directory_whose_name_has_brackets(tmp_path):
    slide = _tiled(Path(tmp_path))
    assert anorak_common.count_tiles(str(slide)) == 3
    assert anorak_common.count_masks(str(slide)) == 3


def test_the_slide_pattern_survives_brackets_in_the_cws_dir(tmp_path):
    root = Path(tmp_path) / "work [1]"
    _tiled(root, "S3.svs")
    pattern = anorak_common.single_slide_pattern(str(root / "cws_tiling"), "S3.svs")
    assert pattern == "S3.svs"


# --- .mrxs --------------------------------------------------------------------

def test_the_tile_count_opens_the_real_slide_not_the_staged_link(tmp_path):
    """A .mrxs's pixels are in <stem>/ beside it; only the real path has that."""
    root = Path(tmp_path)
    raw = root / "raw" / "slide.mrxs"
    raw.parent.mkdir()
    raw.write_bytes(b"")
    (root / "raw" / "slide").mkdir()          # the companion directory
    staged = root / "task" / "slide.mrxs"
    staged.parent.mkdir()
    staged.symlink_to(raw)

    opened = []

    class FakeSlide:
        def __init__(self, path):
            if not Path(path).with_suffix("").is_dir():
                raise RuntimeError(f"Unsupported or missing image file: {path}")
            opened.append(path)
            self.properties = {"objective": "40", "mpp": "0.25"}
            self.level_dimensions = [(8000, 6000)]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("openslide")
    fake.OpenSlide = FakeSlide
    fake.PROPERTY_NAME_OBJECTIVE_POWER = "objective"
    fake.PROPERTY_NAME_MPP_X = "mpp"
    saved = sys.modules.get("openslide")
    sys.modules["openslide"] = fake
    try:
        import anorak_tile
        count = anorak_tile.expected_tile_count(staged, 0.22)
    finally:
        if saved is None:
            sys.modules.pop("openslide", None)
        else:
            sys.modules["openslide"] = saved
    assert count > 0 and opened == [str(raw.resolve())]


# --- the GPU check ------------------------------------------------------------

def _gpu_check(visible_gpus: int, cuda_visible_devices: str | None) -> int | None:
    """Exit status of check_one_gpu, or None if it let the task through."""
    fake = types.ModuleType("tensorflow")
    fake.config = types.SimpleNamespace(
        list_physical_devices=lambda kind: [f"/physical_device:GPU:{i}" for i in range(visible_gpus)])
    saved_tf, saved_env = sys.modules.get("tensorflow"), os.environ.get("CUDA_VISIBLE_DEVICES")
    sys.modules["tensorflow"] = fake
    if cuda_visible_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    try:
        import anorak_predict
        anorak_predict.check_one_gpu()
        return None
    except SystemExit as stop:
        return stop.code
    finally:
        if saved_tf is None:
            sys.modules.pop("tensorflow", None)
        else:
            sys.modules["tensorflow"] = saved_tf
        if saved_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_env


def test_the_one_assigned_gpu_is_accepted_however_it_is_named():
    assert _gpu_check(1, "3") is None        # Slurm's index, no device cgroup
    assert _gpu_check(1, "0") is None        # remapped inside a device cgroup
    assert _gpu_check(1, None) is None       # cgroup shows one card, variable unset


def test_more_than_one_gpu_is_a_refusal_and_none_is_a_retry():
    assert _gpu_check(4, None) == anorak_common.REFUSAL_EXIT_CODE
    assert _gpu_check(2, "0,1") == anorak_common.REFUSAL_EXIT_CODE
    assert _gpu_check(0, "3") == 1


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="anorak_guard_"))) if fn.__code__.co_argcount else fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
