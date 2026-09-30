#!/usr/bin/env python3
"""Tests for tools/bootstrap_gpu_extras.sh, run for real against a fake image.

The extras directory is on every GPU task's PYTHONPATH, ahead of the image's
own packages, for every run until someone rebuilds it. What these guard
against is a build that looks fine and is not: one that installs whatever
version is newest today (so two builds of "the same" directory differ), one
that fetches a module nobody pinned, one that verifies with a PYTHONPATH no
task has, one that writes a lock for a directory that did not verify — or
half a lock — and one that reuses a lock made against a different image.

The fake index (fake_cluster.py) carries a cv2 newer than the pin, so an
unpinned install is visible as the wrong version rather than passing.

    BOOTSTRAP_UNDER_TEST=<other script> python3 tools/test_bootstrap_gpu_extras.py

Runs under pytest and standalone.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_cluster import FakeCluster  # noqa: E402

UNDER_TEST = Path(os.environ.get("BOOTSTRAP_UNDER_TEST", HERE / "bootstrap_gpu_extras.sh"))


class Build:
    def __init__(self, tmp_path):
        self.c = FakeCluster(Path(tmp_path))
        shutil.copy(UNDER_TEST, self.c.pipeline / "tools" / "bootstrap_gpu_extras.sh")
        self.extras = self.c.root / "new extras"   # a fresh directory, with a space
        self.lock = self.extras / "anorak_extras.lock"

    def run(self, *args, anorak=True):
        argv = ["bash", str(self.c.pipeline / "tools" / "bootstrap_gpu_extras.sh"),
                "--image", str(self.c.image), "--extras-dir", str(self.extras)]
        if anorak:
            argv += ["--anorak-dir", str(self.c.anorak)]
        env = self.c.env() if anorak else self.c.env(ANORAK_REPO_DIR="")
        result = subprocess.run(argv + list(args), env=env, capture_output=True,
                                text=True, timeout=300)
        result.out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout + result.stderr)
        return result

    def installed(self) -> dict[str, str]:
        found = {}
        for info in self.extras.glob("*.dist-info"):
            meta = dict(l.split(": ", 1) for l in (info / "METADATA").read_text().splitlines() if ": " in l)
            found[meta["Name"]] = meta["Version"]
        return found

    def pip_requests(self) -> list[str]:
        return [" ".join(call) for call in self.c.calls("pip")]

    def lock_pins(self) -> list[str]:
        return [l for l in self.lock.read_text().splitlines() if l and not l.startswith("#")]


def test_installs_what_the_image_lacks_at_the_pinned_version(tmp_path):
    b = Build(tmp_path)
    r = b.run()
    assert r.returncode == 0, r.out
    # 4.10.0.84 is the newest in the index; the pin is what must land.
    assert b.installed() == {"opencv-python-headless": "4.8.1.78", "pillow": "10.4.0"}, r.out
    assert b.lock_pins() == ["opencv-python-headless==4.8.1.78", "pillow==10.4.0"]
    assert f"# image: {b.c.image}" in b.lock.read_text()


def test_a_module_the_image_has_is_not_installed(tmp_path):
    b = Build(tmp_path)
    (Path(b.c.image_meta["site"]) / "PIL").mkdir()
    (Path(b.c.image_meta["site"]) / "PIL" / "__init__.py").write_text("")
    r = b.run()
    assert r.returncode == 0, r.out
    assert "pillow" not in b.installed()


def test_a_module_nobody_pinned_is_refused_not_fetched(tmp_path):
    b = Build(tmp_path)
    predict = b.c.anorak / "inference_slide" / "predict_gp.py"
    predict.write_text("import skimage\n" + predict.read_text())
    r = b.run()
    assert r.returncode == 1, r.out
    assert "PINNED" in r.out
    assert not any("skimage" in call or "scikit" in call for call in b.pip_requests()), b.pip_requests()
    assert not b.lock.exists()


def test_a_rerun_installs_exactly_the_lock(tmp_path):
    b = Build(tmp_path)
    assert b.run().returncode == 0
    first = b.lock_pins()
    r = b.run()
    assert r.returncode == 0, r.out
    assert b.lock_pins() == first
    assert any("-r" in call for call in b.pip_requests()[-1:]), b.pip_requests()


def test_a_lock_from_another_image_is_refused(tmp_path):
    b = Build(tmp_path)
    assert b.run().returncode == 0
    b.lock.write_text(b.lock.read_text().replace(str(b.c.image), "/elsewhere/old.sif"))
    r = b.run()
    assert r.returncode == 2, r.out
    assert "old.sif" in r.out


def test_a_directory_that_does_not_match_its_lock_is_not_relocked(tmp_path):
    b = Build(tmp_path)
    assert b.run().returncode == 0
    before = b.lock.read_text()
    # Something an earlier, unpinned build left behind.
    info = b.extras / "scikit_image-0.21.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: scikit-image\nVersion: 0.21.0\n")
    r = b.run()
    assert r.returncode == 1, r.out
    assert "does not match" in r.out
    assert b.lock.read_text() == before


def test_a_failed_verification_leaves_the_old_lock_whole(tmp_path):
    b = Build(tmp_path)
    assert b.run().returncode == 0
    before = b.lock.read_text()
    predict = b.c.anorak / "inference_slide" / "predict_gp.py"
    predict.write_text("import skimage\n" + predict.read_text())
    r = b.run("--refresh")
    assert r.returncode == 1, r.out
    assert b.lock.read_text() == before
    assert not [p.name for p in b.extras.iterdir() if ".tmp." in p.name or p.name.startswith(".")], \
        list(b.extras.iterdir())


def test_verification_keeps_the_images_own_pythonpath(tmp_path):
    # PREDICT_GP prepends the extras to the image's PYTHONPATH; replacing it
    # (SINGULARITYENV_PYTHONPATH, as this once did) verifies another interpreter.
    b = Build(tmp_path)
    ngc = b.c.root / "imgsite" / "ngc"
    (ngc / "ngc_only").mkdir(parents=True)
    (ngc / "ngc_only" / "__init__.py").write_text("")
    b.c.image_meta["env_pythonpath"] = str(ngc)
    b.c.save_image()
    predict = b.c.anorak / "inference_slide" / "predict_gp.py"
    predict.write_text("import ngc_only\n" + predict.read_text())
    r = b.run()
    assert r.returncode == 0, r.out


def test_a_quote_in_the_clone_path_is_just_a_character(tmp_path):
    b = Build(tmp_path)
    quoted = b.c.root / "Xi's AIgrading"
    shutil.copytree(b.c.anorak, quoted)
    b.c.anorak = quoted
    r = b.run()
    assert r.returncode == 0, r.out
    assert "predict_gp imports clean" in r.out


def test_no_lock_without_the_upstream_import_check(tmp_path):
    b = Build(tmp_path)
    r = b.run(anorak=False)
    assert r.returncode == 1, r.out
    assert not b.lock.exists()


def test_pip_failing_writes_no_lock(tmp_path):
    b = Build(tmp_path)
    b.c.cluster["pip_offline"] = True
    b.c.save()
    r = b.run()
    assert r.returncode == 1, r.out
    assert not b.lock.exists()


def test_every_pin_is_exact_and_covers_what_the_gpu_task_imports():
    """PINNED against the code: an exact version each, and every third-party
    module the GPU task's own scripts import is either the image's
    (tensorflow, numpy) or pinned. Upstream's predict_gp adds cv2 and PIL."""
    text = UNDER_TEST.read_text()
    block = re.search(r"^PINNED=\((.*?)^\)", text, re.S | re.M).group(1)
    pinned = dict(re.findall(r'"([A-Za-z0-9_]+):([^"]+)"', block))
    assert pinned, "no PINNED entries found"
    for module, requirement in pinned.items():
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==\d+(\.\d+)+", requirement), requirement
    imported = set()
    for script in ("anorak_predict.py", "anorak_common.py"):
        tree = ast.parse((HERE.parent / "bin" / script).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                imported.add(node.module.split(".")[0])
    imported |= {"cv2", "PIL", "numpy", "tensorflow"}   # upstream predict_gp.py's imports
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    local = {p.stem for p in (HERE.parent / "bin").glob("*.py")} | {"predict_gp"}
    third_party = imported - stdlib - local - {"tensorflow", "numpy"}
    if not stdlib:   # Python < 3.10: no list to subtract, so check the known ones only
        third_party &= {"cv2", "PIL"}
    assert third_party <= set(pinned), f"imported by the GPU task but not pinned: {third_party - set(pinned)}"


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures = []
    for name, test in tests:
        with tempfile.TemporaryDirectory(prefix="anorak_bootstrap_test_") as tmp:
            try:
                test(Path(tmp)) if test.__code__.co_argcount else test()
                print(f"PASS  {name}")
            except Exception as error:
                failures.append(name)
                print(f"FAIL  {name}: {type(error).__name__}: {str(error)[:300]}")
                if os.environ.get("VERBOSE"):
                    traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed (bootstrap under test: {UNDER_TEST})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
