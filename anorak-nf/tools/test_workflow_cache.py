#!/usr/bin/env python3
"""What -resume re-runs, proven against real Nextflow rather than asserted.

Nextflow keys a cached task on its script text, its inputs and its container's
name — not on the bin/ scripts it calls, the AIgrading clone they import, the
checkpoint, or an image's bytes. Each of those used to be an edit that changed
what a task did and not whether it re-ran, so a resumed cohort was quietly
processed by two versions of the code. And the tumour a slide belongs to used
to ride through every step, so relabelling one re-tiled and re-segmented it.

Every test here builds its own copy of the pipeline under a temp directory,
swaps the wrappers for fakes that write plausible outputs in milliseconds
(keeping the real anorak_common.py and publish_tree.py), runs real `nextflow
run -profile stub` — the real script blocks, not -stub-run — and reads which
tasks ran from the trace file. Skipped when nextflow is not installed.

ANORAK_NF_DIR points it at another copy of the pipeline, which is how these
were shown to fail against the version before the fix.

Runs under pytest and standalone.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = Path(os.environ.get("ANORAK_NF_DIR", HERE.parent)).resolve()
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

WRAPPERS = ("anorak_tile.py", "anorak_predict.py", "anorak_stitch.py",
            "slide_proportions.py", "tumour_grade.py")

#: One body for every fake, dispatching on its own name like the real ones'
#: command lines. Tiles and counts carry the pid so a re-run is new content —
#: TUMOUR_GRADE is cached on content ('deep'), and would rightly stay cached
#: behind a re-run that reproduced its counts byte for byte.
FAKE = r'''
import os, sys
name = os.path.basename(sys.argv[0]); a = sys.argv[1:]
def arg(k): return a[a.index(k) + 1] if k in a else None
if name == "anorak_tile.py":
    d = os.path.join(arg("--out-dir"), os.path.basename(arg("--slide")))
    os.makedirs(d, exist_ok=True)
    for f in ("Da0.jpg", "Da1.jpg", "Da2.jpg", "param.p"):
        open(os.path.join(d, f), "w").write(f"tile {os.getpid()}")
elif name == "anorak_predict.py":
    d = os.path.join(arg("--out-dir"), arg("--slide-name")); os.makedirs(d, exist_ok=True)
    for i in range(3): open(f"{d}/Da{i}.png", "w").write("m")
elif name == "anorak_stitch.py":
    d = arg("--ss1-final-dir"); os.makedirs(d, exist_ok=True)
    open(f"{d}/{arg('--slide-name')}_Ss1.png", "w").write(f"s {os.getpid()}")
elif name == "slide_proportions.py":
    open(arg("--out"), "w").write(f"slide_id,sample,acinar_px\n{arg('--slide-id')},{arg('--sample')},{os.getpid()}\n")
elif name == "tumour_grade.py":
    for k in ("--out-slides", "--out-tumours", "--out-missing"):
        open(arg(k), "w").write("sample\nT1\n")
'''

#: Names with spaces and parentheses, one in a subdirectory, as on the cluster.
SLIDES = {"BB232000 A1 -2 - 2023-08-29 20.08.31": ("BB232000 A1 -2 - 2023-08-29 20.08.31.ndpi", "T1"),
          "S2 (repeat)": ("sub/S2 (repeat).ndpi", "T1"),
          "S3": ("S3.svs", "T2")}


def have_nextflow() -> bool:
    if shutil.which("nextflow") is None:
        print("SKIP  nextflow not installed")
        return False
    return True


class Harness:
    def __init__(self, root: Path):
        self.root = root
        self.pipe = root / "pipe"
        self.outdir = root / "results"
        self.pipe.mkdir(parents=True)
        for name in ("main.nf", "nextflow.config"):
            shutil.copy(PIPELINE / name, self.pipe / name)
        shutil.copytree(PIPELINE / "conf", self.pipe / "conf")
        (self.pipe / "bin").mkdir()
        for name in WRAPPERS:
            (self.pipe / "bin" / name).write_text(f"# fake {name}\n{FAKE}")
        for name in ("anorak_common.py", "publish_tree.py"):
            if (PIPELINE / "bin" / name).exists():
                shutil.copy(PIPELINE / "bin" / name, self.pipe / "bin" / name)

        for rel, _ in SLIDES.values():
            (root / "raw" / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / "raw" / rel).write_bytes(b"")
        self.aig = root / "aig"
        (self.aig / "models").mkdir(parents=True)
        self.checkpoint = self.aig / "models" / "AIgrading_anorak.h5"
        self.checkpoint.write_bytes(HDF5_SIGNATURE + b"\0" * 100)
        for sub, mod in (("generating_tile", "save_cws.py"), ("inference_slide", "predict_gp.py")):
            (self.aig / sub).mkdir()
            (self.aig / sub / mod).write_text("# upstream\n")
        self.slides_csv = root / "slides.csv"
        self.write_slides({k: v[1] for k, v in SLIDES.items()})

    def write_slides(self, samples: dict[str, str]) -> None:
        with self.slides_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["slide_id", "samples"])
            writer.writerows(samples.items())

    def touch(self, path: Path, text: str = "\n# an edited comment\n") -> None:
        with path.open("a") as handle:
            handle.write(text)

    def run(self, *extra: str, resume: bool = True, env: dict | None = None) -> dict[str, str]:
        """Run the pipeline; returns {task name: trace status}."""
        cmd = ["nextflow", "run", str(self.pipe / "main.nf"), "-profile", "stub",
               "-ansi-log", "false", "-work-dir", str(self.root / "work"),
               "--slides_csv", str(self.slides_csv), "--raw_dir", str(self.root / "raw"),
               "--anorak_dir", str(self.aig), "--outdir", str(self.outdir), *extra]
        if resume:
            cmd.append("-resume")
        result = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True,
                                timeout=600, env={**os.environ, **(env or {})})
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        with (self.outdir / "pipeline_info" / "trace.txt").open() as handle:
            return {row["name"]: row["status"] for row in csv.DictReader(handle, delimiter="\t")}


def ran(statuses: dict[str, str]) -> set[str]:
    return {name for name, status in statuses.items() if status != "CACHED"}


def everything(steps, slides=SLIDES) -> set[str]:
    return {f"{step} ({slide})" for step in steps for slide in slides}


HEAVY_AND_AFTER = ("TILE_SLIDE", "PREDICT_GP", "SS1_STITCH", "PUBLISH_SLIDE", "SLIDE_PROPORTIONS")


def harness(tmp_path) -> Harness:
    return Harness(Path(tmp_path) / "h")


# --- tests ----------------------------------------------------------------

def test_nothing_changed_is_fully_cached(tmp_path):
    if not have_nextflow():
        return
    h = harness(tmp_path)
    first = h.run(resume=False)
    assert len(first) == 5 * len(SLIDES) + 1 and not any(s == "CACHED" for s in first.values()), first
    assert ran(h.run()) == set()


def test_relabelling_a_tumour_reruns_only_its_proportions(tmp_path):
    if not have_nextflow():
        return
    h = harness(tmp_path)
    h.run(resume=False)
    h.write_slides({k: ("T9" if k == "S3" else v[1]) for k, v in SLIDES.items()})
    assert ran(h.run()) == {"SLIDE_PROPORTIONS (S3)", "TUMOUR_GRADE"}


def test_editing_a_wrapper_reruns_it_and_everything_after(tmp_path):
    if not have_nextflow():
        return
    h = harness(tmp_path)
    h.run(resume=False)
    h.touch(h.pipe / "bin" / "anorak_tile.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER) | {"TUMOUR_GRADE"}
    h.touch(h.pipe / "bin" / "anorak_predict.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER[1:]) | {"TUMOUR_GRADE"}
    h.touch(h.pipe / "bin" / "anorak_stitch.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER[2:]) | {"TUMOUR_GRADE"}
    h.touch(h.pipe / "bin" / "anorak_common.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER) | {"TUMOUR_GRADE"}


def test_editing_slide_proportions_reruns_only_the_last_two(tmp_path):
    if not have_nextflow():
        return
    h = harness(tmp_path)
    h.run(resume=False)
    h.touch(h.pipe / "bin" / "slide_proportions.py")
    assert ran(h.run()) == everything(["SLIDE_PROPORTIONS"]) | {"TUMOUR_GRADE"}
    assert ran(h.run()) == set()


def test_upstream_code_and_checkpoint_are_in_the_key(tmp_path):
    if not have_nextflow():
        return
    h = harness(tmp_path)
    h.run(resume=False)
    h.touch(h.aig / "generating_tile" / "save_cws.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER) | {"TUMOUR_GRADE"}
    h.touch(h.aig / "inference_slide" / "predict_gp.py")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER[1:]) | {"TUMOUR_GRADE"}
    h.touch(h.checkpoint, "weights")
    assert ran(h.run()) == everything(HEAVY_AND_AFTER[1:]) | {"TUMOUR_GRADE"}
    # Bytecode the code wrote by being imported is not a change to it.
    (h.aig / "inference_slide" / "__pycache__").mkdir()
    (h.aig / "inference_slide" / "__pycache__" / "predict_gp.cpython-38.pyc").write_bytes(b"pyc")
    assert ran(h.run()) == set()


def test_image_content_is_in_the_key(tmp_path):
    """Nextflow keys on the image's name; a rebuild at the same path is not seen."""
    if not have_nextflow():
        return
    h = harness(tmp_path)
    fakebin = h.root / "fakebin"
    fakebin.mkdir()
    singularity = fakebin / "singularity"
    # Runs whatever follows the image argument, which is the task itself.
    singularity.write_text('#!/bin/bash\nargs=("$@"); for i in "${!args[@]}"; do '
                           'if [[ "${args[$i]}" == *.sif ]]; then exec "${args[@]:$((i+1))}"; fi; done\n'
                           'exit 99\n')
    singularity.chmod(0o755)
    image = h.root / "gpu.sif"
    image.write_bytes(b"SIF v1")
    config = h.root / "sing.config"
    config.write_text(f"params.gpu_container = '{image}'\n"
                      "singularity { enabled = true; autoMounts = false }\n"
                      "process { withLabel: process_gpu { container = params.gpu_container } }\n")
    env = {"PATH": f"{fakebin}:{os.environ['PATH']}"}
    h.run("-c", str(config), resume=False, env=env)
    assert ran(h.run("-c", str(config), env=env)) == set()
    h.touch(image, "rebuilt")
    assert ran(h.run("-c", str(config), env=env)) == everything(HEAVY_AND_AFTER[1:]) | {"TUMOUR_GRADE"}


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_wf_"))
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
