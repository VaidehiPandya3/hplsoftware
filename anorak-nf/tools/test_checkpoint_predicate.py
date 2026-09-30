#!/usr/bin/env python3
"""The launch check and the GPU task must agree on what a checkpoint is.

They used to disagree: main.nf, preflight.sh and check_anorak_model.py asked
`exists()`, anorak_predict.py asked `is_file()`, and once
convert_anorak_model.py --in-place had turned the checkpoint into a SavedModel
directory every launch check passed and every GPU task in the cohort refused.

There are now two copies of one rule — bin/anorak_common.checkpoint_problem,
and main.nf's checkpointProblem, which is Groovy because the workflow body
cannot import Python. This runs both on the same fixtures, the Groovy one by
lifting the function out of main.nf and running it under real Nextflow, and
fails if they ever give different verdicts. Skipped when nextflow is not
installed; the Python side is tested either way.

Runs under pytest and standalone.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_NF = HERE.parent / "main.nf"
sys.path.insert(0, str(HERE.parent / "bin"))
from anorak_common import HDF5_SIGNATURE, checkpoint_problem  # noqa: E402


def fixtures(root: Path) -> dict[str, tuple[Path, bool]]:
    """name -> (path, loadable)."""
    root.mkdir(parents=True, exist_ok=True)
    made = {}

    h5 = root / "download.h5"
    h5.write_bytes(HDF5_SIGNATURE + b"\0" * 64)
    made["hdf5 file"] = (h5, True)

    user_block = root / "user_block.h5"
    user_block.write_bytes(b"\0" * 512 + HDF5_SIGNATURE + b"\0" * 64)
    made["hdf5 with a 512-byte user block"] = (user_block, True)

    # Past the 1 MiB where main.nf's copy used to stop looking: loadable, and
    # was refused at launch while the task would have accepted it.
    big_block = root / "big_user_block.h5"
    big_block.write_bytes(b"\0" * (2 * 1048576) + HDF5_SIGNATURE + b"\0" * 64)
    made["hdf5 with a 2 MiB user block"] = (big_block, True)

    saved = root / "converted.h5"
    saved.mkdir()
    (saved / "saved_model.pb").write_bytes(b"pb")
    made["SavedModel directory under the .h5 name"] = (saved, True)

    half = root / "half_converted.h5"
    half.mkdir()
    made["directory with no saved_model.pb"] = (half, False)

    empty = root / "empty.h5"
    empty.write_bytes(b"")
    made["empty file"] = (empty, False)

    html = root / "error_page.h5"
    html.write_bytes(b"<html><body>429 Too Many Requests</body></html>")
    made["html saved as .h5"] = (html, False)

    made["missing"] = (root / "nothing_here.h5", False)
    return made


def groovy_function() -> str:
    """checkpointProblem and the helper it reads through, lifted from main.nf."""
    text = MAIN_NF.read_text(encoding="utf-8")
    lifted = []
    for signature in (r"def readAt\([^)]*\)", r"def checkpointProblem\(path\)"):
        match = re.search(rf"^{signature} \{{.*?^\}}", text, re.MULTILINE | re.DOTALL)
        assert match, f"main.nf no longer defines {signature}"
        lifted.append(match.group(0))
    return "\n\n".join(lifted)


def groovy_verdicts(paths: list[Path], work: Path) -> list[bool]:
    """True where main.nf's checkpointProblem accepts the path."""
    listing = ", ".join("'" + str(p).replace("'", "\\'") + "'" for p in paths)
    script = work / "probe.nf"
    script.write_text(
        groovy_function() + "\n\n"
        "workflow {\n"
        f"    [{listing}].each {{ p -> println \"VERDICT \" + (checkpointProblem(p) == null) }}\n"
        "}\n", encoding="utf-8")
    result = subprocess.run(
        ["nextflow", "-q", "run", str(script), "-ansi-log", "false"],
        cwd=work, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    verdicts = [line.split()[1] == "true" for line in result.stdout.splitlines()
                if line.startswith("VERDICT ")]
    assert len(verdicts) == len(paths), result.stdout
    return verdicts


def test_python_accepts_exactly_what_load_model_can_load(tmp_path):
    for name, (path, loadable) in fixtures(tmp_path / "f").items():
        problem = checkpoint_problem(path)
        assert (problem is None) == loadable, f"{name}: {problem!r}"


def test_main_nf_agrees_with_the_task(tmp_path):
    if shutil.which("nextflow") is None:
        print("SKIP  nextflow not installed")
        return
    made = fixtures(tmp_path / "f")
    names = list(made)
    paths = [made[name][0] for name in names]
    groovy = groovy_verdicts(paths, tmp_path)
    python = [checkpoint_problem(path) is None for path in paths]
    disagree = [name for name, g, p in zip(names, groovy, python) if g != p]
    assert not disagree, f"main.nf and anorak_common disagree on: {disagree}"


def test_the_task_uses_the_shared_definition():
    """The original bug was a second, private predicate in the task."""
    source = (HERE.parent / "bin" / "anorak_predict.py").read_text(encoding="utf-8")
    assert "checkpoint_problem(" in source
    assert ".is_file()" not in source.split("checkpoint_problem(")[0].split("def main")[-1]


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="checkpoint_predicate_"))
        try:
            fn(tmp_path) if fn.__code__.co_argcount else fn()
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
