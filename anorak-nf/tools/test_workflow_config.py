#!/usr/bin/env python3
"""conf/beatson.config resolves the way the comments in it say.

  - executor.queueSize follows params.queue_size, including from the command
    line, because the preflight check of queue_size + head job + chain
    standbys against MaxSubmitJobs is only meaningful if the number it checks
    is the number Nextflow uses;
  - tiling does not use node-local scratch (a slide's GBs of tiles filled
    $TMPDIR on busy nodes), and nor does publishing, which hard-links into
    outdir and cannot do it from another filesystem.

Resolved by real Nextflow on a probe script; skipped without it. Runs under
pytest and standalone.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEATSON = HERE.parent / "conf" / "beatson.config"

PROBE = """
workflow {
    def c = workflow.session.config
    println "QS=" + c.navigate('executor.queueSize')
    ['process_tiling', 'process_publish', 'process_gpu', 'process_stitch'].each { l ->
        def s = c.process["withLabel:${l}"]?.scratch
        println "SCRATCH ${l}=" + (s == null ? c.process.scratch : s)
    }
}
"""


def resolve(tmp: Path, *extra: str) -> dict[str, str]:
    probe = tmp / "probe.nf"
    probe.write_text(PROBE)
    result = subprocess.run(["nextflow", "-q", "run", str(probe), "-c", str(BEATSON), *extra],
                            cwd=tmp, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    out = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def test_queue_size_comes_from_the_param(tmp_path):
    if shutil.which("nextflow") is None:
        print("SKIP  nextflow not installed")
        return
    assert resolve(Path(tmp_path))["QS"] == "200"
    assert resolve(Path(tmp_path), "--queue_size", "37")["QS"] == "37"


def test_tiling_and_publishing_write_straight_to_the_work_dir(tmp_path):
    if shutil.which("nextflow") is None:
        print("SKIP  nextflow not installed")
        return
    out = resolve(Path(tmp_path))
    assert out["SCRATCH process_tiling"] == "false"
    assert out["SCRATCH process_publish"] == "false"
    assert out["SCRATCH process_gpu"] == "true"
    assert out["SCRATCH process_stitch"] == "true"


def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        try:
            fn(Path(tempfile.mkdtemp(prefix="anorak_cfg_")))
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
