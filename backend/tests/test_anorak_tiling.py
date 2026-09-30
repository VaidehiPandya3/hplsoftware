#!/usr/bin/env python3
"""Tests for submit_anorak_tiling.py, the standalone ANORAK tiler.

Every failure here is the quiet kind: a slide skipped as "already tiled" whose
tiles are at another resolution, two array tasks writing one directory, a
throttle that throttles one batch of eight, a queued array with no record of
its job id. None of them crashes; each leaves a directory of right-looking
tiles.

The worker is exercised through the real `--wrap` string the submitter
generates, run by `sh`, against a fake `openslide` and a fake upstream
`save_cws` — the same stand-ins the audit used — so what is tested is the
command Slurm would actually run. The fake tiler writes each tile's content as
the output-mpp it was run at, which is how a test tells old tiles from new.

Runs under pytest and standalone.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import submit_anorak_tiling as tiler  # noqa: E402

REPO = BACKEND.parent

#: A fake openslide whose "slide" is a JSON file giving its header.
FAKE_OPENSLIDE = '''
import json
PROPERTY_NAME_OBJECTIVE_POWER = "openslide.objective-power"
PROPERTY_NAME_MPP_X = "openslide.mpp-x"
class OpenSlide:
    def __init__(self, path):
        header = json.load(open(path))
        self.properties = {PROPERTY_NAME_OBJECTIVE_POWER: str(header["obj"]),
                           PROPERTY_NAME_MPP_X: str(header["mpp"])}
        self.level_dimensions = [(header["w"], header["h"])]
    def __enter__(self): return self
    def __exit__(self, *a): pass
'''

#: A fake upstream tiler: the geometry of cws_generator, tiles and markers in
#: upstream's order, and FAKE_DIE_AFTER to be killed after that many tiles —
#: after the tiles were rewritten, before param.p, as a preemption would.
FAKE_SAVE_CWS = '''
import json, math, os
def single_file_run(file_name, output_dir, input_dir, out_mpp, **kw):
    h = json.load(open(os.path.join(input_dir, file_name)))
    d = os.path.join(output_dir, file_name)
    os.makedirs(d, exist_ok=True)
    side = 2000 * (h["obj"] / (20 * (h["obj"] / 40) * (h["mpp"] / out_mpp)))
    n = (math.ceil((h["h"] - side) / side + 1)
         * math.ceil((h["w"] - side) / side + 1))
    die = int(os.environ.get("FAKE_DIE_AFTER", "-1"))
    for i in range(n):
        if i == die:
            os._exit(137)
        open(os.path.join(d, f"Da{i}.jpg"), "w").write(str(out_mpp))
    for m in ("param.p", "Ss1.jpg", "FinalScan.ini"):
        open(os.path.join(d, m), "w").write(str(out_mpp))
'''

#: 20000x10000 at 0.25 um/px: 18 tiles at output-mpp 0.22, 15 at 0.25 — so a
#: resolution change is visible in the count as well as in the contents.
HEADER = {"obj": 40, "mpp": 0.25, "w": 20000, "h": 10000}


class Fixture:
    def __init__(self, tmp: Path, names=("S (2).svs",)):
        self.tmp = tmp
        self.fakemods = tmp / "fakemods"
        self.fakemods.mkdir()
        (self.fakemods / "openslide.py").write_text(FAKE_OPENSLIDE)
        self.anorak = tmp / "AIgrading"
        (self.anorak / "generating_tile").mkdir(parents=True)
        (self.anorak / "generating_tile" / "save_cws.py").write_text(FAKE_SAVE_CWS)
        self.raw = tmp / "raw dir"
        self.raw.mkdir()
        for name in names:
            (self.raw / name).write_text(json.dumps(HEADER))
        self.out = tmp / "out dir"

    def slide_list(self, ids, name="slides.csv") -> Path:
        path = self.tmp / name
        pd.DataFrame({"slide_id": list(ids)}).to_csv(path, index=False)
        return path

    def submit(self, ids, output_mpp=0.22, **kwargs) -> dict:
        kwargs.setdefault("metadata_sample", 0)
        kwargs.setdefault("dry_run", True)
        return tiler.submit_array(
            self.slide_list(ids), self.raw, self.out, self.anorak,
            output_mpp=output_mpp, python_executable=Path(sys.executable),
            **kwargs)

    def run_task(self, submission, task=0, batch=0, **env) -> subprocess.CompletedProcess:
        """Run the --wrap of a generated sbatch, as Slurm would, under sh."""
        argv = shlex.split(submission["batches"][batch]["sbatch_command"])
        wrap = argv[argv.index("--wrap") + 1]
        full_env = {**os.environ, "PYTHONPATH": str(self.fakemods),
                    "SLURM_ARRAY_TASK_ID": str(task), **env}
        return subprocess.run(["sh", "-c", wrap], env=full_env, text=True,
                              capture_output=True)

    def slide_dir(self, name="S (2).svs") -> Path:
        return self.out / "cws_tiling" / name

    def tiles(self, name="S (2).svs") -> dict[str, str]:
        return {p.name: p.read_text() for p in self.slide_dir(name).glob("Da*.jpg")}


def _check(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


class FakeSbatch:
    """Stands in for _run_sbatch_with_retry, handing out job ids in order."""

    def __init__(self, first_id=1001, refuse_on=None):
        self.calls = []
        self.next_id = first_id
        self.refuse_on = refuse_on          # 1-based call number to refuse

    def __call__(self, command, *args, **kwargs):
        self.calls.append(list(command))
        if self.refuse_on == len(self.calls):
            raise subprocess.CalledProcessError(
                1, command, output="", stderr="sbatch: error: QOSMaxSubmitJobPerUserLimit")
        job_id, self.next_id = self.next_id, self.next_id + 1
        return subprocess.CompletedProcess(command, 0, stdout=f"Submitted batch job {job_id}\n",
                                           stderr="")


def _with_fake_sbatch(fake):
    original = tiler._run_sbatch_with_retry
    tiler._run_sbatch_with_retry = fake
    return original


# --- the skip check (finding 4) --------------------------------------------

def test_a_clean_resubmission_skips_and_keeps_the_tiles(tmp_path):
    """The guard's other half: whole output at the same settings is reused,
    through the real --wrap, for a slide whose name carries a space."""
    fx = Fixture(tmp_path)
    first = _check(fx.run_task(fx.submit(["S (2)"])))
    assert "[DONE]" in first and "[SKIP]" not in first
    assert len(fx.tiles()) == 18
    record = json.loads((fx.slide_dir() / tiler.COMPLETION_RECORD).read_text())
    assert record["output_mpp"] == 0.22 and record["expected_tiles"] == 18

    again = _check(fx.run_task(fx.submit(["S (2)"])))
    assert "[SKIP]" in again, again


def test_a_resubmission_at_another_mpp_retiles(tmp_path):
    """The confirmed bug: tiled at 0.22, resubmitted at 0.25, "[SKIP] Already
    tiled" — the 15 tiles 0.25 expects are all present among the 18 from 0.22
    and the markers exist, so the old check passed and kept tiles at the wrong
    resolution."""
    fx = Fixture(tmp_path)
    _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.22)))

    out = _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.25)))

    assert "[SKIP]" not in out, out
    tiles = fx.tiles()
    assert len(tiles) == 15, sorted(tiles)
    assert set(tiles.values()) == {"0.25"}


def test_a_retile_killed_before_param_p_is_not_skipped_next_time(tmp_path):
    """Complete at 0.25; a re-tile at 0.22 is killed after 16 tiles, before
    param.p. Its markers are the 0.25 run's and Da0..Da14 all exist, so the old
    check skipped the next 0.25 attempt over a directory of 0.22 tiles."""
    fx = Fixture(tmp_path)
    _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.25)))

    killed = fx.run_task(fx.submit(["S (2)"], output_mpp=0.22), FAKE_DIE_AFTER="16")
    assert killed.returncode != 0
    assert not (fx.slide_dir() / tiler.COMPLETION_RECORD).exists()

    out = _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.25)))
    assert "[SKIP]" not in out, out
    tiles = fx.tiles()
    assert len(tiles) == 15 and set(tiles.values()) == {"0.25"}, tiles


def test_a_stray_extra_tile_is_not_a_whole_tiling(tmp_path):
    """Exactly Da0..Da{n-1}: a tile past the end is left over from other
    settings, and the stitch would pick it up by index."""
    fx = Fixture(tmp_path)
    _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.25)))
    (fx.slide_dir() / "Da15.jpg").write_text("stale")

    out = _check(fx.run_task(fx.submit(["S (2)"], output_mpp=0.25)))

    assert "[SKIP]" not in out, out
    assert len(fx.tiles()) == 15 and "Da15.jpg" not in fx.tiles()


def test_the_expected_count_agrees_with_the_pipelines_tiler(tmp_path):
    """Two copies of cws_generator's grid arithmetic, one here and one in
    anorak-nf/bin/anorak_tile.py. If they drift, one of them calls whole
    output incomplete forever, or the reverse."""
    fx = Fixture(tmp_path)
    sys.path.insert(0, str(fx.fakemods))
    sys.path.insert(0, str(REPO / "anorak-nf" / "bin"))
    sys.modules.pop("openslide", None)
    try:
        import anorak_tile  # noqa: PLC0415
        for header in ({"obj": 40, "mpp": 0.25, "w": 20000, "h": 10000},
                       {"obj": 40, "mpp": 0.2325, "w": 98765, "h": 43210},
                       {"obj": 20, "mpp": 0.5040, "w": 51234, "h": 38001},
                       {"obj": 40, "mpp": 0.2520, "w": 3999, "h": 4001}):
            slide = tmp_path / f"s{header['w']}.svs"
            slide.write_text(json.dumps(header))
            for mpp in (0.22, 0.25):
                assert (tiler.expected_tile_count(slide, mpp)
                        == anorak_tile.expected_tile_count(slide, mpp)), (header, mpp)
    finally:
        sys.path.remove(str(fx.fakemods))
        sys.path.remove(str(REPO / "anorak-nf" / "bin"))
        sys.modules.pop("openslide", None)
        sys.modules.pop("anorak_tile", None)


# --- the slide list (findings 3 and 5) --------------------------------------

def test_two_ids_for_one_file_are_refused(tmp_path):
    """'X' and 'X.ndpi' resolve to one file. Compared as text they were two
    slides, two array tasks, one output directory written concurrently."""
    fx = Fixture(tmp_path, names=("X.ndpi",))
    for ids in (["X", "X.ndpi"], ["X", "X"]):
        try:
            tiler.resolve_slides(ids, fx.raw)
        except ValueError as e:
            assert "more than once" in str(e), e
        else:
            raise AssertionError(f"{ids} both resolved to one slide and were accepted")


def test_slide_ids_are_read_as_written(tmp_path):
    """pandas would read `00123` as 123 and `NA` as a blank; each is then a
    slide that resolves to nothing, or to another file."""
    fx = Fixture(tmp_path, names=("00123.svs", "NA.svs"))
    path = tmp_path / "ids.csv"
    path.write_text("slide_id\n00123\nNA\n")

    ids = tiler.read_slide_ids(path)

    assert ids == ["00123", "NA"]
    assert [p.name for p in tiler.resolve_slides(ids, fx.raw)] == ["00123.svs", "NA.svs"]


# --- submission (findings 6 and 7) -------------------------------------------

def test_batches_are_chained_so_the_limit_is_total(tmp_path):
    """%N throttles one array. Submitted together, three batches at 50 were
    150 tasks at once; chained, each waits for the one before it."""
    fx = Fixture(tmp_path, names=("A.svs", "B.svs", "C.svs"))
    fake = FakeSbatch()
    original = _with_fake_sbatch(fake)
    try:
        submission = fx.submit(["A", "B", "C"], batch_size=1, dry_run=False)
    finally:
        tiler._run_sbatch_with_retry = original

    deps = [[a for a in call if a.startswith("--dependency")] for call in fake.calls]
    assert deps == [[], ["--dependency=afterany:1001"],
                    ["--dependency=afterany:1002"]], deps
    assert submission["job_ids"] == ["1001", "1002", "1003"]
    assert submission["status"] == "submitted"


def test_a_resubmission_does_not_rewrite_a_pending_arrays_manifest(tmp_path):
    """Array tasks read their manifest when they start, maybe hours later. A
    second submission rewriting it in place re-points the first array's task N
    at the new list's slide N."""
    fx = Fixture(tmp_path, names=("A.svs", "B.svs"))
    first = fx.submit(["A", "B"])
    first_manifest = Path(first["batches"][0]["manifest_path"])
    before = first_manifest.read_text()

    second = fx.submit(["B"])

    assert Path(second["batches"][0]["manifest_path"]) != first_manifest
    assert first_manifest.read_text() == before


def test_a_failed_later_batch_still_records_the_queued_ones(tmp_path):
    """Batch 0 queued, batch 1 refused by sbatch: the old code raised before
    writing anything, so batch 0 ran with no record of its job id."""
    fx = Fixture(tmp_path, names=("A.svs", "B.svs"))
    original = _with_fake_sbatch(FakeSbatch(refuse_on=2))
    try:
        fx.submit(["A", "B"], batch_size=1, dry_run=False)
    except RuntimeError as e:
        assert "1001" in str(e), e
    else:
        raise AssertionError("a refused sbatch was reported as a submission")
    finally:
        tiler._run_sbatch_with_retry = original

    record = json.loads((fx.out / "anorak_submission.json").read_text())
    assert record["job_ids"] == ["1001"]
    assert record["status"] == "partial"
    assert record["batches"][0]["job_id"] == "1001"


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_anorak_tiling_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
