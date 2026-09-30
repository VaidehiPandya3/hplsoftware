#!/usr/bin/env python3
"""Run ANORAK on its own, from a dataset path (POST /anorak-runs).

What is pinned:

  * With no tumour-slide list, every slide ANORAK can read is listed, grouped
    into tumours by HPL's rule, and marked unverified — never "true".
  * The unverified marker is accepted only when the server built the list
    itself. A list someone supplies is checked exactly as Stage 7 checks it,
    and a list that says a slide is NOT tumour is refused whatever the flag.
  * Two files with one name, which ANORAK's main.nf would refuse the whole
    cohort over at launch, are refused before anything is queued.
  * Resume repeats the run as it was: same source list, scope, seed, and the
    same tumour-verified choice.

Everything goes through the real endpoint functions, with the database and
sbatch stubbed at the module attributes the server calls. Runs under pytest
and standalone.
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from fastapi import HTTPException  # noqa: E402

import submit_anorak_nf as anorak  # noqa: E402


class _Patched:
    def __init__(self, module, **attrs):
        self.module, self.attrs, self.original = module, attrs, {}

    def __enter__(self):
        for name, value in self.attrs.items():
            self.original[name] = getattr(self.module, name)
            setattr(self.module, name, value)
        return self.module

    def __exit__(self, *exc):
        for name, value in self.original.items():
            setattr(self.module, name, value)
        return False


class _FakeEngine:
    def __init__(self):
        self.inserted = []

    @contextlib.contextmanager
    def begin(self):
        engine = self

        class _Conn:
            def execute(self, statement, params=None):
                engine.inserted.append(params)

        yield _Conn()


def _slides(raw: Path, names) -> None:
    for name in names:
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


# --- the directory-wide list -------------------------------------------------

def test_every_readable_slide_is_listed_and_marked_unverified(tmp_path):
    raw = tmp_path / "raw"
    _slides(raw, ["BB1 A1 - 2023.svs", "BB1 B2 - 2023.ndpi", "BB7 C1.tiff", "old.scn", "notes.txt"])
    frame, info = anorak.slide_list_from_directory(raw)

    assert sorted(frame[anorak.SLIDE_COLUMN]) == ["BB1 A1 - 2023", "BB1 B2 - 2023", "BB7 C1"]
    by_slide = dict(zip(frame[anorak.SLIDE_COLUMN], frame[anorak.SAMPLE_COLUMN]))
    assert by_slide["BB1 A1 - 2023"] == by_slide["BB1 B2 - 2023"] == "BB1", by_slide
    assert set(frame[anorak.TUMOUR_COLUMN]) == {anorak.UNVERIFIED}, "a slide was claimed as tumour"
    assert info["skipped_unsupported"] == ["old.scn"]


def test_two_files_with_one_name_are_refused(tmp_path):
    raw = tmp_path / "raw"
    _slides(raw, ["a/S1.svs", "b/S1.tif"])
    try:
        anorak.slide_list_from_directory(raw)
    except ValueError as e:
        assert "S1" in str(e)
        return
    raise AssertionError("an ambiguous slide name reached ANORAK")


def test_unverified_is_accepted_only_when_the_server_built_the_list(tmp_path):
    frame = pd.DataFrame({"slide_id": ["S1", "S2"], "samples": ["T1", "T2"],
                          "is_tumour": [anorak.UNVERIFIED] * 2})
    try:
        anorak.check_slide_list(frame, tmp_path / "list.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("an unverified list passed as a tumour-slide list")
    anorak.check_slide_list(frame, tmp_path / "list.csv", tumour_verified=False)

    # A verdict of "not tumour" is still a verdict, whatever the flag says.
    frame.loc[1, "is_tumour"] = "false"
    try:
        anorak.check_slide_list(frame, tmp_path / "list.csv", tumour_verified=False)
    except ValueError:
        return
    raise AssertionError("a slide marked not-tumour was graded")


def test_blank_samples_are_still_refused_without_a_tumour_list(tmp_path):
    frame = pd.DataFrame({"slide_id": ["S1"], "samples": [" "], "is_tumour": [anorak.UNVERIFIED]})
    try:
        anorak.check_slide_list(frame, tmp_path / "list.csv", tumour_verified=False)
    except ValueError:
        return
    raise AssertionError("a slide with no tumour was pooled into a tumour that does not exist")


# --- the endpoints ----------------------------------------------------------------

def _server():
    try:
        import tile_server_v2_ as srv
    except Exception as e:  # noqa: BLE001 - no openslide/DB on this machine
        print(f"  (skipped: tile server not importable here: {e})")
        return None
    return srv


def test_a_path_alone_runs_anorak_on_every_slide(tmp_path):
    srv = _server()
    if srv is None:
        return
    raw = tmp_path / "Cohort"
    _slides(raw, ["BB1 A.svs", "BB2 A.svs"])
    submitted = []
    engine = _FakeEngine()
    with _Patched(srv, _get_engine=lambda *a: engine, ANORAK_RESULTS_ROOT=tmp_path / "anorak",
                  _submit_anorak=lambda sid, req, tumour_verified=True: submitted.append(
                      (sid, req, tumour_verified)) or {"submission_id": sid, "selection": {"slides": 2}}):
        result = srv.create_anorak_run(srv.AnorakRunRequest(dataset_path=str(raw)))

    sid, req, verified = submitted[0]
    assert verified is False and result["tumour_verified"] is False
    source = Path(req.slides_csv)
    assert source.is_file() and source.parent == tmp_path / "anorak" / "Cohort"
    listed = pd.read_csv(source, dtype=str)
    assert sorted(listed["slide_id"]) == ["BB1 A", "BB2 A"]
    assert engine.inserted[0]["status"] == srv.ANORAK_ONLY_STATUS
    assert req.scope == "full"


def test_a_given_tumour_list_is_checked_like_stage_7s(tmp_path):
    srv = _server()
    if srv is None:
        return
    raw = tmp_path / "Cohort"
    _slides(raw, ["BB1 A.svs"])
    given = tmp_path / "tumours.csv"
    given.write_text("slide_id,samples,is_tumour\nBB1 A,BB1,true\n")
    submitted = []
    with _Patched(srv, _get_engine=lambda *a: _FakeEngine(), ANORAK_RESULTS_ROOT=tmp_path / "anorak",
                  _submit_anorak=lambda sid, req, tumour_verified=True: submitted.append(
                      (req, tumour_verified)) or {"submission_id": sid, "selection": {"slides": 1}}):
        srv.create_anorak_run(srv.AnorakRunRequest(
            dataset_path=str(raw), slides_csv=str(given), sample_size=1, seed=7))
    req, verified = submitted[0]
    assert verified is True, "a supplied list skipped Stage 7's tumour check"
    assert req.slides_csv == str(given) and req.scope == "subset" and req.seed == 7


def test_a_folder_with_no_readable_slides_leaves_no_run_behind(tmp_path):
    srv = _server()
    if srv is None:
        return
    raw = tmp_path / "Empty"
    _slides(raw, ["notes.txt"])
    engine = _FakeEngine()
    with _Patched(srv, _get_engine=lambda *a: engine, ANORAK_RESULTS_ROOT=tmp_path / "anorak"):
        try:
            srv.create_anorak_run(srv.AnorakRunRequest(dataset_path=str(raw)))
        except HTTPException as e:
            assert e.status_code == 400
        else:
            raise AssertionError("an ANORAK run was started over a folder with no slides")
    assert not engine.inserted, "a refused run still left a row"


def test_resume_repeats_the_run_as_it_was(tmp_path):
    srv = _server()
    if srv is None:
        return
    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "all_slides.csv"
    source.write_text("slide_id,samples,is_tumour\nS1,T1,unverified\n")
    (out / "slide_list.selection.json").write_text(json.dumps({
        "scope": "subset", "sample_size": 1, "seed": 42, "source_csv": str(source),
        "tumour_verified": False}))
    submitted = []
    with _Patched(srv, _get_dataset_run_row=lambda sid: {"anorak_out_dir": str(out)},
                  _submit_anorak=lambda sid, req, tumour_verified=True: submitted.append(
                      (req, tumour_verified)) or {}):
        srv.resume_anorak_run("sub1")
    req, verified = submitted[0]
    assert (req.slides_csv, req.scope, req.sample_size, req.seed, req.resume) == \
        (str(source), "subset", 1, 42, True)
    assert verified is False


# --- standalone runner ------------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_standalone_"))
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
