#!/usr/bin/env python3
"""Tests for the ANORAK run's slide list and the check on its grading table.

The run's slide_list.csv is both the input Nextflow grades and the record the
grading table is checked against, so a wrong one is wrong twice with nothing to
disagree. What these guard against: pandas rewriting sample ids on the way
through (`007` -> 7, `NA` -> blank), a cohort list that still holds its
non-tumour slides (nothing downstream reads is_tumour), blank samples that the
grader pools into one tumour, a half-written list, and a grading table short
of some tumours passing as finished because it has the right columns.

Runs under pytest and standalone.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import submit_anorak_nf as sub  # noqa: E402
from test_anorak_chain import FakeSbatch, _cohort, _patch, teardown_function  # noqa: E402,F401


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _refused(fn, *words):
    try:
        fn()
    except ValueError as refusal:
        for word in words:
            assert word in str(refusal), str(refusal)
        return str(refusal)
    raise AssertionError("accepted")


def _submit_with(tmp_path, slides_text, **kwargs):
    cohort = _cohort(tmp_path)
    _write(cohort["slides_csv"], slides_text)
    fake = FakeSbatch()
    _patch(sub, "_run_sbatch_with_retry", fake)
    return cohort, fake, (lambda: sub.submit_anorak_job(**cohort, **kwargs))


# --- the run's copy is the source, cell for cell -------------------------

def test_the_run_list_keeps_sample_ids_exactly(tmp_path):
    source = ("slide_id,samples,is_tumour,malignant_fraction\n"
              "S1,007,True,0.5\n"
              "S2,1001,True,\n"
              "S3,NA,True,0.1\n")
    cohort, _, go = _submit_with(tmp_path, source)
    go()
    copied = (cohort["out_dir"] / "slide_list.csv").read_text(encoding="utf-8")
    assert copied == source, copied


def test_an_interrupted_write_leaves_the_previous_list_whole(tmp_path):
    """Nextflow would read a truncated list as a smaller cohort, and the
    grading table would then be checked against that and pass."""
    path = tmp_path / "slide_list.csv"
    _write(path, "slide_id,samples,is_tumour\nS1,T1,True\nS2,T2,True\n")
    before = path.read_text()

    class Interrupted(pd.DataFrame):
        def to_csv(self, target, *args, **kwargs):
            Path(target).write_text("slide_id,samples,is_tumour\nS1,")
            raise OSError("disk quota exceeded")

    try:
        sub.write_slide_list(Interrupted({"slide_id": ["S1"]}), path, {})
    except OSError:
        pass
    assert path.read_text() == before


# --- lists that would grade the wrong thing ------------------------------

def test_non_tumour_slides_are_refused(tmp_path):
    """select_tumour_slides.py writes every slide unless --tumour-only."""
    _, fake, go = _submit_with(tmp_path, "slide_id,samples,is_tumour\n"
                                         "S1,T1,True\nS2,T2,False\n")
    _refused(go, "is_tumour", "'S2'", "--tumour-only")
    assert fake.calls == []


def test_a_list_without_is_tumour_is_refused(tmp_path):
    _, fake, go = _submit_with(tmp_path, "slide_id,samples\nS1,T1\n")
    _refused(go, "is_tumour")
    assert fake.calls == []


def test_blank_samples_are_refused(tmp_path):
    """Pooled into one tumour that does not exist."""
    _, fake, go = _submit_with(tmp_path, "slide_id,samples,is_tumour\n"
                                         "S1,T1,True\nS2,  ,True\nS3,,True\n")
    _refused(go, "2 slide(s)", "blank", "'S2'")
    assert fake.calls == []


def test_a_list_without_samples_is_refused(tmp_path):
    _, fake, go = _submit_with(tmp_path, "slide_id,is_tumour\nS1,True\n")
    _refused(go, "'samples'")
    assert fake.calls == []


# --- the grading table against the list ----------------------------------

_GRADES_HEADER = "sample,slides,predominant_pattern,iaslc_grade\n"


def _run_dir(tmp_path, samples):
    out = tmp_path / "out"; out.mkdir()
    _write(out / "slide_list.csv",
           "slide_id,samples,is_tumour\n"
           + "".join(f"S{i},{s},True\n" for i, s in enumerate(samples)))
    return out


def test_a_table_matching_the_list_is_finished(tmp_path):
    out = _run_dir(tmp_path, ["007", "NA", "T3", "T3"])
    grades = _write(sub.grades_csv_path(out), _GRADES_HEADER
                    + "007,1,acinar,2\nNA,1,solid,3\nT3,2,lepidic,1\n")
    assert sub.validate_anorak_output(grades) == (True, "")


def test_a_table_short_of_a_tumour_is_not_finished(tmp_path):
    out = _run_dir(tmp_path, ["T1", "T2", "T3"])
    grades = _write(sub.grades_csv_path(out), _GRADES_HEADER
                    + "T1,1,acinar,2\nT2,1,solid,3\n")
    ok, reason = sub.validate_anorak_output(grades)
    assert not ok and "missing" in reason and "T3" in reason, reason


def test_a_tumour_not_in_the_list_is_refused(tmp_path):
    out = _run_dir(tmp_path, ["T1"])
    grades = _write(sub.grades_csv_path(out), _GRADES_HEADER
                    + "T1,1,acinar,2\nT9,1,solid,3\n")
    ok, reason = sub.validate_anorak_output(grades)
    assert not ok and "T9" in reason, reason


def test_sample_ids_are_compared_as_written(tmp_path):
    """A grader that read '007' as 7 wrote a different tumour."""
    out = _run_dir(tmp_path, ["007"])
    grades = _write(sub.grades_csv_path(out), _GRADES_HEADER + "7,1,acinar,2\n")
    ok, reason = sub.validate_anorak_output(grades)
    assert not ok and "007" in reason, reason


def test_a_table_with_no_slide_list_beside_it_is_not_finished(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    grades = _write(sub.grades_csv_path(out), _GRADES_HEADER + "T1,1,acinar,2\n")
    ok, reason = sub.validate_anorak_output(grades)
    assert not ok and "slide_list.csv" in reason, reason


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="anorak_slide_list_test_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        finally:
            teardown_function(fn)
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
