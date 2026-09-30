#!/usr/bin/env python3
"""Tests for the refusals tumour_grade.py makes before it writes a table.

Each guard here stands between a finished cohort and a grading table that is
well-formed and wrong: blank samples pooled into one tumour that does not
exist, `007` read as 7 and colliding with slide `7`, a slide that never
arrived leaving the table short with nothing to say so. They run the real
script as TUMOUR_GRADE does, because the exit status is part of the contract:
65 is what nextflow.config finishes the run on, where any other code is
retried — a refusal that exited 1 would be retried twice and then finish,
and one that exited 0 would publish.

Runs under pytest and standalone.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "bin" / "tumour_grade.py"
sys.path.insert(0, str(HERE.parent / "bin"))
from anorak_common import REFUSAL_EXIT_CODE  # noqa: E402

PX = ["lepidic_px", "papillary_px", "acinar_px",
      "cribriform_px", "micropapillary_px", "solid_px"]


def write_csv(path: Path, header, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def counts_file(tmp: Path, slide_id: str, sample: str, acinar=100) -> Path:
    """One slide's counts, as SLIDE_PROPORTIONS writes them."""
    values = {column: 0 for column in PX}
    values["acinar_px"] = acinar
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in slide_id)
    return write_csv(tmp / f"{name}.{len(list(tmp.glob('*.counts.csv')))}.counts.csv",
                     ["slide_id", "sample", *PX, "pattern_pixels"],
                     [[slide_id, sample, *values.values(), acinar]])


def grade(tmp: Path, counts, expected=None, allow_missing=False):
    command = [sys.executable, str(SCRIPT), "--slide-counts", *map(str, counts),
               "--out-slides", str(tmp / "slides.csv"),
               "--out-tumours", str(tmp / "tumours.csv")]
    if expected is not None:
        command += ["--expected", str(write_csv(tmp / "expected.csv",
                                                ["slide_id", "sample"], expected)),
                    "--out-missing", str(tmp / "missing.csv")]
    if allow_missing:
        command.append("--allow-missing")
    return subprocess.run(command, capture_output=True, text=True)


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_refusals_exit_with_the_code_the_config_finishes_on():
    """nextflow.config's errorStrategy finishes on 65 and retries everything
    else. Pinned against the config text, so the two cannot drift apart."""
    config = (HERE.parent / "nextflow.config").read_text(encoding="utf-8")
    assert f"task.exitStatus == {REFUSAL_EXIT_CODE} ? 'finish'" in config


def test_a_blank_sample_is_refused_not_pooled(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", ""),
                              counts_file(tmp_path, "S2", "")])
    assert result.returncode == REFUSAL_EXIT_CODE, result.stderr
    assert "no sample" in result.stderr
    assert not (tmp_path / "tumours.csv").exists()


def test_leading_zeros_are_identity_not_formatting(tmp_path):
    """'007' and '7' are two slides of two tumours. Read with type guessing
    they were one id, twice — a duplicate refusal after the whole cohort."""
    result = grade(tmp_path, [counts_file(tmp_path, "007", "0012"),
                              counts_file(tmp_path, "7", "12")])
    assert result.returncode == 0, result.stderr
    samples = sorted(row["sample"] for row in read(tmp_path / "tumours.csv"))
    assert samples == ["0012", "12"]


def test_a_sample_called_NA_is_a_sample(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", "NA")])
    assert result.returncode == 0, result.stderr
    assert [row["sample"] for row in read(tmp_path / "tumours.csv")] == ["NA"]


def test_a_slide_that_never_arrived_is_refused(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", "T1")],
                   expected=[["S1", "T1"], ["S2", "T2"]])
    assert result.returncode == REFUSAL_EXIT_CODE, result.stderr
    assert "1 of 2 listed slides have no counts" in result.stderr
    assert not (tmp_path / "tumours.csv").exists()


def test_a_best_effort_sweep_names_what_is_missing(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", "T1")],
                   expected=[["S1", "T1"], ["S2 (repeat)", "T2"]],
                   allow_missing=True)
    assert result.returncode == 0, result.stderr
    assert [row["slide_id"] for row in read(tmp_path / "missing.csv")] == ["S2 (repeat)"]


def test_a_slide_nobody_listed_is_refused_even_best_effort(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", "T1"),
                              counts_file(tmp_path, "OTHER", "T9")],
                   expected=[["S1", "T1"]], allow_missing=True)
    assert result.returncode == REFUSAL_EXIT_CODE, result.stderr
    assert "not in the slide list" in result.stderr


def test_a_relabelled_slide_is_refused(tmp_path):
    result = grade(tmp_path, [counts_file(tmp_path, "S1", "T1")],
                   expected=[["S1", "T2"]])
    assert result.returncode == REFUSAL_EXIT_CODE, result.stderr
    assert "different sample" in result.stderr


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="tumour_grade_guards_"))
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
