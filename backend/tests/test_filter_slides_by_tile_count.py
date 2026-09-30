#!/usr/bin/env python3
"""Tests for filter_slides_by_tile_count.py.

The script only drops rows, so whatever it reads it writes back out — and the
file it writes is what ANORAK is given. pandas' default read rewrites
identifiers on the way through (a numeric `samples` column with a blank turns
`1001` into `1001.0`, `00123` loses its zeros, `NA` becomes a blank), so a
round trip through it produced a slide list whose ids no longer match the ones
they came from, with nothing to say so.

Runs under pytest and standalone.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import filter_slides_by_tile_count as flt  # noqa: E402

HEADER = "slide_id,dataset_id,samples,malignant_fraction,malignant_tiles,total_tiles,n_malignant_hpcs,is_tumour\n"


def test_kept_rows_are_written_back_exactly_as_read(tmp_path):
    source = tmp_path / "slides.csv"
    kept_rows = [
        "00123,DS,1001,0.5,50,100,1,True\n",
        "NA,DS,NA,0.2,20,100,1,True\n",
        "None,DS,,0.3,30,100,1,True\n",
    ]
    source.write_text(HEADER + "".join(kept_rows) + "THIN,DS,1002,0.01,1,100,1,True\n")
    out = tmp_path / "kept.csv"

    assert flt.main([str(source), "--min-malignant-tiles", "10", "--out", str(out)]) == 0

    assert out.read_text() == HEADER + "".join(kept_rows)
    assert not out.with_suffix(".csv.tmp").exists()


def test_a_blank_count_is_refused_not_read_as_zero(tmp_path):
    """Reading as text must not turn a blank count into a pass: the cut is
    made on this number, so it is refused rather than guessed."""
    source = tmp_path / "slides.csv"
    source.write_text(HEADER + "A,DS,S1,0.5,,100,1,True\n")
    out = tmp_path / "kept.csv"

    assert flt.main([str(source), "--out", str(out)]) == 1
    assert not out.exists()


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_tile_count_"))
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
