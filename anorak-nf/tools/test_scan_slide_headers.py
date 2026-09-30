#!/usr/bin/env python3
"""Tests for scan_slide_headers.

Runs under pytest and standalone (`python3 tools/test_scan_slide_headers.py`),
the way the rest of this project's suites do, because the cluster has no
pytest.

Every test here asks whether a check can come out BAD. A header scan that only
ever sees good headers, and a resolver that only ever sees resolvable ids,
would both pass while reporting nothing — which is the exact failure this tool
exists to prevent, one level up.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_slide_headers as scan

OBJ, MPP = scan.PROP_OBJECTIVE, scan.PROP_MPP_X


# --- header_verdict: the refusal anorak_tile.py makes --------------------

def test_a_complete_header_passes():
    ok, objective, mpp, reason = scan.header_verdict({OBJ: "40", MPP: "0.2325"})
    assert ok and reason == ""
    assert (objective, mpp) == (40.0, 0.2325)


def test_a_slide_with_no_mpp_is_refused():
    ok, _, _, reason = scan.header_verdict({OBJ: "40"})
    assert not ok
    assert "microns-per-pixel" in reason


def test_a_slide_with_no_objective_is_refused():
    ok, _, _, reason = scan.header_verdict({MPP: "0.25"})
    assert not ok
    assert "objective power" in reason


def test_a_slide_missing_both_names_both():
    ok, _, _, reason = scan.header_verdict({})
    assert not ok
    assert "objective power" in reason and "microns-per-pixel" in reason


def test_zero_is_refused_not_treated_as_present():
    # The field is there and parses; upstream would divide by it.
    ok, _, _, reason = scan.header_verdict({OBJ: "40", MPP: "0"})
    assert not ok
    assert "must be > 0" in reason


def test_a_non_numeric_header_is_refused_and_says_so():
    ok, _, _, reason = scan.header_verdict({OBJ: "40", MPP: "unknown"})
    assert not ok
    assert "non-numeric" in reason


def test_the_verdict_matches_anorak_tiles_own_rule():
    """The scan is only worth running if it refuses exactly what the task does."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
    import anorak_tile

    class FakeSlide:
        def __init__(self, properties):
            self.properties = properties

    for properties in ({OBJ: "40"}, {MPP: "0.25"}, {}, {OBJ: "0", MPP: "0.25"},
                       {OBJ: "40", MPP: "-1"}, {OBJ: "40", MPP: "nonsense"}):
        scan_ok = scan.header_verdict(properties)[0]
        try:
            anorak_tile.slide_scale(FakeSlide(properties), Path("x.ndpi"))
            task_ok = True
        except SystemExit:
            task_ok = False
        assert scan_ok == task_ok, f"disagreement on {properties}"


# --- resolve_rows: main.nf's rules ---------------------------------------

def _index(tmp, names):
    for name in names:
        (tmp / name).write_bytes(b"")
    return scan.index_raw_dir(tmp)


def test_an_id_resolves_by_filename_and_by_stem(tmp_path):
    index = _index(tmp_path, ["S1.ndpi"])
    by_stem, _, _ = scan.resolve_rows([{"slide_id": "S1"}], index, "slide_id")
    by_name, _, _ = scan.resolve_rows([{"slide_id": "S1.ndpi"}], index, "slide_id")
    assert by_stem[0][1].name == by_name[0][1].name == "S1.ndpi"


def test_an_unresolvable_id_is_reported_not_skipped(tmp_path):
    index = _index(tmp_path, ["S1.ndpi"])
    resolved, missing, _ = scan.resolve_rows(
        [{"slide_id": "S1"}, {"slide_id": "S404"}], index, "slide_id")
    assert [s for s, _ in resolved] == ["S1"]
    assert missing == ["S404"]


def test_an_id_matching_two_files_is_ambiguous_not_first_wins(tmp_path):
    # Directory order would otherwise decide which slide gets graded.
    index = _index(tmp_path, ["S1.ndpi", "S1.svs"])
    resolved, _, ambiguous = scan.resolve_rows([{"slide_id": "S1"}], index, "slide_id")
    assert resolved == []
    assert set(p.suffix for p in ambiguous["S1"]) == {".ndpi", ".svs"}


def test_an_unsupported_extension_is_not_indexed(tmp_path):
    # save_cws dispatches on extension and returns silently otherwise, so a
    # .jpg named like a slide must not resolve.
    index = _index(tmp_path, ["S1.jpg"])
    _, missing, _ = scan.resolve_rows([{"slide_id": "S1"}], index, "slide_id")
    assert missing == ["S1"]


def test_names_with_spaces_resolve(tmp_path):
    index = _index(tmp_path, ["BB232181 A3-1 - 2023-09-06 22.15.28.ndpi"])
    resolved, missing, _ = scan.resolve_rows(
        [{"slide_id": "BB232181 A3-1 - 2023-09-06 22.15.28"}], index, "slide_id")
    assert not missing and len(resolved) == 1


def test_the_supported_list_is_the_one_main_nf_uses():
    """Drift here means the scan resolves slides the run will not, or worse."""
    main_nf = (Path(__file__).resolve().parent.parent / "main.nf").read_text()
    listed = main_nf.split("def SUPPORTED = [", 1)[1].split("]", 1)[0]
    from_nf = tuple(part.strip().strip("'\"") for part in listed.split(","))
    assert from_nf == scan.SUPPORTED


# --- --resolve-only and main.nf's list ----------------------------------------

def test_the_supported_list_is_read_from_main_nf(tmp_path):
    main_nf = tmp_path / "main.nf"
    main_nf.write_text("    def SUPPORTED = ['.svs', '.ndpi']\n")
    assert scan.supported_from_main_nf(main_nf) == (".svs", ".ndpi")
    assert scan.supported_from_main_nf(tmp_path / "absent.nf") is None


def _resolve_only(tmp_path, capsys_lines):
    import contextlib, io, json
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = scan.main(["--resolve-only", "--slides-csv", str(tmp_path / "list.csv"),
                        "--raw-dir", str(tmp_path / "raw")])
    line = [l for l in out.getvalue().splitlines() if l.startswith("RESOLVE_SUMMARY ")][-1]
    return rc, json.loads(line.split(" ", 1)[1])


def test_resolve_only_names_mrxs_without_their_data_directory(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    for name in ("S1.ndpi", "S2.mrxs", "S3.mrxs"):
        (raw / name).write_bytes(b"")
    (raw / "S3").mkdir(); (raw / "S3" / "Slidedat.ini").write_text("")
    (tmp_path / "list.csv").write_text("slide_id\nS1\nS2\nS3\n")
    rc, summary = _resolve_only(tmp_path, None)
    if ".mrxs" in scan.SUPPORTED:
        assert rc == 0 and summary["resolved"] == 3
    assert sorted(summary["mrxs"]) == ["S2", "S3"]
    assert summary["mrxs_without_data_dir"] == ["S2"]


def test_resolve_only_finds_mrxs_that_main_nf_no_longer_accepts(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    (raw / "S2.mrxs").write_bytes(b"")
    (tmp_path / "list.csv").write_text("slide_id\nS2\n")
    saved = scan.SUPPORTED
    scan.SUPPORTED = tuple(e for e in saved if e != ".mrxs")
    try:
        rc, summary = _resolve_only(tmp_path, None)
    finally:
        scan.SUPPORTED = saved
    assert rc == 1 and summary["missing"] == 1
    assert summary["mrxs"] == ["S2"]


if __name__ == "__main__":
    import tempfile, traceback
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            if "tmp_path" in test.__code__.co_varnames[:test.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as tmp:
                    test(Path(tmp))
            else:
                test()
            print(f"  ok    {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
