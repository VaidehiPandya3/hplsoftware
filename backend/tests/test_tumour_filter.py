"""Selecting the tumour-bearing slides of a cohort.

The filter decides which slides ANORAK runs on at all, so its failure mode is a
cohort that is quietly too small: a slide dropped here is never graded, and
nothing downstream has any way to notice a slide it was never given. Every
refusal in select_tumour_slides.py exists against that, and every test below
proves one of them can actually fail — not that the happy path is happy.

The other half is that the same number is computed two ways — from the KB's
per-slide proportions, and by counting tiles in a Stage 4 CSV — because the KB is
not always ready. Two implementations of one number that could silently disagree
is the bug class this codebase is written against, so the fixture is built by
running the real Stage 6 aggregation over the same CSV and asserting the two
paths land on identical rows.

Runs against SQLite standing in for Postgres, like test_kb_load.py: what is
under test is the aggregation and the guards, not the driver.
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, text  # noqa: E402

import load_hpc_assignments as loader  # noqa: E402
import select_tumour_slides as filt  # noqa: E402
from malignancy import (  # noqa: E402
    UnrecognisedMalignancy,
    describe_malignant,
    parse_malignant,
)

DATASET = "TEST_5x"
REFERENCE = "hpc_reference_leiden_2p5_fold2"

#: Three clusters, two of them malignant, spelled the way the live dictionary
#: spells it ("True"/"False" text in a column of unrecorded type).
CLUSTERS = {"0": "False", "1": "True", "2": "True"}


# --- fixtures -------------------------------------------------------------

def _assignment_csv(tmp_path: Path, per_slide: dict[str, list[str]],
                    name="assignments.csv", margins=None) -> Path:
    """A Stage 4 CSV where each slide's tiles carry the given cluster ids."""
    rows = []
    for slide, clusters in per_slide.items():
        for i, cluster in enumerate(clusters):
            rows.append({
                "samples": slide, "slides": slide, "tiles": f"{i}_{i}.jpeg",
                "leiden_2.5": cluster,
                "vote_margin": (margins or {}).get(slide, 0.8),
                "neighbor_distance": 1.2,
                "hpc_reference": REFERENCE,
            })
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_kb(tmp_path: Path, clusters=None, name="kb.sqlite"):
    """A KB with the dictionary and the two profile tables, and nothing else."""
    engine = create_engine(f"sqlite:///{tmp_path / name}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE hpc_dictionary (hpc_id TEXT, malignant TEXT)"))
        conn.execute(text("""
            CREATE TABLE hpl_profile_summary (
                samples TEXT, slides TEXT, total_tiles INTEGER,
                dominant_hpc TEXT, dataset_id TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE hpl_profile_proportion (
                samples TEXT, slides TEXT, hpc_id TEXT,
                proportion REAL, dataset_id TEXT
            )"""))
        for hpc_id, malignant in (clusters if clusters is not None else CLUSTERS).items():
            conn.execute(text("INSERT INTO hpc_dictionary VALUES (:h, :m)"),
                         {"h": hpc_id, "m": malignant})
    return engine


def _load_profiles(engine, csv_path: Path, dataset_id=DATASET):
    """Fill the profile tables the way Stage 6 would, from the same CSV.

    Deliberately routed through loader.compute_profiles() rather than through
    hand-written proportions: the KB path is only worth comparing against the
    CSV path if the rows it reads are the rows Stage 6 actually writes.
    """
    frame, cluster_column = loader.read_assignments(csv_path)
    proportions, summary = loader.compute_profiles(frame, cluster_column,
                                                 cancer_type=None)
    proportions["dataset_id"] = dataset_id
    summary["dataset_id"] = dataset_id
    with engine.begin() as conn:
        for _, row in summary.iterrows():
            conn.execute(text(
                "INSERT INTO hpl_profile_summary (samples, slides, total_tiles, "
                "dominant_hpc, dataset_id) VALUES (:s, :sl, :t, :d, :ds)"),
                {"s": row["samples"], "sl": row["slides"],
                 "t": int(row["total_tiles"]), "d": row["dominant_hpc"],
                 "ds": dataset_id})
        for _, row in proportions.iterrows():
            conn.execute(text(
                "INSERT INTO hpl_profile_proportion (samples, slides, hpc_id, "
                "proportion, dataset_id) VALUES (:s, :sl, :h, :p, :ds)"),
                {"s": row["samples"], "sl": row["slides"], "h": row["hpc_id"],
                 "p": float(row["proportion"]), "ds": dataset_id})
    return proportions, summary


# --- the malignancy vocabulary -------------------------------------------

def test_an_unrecognised_malignant_value_is_refused_not_coerced(tmp_path):
    """The guard this filter would be worthless without.

    hpc_dictionary.malignant is loosely typed and its column type is one of the
    four things kb_live_schema_2026-08-26.txt never captured, so an unexpected
    spelling is a real possibility. Read as non-malignant — the innocuous
    default — it removes every slide whose tumour sits in that cluster and
    reports a smaller cohort as a success.
    """
    engine = _make_kb(tmp_path, clusters={"0": "False", "1": "True",
                                          "2": "probably"})

    try:
        filt.load_malignancy(engine)
    except SystemExit as e:
        assert "probably" in str(e)
        assert "hpc_id 2" in str(e)
    else:
        raise AssertionError("an unreadable malignant value was accepted")


def test_a_missing_malignant_value_is_not_read_as_benign(tmp_path):
    """NULL is not False. A cluster whose malignancy was never recorded is a
    dictionary row to fix, not a non-malignant cluster."""
    engine = _make_kb(tmp_path, clusters={"0": "False", "1": None})

    try:
        filt.load_malignancy(engine)
    except SystemExit as e:
        assert "hpc_id 1" in str(e)
    else:
        raise AssertionError("a NULL malignant value was read as non-malignant")


def test_a_dictionary_with_no_malignant_cluster_is_refused(tmp_path):
    """Every slide would come out non-tumour, and the cohort would look like
    one with no cancer in it rather than like a reference mismatch."""
    engine = _make_kb(tmp_path, clusters={"0": "False", "1": "False"})

    try:
        filt.load_malignancy(engine)
    except SystemExit as e:
        assert "malignant" in str(e)
    else:
        raise AssertionError("a dictionary with no malignant cluster was accepted")


def test_the_vocabulary_covers_what_the_ui_already_accepted(tmp_path):
    """The rule was lifted out of app_v28.py, so it must not have narrowed on
    the way: both spellings the UI recognised still parse, and an unknown one
    is still shown as "missing" rather than raising in a viewer."""
    for value in ("True", "t", "1", "yes", "Y", "malignant", True, 1):
        assert parse_malignant(value) is True, value
    for value in ("False", "f", "0", "no", "N", "non-malignant",
                  "non malignant", False, 0):
        assert parse_malignant(value) is False, value

    assert describe_malignant("nonsense") == "missing"
    assert describe_malignant(None) == "missing"
    assert describe_malignant("t") == "malignant"

    # An integer that is not a flag is not silently truthy.
    for value in (2, -1):
        try:
            parse_malignant(value)
        except UnrecognisedMalignancy:
            pass
        else:
            raise AssertionError(f"{value!r} was read as a malignancy flag")


# --- the inclusive rule ---------------------------------------------------

def test_a_slide_with_exactly_one_malignant_tile_is_selected(tmp_path):
    """The rule is select, not exclude, and deliberately inclusive: no
    tumour-bearing slide is missed even at the cost of admitting a slide on one
    tile out of thousands. This is the boundary that costs ANORAK the most
    (67x the pixel volume of an HPL pass) and the one a later grade will be
    least supported by, so it is pinned rather than left to the aggregation."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {
        "SLIDE_A": ["0"] * 199 + ["1"],     # one malignant tile in 200
        "SLIDE_B": ["0"] * 200,             # none at all
    })
    malignancy = filt.load_malignancy(engine)

    frame = filt.from_csv(csv, malignancy, dataset_id=DATASET)
    by_slide = frame.set_index("slide_id")

    assert bool(by_slide.loc["SLIDE_A", "is_tumour"]) is True
    assert int(by_slide.loc["SLIDE_A", "malignant_tiles"]) == 1
    assert bool(by_slide.loc["SLIDE_B", "is_tumour"]) is False
    # And the thinness is on the row, not just implied by the verdict.
    assert by_slide.loc["SLIDE_A", "malignant_fraction"] == 1 / 200
    assert filt.report(frame)["single_tile_slides"] == 1


def test_a_rounded_tile_count_cannot_overturn_the_verdict(tmp_path):
    """On the KB path the tile count is derived from a share, so it rounds.
    is_tumour has to come from the fraction — a slide whose malignant tissue
    rounds to zero tiles is still a slide with malignant tissue."""
    frame = filt._finish(pd.DataFrame([{
        "slide_id": "SLIDE_A", "dataset_id": DATASET, "samples": "SLIDE_A",
        "malignant_fraction": 0.0001, "malignant_tiles": 0,
        "total_tiles": 1000, "n_malignant_hpcs": 1,
    }]))

    assert bool(frame.loc[0, "is_tumour"]) is True


# --- the two sources agree ------------------------------------------------

def test_kb_and_csv_paths_agree_on_the_same_cohort(tmp_path):
    """The reason both paths exist is that Stage 6 has not run everywhere. The
    reason this test exists is that they could disagree without anyone
    noticing: one counts tiles, the other multiplies a stored share by a stored
    total, and only their difference would ever show up."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {
        "SLIDE_A": ["0"] * 199 + ["1"],
        "SLIDE_B": ["0"] * 200,
        "SLIDE_C": ["1"] * 40 + ["2"] * 10 + ["0"] * 50,
    })
    _load_profiles(engine, csv)
    malignancy = filt.load_malignancy(engine)

    from_csv = filt.from_csv(csv, malignancy, dataset_id=DATASET)
    from_kb = filt.from_kb(engine, malignancy, dataset_id=DATASET)

    pd.testing.assert_frame_equal(
        from_kb.drop(columns=["malignant_fraction"]),
        from_csv.drop(columns=["malignant_fraction"]),
        check_dtype=False)
    pd.testing.assert_series_equal(
        from_kb["malignant_fraction"], from_csv["malignant_fraction"],
        check_dtype=False)

    # And the numbers themselves, so a test that agreed because both paths were
    # broken the same way would still fail.
    by_slide = from_kb.set_index("slide_id")
    assert int(by_slide.loc["SLIDE_C", "malignant_tiles"]) == 50
    assert int(by_slide.loc["SLIDE_C", "n_malignant_hpcs"]) == 2
    assert int(by_slide.loc["SLIDE_B", "n_malignant_hpcs"]) == 0


# --- the KB path's own refusals -------------------------------------------

def test_an_unknown_cluster_id_is_refused(tmp_path):
    """A cluster with no dictionary row has unknown malignancy, and both ways
    of proceeding are wrong: dropped it loses slides, kept it admits them on
    evidence nobody has. This is a reference mismatch, and it looks exactly
    like a cohort with fewer tumour slides than expected."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 10 + ["7"] * 10})
    malignancy = filt.load_malignancy(engine)

    try:
        filt.from_csv(csv, malignancy, dataset_id=DATASET)
    except SystemExit as e:
        assert "'7'" in str(e) or '"7"' in str(e)
    else:
        raise AssertionError("an unknown cluster id was accepted")


def test_proportions_that_do_not_sum_to_one_are_refused(tmp_path):
    """Stage 6 computes each proportion as a share of the slide's tiles, so
    they sum to 1. A slide short a proportion row understates its malignant
    fraction, in the direction that turns a tumour slide into a clean one."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 50 + ["1"] * 50})
    _load_profiles(engine, csv)
    malignancy = filt.load_malignancy(engine)

    # It passes first, or the deletion below proves nothing.
    assert bool(filt.from_kb(engine, malignancy, dataset_id=DATASET)
                .loc[0, "is_tumour"]) is True

    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM hpl_profile_proportion WHERE hpc_id = '1'"))

    try:
        filt.from_kb(engine, malignancy, dataset_id=DATASET)
    except SystemExit as e:
        assert "sum to 1" in str(e)
    else:
        raise AssertionError("a slide missing half its proportions was accepted")


def test_a_summary_row_without_proportions_is_refused(tmp_path):
    """A half-written Stage 6: a tile count with no composition beside it.
    Read from the proportion table alone the slide would simply be absent —
    indistinguishable from a slide that is not in the cohort — so the query is
    driven from the summary and the gap refused."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 50 + ["1"] * 50})
    _load_profiles(engine, csv)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO hpl_profile_summary (samples, slides, total_tiles, "
            "dominant_hpc, dataset_id) VALUES ('SLIDE_Z', 'SLIDE_Z', 100, '0', "
            f"'{DATASET}')"))
    malignancy = filt.load_malignancy(engine)

    try:
        filt.from_kb(engine, malignancy, dataset_id=DATASET)
    except SystemExit as e:
        assert "SLIDE_Z" in str(e)
    else:
        raise AssertionError("a slide with no composition was accepted")


def test_a_cohort_with_no_rows_is_refused(tmp_path):
    """Stage 4/6 has not run. Reported as such, rather than as a cohort with
    no tumour slides in it — the two read identically in any summary line."""
    engine = _make_kb(tmp_path)
    malignancy = filt.load_malignancy(engine)

    try:
        filt.from_kb(engine, malignancy, dataset_id="NOT_LOADED")
    except SystemExit as e:
        assert "NOT_LOADED" in str(e)
    else:
        raise AssertionError("an unloaded cohort was accepted")


def test_a_dataset_id_disagreement_between_the_profile_tables_is_refused(tmp_path):
    """hpl_profile_summary's UNIQUE constraint is on (samples, slides) and does
    not include dataset_id, so the database permits a slide's proportions to
    claim a different cohort than its summary. Nothing else in the pipeline
    would notice."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 50 + ["1"] * 50})
    _load_profiles(engine, csv)
    with engine.begin() as conn:
        conn.execute(text("UPDATE hpl_profile_proportion SET dataset_id = 'OTHER' "
                          "WHERE hpc_id = '1'"))
    malignancy = filt.load_malignancy(engine)

    try:
        filt.from_kb(engine, malignancy, dataset_id=DATASET)
    except SystemExit as e:
        assert "OTHER" in str(e)
    else:
        raise AssertionError("a cross-cohort proportion row was accepted")


# --- the report -----------------------------------------------------------

def test_the_distribution_shows_what_the_inclusive_rule_admits(tmp_path):
    """The rule is chosen, not offered — but the printed table is the only
    place its cost is visible, so it has to actually count."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {
        "SLIDE_A": ["0"] * 199 + ["1"],        # 0.005 malignant
        "SLIDE_B": ["0"] * 200,                # not tumour
        "SLIDE_C": ["1"] * 100,                # 1.0 malignant
    })
    malignancy = filt.load_malignancy(engine)

    summary = filt.report(filt.from_csv(csv, malignancy, dataset_id=DATASET))

    assert summary["slides"] == 3
    assert summary["tumour_slides"] == 2
    assert summary["non_tumour_slides"] == 1
    # The non-tumour slide's tiles are not counted as work to be done.
    assert summary["tumour_tiles"] == 300
    by_cut = {row["min_fraction"]: row for row in summary["distribution"]}
    assert by_cut[0.0]["slides_kept"] == 2
    assert by_cut[0.01]["slides_kept"] == 1      # SLIDE_A falls out at 1%
    assert by_cut[0.01]["slides_dropped"] == 1


def test_a_clean_cohort_still_comes_out_whole(tmp_path):
    """Every test above proves a refusal fires. This one proves they can all
    hold their peace, which is the other half of a working guard."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {
        "SLIDE_A": ["0"] * 50 + ["1"] * 50,
        "SLIDE_B": ["0"] * 100,
        "SLIDE_C": ["2"] * 100,
    })
    _load_profiles(engine, csv)
    malignancy = filt.load_malignancy(engine)

    frame = filt.from_kb(engine, malignancy, dataset_id=DATASET)

    assert list(frame.columns) == list(filt._OUTPUT_COLUMNS)
    assert len(frame) == 3
    assert frame["is_tumour"].tolist() == [True, False, True]
    assert frame["malignant_tiles"].tolist() == [50, 0, 100]
    assert frame["dataset_id"].unique().tolist() == [DATASET]


# --- what --out writes -----------------------------------------------------

def _run_main(engine, argv):
    """select_tumour_slides.main() against the SQLite KB instead of Postgres."""
    original = filt.make_engine
    filt.make_engine = lambda: engine
    try:
        return filt.main([str(a) for a in argv])
    finally:
        filt.make_engine = original


def test_out_is_a_tumour_only_slide_list_by_default(tmp_path):
    """--out is what the UI tells people to hand ANORAK, and nothing
    downstream reads is_tumour. Written in full by default, every normal slide
    in the cohort was tiled and graded. The composition report is still there,
    behind a flag that says what it is; --tumour-only still parses."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {
        "SLIDE_A": ["0"] * 50 + ["1"] * 50,
        "SLIDE_B": ["0"] * 100,
    })

    default = tmp_path / "default.csv"
    _run_main(engine, ["--from-csv", csv, "--dataset-id", DATASET, "--out", default])
    written = pd.read_csv(default, dtype=str, keep_default_na=False)
    assert written["slide_id"].tolist() == ["SLIDE_A"]
    assert set(written["is_tumour"]) == {"True"}

    legacy = tmp_path / "legacy.csv"
    _run_main(engine, ["--from-csv", csv, "--dataset-id", DATASET, "--out", legacy,
                       "--tumour-only"])
    assert legacy.read_text() == default.read_text()

    full = tmp_path / "full.csv"
    _run_main(engine, ["--from-csv", csv, "--dataset-id", DATASET, "--out", full,
                       "--include-non-tumour"])
    assert pd.read_csv(full)["slide_id"].tolist() == ["SLIDE_A", "SLIDE_B"]


# --- keys that pandas would drop or rewrite --------------------------------

def test_a_null_dataset_id_in_the_kb_is_refused_not_dropped(tmp_path):
    """pandas' groupby drops a NULL key by default, so a malignant slide whose
    summary carries no dataset_id had no row in the list and no message. The
    mismatch guard cannot see it: NULL on both sides is not a mismatch."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 50 + ["1"] * 50,
                                     "SLIDE_B": ["1"] * 100})
    _load_profiles(engine, csv)
    with engine.begin() as conn:
        conn.execute(text("UPDATE hpl_profile_summary SET dataset_id = NULL "
                          "WHERE slides = 'SLIDE_B'"))
        conn.execute(text("UPDATE hpl_profile_proportion SET dataset_id = NULL "
                          "WHERE slides = 'SLIDE_B'"))
    malignancy = filt.load_malignancy(engine)

    try:
        frame = filt.from_kb(engine, malignancy)          # every cohort
    except SystemExit as e:
        assert "SLIDE_B" in str(e) and "dataset_id" in str(e), e
    else:
        raise AssertionError(
            f"a NULL dataset_id was accepted; slides out: {frame['slide_id'].tolist()}")


def test_a_blank_sample_in_the_csv_is_refused_not_dropped(tmp_path):
    """The same silent drop on the CSV path, where a blank `samples` cell
    becomes NaN on the way in and the slide's group disappears."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["0"] * 10 + ["1"] * 10,
                                     "SLIDE_B": ["1"] * 20})
    frame = pd.read_csv(csv, dtype=str, keep_default_na=False)
    frame.loc[frame["slides"] == "SLIDE_B", "samples"] = ""
    frame.to_csv(csv, index=False)
    malignancy = filt.load_malignancy(engine)

    try:
        out = filt.from_csv(csv, malignancy, dataset_id=DATASET)
    except SystemExit as e:
        assert "SLIDE_B" in str(e) and "samples" in str(e), e
    else:
        raise AssertionError(
            f"a blank sample was accepted; slides out: {out['slide_id'].tolist()}")


def test_csv_sample_ids_come_out_as_written(tmp_path):
    """pandas guesses a numeric `samples` column: `007` becomes 7, and `NA`
    becomes a blank. Either is a sample id ANORAK then grades under a name
    that matches nothing it came from."""
    engine = _make_kb(tmp_path)
    csv = _assignment_csv(tmp_path, {"SLIDE_A": ["1"] * 10, "SLIDE_B": ["1"] * 10,
                                     "SLIDE_C": ["1"] * 10})
    frame = pd.read_csv(csv, dtype=str, keep_default_na=False)
    frame["samples"] = frame["slides"].map(
        {"SLIDE_A": "007", "SLIDE_B": "NA", "SLIDE_C": "1001"})
    frame.to_csv(csv, index=False)
    malignancy = filt.load_malignancy(engine)

    out = filt.from_csv(csv, malignancy, dataset_id=DATASET)

    assert out.set_index("slide_id")["samples"].to_dict() == {
        "SLIDE_A": "007", "SLIDE_B": "NA", "SLIDE_C": "1001"}


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_tumour_test_"))
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
