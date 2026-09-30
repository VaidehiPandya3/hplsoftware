"""Stage 4's assigner streams its input in chunks. These tests are about what
streaming could silently have broken.

The dangerous property is that `--centering query` mirrors scanpy's Ingest and
subtracts the mean over *all* queries. That couples every tile to every other
one, so the moment the input is processed in pieces there are two new ways to be
wrong, both of which still emit a complete, well-formed CSV:

  * a chunk centred on its own mean, making a tile's cluster depend on which
    chunk boundary it happened to fall inside;
  * a shard centred on its slice's mean, making cluster IDs depend on how many
    jobs the work was split across.

So the tests here are equivalence tests, not smoke tests: chunked must equal
unchunked, and sharded must equal unsharded, bit for bit.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ASSIGN = BACKEND / "assign_hpc_clusters.py"

REF_ROWS, DIM, NCOMP, NCLUST, K = 4000, 32, 16, 12, 25
QUERY_ROWS = 900


def _write_reference(path: Path, seed: int = 0) -> None:
    """A build_hpc_reference.py-shaped .npz, small enough to be fast."""
    import json
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, NCLUST, REF_ROWS).astype(np.int64)
    np.savez(
        path,
        reference=rng.standard_normal((REF_ROWS, NCOMP)).astype(np.float32),
        components=rng.standard_normal((DIM, NCOMP)).astype(np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )


def _write_queries(path: Path, rows: int = QUERY_ROWS, dim: int = DIM, seed: int = 1) -> None:
    """A projections .h5 as feature extraction leaves it."""
    rng = np.random.default_rng(seed)
    # Offset from zero so the query mean is meaningfully non-zero — with a mean
    # of ~0 every centering bug would look like a passing test.
    emb = (rng.standard_normal((rows, dim)) + 3.0).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("img_z_latent", data=emb)
        f.create_dataset("samples", data=np.array([b"S%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("slides", data=np.array([b"slide_%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("tiles", data=np.array([b"%d_%d.jpeg" % (i // 30, i % 30) for i in range(rows)]))


def _run(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ASSIGN), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    if result.returncode != 0:
        raise AssertionError(f"assign failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _assign(ref: Path, h5: Path, out: Path, **flags) -> Path:
    args = ["--reference", str(ref), "--h5", str(h5), "--out", str(out)]
    for key, value in flags.items():
        flag = f"--{key.replace('_', '-')}"
        # store_true flags take no value; passing True has to mean "present".
        if value is True:
            args.append(flag)
        elif value is not False:
            args += [flag, str(value)]
    _run(*args)
    return out


def test_chunk_size_does_not_change_assignments(tmp_path):
    """A tile's cluster must not depend on which chunk it landed in. This is the
    check that the query mean is computed over the file and not per chunk."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    outputs = []
    # 100 forces ragged chunks against 900 rows; 100_000 is a single chunk.
    for chunk in (100, 256, 900, 100_000):
        out = _assign(ref, h5, tmp_path / f"c{chunk}.csv", chunk_size=chunk)
        outputs.append(out.read_bytes())

    for chunk, data in zip((256, 900, 100_000), outputs[1:]):
        assert data == outputs[0], f"chunk_size={chunk} changed the output"


def test_batch_size_does_not_change_assignments(tmp_path):
    """The search batch is a performance knob only — search is per-query
    independent, so this must hold for any value."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    a = _assign(ref, h5, tmp_path / "b64.csv", batch_size=64).read_bytes()
    b = _assign(ref, h5, tmp_path / "b16k.csv", batch_size=16384).read_bytes()
    assert a == b


def test_sharded_with_shared_mean_equals_unsharded(tmp_path):
    """The property that makes sharding safe. Every shard is handed the same
    precomputed mean, so the concatenation must equal a single run exactly."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    whole = _assign(ref, h5, tmp_path / "whole.csv").read_text()

    mean_path = tmp_path / "mean.npy"
    _run("--reference", str(ref), "--h5", str(h5), "--precompute-mean", str(mean_path))
    assert mean_path.is_file()

    bounds = [(0, 225), (225, 450), (450, 675), (675, QUERY_ROWS)]
    pieces = []
    for lo, hi in bounds:
        _run("--reference", str(ref), "--h5", str(h5),
             "--out", str(tmp_path / "shard.csv"),
             "--query-mean", str(mean_path),
             "--row-start", str(lo), "--row-stop", str(hi))
        part = tmp_path / f"shard.rows{lo}-{hi}.csv"
        assert part.is_file(), f"no part written for [{lo}, {hi})"
        pieces.append(part.read_text())

    header = pieces[0].splitlines()[0]
    merged = header + "\n" + "".join(
        "".join(line + "\n" for line in p.splitlines()[1:]) for p in pieces
    )
    assert merged == whole, "sharded output differs from a single run"


def test_shard_without_shared_mean_is_refused(tmp_path):
    """The guard has to earn its place: without it, each shard would centre on
    its own slice and produce different labels. Proven by showing the labels
    really do differ when the mean is not shared."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    # 1. It is refused, and the message says what to do.
    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(ref), "--h5", str(h5),
         "--out", str(tmp_path / "x.csv"), "--row-start", "0", "--row-stop", "225"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert result.returncode != 0
    assert "--query-mean" in result.stderr, result.stderr

    # 2. And the refusal is not pedantry. Centering a slice by its own mean does
    #    change assignments: 'none' centering makes the slice independent, so
    #    running a slice under it and comparing against the same rows of a whole
    #    run under 'query' shows the two spaces disagree.
    whole = _assign(ref, h5, tmp_path / "w.csv", centering="none").read_text().splitlines()
    _run("--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "s.csv"),
         "--centering", "none", "--row-start", "0", "--row-stop", "225")
    part = (tmp_path / "s.rows0-225.csv").read_text().splitlines()
    # 'none' needs no mean, so a slice IS safe there — same rows, same answers.
    assert part[1:] == whole[1:226], "centering=none should be shard-independent"


def test_query_mean_matches_a_whole_array_mean(tmp_path):
    """The streamed float64 accumulation must agree with the obvious
    computation, or every projected coordinate is slightly off."""
    import assign_hpc_clusters as m

    h5 = tmp_path / "q.h5"
    _write_queries(h5, rows=1001)
    with h5py.File(h5, "r") as f:
        expected = np.asarray(f["img_z_latent"][:], dtype=np.float32).mean(axis=0)

    for chunk in (7, 128, 5000):
        got = m.compute_query_mean(h5, "z_latent", chunk, 1001)
        assert np.allclose(got, expected, atol=1e-6), f"chunk={chunk}"


def test_high_dimensional_rep_key_streams(tmp_path):
    """--rep-key h_latent is 1536-d, the case that motivated streaming: the old
    whole-array read was ~86 GB at registry scale. Nothing else exercises it."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    import json
    rng = np.random.default_rng(3)
    np.savez(
        ref,
        reference=rng.standard_normal((500, NCOMP)).astype(np.float32),
        components=rng.standard_normal((1536, NCOMP)).astype(np.float32),
        codes=rng.integers(0, NCLUST, 500).astype(np.int64),
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )
    rows = 300
    with h5py.File(h5, "w") as f:
        f.create_dataset("img_h_latent", data=rng.standard_normal((rows, 1536)).astype(np.float32))
        f.create_dataset("samples", data=np.array([b"S1"] * rows))
        f.create_dataset("slides", data=np.array([b"slide_1"] * rows))
        f.create_dataset("tiles", data=np.array([b"%d.jpeg" % i for i in range(rows)]))

    out = _assign(ref, h5, tmp_path / "h.csv", rep_key="h_latent", chunk_size=64)
    lines = out.read_text().splitlines()
    assert len(lines) == rows + 1


def test_partial_output_is_not_left_behind(tmp_path):
    """A killed run must not leave a CSV holding some of the tiles: nothing
    downstream checks row counts before merging."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    out = tmp_path / "ok.csv"
    _assign(ref, h5, out)
    assert out.is_file()
    assert not out.with_name(out.name + ".partial").exists()


# --- merging the shards --------------------------------------------------
# Same failure shapes as test_shard_merge.py covers for the HDF5 merge. A CSV
# concatenation is simpler but fails identically: joined across a gap or over a
# half-written part, it produces a file with the right columns and no missing
# values that every downstream consumer accepts.

def _write_csv_parts(final: Path, bounds, rows_per=None, header="a,b,hpc_reference"):
    for lo, hi in bounds:
        n = (hi - lo) if rows_per is None else rows_per.get((lo, hi), hi - lo)
        part = final.with_name(f"{final.stem}.rows{lo}-{hi}{final.suffix}")
        part.write_text(header + "\n" + "".join(f"{i},x,ref\n" for i in range(lo, lo + n)))


def _merge_fails(final, *, contains, expected_rows=None):
    from merge_assignment_shards import merge_assignment_shards
    try:
        merge_assignment_shards(final, expected_rows=expected_rows)
    except (ValueError, FileExistsError) as e:
        assert contains in str(e), f"expected {contains!r} in: {e}"
        return
    raise AssertionError(f"expected a failure mentioning {contains!r}")


def test_csv_merge_round_trips(tmp_path):
    from merge_assignment_shards import merge_assignment_shards
    final = tmp_path / "out.csv"
    _write_csv_parts(final, [(0, 100), (100, 200), (200, 300)])
    info = merge_assignment_shards(final, expected_rows=300, cleanup=True)
    assert info["rows"] == 300 and info["parts"] == 3
    lines = final.read_text().splitlines()
    assert len(lines) == 301
    # Row order must come from the filenames, not the glob.
    assert [int(l.split(",")[0]) for l in lines[1:]] == list(range(300))
    assert not list(tmp_path.glob("*.rows*"))


def test_csv_merge_rejects_gap_overlap_and_short_parts(tmp_path):
    for label, bounds, kw in [
        ("Gap in coverage",   [(0, 100), (200, 300)], {}),
        ("overlaps",          [(0, 150), (100, 300)], {}),
        ("last shard is missing", [(0, 100), (100, 200)], {"expected_rows": 300}),
    ]:
        final = tmp_path / f"{label[:6].replace(' ','_')}.csv"
        _write_csv_parts(final, bounds)
        _merge_fails(final, contains=label, **kw)
        assert not final.exists(), "nothing may be written when coverage is bad"

    # A part that died mid-write: its name claims more rows than it holds.
    final = tmp_path / "short.csv"
    _write_csv_parts(final, [(0, 100), (100, 200)], rows_per={(100, 200): 40})
    _merge_fails(final, contains="incomplete", expected_rows=200)


def test_csv_merge_rejects_mismatched_columns(tmp_path):
    final = tmp_path / "cols.csv"
    _write_csv_parts(final, [(0, 100)])
    _write_csv_parts(final, [(100, 200)], header="a,b,different")
    _merge_fails(final, contains="not parts of one run", expected_rows=200)


def test_reference_keys_match_the_builder(tmp_path):
    """submit_cluster_assignment.check_reference validates the .npz before
    queueing. It once required a "labels" key the builder has never written, so
    a correctly built 2.5M-tile reference was rejected as malformed.

    Round-tripping through the real build_hpc_reference.save() is what keeps the
    reader and the writer in step — listing the keys in both places is how they
    drifted in the first place.
    """
    import build_hpc_reference
    from submit_cluster_assignment import check_reference

    out = tmp_path / "ref.npz"
    build_hpc_reference.save(
        {
            "reference": np.zeros((100, NCOMP), np.float32),
            "components": np.zeros((DIM, NCOMP), np.float32),
            "codes": np.arange(100, dtype=np.int64) % NCLUST,
            "categories": [str(i) for i in range(NCLUST)],
            "n_neighbors": 250,
            "groupby": "leiden_2.5",
            "source": "synthetic",
            "mean": None,
        },
        out,
    )

    info = check_reference(out)
    assert info["reference_rows"] == 100
    assert info["reference_dims"] == NCOMP
    assert info["n_clusters"] == NCLUST
    assert info["groupby"] == "leiden_2.5"
    assert info["k"] == 250
    # This reference stores no mean, so --centering reference is unavailable and
    # sharding must go through --query-mean.
    assert info["has_mean"] is False

    # And with a mean, it is reported.
    out2 = tmp_path / "ref_mean.npz"
    build_hpc_reference.save(
        {
            "reference": np.zeros((100, NCOMP), np.float32),
            "components": np.zeros((DIM, NCOMP), np.float32),
            "codes": np.arange(100, dtype=np.int64) % NCLUST,
            "categories": [str(i) for i in range(NCLUST)],
            "n_neighbors": 250,
            "groupby": "leiden_2.5",
            "source": "synthetic",
            "mean": np.zeros(DIM, np.float32),
        },
        out2,
    )
    assert check_reference(out2)["has_mean"] is True


# --- adaptive k ----------------------------------------------------------
#
# The gate re-votes only the tiles whose base-k vote was nearly tied, at a
# wider prefix of the SAME search. Two things could go wrong quietly: the flag
# could be accepted and ignored (a normal-looking CSV that never re-voted), or
# vote_margin could keep describing the discarded base vote while hpc_id came
# from the wide one — which would make Stage 5's --min-margin drop exactly the
# tiles this rescues. So these check the refusals fire and the columns agree.


def _run_expecting_failure(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ASSIGN), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert result.returncode != 0, f"expected a refusal, got:\n{result.stdout}"
    return result.stdout + result.stderr


def test_adaptive_k_no_wider_than_k_is_refused(tmp_path):
    """A re-vote at a k no wider than the base sees the same neighbours and
    cannot change a single label. Accepting it would produce an output that
    looks adaptive and is not."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1", "--adaptive-k", "10",
    )
    assert "not wider" in message

    # Narrower is refused for the same reason, not silently clamped.
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1", "--adaptive-k", "5",
    )
    assert "not wider" in message


def test_adaptive_k_without_a_margin_is_refused(tmp_path):
    """--adaptive-k alone has nothing to gate on. Ignoring it would mean a
    typo'd sweep silently ran the plain configuration."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-k", "25",
    )
    assert "without a positive" in message


def test_adaptive_k_beyond_the_reference_is_refused(tmp_path):
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    message = _run_expecting_failure(
        "--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "o.csv"),
        "--k", "10", "--adaptive-margin", "0.1",
        "--adaptive-k", str(REF_ROWS + 1),
    )
    assert "exceeds" in message


def test_adaptive_off_is_bit_identical_to_before(tmp_path):
    """The widened search only happens when the gate is on. With it off the
    output must match byte for byte, or every existing assignment in the KB is
    now unreproducible."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    plain = _assign(ref, h5, tmp_path / "plain.csv", k=10, distance_weighted=True, distance_power=3)
    explicit_off = _assign(ref, h5, tmp_path / "off.csv", k=10, distance_weighted=True, distance_power=3,
                           adaptive_margin=0)
    assert plain.read_text() == explicit_off.read_text()

    frame = pd.read_csv(plain)
    assert len(frame) == QUERY_ROWS


def test_adaptive_rewrites_only_low_margin_tiles(tmp_path):
    """Above the threshold nothing may move; below it, the label and the margin
    must both come from the wide vote."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    threshold = 0.30
    base = pd.read_csv(_assign(ref, h5, tmp_path / "base.csv",
                               k=10, distance_weighted=True, distance_power=3))
    adaptive = pd.read_csv(_assign(ref, h5, tmp_path / "adapt.csv",
                                   k=10, distance_weighted=True, distance_power=3,
                                   adaptive_margin=threshold, adaptive_k=25))
    wide = pd.read_csv(_assign(ref, h5, tmp_path / "wide.csv",
                               k=25, distance_weighted=True, distance_power=3))

    assert list(base.columns) == list(adaptive.columns), \
        "adaptive k must not change the CSV schema — the loader reflects on it"

    low = base["vote_margin"] < threshold
    assert low.any(), "threshold too low to exercise the re-vote"
    assert not low.all(), "threshold too high to test the untouched rows"

    # Untouched rows: identical label and identical margin.
    kept = ~low
    assert (adaptive.loc[kept, "leiden_2.5"].to_numpy()
            == base.loc[kept, "leiden_2.5"].to_numpy()).all()
    assert np.allclose(adaptive.loc[kept, "vote_margin"],
                       base.loc[kept, "vote_margin"])

    # Re-voted rows: label and margin both equal a plain k=25 run's, because
    # the wide prefix of one search is the same neighbourhood as searching 25.
    assert (adaptive.loc[low, "leiden_2.5"].to_numpy()
            == wide.loc[low, "leiden_2.5"].to_numpy()).all()
    assert np.allclose(adaptive.loc[low, "vote_margin"],
                       wide.loc[low, "vote_margin"]), \
        "vote_margin still describes the discarded base vote"
    assert np.allclose(adaptive.loc[low, "neighbor_distance"],
                       wide.loc[low, "neighbor_distance"])

    # And it must actually have changed something, or the test proves nothing.
    changed = (adaptive.loc[low, "leiden_2.5"].to_numpy()
               != base.loc[low, "leiden_2.5"].to_numpy())
    assert changed.any(), "the re-vote changed no label at all"


def test_adaptive_survives_chunking_and_reports_the_count(tmp_path):
    """The gate is applied per batch, so it must not become chunk-dependent —
    the same failure mode the centering tests exist for."""
    import pandas as pd
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    frames = []
    for chunk in (100, 100_000):
        out = tmp_path / f"c{chunk}.csv"
        stdout = _run("--reference", str(ref), "--h5", str(h5), "--out", str(out),
                      "--k", "10", "--distance-power", "3",
                      "--adaptive-margin", "0.3", "--adaptive-k", "25",
                      "--distance-weighted", "--chunk-size", str(chunk))
        assert "Re-voted  :" in stdout, "the re-voted count is not reported"
        frames.append(pd.read_csv(out))

    assert frames[0].equals(frames[1])

# --- the submitter forwards the vote -------------------------------------
#
# Every vote knob was absent from the Slurm command before now, so a knob
# measured offline had no way to reach a real assignment and the mismatch was
# invisible: the job ran, wrote a complete CSV, and used the defaults. These
# check the flags arrive and that an inert combination is refused at submit
# time rather than dropped inside the container.


def test_vote_flags_forwards_the_measured_configuration(tmp_path):
    from submit_cluster_assignment import vote_flags
    flags = " ".join(vote_flags(
        k=10, distance_weighted=True, distance_power=3.0, class_weighted=False,
        local_scaling=0, adaptive_margin=0.1, adaptive_k=25,
    ))
    assert flags == ("--distance-weighted --distance-power 3 "
                     "--adaptive-margin 0.1 --adaptive-k 25")


def test_vote_flags_defaults_add_nothing(tmp_path):
    """The default has to stay exactly what Stage 4 already ran, or every
    assignment already in the KB becomes unreproducible."""
    from submit_cluster_assignment import vote_flags
    assert vote_flags(k=None, distance_weighted=False, distance_power=1.0,
                      class_weighted=False, local_scaling=0,
                      adaptive_margin=0.0, adaptive_k=0) == []


def _refused(expected: str, **kwargs) -> None:
    """vote_flags must exit with a message naming the problem.

    try/except rather than pytest.raises: every suite here also has to run
    standalone on the cluster, where pytest is not installed.
    """
    from submit_cluster_assignment import vote_flags
    try:
        flags = vote_flags(**kwargs)
    except SystemExit as e:
        assert expected in str(e), str(e)
    else:
        raise AssertionError(
            f"expected a refusal mentioning {expected!r}, got flags {flags}")


def test_vote_flags_refuses_a_power_without_weighting(tmp_path):
    """distance_power is read only when distance_weighted is on, so this pair
    would otherwise queue a four-hour job that ignored the exponent."""
    _refused("ignored without",
             k=10, distance_weighted=False, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.0, adaptive_k=0)


def test_vote_flags_refuses_adaptive_without_an_explicit_k(tmp_path):
    """k defaults to the reference's own n_neighbors, read inside the job. If
    adaptive_k were checked against that, whether the re-vote does anything
    could not be known until the job was already running."""
    _refused("explicit --k",
             k=None, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.1, adaptive_k=25)


def test_vote_flags_refuses_an_adaptive_k_that_is_not_wider(tmp_path):
    _refused("not wider",
             k=25, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.1, adaptive_k=25)
    _refused("without a positive",
             k=10, distance_weighted=True, distance_power=3.0,
             class_weighted=False, local_scaling=0,
             adaptive_margin=0.0, adaptive_k=25)


def test_the_slurm_command_actually_carries_the_vote_flags(tmp_path):
    """vote_flags could be correct and still never reach the command line."""
    from submit_cluster_assignment import _build_assignment_command, vote_flags
    vote = vote_flags(k=10, distance_weighted=True, distance_power=3.0,
                      class_weighted=False, local_scaling=0,
                      adaptive_margin=0.1, adaptive_k=25)
    command = _build_assignment_command(
        singularity_bin="singularity",
        singularity_image=tmp_path / "image.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz",
        projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv",
        rep_key="z_latent",
        k=10,
        batch_size=16_384,
        validate_against=None,
        vote=vote,
    )
    for flag in ("--distance-weighted", "--distance-power 3",
                 "--adaptive-margin 0.1", "--adaptive-k 25", "--k 10"):
        assert flag in command, f"{flag} never reached the Slurm command"

    # And the default stays clean: no vote flags at all.
    plain = _build_assignment_command(
        singularity_bin="singularity",
        singularity_image=tmp_path / "image.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz",
        projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv",
        rep_key="z_latent",
        k=None,
        batch_size=16_384,
        validate_against=None,
        vote=[],
    )
    for flag in ("--distance-weighted", "--adaptive-margin", "--class-weighted",
                 "--local-scaling"):
        assert flag not in plain

# --- the named vote presets ----------------------------------------------
#
# The presets are what makes the tuned configuration one click instead of seven
# numbers typed correctly. Their whole value is that the server, the API client
# and the UI cannot disagree about what "tuned" means, so these pin the numbers
# and the wiring rather than the plumbing.


def test_the_tuned_preset_is_the_configuration_that_was_measured(tmp_path):
    """97.27% was measured for exactly these settings. If a preset drifts from
    them, the UI goes on displaying the accuracy of a configuration it is no
    longer running — which is worse than displaying nothing."""
    from submit_cluster_assignment import VOTE_PRESETS
    assert VOTE_PRESETS["tuned"]["flags"] == {
        "k": 10,
        "distance_weighted": True,
        "distance_power": 3.0,
        "class_weighted": False,
        "local_scaling": 0,
        "adaptive_margin": 0.15,
        "adaptive_k": 25,
    }
    # 0.15, not 0.1: the sweep's original grid was 0/0.10/0.25 and could not see
    # its own optimum.
    assert VOTE_PRESETS["tuned"]["flags"]["adaptive_margin"] == 0.15


def test_the_legacy_preset_is_what_stage_4_used_to_run(tmp_path):
    """It exists to reproduce an existing assignment exactly, so it has to be
    the plain unweighted vote and produce no flags at all."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote, vote_flags
    assert vote_flags(**resolve_vote("legacy")) == []


def test_every_preset_resolves_to_a_usable_configuration(tmp_path):
    """A preset that vote_flags refuses would be a one-click refusal. Checked
    for all of them so a new preset cannot be added inert."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote, vote_flags
    for name, spec in VOTE_PRESETS.items():
        vote_flags(**resolve_vote(name))          # must not raise
        for field in ("label", "accuracy", "summary", "why", "flags"):
            assert spec.get(field), f"{name} has no {field}"
        assert 0.5 < spec["accuracy"] < 1.0, name


def test_an_unset_override_does_not_erase_the_preset(tmp_path):
    """Every override arrives from an HTTP body where absent fields are None.
    If None meant "set to None" rather than "not specified", sending a preset
    with no overrides would strip it down to nothing."""
    from submit_cluster_assignment import VOTE_PRESETS, resolve_vote
    everything_none = dict.fromkeys(VOTE_PRESETS["tuned"]["flags"], None)
    assert resolve_vote("tuned", **everything_none) == VOTE_PRESETS["tuned"]["flags"]


def test_an_override_applies_on_top_of_the_preset(tmp_path):
    from submit_cluster_assignment import resolve_vote
    tuned = resolve_vote("tuned")
    changed = resolve_vote("tuned", adaptive_margin=0.1)
    assert changed["adaptive_margin"] == 0.1
    assert {k: v for k, v in changed.items() if k != "adaptive_margin"} \
        == {k: v for k, v in tuned.items() if k != "adaptive_margin"}


def test_an_unknown_preset_and_an_unknown_setting_are_both_refused(tmp_path):
    from submit_cluster_assignment import resolve_vote
    try:
        resolve_vote("whatever-sounds-good")
    except SystemExit as e:
        assert "Unknown vote preset" in str(e)
    else:
        raise AssertionError("an unknown preset must be refused")

    try:
        resolve_vote("tuned", distnace_power=3.0)   # typo, deliberately
    except SystemExit as e:
        assert "Not vote settings" in str(e), str(e)
    else:
        raise AssertionError("a misspelled setting must be refused, not ignored")


def test_describe_vote_says_when_a_preset_was_modified(tmp_path):
    """The run record's one line about the vote. A preset name alone would be a
    lie the moment anything was overridden, and that line is the only place two
    CSVs from one reference but different votes can be told apart."""
    from submit_cluster_assignment import describe_vote, resolve_vote
    plain = describe_vote(resolve_vote("tuned"), "tuned")
    assert plain.startswith("tuned:") and "modified" not in plain
    assert "--adaptive-margin 0.15" in plain

    modified = describe_vote(resolve_vote("tuned", adaptive_margin=0.1), "tuned")
    assert "(modified)" in modified
    assert "--adaptive-margin 0.1 " in modified + " "


def test_an_explicit_k_beats_the_presets_k(tmp_path):
    """k is both a plain argument of the submitter and part of a preset. Two
    resolution paths would mean a caller passing k=15 with the tuned preset
    silently getting the preset's 10."""
    from submit_cluster_assignment import resolve_vote
    assert resolve_vote("tuned", k=15)["k"] == 15
    assert resolve_vote("tuned", k=None)["k"] == 10


def test_the_presets_reach_the_slurm_command(tmp_path):
    """A preset that resolves correctly and never reaches the command line
    would be the same bug as before, one level up."""
    from submit_cluster_assignment import (_build_assignment_command,
                                           resolve_vote, vote_flags)
    common = dict(
        singularity_bin="singularity", singularity_image=tmp_path / "i.sif",
        extras_dir=tmp_path / "extras",
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=tmp_path / "ref.npz", projections_h5=tmp_path / "q.h5",
        out_csv=tmp_path / "out.csv", rep_key="z_latent",
        batch_size=16_384, validate_against=None,
    )
    tuned = resolve_vote("tuned")
    command = _build_assignment_command(
        k=tuned["k"], vote=vote_flags(**tuned), **common)
    for flag in ("--k 10", "--distance-weighted", "--distance-power 3",
                 "--adaptive-margin 0.15", "--adaptive-k 25"):
        assert flag in command, f"{flag} never reached the Slurm command"

    legacy = resolve_vote("legacy")
    plain = _build_assignment_command(
        k=legacy["k"], vote=vote_flags(**legacy), **common)
    for flag in ("--distance-weighted", "--adaptive-margin", "--class-weighted",
                 "--local-scaling", "--k "):
        assert flag not in plain, f"legacy must not pass {flag}"


# --- the server / client / UI contract -----------------------------------
#
# Four modules have to agree about the vote: the submitter defines it, the
# server forwards it, the client sends it, the UI displays it. Each seam fails
# differently and none fails visibly, so each is pinned here. These import
# tile_server_v2_, which is heavy but does import cleanly; if that ever stops
# being true these tests say so loudly rather than being skipped.


def test_the_servers_vote_kwargs_are_all_accepted_by_the_submitter(tmp_path):
    """The seam that would 500 at submit time. The server unpacks vote_kwargs()
    into submit_cluster_assignment_job, so a key it does not accept is a
    TypeError on a real submission and on nothing before it."""
    import inspect
    import tile_server_v2_ as srv
    from submit_cluster_assignment import submit_cluster_assignment_job

    accepted = set(inspect.signature(submit_cluster_assignment_job).parameters)
    sent = set(srv.ClusterAssignmentRequest().vote_kwargs())
    assert sent <= accepted, f"the server sends {sorted(sent - accepted)}, which "\
                             f"submit_cluster_assignment_job does not accept"


def test_the_server_defaults_to_the_tuned_preset(tmp_path):
    """The submitter defaults to legacy so no programmatic caller changes
    behaviour by being upgraded; the request model defaults to tuned so the UI
    queues the measured configuration. Both halves matter, so both are pinned."""
    import inspect
    import tile_server_v2_ as srv
    from submit_cluster_assignment import (DEFAULT_VOTE_PRESET,
                                           submit_cluster_assignment_job)

    assert srv.ClusterAssignmentRequest().vote_preset == DEFAULT_VOTE_PRESET == "tuned"
    # The test endpoint's model inherits it, so a sample run and a full run
    # cannot silently use different votes.
    assert srv.ClusterAssignmentTestRequest(
        projections_h5="/x.h5").vote_preset == "tuned"
    submitter_default = inspect.signature(
        submit_cluster_assignment_job).parameters["vote_preset"].default
    assert submitter_default is None, (
        "the submitter must not default to a preset — an existing caller would "
        "change behaviour just by being upgraded")


def test_the_served_presets_carry_everything_the_ui_displays(tmp_path):
    """The UI reads label/accuracy/summary/why/flags off this payload and shows
    an accuracy figure next to a named configuration. A field missing here is a
    blank caption; a field wrong here is a confident lie."""
    import tile_server_v2_ as srv
    from submit_cluster_assignment import VOTE_PRESETS

    served = srv.list_vote_presets()
    assert served["default"] in served["presets"]
    assert set(served["presets"]) == set(VOTE_PRESETS)
    for name, spec in served["presets"].items():
        for field in ("label", "accuracy", "summary", "why", "flags"):
            assert spec.get(field), f"{name} is missing {field}"
        # And it must be the same numbers, not a copy that has drifted.
        assert spec["flags"] == VOTE_PRESETS[name]["flags"]
    # The UI's override inputs read these three off flags and format them with
    # :g, so they have to be numbers rather than None.
    tuned = served["presets"]["tuned"]["flags"]
    for field in ("distance_power", "adaptive_margin", "adaptive_k"):
        assert isinstance(tuned[field], (int, float)), field


def test_the_api_client_sends_the_preset_and_the_overrides(tmp_path):
    """Checked on the request body the client builds, because the UI passes the
    preset and overrides separately and a dropped one is invisible: the server
    would apply its default and the run would look fine."""
    import sys
    sys.path.insert(0, str(BACKEND.parent / "app"))
    import api_client

    sent = {}

    class _Spy(api_client.TileServerClient):
        def __init__(self):
            pass

        def _post_json(self, path, body, **kw):
            sent["path"], sent["body"] = path, body
            return {}

    _Spy().start_cluster_assignment(
        "run1", reference=None, overwrite=True,
        vote_preset="tuned", vote_overrides={"adaptive_margin": 0.1})
    assert sent["path"].endswith("/assign-clusters")
    assert sent["body"]["vote_preset"] == "tuned"
    assert sent["body"]["adaptive_margin"] == 0.1
    assert sent["body"]["overwrite"] is True

    # No preset and no overrides must leave the body clean, so the server's own
    # default applies rather than a null overriding it.
    _Spy().start_cluster_assignment("run1")
    assert "vote_preset" not in sent["body"]
    assert not any(k.startswith("adaptive") for k in sent["body"])

    # And the test path carries it too, or a sample run would use a different
    # vote from the full run it is meant to preview.
    _Spy().start_test_cluster_assignment(
        "run1", "/x.h5", vote_preset="legacy")
    assert sent["path"].endswith("/assign-clusters-test")
    assert sent["body"]["vote_preset"] == "legacy"


# --- a truth file that disagrees with itself -----------------------------
#
# Kai's TCGA label CSV lists 100 tiles twice, 96 with two different Leiden
# labels, all on one slide. validate() merges one_to_one so a duplicated key
# cannot silently multiply rows, which meant the acceptance test -- the gate
# CLAUDE.md calls a defect below 99% -- died on a pandas MergeError naming
# neither the file nor the slide. Dropping duplicates blindly would be worse:
# 96 tiles would be scored against whichever label happened to come first.


def _truth(rows) -> "object":
    import pandas as pd
    return pd.DataFrame(rows, columns=["samples", "slides", "tiles", "leiden_2.5"])


def test_conflicting_duplicates_are_excluded_not_resolved(tmp_path):
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "1_1.jpeg", 9),      # same tile, different label
        ("S1", "sl1", "2_2.jpeg", 7),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [("sl1", "1_1.jpeg")]
    assert redundant == 0
    # Excluded entirely — neither 5 nor 9 may survive as "the" answer.
    assert list(clean["tiles"]) == ["2_2.jpeg"]


def test_redundant_duplicates_are_deduplicated_and_kept(tmp_path):
    """The same label twice carries no ambiguity, so the tile stays scoreable."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "1_1.jpeg", 5),
        ("S1", "sl1", "2_2.jpeg", 7),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [] and redundant == 1
    assert sorted(clean["tiles"]) == ["1_1.jpeg", "2_2.jpeg"]
    assert clean.loc[clean["tiles"] == "1_1.jpeg", "leiden_2.5"].tolist() == [5]


def test_every_row_is_accounted_for(tmp_path):
    """kept + 2 per conflicting pair + redundant must equal the input, or rows
    are going missing somewhere other than the two documented reasons."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "1_1.jpeg", 9),
        ("S1", "sl1", "2_2.jpeg", 7), ("S1", "sl1", "2_2.jpeg", 7),
        ("S1", "sl2", "3_3.jpeg", 1),
    ])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert len(clean) + 2 * len(ambiguous) + redundant == len(truth)


def test_a_clean_truth_file_is_returned_untouched(tmp_path):
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "2_2.jpeg", 7)])
    clean, ambiguous, redundant = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [] and redundant == 0
    assert clean.equals(truth)


def test_a_tile_is_dropped_by_key_not_by_position(tmp_path):
    """The exclusion is an anti-join on (slides, tiles). Matching on the tile
    name alone would drop the same tile coordinate from every other slide —
    silently shrinking the comparison across the whole cohort."""
    from assign_hpc_clusters import _dedupe_truth
    truth = _truth([
        ("S1", "sl1", "1_1.jpeg", 5), ("S1", "sl1", "1_1.jpeg", 9),
        ("S2", "sl2", "1_1.jpeg", 3),      # same tile name, different slide
    ])
    clean, ambiguous, _ = _dedupe_truth(truth, "leiden_2.5")
    assert ambiguous == [("sl1", "1_1.jpeg")]
    assert list(clean["slides"]) == ["sl2"], "the other slide's tile was dropped too"


def test_validate_scores_no_tile_against_an_arbitrary_label(tmp_path):
    """Goes through validate(), not the helper, because that is where the naive
    fix lives. `drop_duplicates(keep="first")` would score a conflicting tile
    against whichever of its two labels the CSV happened to list first — this
    hands it the OTHER one, so the naive path reports a disagreement where the
    honest path reports an excluded tile."""
    import contextlib
    import io
    import pandas as pd
    from assign_hpc_clusters import validate

    truth_path = tmp_path / "truth.csv"
    _truth([
        ("S1", "sl1", "1_1.jpeg", 5),      # listed first
        ("S1", "sl1", "1_1.jpeg", 9),      # and again, differently
        ("S1", "sl1", "2_2.jpeg", 7),
    ]).to_csv(truth_path, index=False)

    frame = pd.DataFrame({
        "samples": ["S1", "S1"], "slides": ["sl1", "sl1"],
        "tiles": ["1_1.jpeg", "2_2.jpeg"],
        "leiden_2.5": [9, 7],              # 9 is the label keep="first" discards
        "vote_margin": [0.9, 0.9],
    })

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        agreed = validate(frame, truth_path, "leiden_2.5")
    text = out.getvalue()

    # Honest: the ambiguous tile is excluded, so 1 tile matched and it agrees.
    # Naive keep-first: 2 matched, one scored against 5, agreement 50%.
    assert agreed is True, text
    assert "1 of 2 tiles matched" in text, text
    assert "excluded" in text and "DIFFERENT" in text, text
    assert "agreement 100.000%" in text, text


def test_validate_survives_the_real_label_files_duplicates(tmp_path):
    """End to end on Kai's actual CSV if it is present, since that is the file
    the acceptance test names and the one that used to crash."""
    import pandas as pd
    from assign_hpc_clusters import validate
    real = BACKEND.parent / "TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv"
    if not real.is_file():
        return          # not on this machine; the unit tests above still hold
    truth = pd.read_csv(real)
    # Present but not Kai's file. On the cluster this filename holds a derived
    # copy whose cluster column has been renamed to hpc_id, and validate() quite
    # correctly refuses a truth file with no groupby column — which is a finding
    # about that file, not about the join this test covers. Skipping keeps the
    # test honest rather than asserting against whatever happens to sit at the
    # path; the same refusal would meet --validate-against, so it is worth
    # saying out loud.
    if "leiden_2.5" not in truth.columns:
        print(f"    (skipped: {real.name} has no 'leiden_2.5' column — "
              f"{list(truth.columns)}. --validate-against would refuse this "
              f"file too.)")
        return

    # An "assignment" that agrees with the truth everywhere it is defined, built
    # from the truth itself so the only thing under test is the join.
    frame = truth.drop_duplicates(["slides", "tiles"], keep=False).copy()
    frame["vote_margin"] = 0.9
    assert validate(frame, real, "leiden_2.5") is True


# --- container binds -----------------------------------------------------
#
# A real failure: `--validate-against <bare filename.csv>` made
# validate_against.parent == Path("."), which _bind_args turned into
# "--bind .:.". Singularity resolves the source against the job's cwd but leaves
# the destination relative, so it refused with an error naming an ABSOLUTE source
# path and complaining the destination was not absolute -- pointing nowhere near
# the relative argument that caused it. The job died in seconds after queueing.


def test_a_relative_path_never_produces_a_relative_bind(tmp_path):
    from submit_feature_extraction import _bind_args
    specs = [b for b in _bind_args("assign_hpc_clusters.py") if ":" in b]
    assert specs, "no binds produced at all"
    for spec in specs:
        source, destination = spec.split(":", 1)
        assert destination.startswith("/"), f"relative bind destination: {spec}"
        assert source.startswith("/"), f"relative bind source: {spec}"


def test_the_symlink_and_its_target_are_still_bound_separately(tmp_path):
    """The fix uses abspath, not resolve(). resolve() would follow the symlink
    and collapse the two candidates into one, losing the /hpc-home side -- which
    is the entire reason _bind_args binds each path twice."""
    from submit_feature_extraction import _bind_args
    target = tmp_path / "real"
    target.mkdir()
    (target / "file.txt").write_text("x")
    link = tmp_path / "link"
    link.symlink_to(target)

    specs = [b.split(":", 1)[0] for b in _bind_args(link / "file.txt") if ":" in b]
    # Compared by realpath, not by string: on macOS tempfile hands back
    # /var/folders/... where /var is itself a symlink to /private/var, so the
    # bound realpath form resolves two links at once and never equals
    # str(target). Under pytest tmp_path is already resolved and it does — which
    # is why this passed there and failed standalone.
    import os
    assert str(link) in specs, f"the symlink form was not bound: {specs}"
    resolved = {os.path.realpath(spec) for spec in specs}
    assert os.path.realpath(target) in resolved, \
        f"the realpath form was not bound: {specs}"
    # Two distinct entries, which is the property that matters: collapsing them
    # is what resolve() would do and what would break the /hpc-home side.
    assert len({os.path.realpath(link), os.path.realpath(target)}) == 1
    assert str(link) not in {str(target)}, "the fixture did not create a symlink"


def test_a_relative_validate_against_is_bound_absolutely(tmp_path):
    """End to end through the command builder, which is where the relative path
    actually entered — _bind_args is only reached via validate_against.parent."""
    from submit_cluster_assignment import _build_assignment_command
    truth = tmp_path / "truth.csv"
    truth.write_text("samples,slides,tiles,leiden_2.5\n")

    import os
    previous = os.getcwd()
    os.chdir(tmp_path)
    try:
        command = _build_assignment_command(
            singularity_bin="singularity", singularity_image=tmp_path / "i.sif",
            extras_dir=tmp_path / "extras",
            assign_script=BACKEND / "assign_hpc_clusters.py",
            reference=tmp_path / "ref.npz", projections_h5=tmp_path / "q.h5",
            out_csv=tmp_path / "out.csv", rep_key="z_latent", k=None,
            batch_size=16_384,
            validate_against=Path("truth.csv"),   # bare filename, as a user types
            vote=[],
        )
    finally:
        os.chdir(previous)

    assert " .:." not in command and "--bind .:." not in command, command
    # And the truth file still reaches the job as an absolute path.
    assert str(truth.resolve()) in command or str(truth) in command, command


# --- Slurm walltime ------------------------------------------------------
#
# The assigner has no resume: it opens its output with mode='w' before encoding
# and its "output already exists" path crashes on an unbound local. So hitting
# the walltime does not cost the remaining fraction, it costs the whole run —
# and the retry then fails in seconds on the stale output with an error pointing
# nowhere near the cause.
#
# This path submits up to three jobs. Two of them were hardcoded at 2 hours
# while only the third was settable, so raising the limit for a long run left
# two that could still kill it late, after the expensive part had succeeded.


def test_all_three_walltimes_are_settable(tmp_path):
    import inspect
    from submit_cluster_assignment import submit_cluster_assignment_job

    parameters = inspect.signature(submit_cluster_assignment_job).parameters
    for name in ("time_limit", "mean_time_limit", "merge_time_limit"):
        assert name in parameters, f"{name} is not settable"


def test_no_walltime_is_hardcoded_in_an_sbatch(tmp_path):
    """The regression guard. A literal --time= next to an sbatch is a limit
    nobody can raise from the command line."""
    import re
    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    # A literal is "--time=" followed by a digit; "--time={...}" is an
    # interpolation and is what we want. Grepping for the flag alone matches
    # both, which is how the first version of this test failed on correct code.
    literal = re.compile(r"--time=\d")
    hardcoded = [line.strip() for line in source.splitlines()
                 if literal.search(line)]
    assert not hardcoded, f"hardcoded walltime(s): {hardcoded}"


def test_the_assignment_gets_the_longest_limit(tmp_path):
    """It is the only one whose cost is queries x reference rows; the other two
    are single passes over the input and the output. A mean job outliving the
    assignment would mean the numbers were picked without thinking about which
    job actually takes the time."""
    from submit_cluster_assignment import (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT,
                                           MERGE_TIME_LIMIT)

    def seconds(limit: str) -> int:
        days, _, rest = limit.partition("-")
        if not rest:
            rest, days = days, "0"
        hours, minutes, secs = (int(x) for x in rest.split(":"))
        return int(days) * 86400 + hours * 3600 + minutes * 60 + secs

    # A floor, not an exact value. This number is headroom over a measured
    # extrapolation and moves with the cohort size — it went 2 days -> 4 days
    # when 14,000 slides put the unsharded estimate near 20 hours. Pinning it
    # exactly made this test fail for a reason that is not a defect, which is
    # the opposite of what it is for. The ordering below is the real invariant.
    assert seconds(ASSIGN_TIME_LIMIT) >= 2 * 86400
    assert seconds(ASSIGN_TIME_LIMIT) > seconds(MEAN_TIME_LIMIT)
    assert seconds(MEAN_TIME_LIMIT) >= seconds(MERGE_TIME_LIMIT)


def test_every_default_is_a_walltime_slurm_accepts(tmp_path):
    """A malformed --time is rejected by sbatch at submit, which is at least
    loud — but only once someone tries, and these are defaults."""
    import re
    from submit_cluster_assignment import (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT,
                                           MERGE_TIME_LIMIT)
    pattern = re.compile(r"^(\d+-)?\d{1,2}:\d{2}:\d{2}$")
    for limit in (ASSIGN_TIME_LIMIT, MEAN_TIME_LIMIT, MERGE_TIME_LIMIT):
        assert pattern.match(limit), f"{limit!r} is not a Slurm walltime"


# --- the threads the job actually gets ------------------------------------
#
# `singularity exec --cleanenv` wipes the environment before the inner shell
# runs, so a thread count written as "${SLURM_CPUS_PER_TASK:-1}" and expanded
# in there cannot see Slurm's value: the fallback wins and every assignment
# runs on one core. Nothing fails, nothing warns; a 2.5M-row reference at 127
# dimensions came to 49 tiles/s, which is one core's fp32 rate, and an
# 18.5M-tile cohort took three days instead of hours.
#
# The number therefore has to be baked in at submit time, and these check it is
# — including that it agrees with what the sbatch asks Slurm for, since the two
# now live in different strings and drift silently.


def _assignment_command(cpus=16, shards=1):
    """The --wrap payload for a submission, without submitting anything."""
    import submit_cluster_assignment as sca

    return sca._build_assignment_command(
        singularity_bin="singularity",
        singularity_image=Path("/img.sif"),
        extras_dir=Path("/extras"),
        assign_script=BACKEND / "assign_hpc_clusters.py",
        reference=Path("/ref/hpc_reference.npz"),
        projections_h5=Path("/proj/hdf5_x_he_train.h5"),
        out_csv=Path("/out/x_hpc_assignments.csv"),
        rep_key="z_latent",
        k=None,
        batch_size=16384,
        validate_against=None,
        threads=cpus,
    )


def test_the_thread_count_survives_cleanenv(tmp_path):
    """A literal number, because the variable it used to read is gone by then."""
    command = _assignment_command(cpus=32)

    assert "export OMP_NUM_THREADS=32;" in command
    assert "SLURM_CPUS_PER_TASK" not in command, (
        "the thread count is read from a variable --cleanenv has already wiped")


def test_every_container_step_pins_its_threads(tmp_path):
    """The mean and merge steps run through a different builder, which had the
    same bug."""
    import submit_cluster_assignment as sca

    command = sca._build_simple_command(
        singularity_bin="singularity", singularity_image=Path("/img.sif"),
        extras_dir=Path("/extras"), script=BACKEND / "assign_hpc_clusters.py",
        args=["--precompute-mean /out/mean.npy"],
        extra_binds=[Path("/proj")], banner="Query mean", threads=4,
    )

    assert "export OMP_NUM_THREADS=4;" in command
    assert "SLURM_CPUS_PER_TASK" not in command


def test_no_container_command_expands_slurms_cpu_variable(tmp_path):
    """The whole file, so a third builder cannot reintroduce it. Naming the
    variable in a comment is fine — what does not work is *expanding* it, which
    only happens inside the container where --cleanenv has already run."""
    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    offenders = [line.strip() for line in source.splitlines()
                 if "${SLURM_CPUS_PER_TASK" in line]

    assert not offenders, f"expanded inside --cleanenv: {offenders}"


def test_the_threads_asked_for_match_the_cpus_requested(tmp_path):
    """The two numbers are in different strings now — the export in the wrapped
    command, and --cpus-per-task in the sbatch argv. Asking Slurm for 32 cores
    and pinning faiss to 1 is exactly the bug this replaced, in reverse."""
    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    # The mean and merge steps hardcode both numbers; they must agree.
    assert 'threads=4,' in source and '"--cpus-per-task=4"' in source
    assert 'threads=2,' in source and '"--cpus-per-task=2"' in source
    # And the array job passes through whatever was requested.
    assert "threads=cpus," in source


# --- the GPU search ------------------------------------------------------
#
# GpuIndexFlat is the same exhaustive scan as IndexFlat: every query is compared
# against every reference vector. That is what makes it admissible where
# faiss-ivf was not — ivf changed which neighbour came back (33% agreement on
# this reference), and this changes only the hardware.
#
# What cannot be assumed is that a given faiss build's GPU path works. One does
# expose StandardGpuResources, report get_num_gpus() == 1, accept
# index_cpu_to_gpu without error, and still be a stub. So the Searcher verifies
# its GPU index against the CPU index at startup, and these tests are about that
# refusal — the search itself needs no test beyond the equivalence one below,
# which runs only where a GPU is actually present.


def _has_working_gpu_faiss() -> bool:
    import faiss
    if not hasattr(faiss, "StandardGpuResources"):
        return False
    try:
        return faiss.get_num_gpus() > 0
    except Exception:  # noqa: BLE001
        return False


def test_an_unknown_device_is_refused(tmp_path):
    from assign_hpc_clusters import Searcher

    try:
        Searcher(np.zeros((4, 3), dtype=np.float32), device="cuda")
    except ValueError as e:
        assert "device" in str(e)
    else:
        raise AssertionError("'cuda' was accepted; the choices are cpu and gpu")


def test_cpu_is_honoured_when_asked_for(tmp_path):
    """The default is "auto" now, so what has to be pinned is that an explicit
    cpu never tries a GPU — on a host that has one, this is the only way to get
    the CPU index, and every previously-measured number came from it."""
    from assign_hpc_clusters import Searcher

    searcher = Searcher(np.eye(8, dtype=np.float32), device="cpu")
    assert searcher.device == "cpu"
    assert searcher.backend == "faiss-flat"


def test_auto_falls_back_to_cpu_and_says_why(tmp_path):
    """auto is the default, so the fallback has to be both correct and visible:
    a GPU quietly becoming a CPU is a 30x slowdown that looks like nothing."""
    import faiss

    from assign_hpc_clusters import Searcher

    saved = getattr(faiss, "StandardGpuResources", None)
    if saved is not None:
        del faiss.StandardGpuResources
    try:
        searcher = Searcher(np.eye(8, dtype=np.float32))      # auto
        assert searcher.device == "cpu"
        assert searcher.backend == "faiss-flat"
        assert "bootstrap-extras-gpu" in searcher._auto_note, searcher._auto_note
    finally:
        if saved is not None:
            faiss.StandardGpuResources = saved


def test_auto_still_refuses_a_gpu_that_disagrees(tmp_path):
    """The one thing auto must never do is lower the bar on correctness. A
    disagreeing GPU index means a broken build, and a broken build produces a
    complete CSV of wrong cluster IDs whichever flag asked for it — so it is
    refused rather than fallen back from."""
    import faiss

    from assign_hpc_clusters import Searcher

    reference = np.random.default_rng(0).standard_normal((256, 16), dtype=np.float32)

    real_move = getattr(faiss, "index_cpu_to_gpu", None)
    real_resources = getattr(faiss, "StandardGpuResources", None)
    real_count = getattr(faiss, "get_num_gpus", None)

    class _WrongIndex:
        def search(self, queries, k):
            rows = queries.shape[0]
            return (np.full((rows, k), 7.0, dtype=np.float32),
                    np.zeros((rows, k), dtype=np.int64))

    faiss.StandardGpuResources = lambda: object()
    faiss.get_num_gpus = lambda: 1
    faiss.index_cpu_to_gpu = lambda *a, **k: _WrongIndex()
    try:
        try:
            Searcher(reference)                                # auto, not "gpu"
        except SystemExit as e:
            assert "REFUSING --device gpu" in str(e), str(e)
        else:
            raise AssertionError("auto accepted a disagreeing GPU index")
    finally:
        for name, value in (("index_cpu_to_gpu", real_move),
                            ("StandardGpuResources", real_resources),
                            ("get_num_gpus", real_count)):
            if value is not None:
                setattr(faiss, name, value)


def test_a_gpu_index_that_disagrees_with_cpu_is_refused(tmp_path):
    """The stub-build case, forced: a GPU index that answers differently must
    stop the run, because nothing downstream can see the difference. Every tile
    would still get a cluster ID and a plausible margin."""
    import faiss

    from assign_hpc_clusters import Searcher

    reference = np.random.default_rng(0).standard_normal((256, 16), dtype=np.float32)

    real_move = getattr(faiss, "index_cpu_to_gpu", None)
    real_resources = getattr(faiss, "StandardGpuResources", None)

    class _WrongIndex:
        """Answers with the right shapes and the wrong content."""

        def search(self, queries, k):
            rows = queries.shape[0]
            return (np.full((rows, k), 7.0, dtype=np.float32),
                    np.zeros((rows, k), dtype=np.int64))

    faiss.StandardGpuResources = lambda: object()
    faiss.index_cpu_to_gpu = lambda *a, **k: _WrongIndex()
    try:
        try:
            Searcher(reference, device="gpu")
        except SystemExit as e:
            assert "REFUSING --device gpu" in str(e), str(e)
            assert "nearest neighbour" in str(e)
        else:
            raise AssertionError("a disagreeing GPU index was accepted")
    finally:
        if real_move is not None:
            faiss.index_cpu_to_gpu = real_move
        if real_resources is not None:
            faiss.StandardGpuResources = real_resources


def test_a_faiss_without_gpu_support_says_what_to_install(tmp_path):
    import faiss

    from assign_hpc_clusters import Searcher

    real_resources = getattr(faiss, "StandardGpuResources", None)
    if real_resources is not None:
        del faiss.StandardGpuResources
    try:
        try:
            Searcher(np.eye(8, dtype=np.float32), device="gpu")
        except SystemExit as e:
            assert "--bootstrap-extras-gpu" in str(e), str(e)
        else:
            raise AssertionError("device=gpu was accepted without GPU support")
    finally:
        if real_resources is not None:
            faiss.StandardGpuResources = real_resources


def test_gpu_and_cpu_agree_on_the_same_reference(tmp_path):
    """The equivalence claim itself, where there is a GPU to check it on.

    Measured here: 100% agreement on the nearest neighbour, 99.99% across all
    25, distances differing by ~1e-4 from float accumulation order. Near-ties
    can reorder inside the k-list, which is why the assertion is on the top-1
    and on distance closeness rather than on identical arrays.
    """
    if not _has_working_gpu_faiss():
        print("    (skipped: no GPU faiss on this machine)")
        return

    from assign_hpc_clusters import Searcher

    rng = np.random.default_rng(0)
    reference = rng.standard_normal((4000, 64), dtype=np.float32)
    queries = rng.standard_normal((500, 64), dtype=np.float32)

    cpu_i, cpu_d = Searcher(reference, device="cpu").search(queries, 25)
    gpu_i, gpu_d = Searcher(reference, device="gpu").search(queries, 25)

    assert (cpu_i[:, 0] == gpu_i[:, 0]).all(), "the nearest neighbour must match"
    assert np.abs(cpu_d - gpu_d).max() < 1e-2, "distances must agree"


def test_the_gpu_job_asks_for_a_gpu_and_the_cpu_job_does_not(tmp_path):
    """--nv and --gres go together: without the runtime the container sees no
    driver, and without the allocation Slurm gives it no card."""
    import submit_cluster_assignment as sca

    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    assert '"--gres=gpu:1"' in source

    for device, expected in (("cpu", False), ("gpu", True)):
        command = sca._build_assignment_command(
            singularity_bin="singularity", singularity_image=Path("/img.sif"),
            extras_dir=Path("/extras"),
            assign_script=BACKEND / "assign_hpc_clusters.py",
            reference=Path("/ref/r.npz"), projections_h5=Path("/p/x.h5"),
            out_csv=Path("/o/x.csv"), rep_key="z_latent", k=None,
            batch_size=16384, validate_against=None, threads=16, device=device,
        )
        assert ("--nv" in command) is expected, f"{device}: --nv wrong"
        assert f"--device {device}" in command


# --- threads the process can actually use --------------------------------
#
# Baking OMP_NUM_THREADS in at submit time fixed the first half of this: the
# container could not read Slurm's variable, so it always ran on one core. The
# second half is that the variable says what Slurm *allocated*, not what the
# process may touch. Told to use 16 threads on a node whose affinity mask gave
# it fewer usable CPUs, the assignment ran at 20 tiles/s — against 49 tiles/s
# for the same work on a single thread. Oversubscription is worse than serial,
# and nothing about it fails.


def test_threads_never_exceed_the_cpus_the_process_may_use(tmp_path):
    from assign_hpc_clusters import resolve_thread_count

    assert resolve_thread_count(16, 1) == 1, "16 threads on one core is slower than one"
    assert resolve_thread_count(16, 4) == 4
    assert resolve_thread_count(16, 16) == 16
    # Asking for fewer than are available is a deliberate choice, not a mistake.
    assert resolve_thread_count(4, 16) == 4


def test_an_unset_request_takes_what_is_available(tmp_path):
    from assign_hpc_clusters import resolve_thread_count

    assert resolve_thread_count(None, 8) == 8
    assert resolve_thread_count(0, 8) == 8


def test_a_nonsense_affinity_still_yields_one_thread(tmp_path):
    """The guard has to survive its own inputs: zero usable CPUs is not a
    reason to run zero threads."""
    from assign_hpc_clusters import resolve_thread_count

    assert resolve_thread_count(16, 0) == 1
    assert resolve_thread_count(0, 0) == 1
    assert resolve_thread_count(-4, 8) == 8


def test_the_thread_count_is_reported(tmp_path, capsys=None):
    """A job running 16 threads on one core and a job running one thread look
    identical from outside. The log has to say which."""
    import io
    from contextlib import redirect_stdout

    from assign_hpc_clusters import _configure_threads

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        threads = _configure_threads()
    printed = buffer.getvalue()

    assert threads >= 1
    assert "Threads" in printed
    assert "usable by this process" in printed


# --- where the time goes -------------------------------------------------
#
# The loop is read -> project -> search -> vote -> write, strictly in sequence.
# Only the search is threaded (inside faiss) or GPU-accelerable; the rest is
# sequential Python that more cores do nothing for. Which one dominates decides
# whether threads, a GPU, or sharding is the right lever — and the only way that
# was ever established here was by comparing wall clocks by hand.


def test_the_run_reports_which_phase_the_time_went_to(tmp_path):
    reference = tmp_path / "ref.npz"
    queries = tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=1200)

    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(reference),
         "--h5", str(queries), "--out", str(tmp_path / "a.csv"),
         "--chunk-size", "256"],
        capture_output=True, text=True, timeout=600)

    assert result.returncode == 0, result.stderr[-2000:]
    line = [l for l in result.stdout.splitlines() if l.startswith("Time spent:")]
    assert line, f"no phase breakdown in:\n{result.stdout}"
    for phase in ("read", "project", "search", "vote", "write"):
        assert phase in line[0], f"{phase} missing from {line[0]!r}"


def test_the_progress_lines_carry_the_split_too(tmp_path):
    """A run long enough for the split to matter is one nobody wants to wait out
    before learning which phase to attack. This one was three days."""
    reference = tmp_path / "ref.npz"
    queries = tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=1200)

    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(reference),
         "--h5", str(queries), "--out", str(tmp_path / "a.csv"),
         "--chunk-size", "128", "--progress", "256"],
        capture_output=True, text=True, timeout=600)

    assert result.returncode == 0, result.stderr[-2000:]
    progress = [l for l in result.stdout.splitlines() if "tiles/s [" in l]
    assert progress, f"no progress line with a split in:\n{result.stdout}"
    assert "search" in progress[-1], progress[-1]


def test_the_thread_count_appears_in_the_log(tmp_path):
    """So "why is this slow" is answerable from the log alone. 16 threads on one
    core and one thread look identical otherwise, and differ 2.4x."""
    reference = tmp_path / "ref.npz"
    queries = tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=600)

    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(reference),
         "--h5", str(queries), "--out", str(tmp_path / "a.csv")],
        capture_output=True, text=True, timeout=600)

    assert result.returncode == 0, result.stderr[-2000:]
    threads = [l for l in result.stdout.splitlines() if l.startswith("Threads")]
    assert threads, f"no thread line in:\n{result.stdout}"
    assert "usable by this process" in threads[0]


# --- the walltime the partition will actually grant ----------------------
#
# ASSIGN_TIME_LIMIT is four days, chosen for the GPU partition, which allows
# five. `compute` allows two — so submitting a sharded CPU run with the defaults
# always failed, and failed in the worst available way: sbatch refuses with
# "Requested time limit is invalid (missing or exceeds some limit)", naming
# neither the limit nor the value, at the bottom of a traceback carrying a
# 3,000-character --wrap string — and *after* the mean job was queued, leaving an
# orphan waiting on a dependency that would never exist.


def test_slurm_walltimes_parse_in_every_form_both_sides_use(tmp_path):
    from submit_cluster_assignment import parse_slurm_walltime

    assert parse_slurm_walltime("4-00:00:00") == 4 * 86400
    assert parse_slurm_walltime("08:00:00") == 8 * 3600
    assert parse_slurm_walltime("1-12:30:45") == 86400 + 12 * 3600 + 30 * 60 + 45
    assert parse_slurm_walltime("30:00") == 1800
    # sinfo's word for no limit, which must not read as zero seconds.
    for unlimited in ("infinite", "INFINITE", "unlimited", "", "  "):
        assert parse_slurm_walltime(unlimited) is None


def test_a_walltime_over_the_partition_limit_is_refused(tmp_path):
    import submit_cluster_assignment as sca

    original = sca.partition_time_limit
    sca.partition_time_limit = lambda partition: 2 * 86400      # compute
    try:
        try:
            sca.check_time_limit("compute", "4-00:00:00")
        except ValueError as e:
            assert "exceeds partition" in str(e), str(e)
            assert "1-00:00:00" in str(e), "the message must name a value that works"
        else:
            raise AssertionError("4 days was accepted against a 2-day partition")
    finally:
        sca.partition_time_limit = original


def test_a_walltime_within_the_limit_passes(tmp_path):
    """The guard has to be able to pass — and an unknown limit must never block
    a submission, since sinfo being unavailable is not a reason to refuse."""
    import submit_cluster_assignment as sca

    original = sca.partition_time_limit
    try:
        sca.partition_time_limit = lambda partition: 2 * 86400
        sca.check_time_limit("compute", "1-00:00:00")
        sca.check_time_limit("compute", "2-00:00:00")           # exactly the limit
        sca.partition_time_limit = lambda partition: None       # unlimited, or unknown
        sca.check_time_limit("anything", "9-00:00:00")
    finally:
        sca.partition_time_limit = original


def test_the_limit_is_checked_before_anything_is_submitted(tmp_path):
    """Ordering is the point: the mean job goes first, so a walltime rejected by
    sbatch stranded it against a dependency that never came."""
    source = (BACKEND / "submit_cluster_assignment.py").read_text()
    checked = source.index("check_time_limit(partition, time_limit)")
    first_sbatch = source.index("mean_sbatch = [")

    assert checked < first_sbatch, \
        "the walltime is checked after the mean job is built"


# --- a second identical pipeline -----------------------------------------
#
# Retyping the submission queued two full pipelines: two mean jobs writing one
# query_mean.npy, two arrays writing the same 32 part files, and two merges —
# the second of which runs --cleanup and deletes parts the first one's tasks are
# still writing. Two processes writing one part file can produce a file whose
# row count is right and whose contents interleave, which is exactly the failure
# this codebase exists to refuse.


def test_a_second_pipeline_with_the_same_name_is_refused(tmp_path):
    import submit_cluster_assignment as sca

    original = sca.jobs_in_flight_named
    sca.jobs_in_flight_named = lambda name: ["1241571", "1241572", "1241573"]
    try:
        try:
            sca.refuse_if_already_queued("hpl_cluster_assign")
        except ValueError as e:
            assert "already queued or running" in str(e), str(e)
            # The message has to carry the ids, or cancelling means hunting.
            assert "1241572" in str(e)
            assert "scancel" in str(e)
        else:
            raise AssertionError("a duplicate pipeline was accepted")
    finally:
        sca.jobs_in_flight_named = original


def test_nothing_queued_means_nothing_refused(tmp_path):
    """The guard has to pass in the normal case, and must not turn an
    unreachable squeue into a refusal — it prevents an accident, it is not a
    precondition for submitting."""
    import submit_cluster_assignment as sca

    original = sca.jobs_in_flight_named
    sca.jobs_in_flight_named = lambda name: []
    try:
        sca.refuse_if_already_queued("hpl_cluster_assign")
    finally:
        sca.jobs_in_flight_named = original


def test_force_duplicate_overrides_it(tmp_path):
    import submit_cluster_assignment as sca

    original = sca.jobs_in_flight_named
    sca.jobs_in_flight_named = lambda name: ["1241571"]
    try:
        sca.refuse_if_already_queued("hpl_cluster_assign", force=True)
    finally:
        sca.jobs_in_flight_named = original


def test_the_whole_pipeline_is_matched_not_just_the_array(tmp_path):
    """The mean and merge steps are named <job_name>_mean / _merge. A match on
    the exact name only would miss a pipeline whose array had finished while its
    merge was still pending."""
    import submit_cluster_assignment as sca

    class _Result:
        returncode = 0
        stdout = ("1241571|hpl_cluster_assign_mean\n"
                  "1241573|hpl_cluster_assign_merge\n"
                  "1241999|something_else\n")
        stderr = ""

    original = sca.subprocess.run
    sca.subprocess.run = lambda *a, **k: _Result()
    try:
        found = sca.jobs_in_flight_named("hpl_cluster_assign")
    finally:
        sca.subprocess.run = original

    assert found == ["1241571", "1241573"], found


# --- reusing a query mean ------------------------------------------------
#
# The mean is one streamed pass over the projections and needs neither faiss nor
# the container, so `assign_hpc_clusters.py --precompute-mean` runs it anywhere.
# When the mean *job* will not start — a dependency that can never be satisfied
# strands the whole array — that should not block a submission, and a retry
# should not repeat a pass it already has.
#
# Every shard centres on this one file, so it is validated rather than trusted:
# a truncated or wrong-width mean produces 32 well-formed CSVs of wrong cluster
# IDs, with nothing downstream able to tell.


def _stub_submitter(monkey_target, submitted):
    import subprocess as sp

    def _fake(argv, *a, **k):
        submitted.append(argv)
        return sp.CompletedProcess(argv, 0, stdout="Submitted batch job 1", stderr="")
    monkey_target._run_sbatch_with_retry = _fake
    monkey_target._check_singularity_image = lambda *a, **k: None
    monkey_target._check_container_extras = lambda *a, **k: None
    monkey_target.partition_time_limit = lambda p: None
    monkey_target.jobs_in_flight_named = lambda n: []


def _submit_with_mean(tmp_path, mean_path, shards=4):
    import submit_cluster_assignment as sca

    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=900)
    submitted = []
    saved = (sca._run_sbatch_with_retry, sca._check_singularity_image,
             sca._check_container_extras, sca.partition_time_limit,
             sca.jobs_in_flight_named)
    _stub_submitter(sca, submitted)
    try:
        info = sca.submit_cluster_assignment_job(
            projections_h5=queries, out_csv=tmp_path / "out.csv",
            reference=reference, shards=shards, query_mean=mean_path,
            overwrite=True)
        return info, submitted
    finally:
        (sca._run_sbatch_with_retry, sca._check_singularity_image,
         sca._check_container_extras, sca.partition_time_limit,
         sca.jobs_in_flight_named) = saved


def _mean_width(tmp_path) -> int:
    """The width a query mean must have: the PCA basis's *input* dimension.

    Not the component count. The basis is (input dims, components) — DIM=32 and
    NCOMP=16 in this fixture, 128 and 127 in production — and project()
    subtracts the mean from raw embeddings before multiplying. Checking a mean
    against the component count rejects the correct file, which is exactly what
    the first version of this guard did to a real 128-wide mean.
    """
    reference = tmp_path / "ref.npz"
    if not reference.is_file():
        _write_reference(reference)
    return int(np.load(reference)["components"].shape[0])


def test_a_reused_mean_submits_no_mean_job(tmp_path):
    _write_reference(tmp_path / "ref.npz")
    mean = tmp_path / "mean.npy"
    np.save(mean, np.zeros(_mean_width(tmp_path), dtype=np.float32))

    info, submitted = _submit_with_mean(tmp_path, mean)

    assert info["mean_job_id"] is None, "a mean job was queued anyway"
    names = [a[a.index("--job-name=hpl_cluster_assign") if False else 1]
             for a in submitted]
    assert not any(n.endswith("_mean") for n in names), names
    # And the array must not depend on a job that was never submitted.
    array_argv = submitted[0]
    assert not any(a.startswith("--dependency") for a in array_argv), array_argv


def test_a_mean_of_the_wrong_width_is_refused_before_sbatch(tmp_path):
    """The failure with no downstream check: every shard centres on this file,
    so a wrong width silently reprojects the whole cohort."""
    _write_reference(tmp_path / "ref.npz")
    mean = tmp_path / "mean.npy"
    # The component count rather than the embedding width — 127 against 128 in
    # production. This is the near-miss, not an arbitrary wrong number: a mean
    # this wide is what someone gets by reading the reference's own shape, and
    # it is the case the first version of this guard got backwards.
    components = int(np.load(tmp_path / "ref.npz")["reference"].shape[1])
    np.save(mean, np.zeros(components, dtype=np.float32))
    assert components != _mean_width(tmp_path), "the fixture must distinguish them"

    try:
        _submit_with_mean(tmp_path, mean)
    except ValueError as e:
        assert "embeddings" in str(e), str(e)
        assert "wrong cluster IDs" in str(e)
    else:
        raise AssertionError("a wrong-width mean was accepted")


def test_a_non_finite_mean_is_refused(tmp_path):
    """What an interrupted --precompute-mean leaves behind."""
    _write_reference(tmp_path / "ref.npz")
    mean = tmp_path / "mean.npy"
    values = np.zeros(_mean_width(tmp_path), dtype=np.float32)
    values[3] = np.nan
    np.save(mean, values)

    try:
        _submit_with_mean(tmp_path, mean)
    except ValueError as e:
        assert "non-finite" in str(e), str(e)
    else:
        raise AssertionError("a mean containing NaN was accepted")


def test_a_missing_mean_file_is_refused(tmp_path):
    _write_reference(tmp_path / "ref.npz")
    try:
        _submit_with_mean(tmp_path, tmp_path / "absent.npy")
    except FileNotFoundError as e:
        assert "absent.npy" in str(e)
    else:
        raise AssertionError("a missing mean file was accepted")


# --- the shard index has to survive --cleanenv ---------------------------
#
# SLURM_ARRAY_TASK_ID is one more thing `singularity exec --cleanenv` wipes, and
# the shard preamble indexed the bounds arrays with it *inside* the container.
# Under `set -u` that aborted every task the instant the import check finished:
# stdout ended at "container packages: ok", the whole 32-task array died, and
# the merge sat on DependencyNeverSatisfied with nothing in the log but the
# place it stopped. Twice, because the same mechanism had already been found and
# fixed for the thread count without anyone asking what else came through the
# same door.
#
# So the bounds are resolved outside the container and passed in through
# SINGULARITYENV_/APPTAINERENV_, and this test runs the generated shell for real
# against a stand-in that strips the environment the way --cleanenv does. A
# string assertion would not have caught the original: it looked correct.


def _fake_singularity(tmp_path: Path) -> Path:
    """A `singularity` that keeps only what the SINGULARITYENV_ prefix passes."""
    script = tmp_path / "singularity"
    script.write_text(
        "#!/bin/bash\n"
        'inner="${@: -1}"\n'
        "clean_env=()\n"
        "while IFS='=' read -r name value; do\n"
        "  case \"$name\" in\n"
        "    SINGULARITYENV_*) clean_env+=(\"${name#SINGULARITYENV_}=$value\");;\n"
        "  esac\n"
        "done < <(env)\n"
        'exec env -i PATH="$PATH" HOME="$HOME" "${clean_env[@]}" bash -lc "$inner"\n'
    )
    script.chmod(0o755)
    return script


def _runnable_shard_command(tmp_path: Path, bounds):
    """The real generated command, with the two container-only python calls
    swapped for shell equivalents — everything about quoting, the preamble and
    the environment crossing stays exactly as submitted."""
    import submit_cluster_assignment as sca

    command = sca._build_assignment_command(
        singularity_bin=str(_fake_singularity(tmp_path)),
        singularity_image=Path("/img.sif"), extras_dir=Path("/extras"),
        assign_script=Path("/bin/echo"), reference=Path("/ref/r.npz"),
        projections_h5=Path("/p/x.h5"), out_csv=Path("/o/x.csv"),
        rep_key="z_latent", k=None, batch_size=16384, validate_against=None,
        threads=2, shard_bounds=bounds)
    return (command.replace("python -c ", "true ")
                   .replace("python /bin/echo", "echo ASSIGN_ARGS:"))


def test_each_shard_gets_its_own_rows_through_cleanenv(tmp_path):
    import os

    command = _runnable_shard_command(tmp_path, [(0, 100), (100, 200), (200, 300)])

    for task, expected in (("0", ("0", "100")), ("2", ("200", "300"))):
        result = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True,
            env={**os.environ, "SLURM_ARRAY_TASK_ID": task}, timeout=120)

        assert result.returncode == 0, (
            f"task {task} failed: {result.stderr[-400:]}")
        lo, hi = expected
        assert f"=== Shard {task}: rows {lo}-{hi} ===" in result.stdout, result.stdout
        assert f"--row-start {lo} --row-stop {hi}" in result.stdout, result.stdout


def test_a_sharded_command_run_as_a_plain_job_still_aborts(tmp_path):
    """The guard worth keeping. Without an array index there is no correct range,
    and one task quietly assigning the wrong rows is worse than a failure."""
    import os

    command = _runnable_shard_command(tmp_path, [(0, 100), (100, 200)])
    env = {k: v for k, v in os.environ.items() if k != "SLURM_ARRAY_TASK_ID"}

    result = subprocess.run(["bash", "-lc", command], capture_output=True,
                            text=True, env=env, timeout=120)

    assert result.returncode != 0
    assert "SLURM_ARRAY_TASK_ID" in result.stderr


def test_the_shard_bounds_are_resolved_outside_the_container(tmp_path):
    """Belt and braces on the mechanism, so a future edit that moves the
    preamble back inside is caught by reading as well as by running."""
    import submit_cluster_assignment as sca

    command = sca._build_assignment_command(
        singularity_bin="singularity", singularity_image=Path("/img.sif"),
        extras_dir=Path("/extras"), assign_script=Path("/b/assign.py"),
        reference=Path("/ref/r.npz"), projections_h5=Path("/p/x.h5"),
        out_csv=Path("/o/x.csv"), rep_key="z_latent", k=None, batch_size=16384,
        validate_against=None, threads=2, shard_bounds=[(0, 10), (10, 20)])

    # The array index is read before singularity is invoked...
    assert command.index("SLURM_ARRAY_TASK_ID") < command.index("singularity")
    # ...and the resolved values cross the boundary by the documented route.
    assert "SINGULARITYENV_ROW_START" in command
    assert "APPTAINERENV_ROW_START" in command


# --- deciding the device at submit time ----------------------------------
#
# The job's own "auto" can look at the GPU in front of it. A submission cannot:
# --nv, --gres and which extras directory to bind are all chosen before a node
# is allocated. So the submitter resolves "auto" from the one observable it has
# — whether the GPU extras were ever bootstrapped — and prints the reason,
# because "why is this on the CPU partition" should not need investigating.


def test_auto_picks_cpu_when_the_gpu_extras_are_absent(tmp_path):
    from submit_cluster_assignment import resolve_device

    device, why = resolve_device("auto", tmp_path / "nothing-here")

    assert device == "cpu"
    assert "bootstrap-extras-gpu" in why, why


def test_auto_picks_gpu_once_the_extras_exist(tmp_path):
    """Bootstrapped means a `faiss` package inside the GPU extras directory —
    the directory alone is not enough, since a failed bootstrap leaves one."""
    from submit_cluster_assignment import resolve_device

    extras = tmp_path / "extras-py38-gpu"
    (extras / "faiss").mkdir(parents=True)

    device, why = resolve_device("auto", extras)

    assert device == "gpu"
    assert str(extras) in why

    # An empty directory is what a failed bootstrap leaves behind.
    empty = tmp_path / "half-done"
    empty.mkdir()
    assert resolve_device("auto", empty)[0] == "cpu"


def test_an_explicit_device_is_never_second_guessed(tmp_path):
    from submit_cluster_assignment import resolve_device

    assert resolve_device("cpu", tmp_path)[0] == "cpu"
    assert resolve_device("gpu", tmp_path)[0] == "gpu"
    try:
        resolve_device("cuda", tmp_path)
    except ValueError as e:
        assert "auto" in str(e)
    else:
        raise AssertionError("'cuda' was accepted as a device")


def test_a_gpu_on_a_partition_with_no_gpus_is_refused(tmp_path):
    """--gres=gpu:1 on a GPU-less partition pends forever as ReqNodeNotAvail,
    which reads like a busy queue rather than a misconfiguration."""
    import submit_cluster_assignment as sca

    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=300)

    submitted = []
    saved = (sca._run_sbatch_with_retry, sca._check_singularity_image,
             sca._check_container_extras, sca.partition_time_limit,
             sca.jobs_in_flight_named, sca.partition_has_gpus)
    _stub_submitter(sca, submitted)
    sca.partition_has_gpus = lambda partition: False
    try:
        try:
            sca.submit_cluster_assignment_job(
                projections_h5=queries, out_csv=tmp_path / "out.csv",
                reference=reference, device="gpu", partition="compute",
                overwrite=True)
        except ValueError as e:
            assert "advertises none" in str(e), str(e)
            assert not submitted, "sbatch was called anyway"
        else:
            raise AssertionError("a GPU job was submitted to a GPU-less partition")
    finally:
        (sca._run_sbatch_with_retry, sca._check_singularity_image,
         sca._check_container_extras, sca.partition_time_limit,
         sca.jobs_in_flight_named, sca.partition_has_gpus) = saved


def test_an_unknown_partition_gpu_count_does_not_block(tmp_path):
    """sinfo being unavailable is not a reason to refuse — same posture as the
    walltime check."""
    import submit_cluster_assignment as sca

    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=300)

    submitted = []
    saved = (sca._run_sbatch_with_retry, sca._check_singularity_image,
             sca._check_container_extras, sca.partition_time_limit,
             sca.jobs_in_flight_named, sca.partition_has_gpus)
    _stub_submitter(sca, submitted)
    sca.partition_has_gpus = lambda partition: None
    try:
        sca.submit_cluster_assignment_job(
            projections_h5=queries, out_csv=tmp_path / "out.csv",
            reference=reference, device="gpu", partition="gpu", overwrite=True)
        assert submitted, "nothing was submitted"
        assert any("--gres=gpu:1" in a for a in submitted[0]), submitted[0]
    finally:
        (sca._run_sbatch_with_retry, sca._check_singularity_image,
         sca._check_container_extras, sca.partition_time_limit,
         sca.jobs_in_flight_named, sca.partition_has_gpus) = saved


# --- --help has to render -------------------------------------------------
#
# argparse %-interpolates help text, so a literal "%P" in a help string — from
# `sinfo -o "%P %l"`, which is genuinely the command worth quoting there — makes
# --help itself raise ValueError. Nothing else notices: the module imports, the
# parser builds, every submission works, and only asking for help fails. Cheap
# to check, and it covers the whole class.


def test_every_submitter_can_print_its_own_help(tmp_path):
    for module in ("submit_cluster_assignment.py", "submit_feature_extraction.py",
                   "submit_kb_write.py"):
        result = subprocess.run([sys.executable, str(BACKEND / module), "--help"],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (
            f"{module} --help failed:\n{result.stderr[-800:]}")
        assert "usage" in result.stdout.lower(), module


def test_the_gpu_bootstrap_is_offered_by_the_stage_that_uses_it(tmp_path):
    """The package is for the assignment's search, so the assignment's own CLI
    installs it. It lived only on submit_feature_extraction.py, which reads as
    "run feature extraction on the GPU" and is not what it does."""
    result = subprocess.run(
        [sys.executable, str(BACKEND / "submit_cluster_assignment.py"), "--help"],
        capture_output=True, text=True, timeout=120)

    assert "--bootstrap-gpu-faiss" in result.stdout
    assert "no feature extraction" in result.stdout


# --- the benchmark ---------------------------------------------------------
#
# Every estimate of this stage has been wrong so far, in both directions: FLOP
# counting said the search would be ~99% of the time, a real run's CPU
# accounting said nearer 30%, and a thread count that read correctly in the
# submitter ran on one core. So the shape of a run gets chosen from a measured
# slice, and the thing doing the measuring has to be trustworthy itself.


def _bench(args, timeout=1800):
    return subprocess.run([sys.executable, str(BACKEND / "benchmark_assignment.py"),
                           *args], capture_output=True, text=True, timeout=timeout)


def test_a_slice_without_a_shared_mean_is_refused(tmp_path):
    """--centering query centres on the mean of every query, so a slice that
    computed its own would time a different computation from the real run and
    label the same tiles differently. Refused with the command that fixes it."""
    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=600)

    result = _bench(["--projections-h5", str(queries), "--reference", str(reference),
                     "--rows", "300"], timeout=300)

    assert result.returncode == 1
    assert "--precompute-mean" in result.stderr, result.stderr
    assert "different cluster IDs" in result.stderr


def test_a_slice_too_short_to_time_is_not_extrapolated(tmp_path):
    """The first version reported 2e12 tiles/s: the assigner prints elapsed to
    one decimal, a sub-second slice reads as 0.0s, and the division exploded.
    Projecting a full cohort off that would have been worse than no number."""
    reference, queries, mean = tmp_path / "ref.npz", tmp_path / "q.h5", tmp_path / "m.npy"
    _write_reference(reference)
    _write_queries(queries, rows=900)
    subprocess.run([sys.executable, str(ASSIGN), "--reference", str(reference),
                    "--h5", str(queries), "--precompute-mean", str(mean)],
                   check=True, capture_output=True, timeout=300)

    result = _bench(["--projections-h5", str(queries), "--reference", str(reference),
                     "--query-mean", str(mean), "--rows", "400",
                     "--threads", "1", "--device", "cpu"], timeout=600)

    assert result.returncode == 0, result.stderr[-500:]
    assert "too fast to time" in result.stdout
    assert "Raise --rows" in result.stdout
    # And no projection table, because it would be extrapolated from noise.
    assert "Projected wall clock" not in result.stdout


def test_the_benchmark_reports_the_rate_the_assigner_measured(tmp_path):
    """It parses the assigner's own printed rate rather than recomputing one,
    and carries the phase split through, because which phase dominates is the
    decision the benchmark exists to inform."""
    import json

    reference, queries, mean = tmp_path / "ref.npz", tmp_path / "q.h5", tmp_path / "m.npy"
    rng = np.random.default_rng(0)
    # A reference large enough for the search to take measurable time.
    ref_rows, dim, ncomp, nclust = 60_000, 32, 24, 12
    np.savez(reference,
             reference=rng.standard_normal((ref_rows, ncomp)).astype(np.float32),
             components=rng.standard_normal((dim, ncomp)).astype(np.float32),
             codes=rng.integers(0, nclust, ref_rows).astype(np.int64),
             categories=np.array([str(i) for i in range(nclust)]),
             n_neighbors=np.int64(10), meta=json.dumps({"groupby": "leiden_2.5"}))
    rows = 30_000
    with h5py.File(queries, "w") as f:
        f.create_dataset("z_latent",
                         data=(rng.standard_normal((rows, dim)) + 3).astype(np.float32))
        for name in ("samples", "slides", "tiles"):
            f.create_dataset(name, data=np.array(
                [f"{name[0]}{i % 53:04d}".encode() for i in range(rows)]))
    subprocess.run([sys.executable, str(ASSIGN), "--reference", str(reference),
                    "--h5", str(queries), "--precompute-mean", str(mean)],
                   check=True, capture_output=True, timeout=600)

    result = _bench(["--projections-h5", str(queries), "--reference", str(reference),
                     "--query-mean", str(mean), "--rows", str(rows),
                     "--threads", "1", "--device", "cpu", "--shards", "1", "8"])

    assert result.returncode == 0, result.stderr[-800:]
    # A plausible rate, not 2e12 and not zero.
    rates = [int(m.replace(",", "")) for m in
             re.findall(r"([\d,]+) tiles/s", result.stdout)]
    assert rates, result.stdout
    assert all(1 <= rate < 10_000_000 for rate in rates), rates
    assert "search" in result.stdout, "the phase split did not carry through"


# --- a shard's log has to be findable -------------------------------------
#
# %j in an array task expands to that task's OWN JobId, which appears nowhere in
# squeue — squeue shows 1241672_0. So the logs existed under unpredictable
# names, and every attempt to tail a shard's output hit "no such file", for a
# 32-task array where reading one shard's progress is the whole diagnostic.


def test_an_array_names_its_logs_by_array_id_and_task(tmp_path):
    import submit_cluster_assignment as sca

    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    mean = tmp_path / "mean.npy"
    _write_reference(reference)
    _write_queries(queries, rows=600)
    np.save(mean, np.zeros(_mean_width(tmp_path), dtype=np.float32))

    submitted = []
    saved = (sca._run_sbatch_with_retry, sca._check_singularity_image,
             sca._check_container_extras, sca.partition_time_limit,
             sca.jobs_in_flight_named, sca.partition_has_gpus)
    _stub_submitter(sca, submitted)
    sca.partition_has_gpus = lambda partition: None
    try:
        sca.submit_cluster_assignment_job(
            projections_h5=queries, out_csv=tmp_path / "out.csv",
            reference=reference, shards=4, query_mean=mean, device="cpu",
            overwrite=True)
        array_argv = submitted[0]
        outputs = [a for a in array_argv if a.startswith("--output=")]
        assert outputs, array_argv
        assert "%A_%a" in outputs[0], outputs[0]
        assert "%j" not in outputs[0], outputs[0]
    finally:
        (sca._run_sbatch_with_retry, sca._check_singularity_image,
         sca._check_container_extras, sca.partition_time_limit,
         sca.jobs_in_flight_named, sca.partition_has_gpus) = saved


def test_a_plain_job_still_names_its_log_by_job_id(tmp_path):
    """%A_%a is empty for a non-array job, so the two cases need the two
    patterns — the same %j that is wrong for an array is right here."""
    import submit_cluster_assignment as sca

    reference, queries = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(reference)
    _write_queries(queries, rows=600)

    submitted = []
    saved = (sca._run_sbatch_with_retry, sca._check_singularity_image,
             sca._check_container_extras, sca.partition_time_limit,
             sca.jobs_in_flight_named, sca.partition_has_gpus)
    _stub_submitter(sca, submitted)
    sca.partition_has_gpus = lambda partition: None
    try:
        sca.submit_cluster_assignment_job(
            projections_h5=queries, out_csv=tmp_path / "out.csv",
            reference=reference, shards=1, device="cpu", overwrite=True)
        outputs = [a for a in submitted[0] if a.startswith("--output=")]
        assert "%j" in outputs[0], outputs[0]
    finally:
        (sca._run_sbatch_with_retry, sca._check_singularity_image,
         sca._check_container_extras, sca.partition_time_limit,
         sca.jobs_in_flight_named, sca.partition_has_gpus) = saved


# --- which GPU a shard gets ----------------------------------------------
#
# The third thing --cleanenv takes, after the thread count and the array index:
# CUDA_VISIBLE_DEVICES, which is how Slurm tells each task with --gres=gpu:1
# which physical card is its own. Stripped, every task on a node sees all the
# cards and index_cpu_to_gpu(res, 0, ...) puts them all on GPU 0 — N shards
# contending for one device while the rest idle, and at no point failing.
#
# Tested by running the generated shell against a stand-in that strips the
# environment the way --cleanenv does, because this is precisely the class a
# string assertion cannot catch.


def test_each_gpu_shard_sees_the_card_slurm_gave_it(tmp_path):
    import os

    import submit_cluster_assignment as sca

    command = sca._build_assignment_command(
        singularity_bin=str(_fake_singularity(tmp_path)),
        singularity_image=Path("/img.sif"), extras_dir=Path("/extras"),
        assign_script=Path("/bin/echo"), reference=Path("/ref/r.npz"),
        projections_h5=Path("/p/x.h5"), out_csv=Path("/o/x.csv"),
        rep_key="z_latent", k=None, batch_size=16384, validate_against=None,
        threads=4, device="gpu", shard_bounds=[(0, 100), (100, 200), (200, 300)])
    command = (command.replace("python -c ", "true ")
                      .replace("python /bin/echo",
                               'echo "GPU_INSIDE=$CUDA_VISIBLE_DEVICES"; echo ARGS:'))

    for task, gpu in (("0", "3"), ("2", "5")):
        result = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True, timeout=120,
            env={**os.environ, "SLURM_ARRAY_TASK_ID": task,
                 "CUDA_VISIBLE_DEVICES": gpu})

        assert result.returncode == 0, result.stderr[-400:]
        assert f"GPU_INSIDE={gpu}" in result.stdout, (
            f"task {task} did not receive GPU {gpu}:\n{result.stdout}")


def test_a_gpu_run_outside_slurm_still_picks_a_device(tmp_path):
    """CUDA_VISIBLE_DEVICES is legitimately unset for a hand-run job, where
    device 0 is the only sensible answer — and an unset variable must not abort
    under `set -u`."""
    import os

    import submit_cluster_assignment as sca

    command = sca._build_assignment_command(
        singularity_bin=str(_fake_singularity(tmp_path)),
        singularity_image=Path("/img.sif"), extras_dir=Path("/extras"),
        assign_script=Path("/bin/echo"), reference=Path("/ref/r.npz"),
        projections_h5=Path("/p/x.h5"), out_csv=Path("/o/x.csv"),
        rep_key="z_latent", k=None, batch_size=16384, validate_against=None,
        threads=2, device="gpu")
    command = (command.replace("python -c ", "true ")
                      .replace("python /bin/echo",
                               'echo "GPU_INSIDE=$CUDA_VISIBLE_DEVICES"; echo ARGS:'))
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}

    result = subprocess.run(["bash", "-lc", command], capture_output=True,
                            text=True, timeout=120, env=env)

    assert result.returncode == 0, result.stderr[-400:]
    assert "GPU_INSIDE=0" in result.stdout, result.stdout


def test_a_cpu_run_does_not_pin_a_gpu(tmp_path):
    """Nothing GPU-related should appear in a CPU submission — including the
    passthrough, which would otherwise pin device 0 for a job that never asked
    for one."""
    import submit_cluster_assignment as sca

    command = sca._build_assignment_command(
        singularity_bin="singularity", singularity_image=Path("/img.sif"),
        extras_dir=Path("/extras"), assign_script=Path("/b/assign.py"),
        reference=Path("/ref/r.npz"), projections_h5=Path("/p/x.h5"),
        out_csv=Path("/o/x.csv"), rep_key="z_latent", k=None, batch_size=16384,
        validate_against=None, threads=2, device="cpu",
        shard_bounds=[(0, 10), (10, 20)])

    assert "CUDA_VISIBLE_DEVICES" not in command
    assert "--nv" not in command
    # The shard passthrough is still there — the two are independent.
    assert "SINGULARITYENV_ROW_START" in command


# --- resuming a killed shard ---------------------------------------------
#
# A preempted or requeued task used to restart its whole range: the .partial was
# deleted and the shard began again at row 0. With 32 shards that was minutes,
# but it is also the only reason a GPU run on a preemptible partition was a bad
# trade. Chunks are checkpoints now — each written under a .tmp name and
# renamed, so a chunk file exists only if it is whole — and the identical
# command line resumes, which matters because a requeued job re-runs exactly
# what it ran before.
#
# The property under test is the one that makes it safe: resumed output must
# equal uninterrupted output byte for byte. Anything less is two computations
# concatenated.


def _kill_after_first_chunk(command, env):
    """Start the assigner and SIGKILL it once a chunk has certainly landed.

    Driven by its own progress line rather than a sleep, so the test does not
    depend on how fast the machine is.
    """
    import signal

    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, env=env)
    try:
        for line in proc.stdout:                       # blocks until progress
            if "tiles/s [" in line:
                break
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=60)


def _slow_fixture(tmp_path: Path, ref_rows=40_000, query_rows=6_000):
    """A reference big enough that chunks take measurable time."""
    import json

    rng = np.random.default_rng(0)
    dim, ncomp, nclust = 48, 32, 15
    reference = tmp_path / "ref.npz"
    np.savez(reference,
             reference=rng.standard_normal((ref_rows, ncomp)).astype(np.float32),
             components=rng.standard_normal((dim, ncomp)).astype(np.float32),
             codes=rng.integers(0, nclust, ref_rows).astype(np.int64),
             categories=np.array([str(i) for i in range(nclust)]),
             n_neighbors=np.int64(10), meta=json.dumps({"groupby": "leiden_2.5"}))
    queries = tmp_path / "proj.h5"
    with h5py.File(queries, "w") as f:
        f.create_dataset("z_latent", data=(
            rng.standard_normal((query_rows, dim)) + 3).astype(np.float32))
        for name in ("samples", "slides", "tiles"):
            f.create_dataset(name, data=np.array(
                [f"{name[0]}{i % 71:04d}".encode() for i in range(query_rows)]))
    mean = tmp_path / "mean.npy"
    subprocess.run([sys.executable, str(ASSIGN), "--reference", str(reference),
                    "--h5", str(queries), "--precompute-mean", str(mean)],
                   check=True, capture_output=True, timeout=600)
    return reference, queries, mean


def test_a_killed_shard_resumes_and_matches_an_uninterrupted_run(tmp_path):
    import os

    reference, queries, mean = _slow_fixture(tmp_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    base = [sys.executable, str(ASSIGN), "--reference", str(reference),
            "--h5", str(queries), "--query-mean", str(mean),
            "--chunk-size", "500", "--progress", "500",
            "--row-start", "0", "--row-stop", "4000", "--device", "cpu"]

    subprocess.run(base + ["--out", str(tmp_path / "whole.csv")],
                   check=True, capture_output=True, env=env, timeout=900)
    whole = (tmp_path / "whole.rows0-4000.csv").read_bytes()

    _kill_after_first_chunk(base + ["--out", str(tmp_path / "resumed.csv")], env)
    chunk_dir = tmp_path / "resumed.rows0-4000.csv.chunks"
    assert chunk_dir.is_dir(), "the chunk directory did not survive the kill"
    survived = sorted(chunk_dir.glob("chunk_*.csv"))
    assert survived, "no complete chunk survived, so there is nothing to resume"
    # And no output masquerading as finished.
    assert not (tmp_path / "resumed.rows0-4000.csv").exists()

    result = subprocess.run(base + ["--out", str(tmp_path / "resumed.csv")],
                            capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-800:]
    assert "Resuming" in result.stdout, result.stdout

    resumed = (tmp_path / "resumed.rows0-4000.csv").read_bytes()
    assert resumed == whole, "the resumed output differs from an uninterrupted run"
    assert not chunk_dir.exists(), "the chunk directory was not cleaned up"


def test_a_truncated_chunk_is_recomputed_not_trusted(tmp_path):
    """A chunk file is only skipped when its row count matches its range, so a
    short one is redone rather than assembled into the output."""
    import os

    reference, queries, mean = _slow_fixture(tmp_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    base = [sys.executable, str(ASSIGN), "--reference", str(reference),
            "--h5", str(queries), "--query-mean", str(mean),
            "--chunk-size", "500", "--progress", "500",
            "--row-start", "0", "--row-stop", "2000", "--device", "cpu"]

    subprocess.run(base + ["--out", str(tmp_path / "whole.csv")],
                   check=True, capture_output=True, env=env, timeout=900)
    whole = (tmp_path / "whole.rows0-2000.csv").read_bytes()

    _kill_after_first_chunk(base + ["--out", str(tmp_path / "resumed.csv")], env)
    chunk_dir = tmp_path / "resumed.rows0-2000.csv.chunks"
    survived = sorted(chunk_dir.glob("chunk_*.csv"))
    assert survived
    # Lop the last row off a "complete" chunk.
    lines = survived[0].read_text().splitlines(keepends=True)
    survived[0].write_text("".join(lines[:-1]))

    result = subprocess.run(base + ["--out", str(tmp_path / "resumed.csv")],
                            capture_output=True, text=True, env=env, timeout=900)

    assert result.returncode == 0, result.stderr[-800:]
    assert (tmp_path / "resumed.rows0-2000.csv").read_bytes() == whole


def test_resuming_a_different_configuration_is_refused(tmp_path):
    """Chunks from two configurations concatenated is a complete CSV where some
    rows came from each, with nothing to say which. Refused, naming the field
    that differs and the directory to remove."""
    import os

    reference, queries, mean = _slow_fixture(tmp_path)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    base = [sys.executable, str(ASSIGN), "--reference", str(reference),
            "--h5", str(queries), "--query-mean", str(mean),
            "--chunk-size", "500", "--progress", "500",
            "--row-start", "0", "--row-stop", "2000", "--device", "cpu"]

    _kill_after_first_chunk(base + ["--out", str(tmp_path / "out.csv")], env)
    assert (tmp_path / "out.rows0-2000.csv.chunks").is_dir()

    # Same range, different vote.
    result = subprocess.run(
        base + ["--out", str(tmp_path / "out.csv"), "--k", "5"],
        capture_output=True, text=True, env=env, timeout=900)

    assert result.returncode != 0
    assert "different configuration" in result.stderr, result.stderr[-600:]
    assert "vote" in result.stderr
    assert "rm -rf" in result.stderr


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_assign_test_"))
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
