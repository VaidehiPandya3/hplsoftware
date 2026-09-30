#!/usr/bin/env python3
"""Read every slide's header before the cohort queues, and report the refusals.

`anorak_tile.py` refuses a slide whose header does not carry both objective
power and microns-per-pixel, because upstream would otherwise fall back to
objective-power scaling and tile that slide at a resolution the rest of the
cohort does not share. That refusal is right, and at ten slides it costs
nothing. At 7,221 it costs the run: `errorStrategy` is `finish`, so one such
slide stops the cohort, and it stops it wherever the scheduler happened to
reach — which for a header problem is days of tiling later, for a fact that was
readable in milliseconds on day zero.

This reads all of them up front and prints the list.

    ./scan_slide_headers.py --slides-csv <list.csv> --raw-dir <slides/>

Exits non-zero if any slide would be refused, so it can gate a submission.

Two things it deliberately does NOT do its own way:

  - slide resolution. The rules are main.nf's `resolveSlides`: the same
    extension list, ids matched against both filename and stem, an id matching
    two files fatal rather than resolved by directory order. A scan that
    resolved slides differently from the run would answer a question nobody
    asked.

  - the tile count. It calls `anorak_tile.expected_tile_count`, the function
    the tiling task itself checks its output against, rather than restating
    the arithmetic here where the two could drift apart.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from anorak_tile import expected_tile_count  # noqa: E402

#: main.nf's list, and for its reason: upstream's `save_cws.single_file_run`
#: dispatches on extension and returns silently for anything else, so an
#: unlisted slide is skipped rather than reported. Read out of main.nf itself
#: when it is there, so an edit to that list (dropping .mrxs, say) cannot leave
#: this resolving slides the run no longer will; this copy is the fallback for
#: a scan run away from the pipeline, and test_the_supported_list_is_the_one_
#: main_nf_uses pins the two together.
_FALLBACK_SUPPORTED = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".png", ".qptiff")
MAIN_NF = Path(__file__).resolve().parent.parent / "main.nf"


def supported_from_main_nf(main_nf: Path = MAIN_NF) -> tuple[str, ...] | None:
    """main.nf's `def SUPPORTED = [...]`, or None if it cannot be read."""
    try:
        text = main_nf.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"def\s+SUPPORTED\s*=\s*\[([^\]]*)\]", text)
    if not match:
        return None
    return tuple(part.strip().strip("'\"") for part in match.group(1).split(",")
                 if part.strip())


SUPPORTED = supported_from_main_nf() or _FALLBACK_SUPPORTED

#: openslide's documented property names. Spelled out so the pure functions
#: below can be exercised without openslide installed; checked against the
#: library's own constants at read time, where it is present.
PROP_OBJECTIVE = "openslide.objective-power"
PROP_MPP_X = "openslide.mpp-x"


def index_raw_dir(raw_dir: Path) -> dict[str, list[Path]]:
    """Every supported slide under raw_dir, keyed by filename AND by stem."""
    index: dict[str, list[Path]] = {}
    for entry in Path(raw_dir).rglob("*"):
        if entry.is_file() and entry.name.lower().endswith(SUPPORTED):
            index.setdefault(entry.name, []).append(entry)
            index.setdefault(entry.stem, []).append(entry)
    return index


def resolve_rows(rows, index, slide_column):
    """(resolved, missing, ambiguous) — main.nf's rules, not new ones.

    `resolved` is [(slide_id, path)]. An id matching two distinct files is
    ambiguous rather than resolved, because otherwise which file wins is a
    property of directory order.
    """
    resolved, missing, ambiguous = [], [], {}
    for row in rows:
        slide_id = (row.get(slide_column) or "").strip()
        if not slide_id:
            continue
        matches, seen = [], set()
        for path in index.get(slide_id, []):
            real = str(path.resolve())
            if real not in seen:
                seen.add(real)
                matches.append(path)
        if not matches:
            missing.append(slide_id)
        elif len(matches) > 1:
            ambiguous[slide_id] = matches
        else:
            resolved.append((slide_id, matches[0]))
    return resolved, missing, ambiguous


def header_verdict(properties: dict) -> tuple[bool, float | None, float | None, str]:
    """(ok, objective, mpp, reason) for one slide's openslide properties.

    Mirrors `anorak_tile.slide_scale`: both fields must be present, numeric and
    greater than zero. Kept as a pure function over a dict so the refusal can
    be tested — the bugs this repository keeps finding are guards that cannot
    come out bad, and a header check that only ever sees good headers is one.
    """
    raw_objective = properties.get(PROP_OBJECTIVE)
    raw_mpp = properties.get(PROP_MPP_X)
    try:
        objective = float(raw_objective)
        mpp = float(raw_mpp)
    except (TypeError, ValueError):
        absent = []
        if raw_objective in (None, ""):
            absent.append("objective power")
        if raw_mpp in (None, ""):
            absent.append("microns-per-pixel")
        if not absent:
            return False, None, None, (
                f"non-numeric header: objective {raw_objective!r}, mpp {raw_mpp!r}"
            )
        return False, None, None, f"header carries no {' and no '.join(absent)}"
    if objective <= 0 or mpp <= 0:
        return False, objective, mpp, f"objective {objective}, mpp {mpp}; both must be > 0"
    return True, objective, mpp, ""


def read_slide(slide_id: str, path: Path, output_mpp: float) -> dict:
    """One slide's verdict. Never raises: a scan that stops at the first bad
    slide is the behaviour this exists to replace."""
    import openslide

    assert openslide.PROPERTY_NAME_OBJECTIVE_POWER == PROP_OBJECTIVE
    assert openslide.PROPERTY_NAME_MPP_X == PROP_MPP_X

    record = {"slide_id": slide_id, "path": str(path), "ok": False,
              "objective": "", "mpp": "", "expected_tiles": "", "reason": ""}
    try:
        with openslide.OpenSlide(str(path)) as slide:
            ok, objective, mpp, reason = header_verdict(dict(slide.properties))
    except Exception as error:
        record["reason"] = f"{type(error).__name__}: {error}"
        return record

    record.update(ok=ok, reason=reason,
                  objective="" if objective is None else objective,
                  mpp="" if mpp is None else mpp)
    if ok:
        # Only for a slide that passed: expected_tile_count goes through
        # slide_scale, whose refusal raises SystemExit and would end the scan.
        try:
            record["expected_tiles"] = expected_tile_count(path, output_mpp)
        except Exception as error:
            record["ok"] = False
            record["reason"] = f"tile count could not be derived: {error}"
    return record


def mrxs_data_dir(path: Path) -> Path:
    """Where openslide looks for a MIRAX slide's pixel data: a directory with
    the slide's stem, beside the .mrxs, holding Slidedat.ini."""
    return path.with_suffix("") / "Slidedat.ini"


def report_resolution(resolved, missing, ambiguous, raw_dir: Path) -> int:
    """--resolve-only: what the list resolves to, without opening anything.

    .mrxs slides are reported whether or not SUPPORTED lets them resolve: an
    id that is missing only because main.nf stopped accepting .mrxs is a
    different fix (drop or convert the slide) from an id with no file at all.
    """
    extensions: dict[str, int] = {}
    for _, path in resolved:
        extensions[path.suffix.lower()] = extensions.get(path.suffix.lower(), 0) + 1
    mrxs_paths = [(sid, path) for sid, path in resolved if path.suffix.lower() == ".mrxs"]
    if missing:
        by_name: dict[str, list[Path]] = {}
        for entry in Path(raw_dir).rglob("*"):
            if entry.is_file() and entry.name.lower().endswith(".mrxs"):
                by_name.setdefault(entry.name, []).append(entry)
                by_name.setdefault(entry.stem, []).append(entry)
        mrxs_paths += [(sid, by_name[sid][0]) for sid in missing if sid in by_name]
    mrxs = [sid for sid, _ in mrxs_paths]
    no_data = [sid for sid, path in mrxs_paths if not mrxs_data_dir(path).is_file()]
    print("by extension: " + ", ".join(f"{e} {n}" for e, n in sorted(extensions.items())))
    if missing:
        print(f"{len(missing)} ids resolve to no file, e.g. {missing[:3]}")
    if ambiguous:
        print(f"{len(ambiguous)} ids resolve to more than one file, e.g. {list(ambiguous)[:3]}")
    print("RESOLVE_SUMMARY " + json.dumps({
        "resolved": len(resolved), "missing": len(missing),
        "ambiguous": len(ambiguous), "extensions": extensions,
        "mrxs": mrxs, "mrxs_without_data_dir": no_data,
    }))
    return 1 if (missing or ambiguous) else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slides-csv", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--slide-column", default="slide_id")
    parser.add_argument("--output-mpp", type=float, default=0.22,
                        help="must match the pipeline's --output_mpp, since it "
                             "decides the tile count reported here")
    parser.add_argument("--report", type=Path,
                        help="write every slide's verdict to this CSV")
    parser.add_argument("--write-clean-list", type=Path,
                        help="write a copy of the slide list with the refused "
                             "slides removed. Not written unless asked for: "
                             "dropping slides is a decision about the cohort.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resolve-only", action="store_true",
                        help="resolve the list against --raw-dir and report "
                             "which files it names (and any .mrxs without its "
                             "data directory) without opening a slide — needs "
                             "no openslide, so it runs on a login node")
    parser.add_argument("--machine-summary", action="store_true",
                        help="end with one SCAN_SUMMARY {json} line (largest "
                             "and mean tile count) for preflight.sh to read")
    args = parser.parse_args(argv)

    with open(args.slides_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        print(f"{args.slides_csv} has no rows.", file=sys.stderr)
        return 2
    if args.slide_column not in fieldnames:
        print(f"{args.slides_csv} has no '{args.slide_column}' column; "
              f"found {', '.join(fieldnames)}", file=sys.stderr)
        return 2

    print(f"indexing {args.raw_dir}", flush=True)
    index = index_raw_dir(args.raw_dir)
    resolved, missing, ambiguous = resolve_rows(rows, index, args.slide_column)
    print(f"{len(rows)} slides listed, {len(resolved)} resolved to a file", flush=True)
    if args.resolve_only:
        return report_resolution(resolved, missing, ambiguous, args.raw_dir)

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(read_slide, sid, path, args.output_mpp)
                   for sid, path in resolved]
        for done, future in enumerate(futures, 1):
            records.append(future.result())
            if done % 500 == 0:
                print(f"  read {done}/{len(futures)} headers", flush=True)

    bad = [r for r in records if not r["ok"]]
    good = [r for r in records if r["ok"]]

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys())
                                    if records else ["slide_id"])
            writer.writeheader()
            writer.writerows(records)
        print(f"\nper-slide verdicts: {args.report}")

    print()
    if missing:
        print(f"{len(missing)} slide id{'s' if len(missing) != 1 else ''} "
              f"ha{'ve' if len(missing) != 1 else 's'} no file under {args.raw_dir}:")
        for slide_id in missing[:20]:
            print(f"    {slide_id}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
    if ambiguous:
        print(f"{len(ambiguous)} slide id{'s' if len(ambiguous) != 1 else ''} "
              f"match{'' if len(ambiguous) != 1 else 'es'} more than one file:")
        for slide_id, paths in list(ambiguous.items())[:5]:
            print(f"    {slide_id}:")
            for path in paths:
                print(f"      - {path}")
    if bad:
        print(f"{len(bad)} slide{'s' if len(bad) != 1 else ''} would be "
              f"refused by anorak_tile.py:")
        for record in bad[:20]:
            print(f"    {record['slide_id']}: {record['reason']}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")

    total_tiles = sum(r["expected_tiles"] for r in good if r["expected_tiles"] != "")
    print(f"\n{len(good)} slides pass, {len(bad) + len(missing) + len(ambiguous)} would stop the run.")
    if good:
        print(f"{total_tiles:,} tiles expected across the slides that pass "
              f"({total_tiles // max(len(good), 1):,} per slide on average).")

    if args.machine_summary:
        counted = [r for r in good if r["expected_tiles"] != ""]
        largest = max(counted, key=lambda r: r["expected_tiles"], default=None)
        print("SCAN_SUMMARY " + json.dumps({
            "slides": len(counted),
            "largest_tiles": largest["expected_tiles"] if largest else 0,
            "largest_slide": largest["slide_id"] if largest else "",
            "mean_tiles": total_tiles / len(counted) if counted else 0,
        }))

    if args.write_clean_list:
        keep = {r["slide_id"] for r in good}
        kept = [row for row in rows
                if (row.get(args.slide_column) or "").strip() in keep]
        with open(args.write_clean_list, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        print(f"\nclean list ({len(kept)} slides): {args.write_clean_list}")

    return 1 if (bad or missing or ambiguous) else 0


if __name__ == "__main__":
    raise SystemExit(main())
