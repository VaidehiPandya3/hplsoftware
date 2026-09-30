from pathlib import Path
import argparse
import json
import math

import openslide
import numpy as np
import pandas as pd
from PIL import Image

from slide_naming import slide_id_from_raw_path


SUPPORTED_EXTENSIONS = [".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn"]

# Matches Kai's paper: tiles tessellated at 1.8 µm/pixel (~5x magnification,
# tile diameter 403.2 µm), saved as 224x224 px JPEGs. Native mpp varies per
# slide/scanner (e.g. 0.252 for 40x Aperio scans, ~0.50 for 20x), so the
# native read size is derived per slide rather than assumed fixed.
TARGET_MPP = 1.8
TARGET_TILE_PX = 224
DEFAULT_NATIVE_MPP = 0.252  # fallback if a slide has no mpp metadata


def find_newest_slide_with_mask(raw_dir: str, mask_dir: str, output_dir: str):
    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    slides = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not slides:
        raise FileNotFoundError(f"No supported WSI files found in: {raw_dir}")

    slides = sorted(slides, key=lambda p: p.stat().st_mtime, reverse=True)

    for slide_path in slides:
        slide_id = slide_id_from_raw_path(slide_path)
        mask_path = mask_dir / f"{slide_id}_tissue_mask.png"
        metadata_path = output_dir / slide_id / f"{slide_id}_tile_metadata.csv"

        if mask_path.exists() and not metadata_path.exists():
            return slide_path, mask_path

    raise RuntimeError("No uploaded slide found that has a tissue mask and has not been tiled yet.")


def load_mask(mask_path: str):
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img) > 0
    return mask


def get_tissue_percent_from_mask(mask, x, y, tile_size, slide_w, slide_h):
    mask_h, mask_w = mask.shape

    mx1 = int(x / slide_w * mask_w)
    my1 = int(y / slide_h * mask_h)
    mx2 = int((x + tile_size) / slide_w * mask_w)
    my2 = int((y + tile_size) / slide_h * mask_h)

    mx1 = max(0, min(mx1, mask_w))
    mx2 = max(0, min(mx2, mask_w))
    my1 = max(0, min(my1, mask_h))
    my2 = max(0, min(my2, mask_h))

    tile_mask = mask[my1:my2, mx1:mx2]

    if tile_mask.size == 0:
        return 0.0

    return float(tile_mask.mean() * 100)


def parse_native_mpp(mpp_x) -> float | None:
    """The µm/px a slide reports, or None if it does not report a usable one.

    Separate from get_native_mpp because a caller that has to *say* where its
    number came from needs to know whether the slide supplied one — and
    because a slide reporting "0" used to reach the division below as a
    ZeroDivisionError rather than the documented fallback."""
    try:
        mpp = float(mpp_x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mpp) or mpp <= 0:
        return None
    return mpp


def get_native_mpp(slide: openslide.OpenSlide) -> float:
    mpp = parse_native_mpp(slide.properties.get("openslide.mpp-x"))
    return DEFAULT_NATIVE_MPP if mpp is None else mpp


def native_tile_px(
    native_mpp: float,
    target_mpp: float = TARGET_MPP,
    target_tile_px: int = TARGET_TILE_PX,
) -> int:
    """The native-pixel side of one tile, which is also the stride between
    tiles: the loop below tessellates, so pitch and tile size are one number.

    This is the only place that arithmetic lives, because anything drawing a
    box around a tile has to land on exactly the same integer. The tile server
    hardcoded `int(224 * 1.8 / 0.252)` = 1600 for every slide, while this
    derives it per slide from the slide's own mpp — so on the 1,078 of 1,598
    TCGA slides that are not 0.252 µm/px the viewer's grid was drawn at the
    wrong size, gapped on a finer scan and overlapping on a 20x one."""
    scale = target_mpp / native_mpp
    return round(target_tile_px * scale)


def tile_slide_from_mask(
    slide_path: str,
    mask_path: str,
    output_dir: str,
    target_mpp: float = TARGET_MPP,
    target_tile_px: int = TARGET_TILE_PX,
    min_tissue_percent: float = 30.0,
    level: int = 0,
    jpeg_quality: int = 90,
    slide_id: str | None = None,
):
    """slide_id defaults to slide_id_from_raw_path(slide_path) — see
    run_tissue_detection's matching docstring in tile_mask.py; both need to
    agree on the same slide_id for a given slide_path, since mask lookups
    and tile output both key off it independently."""
    slide_path = Path(slide_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)

    slide_id = slide_id or slide_id_from_raw_path(slide_path)
    slide_output_dir = output_dir / slide_id
    slide_output_dir.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(str(slide_path))
    slide_w, slide_h = slide.level_dimensions[level]

    native_mpp = get_native_mpp(slide)
    tile_px_native = native_tile_px(native_mpp, target_mpp, target_tile_px)

    mask = load_mask(mask_path)

    records = []
    saved_tiles = 0
    checked_tiles = 0
    skipped_tiles = 0

    for y in range(0, slide_h, tile_px_native):
        for x in range(0, slide_w, tile_px_native):
            checked_tiles += 1

            tissue_percent = get_tissue_percent_from_mask(
                mask=mask,
                x=x,
                y=y,
                tile_size=tile_px_native,
                slide_w=slide_w,
                slide_h=slide_h,
            )

            if tissue_percent < min_tissue_percent:
                skipped_tiles += 1
                continue

            tile = slide.read_region(
                (x, y),
                level,
                (tile_px_native, tile_px_native)
            ).convert("RGB")

            if tile_px_native != target_tile_px:
                tile = tile.resize((target_tile_px, target_tile_px), Image.LANCZOS)

            col = x // tile_px_native
            row = y // tile_px_native
            tile_filename = f"{col}_{row}.jpeg"
            tile_path = slide_output_dir / tile_filename
            tile.save(tile_path, "JPEG", quality=jpeg_quality)

            records.append({
                "slides": slide_id,
                "tiles": tile_filename,
                "slide_tile": f"{slide_id}_{tile_filename}",
                "col": col,
                "row": row,
                "x_5x": col * target_tile_px,
                "y_5x": row * target_tile_px,
                "x_native": x,
                "y_native": y,
                "tissue_percent": tissue_percent,
            })

            saved_tiles += 1

    metadata_csv_path = slide_output_dir / f"{slide_id}_tile_metadata.csv"
    metadata_json_path = slide_output_dir / f"{slide_id}_tiling_summary.json"

    df = pd.DataFrame(records)
    df.to_csv(metadata_csv_path, index=False)

    summary = {
        "slide_id": slide_id,
        "slide_path": str(slide_path),
        "mask_path": str(mask_path),
        "output_dir": str(slide_output_dir),
        "slide_width": slide_w,
        "slide_height": slide_h,
        "level": level,
        "level_count": slide.level_count,
        "vendor": slide.properties.get("openslide.vendor"),
        "objective_power": slide.properties.get("openslide.objective-power"),
        "native_mpp": native_mpp,
        "target_mpp": target_mpp,
        "target_tile_px": target_tile_px,
        "tile_px_native": tile_px_native,
        "min_tissue_percent": min_tissue_percent,
        "checked_tiles": checked_tiles,
        "saved_tiles": saved_tiles,
        "skipped_tiles": skipped_tiles,
        "metadata_csv": str(metadata_csv_path),
    }

    with open(metadata_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Slide: {slide_id}")
    print(f"Native mpp: {native_mpp} -> tile_px_native: {tile_px_native} (target {target_tile_px}px @ {target_mpp} mpp)")
    print(f"Checked tiles: {checked_tiles}")
    print(f"Saved tiles: {saved_tiles}")
    print(f"Skipped tiles: {skipped_tiles}")
    print(f"Metadata CSV: {metadata_csv_path}")
    print(f"Summary JSON: {metadata_json_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--slide", required=False)
    parser.add_argument("--mask", required=False)

    parser.add_argument(
        "--raw_dir",
        default="/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi/raw"
    )

    parser.add_argument(
        "--mask_dir",
        default="/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks"
    )

    parser.add_argument(
        "--output",
        default="/hpc-home/home/users/vpandya/long-term-scratch/processed_tiles"
    )

    parser.add_argument("--target_mpp", type=float, default=TARGET_MPP)
    parser.add_argument("--target_tile_px", type=int, default=TARGET_TILE_PX)
    parser.add_argument("--min_tissue", type=float, default=30.0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--jpeg_quality", type=int, default=90)

    args = parser.parse_args()

    if args.slide and args.mask:
        slide_path = Path(args.slide)
        mask_path = Path(args.mask)
    else:
        slide_path, mask_path = find_newest_slide_with_mask(
            raw_dir=args.raw_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output,
        )
        print(f"Auto-selected slide: {slide_path}")
        print(f"Using mask: {mask_path}")

    tile_slide_from_mask(
        slide_path=str(slide_path),
        mask_path=str(mask_path),
        output_dir=args.output,
        target_mpp=args.target_mpp,
        target_tile_px=args.target_tile_px,
        min_tissue_percent=args.min_tissue,
        level=args.level,
        jpeg_quality=args.jpeg_quality,
    )
