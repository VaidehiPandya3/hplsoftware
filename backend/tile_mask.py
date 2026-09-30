from pathlib import Path
import argparse

import openslide
import numpy as np
from PIL import Image

from skimage.color import rgb2hsv
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
from skimage.measure import label, regionprops

from slide_naming import slide_id_from_raw_path

SUPPORTED_EXTENSIONS = [".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn"]


def find_newest_unprocessed_slide(raw_dir: str, output_dir: str):
    raw_dir = Path(raw_dir)
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
        expected_mask = output_dir / f"{slide_id}_tissue_mask.png"
        expected_overlay = output_dir / f"{slide_id}_tissue_overlay.png"

        if not expected_mask.exists() and not expected_overlay.exists():
            return slide_path

    raise RuntimeError("All uploaded slides already have tissue masks.")

def load_slide(slide_path: str):
    slide_path = Path(slide_path)

    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")

    slide = openslide.OpenSlide(str(slide_path))
    return slide


def create_thumbnail(slide, max_size: int = 2000):
    width, height = slide.dimensions

    scale = max(width, height) / max_size
    thumb_w = int(width / scale)
    thumb_h = int(height / scale)

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    return thumbnail


def create_tissue_mask(
    thumbnail: Image.Image,
    saturation_threshold: float = 0.08,
    value_threshold: float = 0.95,
    min_object_size: int = 500,
):
    """
    Creates a binary tissue mask from a WSI thumbnail.

    Tissue is usually pink/purple and has higher saturation.
    Background is usually white/grey and has low saturation/high value.
    """

    arr = np.array(thumbnail) / 255.0
    hsv = rgb2hsv(arr)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (saturation > saturation_threshold) & (value < value_threshold)

    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = binary_opening(mask)
    mask = binary_closing(mask)

    return mask


def save_mask(mask: np.ndarray, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img.save(output_path)

    return output_path


def save_overlay(thumbnail: Image.Image, mask: np.ndarray, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thumb_arr = np.array(thumbnail).copy()

    overlay = thumb_arr.copy()
    overlay[mask] = [255, 0, 0]

    blended = (0.65 * thumb_arr + 0.35 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(output_path)

    return output_path


def get_tissue_regions(mask: np.ndarray, min_region_area: int = 1000):
    """
    Returns bounding boxes of detected tissue regions on thumbnail scale.
    bbox format: min_row, min_col, max_row, max_col
    """

    labelled = label(mask)
    regions = []

    for region in regionprops(labelled):
        if region.area < min_region_area:
            continue

        regions.append({
            "bbox": region.bbox,
            "area": int(region.area),
            "centroid": tuple(float(v) for v in region.centroid),
        })

    return regions


def run_tissue_detection(
    slide_path: str,
    output_dir: str,
    max_size: int = 2000,
    saturation_threshold: float = 0.08,
    value_threshold: float = 0.95,
    slide_id: str | None = None,
):
    """slide_id defaults to slide_id_from_raw_path(slide_path) — the raw
    filename convention bulk/GDC datasets already use. Pass it explicitly
    for a caller that already knows which slide this is (e.g. the tile
    server's ad-hoc uploads, where the id is the one the user chose and is
    already in wsi_registry — agreeing with that record matters more than
    what the stored filename happens to spell)."""
    slide_path = Path(slide_path)
    slide_id = slide_id or slide_id_from_raw_path(slide_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slide = load_slide(slide_path)
    thumbnail = create_thumbnail(slide, max_size=max_size)

    mask = create_tissue_mask(
        thumbnail,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
    )

    thumbnail_path = output_dir / f"{slide_id}_thumbnail.png"
    mask_path = output_dir / f"{slide_id}_tissue_mask.png"
    overlay_path = output_dir / f"{slide_id}_tissue_overlay.png"

    thumbnail.save(thumbnail_path)
    save_mask(mask, mask_path)
    save_overlay(thumbnail, mask, overlay_path)

    regions = get_tissue_regions(mask)

    print(f"Slide: {slide_id}")
    print(f"Thumbnail saved: {thumbnail_path}")
    print(f"Tissue mask saved: {mask_path}")
    print(f"Overlay saved: {overlay_path}")
    print(f"Tissue regions detected: {len(regions)}")

    return {
        "slide_id": slide_id,
        "thumbnail_path": str(thumbnail_path),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "regions": regions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slide",
        required=False,
        help="Optional direct slide path. If not provided, newest unprocessed slide from raw folder is used."
    )

    parser.add_argument(
        "--raw_dir",
        default="/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi/raw"
    )

    parser.add_argument(
        "--output",
        default="/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks"
    )

    parser.add_argument("--max_size", type=int, default=2000)
    parser.add_argument("--sat", type=float, default=0.08)
    parser.add_argument("--val", type=float, default=0.95)

    args = parser.parse_args()

    if args.slide:
        slide_path = Path(args.slide)
    else:
        slide_path = find_newest_unprocessed_slide(
            raw_dir=args.raw_dir,
            output_dir=args.output
        )
        print(f"Auto-selected newest unprocessed slide: {slide_path}")

    run_tissue_detection(
        slide_path=str(slide_path),
        output_dir=args.output,
        max_size=args.max_size,
        saturation_threshold=args.sat,
        value_threshold=args.val,
    )