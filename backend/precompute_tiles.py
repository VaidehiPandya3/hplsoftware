"""
Deep Zoom tile precomputation script.

Run on HPCC for "hot" slides to pre-generate multi-resolution JPEG tiles
so the tile server can serve them instantly without touching the .svs.

Usage:
    python precompute_tiles.py                          # all slides in wsi_registry
    python precompute_tiles.py TCGA-55-7574-01Z-00-DX1  # single slide

Output structure (per slide):
    <OUTPUT_DIR>/<SLIDE_ID>/
        thumbnail.jpg           ← overview JPEG
        <level>/
            <col>_<row>.jpg     ← 256x256 tile at that pyramid level

The tile_server can be extended to check for precomputed tiles before
calling OpenSlide (zero-cost serving for hot slides).
"""

import os
import sys
from pathlib import Path

import openslide
import pandas as pd
from PIL import Image
from sqlalchemy import create_engine
from db_url import database_url

# ---------------------------------------------------------------------------
# Config — match your HPCC environment
# ---------------------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "vpandya")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hpl_kb")

OUTPUT_DIR = Path(os.getenv("TILE_OUTPUT_DIR", "/tmp/hpc_precomputed_tiles"))
TILE_SIZE = 256
THUMB_WIDTH = 3000
JPEG_QUALITY = 85


def get_wsi_map():
    engine = create_engine(
        database_url(DB_NAME, user=DB_USER, password=DB_PASS,
                     host=DB_HOST, port=DB_PORT),
        pool_pre_ping=True,
    )
    df = pd.read_sql("SELECT slide_id, hpc_path FROM wsi_registry", engine)
    df["slide_id"] = df["slide_id"].astype(str).str.strip().str.upper()
    return dict(zip(df["slide_id"], df["hpc_path"]))


def precompute_slide(slide_id: str, svs_path: str):
    print(f"[{slide_id}] Opening {svs_path}")
    slide = openslide.OpenSlide(svs_path)
    out = OUTPUT_DIR / slide_id
    out.mkdir(parents=True, exist_ok=True)

    # 1. Thumbnail
    w0, h0 = slide.level_dimensions[0]
    thumb_h = int(h0 * (THUMB_WIDTH / w0))
    thumb = slide.get_thumbnail((THUMB_WIDTH, thumb_h)).convert("RGB")
    thumb.save(out / "thumbnail.jpg", "JPEG", quality=JPEG_QUALITY)
    print(f"  thumbnail saved ({THUMB_WIDTH}x{thumb_h})")

    # 2. Tiles at each pyramid level
    for level in range(slide.level_count):
        lw, lh = slide.level_dimensions[level]
        ds = slide.level_downsamples[level]
        level_dir = out / str(level)
        level_dir.mkdir(exist_ok=True)

        cols = (lw + TILE_SIZE - 1) // TILE_SIZE
        rows = (lh + TILE_SIZE - 1) // TILE_SIZE
        total = cols * rows
        print(f"  level {level}: {lw}x{lh} → {cols}x{rows} tiles ({total} total)")

        for c in range(cols):
            for r in range(rows):
                tile_path = level_dir / f"{c}_{r}.jpg"
                if tile_path.exists():
                    continue
                origin_x = int(c * TILE_SIZE * ds)
                origin_y = int(r * TILE_SIZE * ds)
                region = slide.read_region((origin_x, origin_y), level, (TILE_SIZE, TILE_SIZE))
                region.convert("RGB").save(tile_path, "JPEG", quality=JPEG_QUALITY)

        print(f"  level {level} done")

    slide.close()
    print(f"[{slide_id}] complete → {out}")


def main():
    wsi_map = get_wsi_map()
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(wsi_map.keys())

    for sid in targets:
        sid = sid.strip().upper()
        svs = wsi_map.get(sid)
        if not svs:
            print(f"[{sid}] SKIP — not in wsi_registry")
            continue
        if not os.path.isfile(svs):
            print(f"[{sid}] SKIP — file not found: {svs}")
            continue
        precompute_slide(sid, svs)

    print("\nAll done.")


if __name__ == "__main__":
    main()
