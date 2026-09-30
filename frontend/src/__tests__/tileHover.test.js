// "Which tile is under the pointer" in the pyramid viewer.
//
// The lookup is the whole feature: hover and click both answer from it, and
// the ways it can be wrong are all quiet. A key built at the wrong pitch does
// not fail — it names a neighbouring tile, or names a tile where there is
// none, and the readout looks exactly as confident either way. That is the
// same class of bug test_tile_pitch.py was written for on the drawing side:
// 1,078 of 1,598 slides are not the 0.252 µm/px the old code assumed, so a
// hit test hard-coded to one pitch would mislabel tiles on two thirds of the
// cohort while looking perfectly healthy on the slide anyone checked.
import { describe, expect, it } from "vitest";

import {
  buildTileLookup,
  GRID_STROKE_MAX,
  GRID_STROKE_MIN,
  gridStrokeWidth,
  tileAtImagePoint,
} from "../components/viewer/overlayBuilders.js";

// A 3x3 patch at a real pitch — 1,734 px, the 0.2325 µm/px scanner — with one
// cell deliberately absent, because "no tile here" has to be an answer.
const PITCH = 1734;
const TILES = [];
for (let col = 0; col < 3; col += 1) {
  for (let row = 0; row < 3; row += 1) {
    if (col === 2 && row === 2) continue; // below the tissue threshold
    TILES.push({
      col,
      row,
      x_native: col * PITCH,
      y_native: row * PITCH,
      slide_tile: `SLIDE_${col}_${row}.JPEG`,
      hpc_id: col * 3 + row,
    });
  }
}

describe("buildTileLookup / tileAtImagePoint", () => {
  const lookup = buildTileLookup(TILES, PITCH);

  it("indexes every tile once", () => {
    expect(lookup.size).toBe(TILES.length);
  });

  it("answers with the tile a point falls inside", () => {
    // Middle of (1,1), and its two far corners — a cell is closed at its top
    // left and open at its bottom right, which is what makes the tiles a
    // partition rather than a set of boxes with contested edges.
    expect(tileAtImagePoint(lookup, 1.5 * PITCH, 1.5 * PITCH, PITCH).slide_tile).toBe("SLIDE_1_1.JPEG");
    expect(tileAtImagePoint(lookup, PITCH, PITCH, PITCH).slide_tile).toBe("SLIDE_1_1.JPEG");
    expect(tileAtImagePoint(lookup, 2 * PITCH - 1, 2 * PITCH - 1, PITCH).slide_tile).toBe("SLIDE_1_1.JPEG");
  });

  it("does not bleed one tile into its neighbour", () => {
    // One pixel past the edge is the next tile, not this one. An off-by-one
    // here is the version of this bug that survives review: every readout is
    // plausible and every one of them is the tile next door.
    expect(tileAtImagePoint(lookup, 2 * PITCH, 1.5 * PITCH, PITCH).slide_tile).toBe("SLIDE_2_1.JPEG");
    expect(tileAtImagePoint(lookup, 1.5 * PITCH, 2 * PITCH, PITCH).slide_tile).toBe("SLIDE_1_2.JPEG");
  });

  it("returns null where the slide has no tile", () => {
    // The skipped cell, and a point off the slide entirely.
    expect(tileAtImagePoint(lookup, 2.5 * PITCH, 2.5 * PITCH, PITCH)).toBeNull();
    expect(tileAtImagePoint(lookup, 40 * PITCH, 0, PITCH)).toBeNull();
    expect(tileAtImagePoint(lookup, -1, -1, PITCH)).toBeNull();
  });

  it("would name the wrong tile at the wrong pitch", () => {
    // The companion that makes the rest load-bearing. This point is 1,700 px
    // across: inside column 0 on a slide tiled at 1,734, and inside column 1
    // if you read it at the 1,600 the server used to publish for every slide.
    // Both answers are real tiles and neither looks uncertain — which is the
    // whole reason the pitch is resolved per slide rather than assumed.
    expect(tileAtImagePoint(lookup, 1700, 100, PITCH).slide_tile).toBe("SLIDE_0_0.JPEG");
    expect(tileAtImagePoint(lookup, 1700, 100, 1600).slide_tile).toBe("SLIDE_1_0.JPEG");
  });

  it("places a record that carries coordinates but no col/row", () => {
    // tiles_meta selects col/row, but nothing guarantees a caller passes a
    // frame that has them; falling back to x_native/pitch keeps such a record
    // in the lattice instead of dropping it from the lookup silently.
    const coordsOnly = buildTileLookup(
      [{ x_native: 2 * PITCH, y_native: 0, slide_tile: "SLIDE_2_0.JPEG", hpc_id: 6 }],
      PITCH
    );
    expect(tileAtImagePoint(coordsOnly, 2.5 * PITCH, 10, PITCH).slide_tile).toBe("SLIDE_2_0.JPEG");
  });

  it("outlines a tile as heavily as the click inspector does", () => {
    // The click inspector draws a fixed 2 units on a viewBox that is the
    // thumbnail, i.e. ~4.2% of a tile at any display scale. The pyramid
    // viewer has to match that fraction rather than a pixel count, or the
    // same grid reads differently in the two viewers.
    for (const cell of [60, 140, 200]) {
      expect(gridStrokeWidth(cell) / cell).toBeCloseTo(0.042, 3);
    }
  });

  it("keeps the outline visible where the old rule went to a hairline", () => {
    // The companion that says why this changed. min(4, w/35) held at 4 px
    // however large the cell grew: 0.67% of a 600 px tile, a thread around a
    // slab, at exactly the zoom where tile boundaries are what you are
    // looking at.
    const oldRule = (w) => Math.max(1, Math.min(4, w / 35));
    expect(oldRule(600)).toBe(4);
    expect(gridStrokeWidth(600)).toBeGreaterThan(2 * oldRule(600));
  });

  it("neither vanishes when zoomed out nor swallows the tile when zoomed in", () => {
    expect(gridStrokeWidth(4)).toBe(GRID_STROKE_MIN);
    expect(gridStrokeWidth(100000)).toBe(GRID_STROKE_MAX);
    // A tile drawn at 40 px across must not be mostly outline.
    expect(gridStrokeWidth(40)).toBeLessThan(40 / 4);
  });

  it("answers nothing rather than throwing when there is no pitch", () => {
    // /slide/{id}/info reports tile_size_native; a slide it cannot resolve one
    // for must leave hovering inert, not crash the viewer around it.
    expect(buildTileLookup(TILES, 0).size).toBe(0);
    expect(tileAtImagePoint(lookup, 100, 100, 0)).toBeNull();
    expect(tileAtImagePoint(buildTileLookup(null, PITCH), 100, 100, PITCH)).toBeNull();
  });
});
