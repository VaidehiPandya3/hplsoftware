// Pure-function port of the overlay/coloring math in app/app_v28.py:
//   - color_for_malignant / color_for_adjacency_group (not in colors.js, so they live here)
//   - add_survival_risk_score               (~line 3280)
//   - build_survival_osd_overlay_records    (~line 4622)
//   - build_osd_overlay_records             (~line 4664)
//   - _draw_heatmap's log-scale gradient     (~line 4513, reused by ClickInspectorViewer)
//   - _draw_grid's per-mode color selection  (~line 4569, reused by ClickInspectorViewer)
//   - _legend_coverage_pairs (Adjacency / Heatmap branches only, ~line 4773)
//
// No React and no DOM here — everything takes plain tile-record arrays (one
// object per row of what app_v28.py calls `df`) and returns plain data.

import { colorForHpc, colorForInflammation, colorForNecrosis } from "../../colors.js";

// -- colors not already ported to colors.js --------------------------------

// Port of backend/malignancy.py. That module exists because app_v28.py used to
// carry two independent normalisations of `hpc_dictionary.malignant` — the
// viewer's colour lookup and its filter — and they disagreed: the colour
// lookup did not recognise the spellings "malignant"/"non-malignant", so a
// dictionary row written that way rendered grey while the filter counted it
// correctly. malignancy.py fixed that by making both read one rule; this is
// that same rule; ported so the two JS call sites (colour, filter/legend
// bucketing) can't drift from each other the same way again.
//
// Recognised spellings, lower-cased and stripped — verbatim from
// backend/malignancy.py's _TRUE / _FALSE, which were themselves taken
// verbatim from the two normalisations this replaced.
const MALIGNANT_TRUE = new Set(["true", "t", "1", "yes", "y", "malignant"]);
const MALIGNANT_FALSE = new Set(["false", "f", "0", "no", "n", "non-malignant", "non malignant"]);

// Port of malignant_flag(): true / false / null. null means "this row does
// not say" — a NULL or a spelling not in the recognised vocabulary — kept
// distinct from non-malignant so a dictionary row that needs attention stays
// visible (grey) rather than being counted as benign.
export function malignantFlag(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (Number.isNaN(value)) return null;
    if (value === 0 || value === 1) return Boolean(value);
    return null;
  }
  const s = String(value).trim().toLowerCase();
  if (MALIGNANT_TRUE.has(s)) return true;
  if (MALIGNANT_FALSE.has(s)) return false;
  return null;
}

// Port of describe_malignant(): "malignant" / "non-malignant" / "missing",
// never null. Used for filtering and legend bucketing.
export function describeMalignant(value) {
  const flag = malignantFlag(value);
  if (flag === null) return "missing";
  return flag ? "malignant" : "non-malignant";
}

export function colorForMalignant(flag) {
  const f = malignantFlag(flag);
  if (f === null) return [160, 160, 160];
  return f ? [255, 80, 80] : [80, 200, 120];
}

export function colorForAdjacencyGroup(groupName) {
  if (groupName === "a_touch") return [80, 200, 120];
  if (groupName === "b_touch") return [255, 165, 0];
  return [160, 160, 160];
}

// -- category-key helpers (used for filtering + legend bucketing) ----------

// Alias kept for callers already using this name (matches app_v28.py's
// `_mal_filter_value`/inline legend bucketing, now both `describe_malignant`).
export const malignantLabelKey = describeMalignant;

export function inflammationLabelKey(v) {
  if (v === null || v === undefined) return "missing";
  const s = String(v).trim().toLowerCase();
  return s === "" ? "missing" : s;
}

export function necrosisLabelKey(v) {
  return inflammationLabelKey(v); // same fillna('missing')+lower+trim rule
}

// -- generic category counter, backs the clickable HPC/Inflammation/Necrosis/
// -- Malignant legend rows (app_v28.py builds each of these inline with its
// -- own value_counts() call; this is the shared shape) --------------------

export function categoryCounts(tiles, getKey) {
  const counts = new Map();
  for (const t of tiles) {
    const k = getKey(t);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const n = tiles.length || 1;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([key, count]) => ({ key, count, pct: (100 * count) / n }));
}

// hpc_id-only counts, top 14, dropping nulls — matches the "HPC clusters"
// legend loop (`hpc_counts.head(14)`), which does not include an "unlabeled"
// row (unlike Inflammation/Necrosis/Malignant, which fold missing into a
// "missing" bucket).
export function hpcCounts(tiles, limit = 14) {
  const counts = new Map();
  for (const t of tiles) {
    const hid = t.hpc_id;
    if (hid === null || hid === undefined || Number.isNaN(hid)) continue;
    const k = Number(hid);
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const n = tiles.length || 1;
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([hpcId, count]) => ({ hpcId, count, pct: (100 * count) / n }));
}

// -- survival risk score (add_survival_risk_score, ~line 3280) -------------

/**
 * @param tiles array of tile records (each may have p_hpc_<id> keys)
 * @param survivalMap Map<hpcId, {coef, expcoef, p, ciLow, ciHigh}> — already
 *   filtered to p < 0.05 by hpcReferenceCache.getHpcSurvivalMapFor
 * @returns {{ tiles: array, usableHpcs: number[] }} new tile objects carrying
 *   survival_risk_score / survival_risk_norm / survival_risk_abs
 */
// Union of ids from every p_hpc_<id> column present anywhere in `tiles`.
// Exported so callers (e.g. hpcReferenceCache fetches, the Heatmap HPC
// dropdown) can compute "which HPCs have a probability column on this
// slide" the same way add_survival_risk_score does.
export function collectProbHpcIds(tiles) {
  const ids = new Set();
  for (const t of tiles) {
    for (const key of Object.keys(t)) {
      if (key.startsWith("p_hpc_")) {
        const hid = Number(key.slice("p_hpc_".length));
        if (Number.isFinite(hid)) ids.add(hid);
      }
    }
  }
  return ids;
}

export function addSurvivalRiskScore(tiles, survivalMap) {
  const probHpcIds = collectProbHpcIds(tiles);

  const usableHpcs = [...probHpcIds].filter((h) => survivalMap.has(h)).sort((a, b) => a - b);

  const risks = tiles.map((t) => {
    let risk = 0;
    for (const hid of usableHpcs) {
      const coef = survivalMap.get(hid).coef;
      const raw = t[`p_hpc_${hid}`];
      const p = Number(raw);
      risk += (Number.isFinite(p) ? p : 0) * coef;
    }
    return risk;
  });

  let maxAbs = 0;
  for (const r of risks) maxAbs = Math.max(maxAbs, Math.abs(r));

  const newTiles = tiles.map((t, i) => {
    const score = risks[i];
    const norm = maxAbs === 0 || !Number.isFinite(maxAbs) ? 0.0 : score / maxAbs;
    return { ...t, survival_risk_score: score, survival_risk_norm: norm, survival_risk_abs: Math.abs(norm) };
  });

  return { tiles: newTiles, usableHpcs };
}

// -- OSD (pyramid viewer) overlay builders ----------------------------------

const MAX_TILES_DEFAULT = 6000;

/**
 * Port of build_survival_osd_overlay_records (~line 4622).
 */
export function buildSurvivalOsdOverlayRecords(tiles, tileSizeNative, maxTiles = MAX_TILES_DEFAULT) {
  const total = tiles ? tiles.length : 0;
  if (!tiles || tiles.length === 0) return { records: [], total, truncated: false };
  const useTiles = tiles.slice(0, maxTiles);
  const records = [];

  for (const r of useTiles) {
    let score = Number(r.survival_risk_norm);
    if (!Number.isFinite(score)) score = 0;
    score = Math.max(-1.0, Math.min(1.0, score));
    const intensity = Math.abs(score);
    if (intensity < 0.02) continue;

    const alpha = 0.25 + 0.65 * intensity;
    let fill, stroke;
    if (score > 0) {
      fill = `rgba(255, 0, 0, ${alpha.toFixed(3)})`;
      stroke = "rgba(180, 0, 0, 0.95)";
    } else {
      fill = `rgba(0, 80, 255, ${alpha.toFixed(3)})`;
      stroke = "rgba(0, 45, 180, 0.95)";
    }

    records.push({
      x: Number(r.x_native),
      y: Number(r.y_native),
      w: Math.round(tileSizeNative),
      h: Math.round(tileSizeNative),
      fill,
      stroke,
      stroke_width: 2,
      opacity: "1.0",
      slide_tile: String(r.slide_tile || ""),
      survival_risk_norm: score,
    });
  }

  return { records, total, truncated: total > maxTiles };
}

/**
 * Port of build_osd_overlay_records (~line 4664).
 *
 * NOTE on fidelity (both preserved from the source rather than "fixed",
 * since the task is a precise port — see the final report for both):
 *   - `mal_f`/`malF` is accepted but never applied inside the per-tile loop
 *     in the original (only infl_f/nec_f are checked there) — the malignant
 *     filter only reaches the *click inspector*'s `_draw_grid`, not the
 *     pyramid/OSD overlay. This function doesn't even take a malF param.
 *   - `heat_alpha`/`heatAlpha` is likewise accepted but never read in the
 *     Heatmap branch below (fill alpha is hardcoded 0.55) — the intensity
 *     slider only affects the click-inspector's raster heatmap
 *     (_draw_heatmap), not this OSD overlay.
 */
export function buildOsdOverlayRecords(
  tiles,
  { highlightMode, inflF, necF, tileSizeNative, heatHpc, heatAlpha = 0.6, adjTileSets, maxTiles = MAX_TILES_DEFAULT } // eslint-disable-line no-unused-vars
) {
  const total = tiles ? tiles.length : 0;
  if (!tiles || tiles.length === 0) return { records: [], total, truncated: false };
  const useTiles = tiles.slice(0, maxTiles);
  const records = [];

  // Heatmap needs pmin/pmax over the (already-truncated) working set once,
  // not per-row — same result as recomputing every iteration in the Python
  // version, just done outside the loop.
  let heatCol = null;
  let pmin = 0;
  let pmax = 1;
  if (highlightMode === "Heatmap" && heatHpc !== null && heatHpc !== undefined) {
    heatCol = `p_hpc_${Number(heatHpc)}`;
    if (useTiles.length > 0 && heatCol in useTiles[0]) {
      const vals = useTiles.map((r) => {
        const v = Number(r[heatCol]);
        return Number.isFinite(v) ? v : 0.0;
      });
      pmin = Math.min(...vals);
      pmax = Math.max(...vals);
      if (pmax <= pmin) pmax = pmin + 1e-9;
    } else {
      heatCol = null; // column absent on this slide -> skip records like Python's `continue`
    }
  }

  for (const r of useTiles) {
    if (inflF !== null && inflF !== undefined) {
      if (String(r.inflammation ?? "").trim().toLowerCase() !== inflF) continue;
    }
    if (necF !== null && necF !== undefined) {
      if (String(r.necrosis ?? "").trim().toLowerCase() !== necF) continue;
    }

    if (highlightMode === "Heatmap") {
      if (heatCol === null) continue;
      const pval = Number(r[heatCol]);
      const p = Number.isFinite(pval) ? pval : 0.0;
      const t = (p - pmin) / (pmax - pmin);
      const red = Math.round(255 * t);
      const green = 255;
      const blue = Math.round(255 * (1 - t));
      records.push({
        x: Number(r.x_native),
        y: Number(r.y_native),
        w: Math.round(tileSizeNative),
        h: Math.round(tileSizeNative),
        fill: `rgba(${red}, ${green}, ${blue}, 0.55)`,
        stroke: "rgba(255,255,255,0.25)",
        stroke_width: 1,
        opacity: "1.0",
      });
      continue;
    }

    if (highlightMode === "Survival Risk Heatmap") {
      const scoreRaw = Number(r.survival_risk_norm);
      let score = Number.isFinite(scoreRaw) ? scoreRaw : 0.0;
      score = Math.max(-1.0, Math.min(1.0, score));
      const intensity = Math.abs(score);
      if (intensity < 0.02) continue;
      const alpha = 0.15 + 0.6 * intensity;
      const fill = score > 0 ? `rgba(255, 0, 0, ${alpha.toFixed(3)})` : `rgba(0, 80, 255, ${alpha.toFixed(3)})`;
      records.push({
        x: Number(r.x_native),
        y: Number(r.y_native),
        w: Math.round(tileSizeNative),
        h: Math.round(tileSizeNative),
        fill,
        stroke: "rgba(255,255,255,0.20)",
        stroke_width: 1,
        opacity: "1.0",
      });
      continue;
    }

    let color;
    if (highlightMode === "Inflammation") {
      color = colorForInflammation(r.inflammation);
    } else if (highlightMode === "Necrosis") {
      color = colorForNecrosis(r.necrosis);
    } else if (highlightMode === "Malignant") {
      color = colorForMalignant(r.malignant);
    } else if (highlightMode === "Adjacency") {
      if (!adjTileSets) continue;
      const key = String(r.slide_tile);
      if (adjTileSets.aTouch.has(key)) color = colorForAdjacencyGroup("a_touch");
      else if (adjTileSets.bTouch.has(key)) color = colorForAdjacencyGroup("b_touch");
      else continue;
    } else {
      color = colorForHpc(r.hpc_id);
    }

    const [cr, cg, cb] = color;
    records.push({
      x: Number(r.x_native),
      y: Number(r.y_native),
      w: Math.round(tileSizeNative),
      h: Math.round(tileSizeNative),
      color: `rgb(${cr}, ${cg}, ${cb})`,
      slide_tile: String(r.slide_tile || ""),
      hpc_id: r.hpc_id === null || r.hpc_id === undefined || Number.isNaN(r.hpc_id) ? "" : String(Math.trunc(r.hpc_id)),
    });
  }

  return { records, total, truncated: total > maxTiles };
}

// -- pyramid-viewer hit testing --------------------------------------------
//
// "Which tile is under this point" for the OpenSeadragon viewer, kept here
// with the rest of the DOM-free overlay math so it can be tested without a
// browser — the alternative is a hit test that only the pointer can exercise.
//
// The lattice is what makes this a Map get rather than a scan over every tile
// on the slide: tile_coordinates stores x_native = col * pitch (the tiler's
// own stride, whatever the slide's mpp), so the cell containing a point is
// floor(point / pitch). A scan would also have to run on every mouse move.

// How heavy a grid outline is drawn in the pyramid viewer, given the tile's
// current width in screen pixels.
//
// A constant fraction of the cell, because that is what the click inspector
// has always drawn — its viewBox is the thumbnail, so its fixed 2-unit stroke
// is ~4.2% of a tile however the image is scaled on screen. The pyramid
// viewer's rule was min(4, w/35), which matches that only while a tile is
// under ~140 px: past there the 4 px cap holds while the cell keeps growing,
// so at 600 px the outline is 0.67% of the cell — a hairline around a slab,
// and the reason the same grid reads clearly in one viewer and faintly in the
// other at exactly the zoom where you are looking at tile boundaries.
//
// The floor keeps the line visible when zoomed out; the ceiling stops it
// eating the tile it is supposed to be framing at extreme zoom.
export const GRID_STROKE_FRACTION = 0.042;
export const GRID_STROKE_MIN = 1.5;
export const GRID_STROKE_MAX = 9;

export function gridStrokeWidth(onScreenTileWidth) {
  const w = Number(onScreenTileWidth);
  if (!Number.isFinite(w) || w <= 0) return GRID_STROKE_MIN;
  return Math.max(GRID_STROKE_MIN, Math.min(GRID_STROKE_MAX, w * GRID_STROKE_FRACTION));
}

// Drawn under the coloured outline, slightly wider, so the grid reads against
// both pale H&E and the dark background behind a slide's edge. The HPC colours
// themselves are not darkened — they are what the legend is keyed on, and a
// tile whose outline does not match its legend swatch is worse than a faint
// one.
export const GRID_HALO_COLOR = "rgba(17, 24, 39, 0.72)";
export const GRID_HALO_EXTRA = 2.5;

/** Map from "<col>_<row>" to the tile record. */
export function buildTileLookup(tiles, pitch) {
  const map = new Map();
  const p = Number(pitch);
  if (!tiles || !Number.isFinite(p) || p <= 0) return map;
  for (const t of tiles) {
    // col/row come off tile_coordinates when present; a record carrying only
    // coordinates is placed by them, so this cannot silently drop tiles from
    // a source that does not have the grid indices.
    const col = Number.isFinite(Number(t.col)) ? Number(t.col) : Math.floor(Number(t.x_native) / p);
    const row = Number.isFinite(Number(t.row)) ? Number(t.row) : Math.floor(Number(t.y_native) / p);
    if (!Number.isFinite(col) || !Number.isFinite(row)) continue;
    map.set(`${col}_${row}`, t);
  }
  return map;
}

/** The tile containing an image-space point, or null where the slide has no
 *  tile — background, or tissue below the threshold Stage 1 skipped. */
export function tileAtImagePoint(lookup, imageX, imageY, pitch) {
  const p = Number(pitch);
  if (!lookup || lookup.size === 0 || !Number.isFinite(p) || p <= 0) return null;
  if (!Number.isFinite(imageX) || !Number.isFinite(imageY)) return null;
  if (imageX < 0 || imageY < 0) return null;
  return lookup.get(`${Math.floor(imageX / p)}_${Math.floor(imageY / p)}`) || null;
}

// -- click-inspector (raster-equivalent) helpers ----------------------------

// Port of _draw_heatmap's log-scale gradient (~line 4513), factored so the
// caller (ClickInspectorViewer) supplies pmin/pmax once and asks per-tile.
export function computeMinMax(tiles, col) {
  let pmin = Infinity;
  let pmax = -Infinity;
  let any = false;
  for (const t of tiles) {
    const v = Number(t[col]);
    if (Number.isFinite(v)) {
      any = true;
      if (v < pmin) pmin = v;
      if (v > pmax) pmax = v;
    }
  }
  if (!any) return null;
  if (pmax <= pmin) pmax = pmin + 1e-9;
  return { pmin, pmax };
}

/** Returns {r,g,b,alpha(0-1)} — alpha uses the same 0.10..0.95 clamp as Python. */
export function heatmapColor(p, pmin, pmax, alphaMaxSetting) {
  const epsilon = 1e-6;
  const denom = Math.log10(pmax + epsilon) - Math.log10(pmin + epsilon) || 1e-9;
  const t = Math.min(1, Math.max(0, (Math.log10(p + epsilon) - Math.log10(pmin + epsilon)) / denom));
  const alphaMin = 0.1;
  const alphaMax = Math.max(0.2, Math.min(alphaMaxSetting, 0.95));
  const alpha = alphaMin + (alphaMax - alphaMin) * t;
  return { r: Math.round(255 * t), g: 255, b: Math.round(255 * (1 - t)), alpha };
}

// Port of _draw_grid's per-mode color selection (~line 4569) plus its
// infl_f/nec_f/mal_f skip logic (the click-inspector grid DOES honor mal_f,
// unlike the OSD overlay above — this matches the source exactly).
export function passesGridFilters(tile, inflF, necF, malF) {
  if (inflF !== null && inflF !== undefined) {
    if (String(tile.inflammation ?? "").trim().toLowerCase() !== inflF) return false;
  }
  if (necF !== null && necF !== undefined) {
    if (String(tile.necrosis ?? "").trim().toLowerCase() !== necF) return false;
  }
  if (malF !== null && malF !== undefined) {
    if (malignantLabelKey(tile.malignant) !== malF) return false;
  }
  return true;
}

/** Returns an [r,g,b] color, or null to mean "skip this tile" (Adjacency-mode non-match). */
export function gridColorForTile(highlightMode, tile, adjTileSets) {
  if (highlightMode === "Inflammation") return colorForInflammation(tile.inflammation);
  if (highlightMode === "Necrosis") return colorForNecrosis(tile.necrosis);
  if (highlightMode === "Malignant") return colorForMalignant(tile.malignant);
  if (highlightMode === "Adjacency") {
    if (!adjTileSets) return null;
    const key = String(tile.slide_tile);
    if (adjTileSets.aTouch.has(key)) return colorForAdjacencyGroup("a_touch");
    if (adjTileSets.bTouch.has(key)) return colorForAdjacencyGroup("b_touch");
    return null;
  }
  return colorForHpc(tile.hpc_id);
}

// -- static legend coverage pairs (Adjacency / Heatmap modes only) ---------
// Port of the Adjacency and Heatmap branches of _legend_coverage_pairs
// (~line 4773). The other modes (HPC clusters/Inflammation/Necrosis/
// Malignant) render as clickable rows built from hpcCounts/categoryCounts
// instead, matching how app_v28.py builds those inline rather than via
// build_legend_panel_html.

export function legendCoveragePairsAdjacency(tiles, adjTileSets, hpcA, hpcB) {
  const n = tiles.length || 1;
  if (!adjTileSets) return [["(select an HPC pair)", 100]];
  let inA = 0;
  let inB = 0;
  let other = 0;
  for (const t of tiles) {
    const key = String(t.slide_tile);
    const a = adjTileSets.aTouch.has(key);
    const b = adjTileSets.bTouch.has(key);
    if (a) inA += 1;
    if (b) inB += 1;
    if (!a && !b) other += 1;
  }
  return [
    [`HPC ${hpcA} → ${hpcB} (green)`, (100 * inA) / n],
    [`HPC ${hpcB} → ${hpcA} (orange)`, (100 * inB) / n],
    ["Not in selected pair", (100 * other) / n],
  ];
}

function quantileSorted(sortedArr, q) {
  if (sortedArr.length === 0) return NaN;
  const idx = q * (sortedArr.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
}

export function legendCoveragePairsHeatmap(tiles, heatHpc) {
  const n = tiles.length || 1;
  if (heatHpc === null || heatHpc === undefined) return [];
  const col = `p_hpc_${Number(heatHpc)}`;
  const hasCol = tiles.some((t) => col in t);
  if (!hasCol) return [["(no probability column)", 100]];

  const vals = [];
  for (const t of tiles) {
    const v = Number(t[col]);
    if (Number.isFinite(v)) vals.push(v);
  }
  if (vals.length === 0) return [["Missing score", 100]];

  const sorted = [...vals].sort((a, b) => a - b);
  const q1 = quantileSorted(sorted, 0.33);
  const q2 = quantileSorted(sorted, 0.66);

  const counts = { "Low P": 0, "Mid P": 0, "High P": 0, Missing: 0 };
  for (const t of tiles) {
    const raw = t[col];
    const v = Number(raw);
    if (raw === null || raw === undefined || !Number.isFinite(v)) {
      counts.Missing += 1;
      continue;
    }
    if (v <= q1) counts["Low P"] += 1;
    else if (v <= q2) counts["Mid P"] += 1;
    else counts["High P"] += 1;
  }

  return Object.entries(counts)
    .filter(([, c]) => c > 0)
    .map(([label, c]) => [label, (100 * c) / n]);
}
