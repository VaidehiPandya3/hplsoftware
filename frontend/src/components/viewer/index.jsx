import { useEffect, useMemo, useState } from "react";
import { api } from "../../api.js";
import { colorForHpc, rgbToCss } from "../../colors.js";
import {
  addSurvivalRiskScore,
  buildOsdOverlayRecords,
  buildSurvivalOsdOverlayRecords,
  collectProbHpcIds,
  malignantLabelKey,
} from "./overlayBuilders.js";
import { getHpcSurvivalMapFor, getHpcTitleMapFor } from "./hpcReferenceCache.js";
import PyramidViewer from "./PyramidViewer.jsx";
import ClickInspectorViewer from "./ClickInspectorViewer.jsx";
import TileInfo from "./TileInfo.jsx";
import Legend from "./Legend.jsx";
import HpcAnnotation from "./HpcAnnotation.jsx";
import AdjacencyControls from "./AdjacencyControls.jsx";
import "./viewer.css";

const THUMB_WIDTH = 3000;
const HIGHLIGHT_MODES = ["HPC clusters", "Inflammation", "Necrosis", "Malignant", "Adjacency", "Heatmap", "Survival Risk Heatmap"];
const NON_ISOLATE_MODES = ["Adjacency", "Heatmap", "Survival Risk Heatmap"];

function normalizeTiles(raw) {
  return (raw || []).map((row) => {
    const r = { ...row };
    if (r.slide_tile !== undefined && r.slide_tile !== null) r.slide_tile = String(r.slide_tile).trim().toUpperCase();
    if (r.slides !== undefined && r.slides !== null) r.slides = String(r.slides).trim().toUpperCase();
    const n = Number(r.hpc_id);
    r.hpc_id = r.hpc_id !== undefined && r.hpc_id !== null && Number.isFinite(n) ? n : null;
    return r;
  });
}

function parseAdjacency(data) {
  const pairEdgeCounts = [];
  for (const [k, v] of Object.entries(data.pair_edge_counts || {})) {
    const [a, b] = k.split("_").map(Number);
    pairEdgeCounts.push({ a, b, count: v });
  }
  const tileNeighborPairs = new Map();
  for (const [k, v] of Object.entries(data.tile_neighbor_pairs || {})) {
    const [a, b] = k.split("_").map(Number);
    tileNeighborPairs.set(`${a}_${b}`, {
      aTouch: new Set(v.a_touch || []),
      bTouch: new Set(v.b_touch || []),
    });
  }
  return { pairEdgeCounts, tileNeighborPairs };
}

// Main WSI viewer — a port of show_wsi() (app_v28.py ~3877-4508). Takes a
// single slideId (the side-by-side "compare" mode from the original is out
// of scope here — see the task notes) and renders one OpenSeadragon pyramid
// viewer or one click-inspector thumbnail, plus the mode-dependent legend
// column.
export default function SlideViewer({ slideId }) {
  const [slideInfo, setSlideInfo] = useState(null); // { w0, h0, tileSizeNative }
  const [tiles, setTiles] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(false);

  const [viewerMode, setViewerMode] = useState("Pyramid zoom");
  const [showGrid, setShowGrid] = useState(true);
  const [highlightMode, setHighlightMode] = useState("HPC clusters");

  const [selectedHpc, setSelectedHpc] = useState(null);
  const [inflF, setInflF] = useState(null);
  const [necF, setNecF] = useState(null);
  const [malF, setMalF] = useState(null);

  const [heatHpc, setHeatHpc] = useState(null);
  const [heatAlpha, setHeatAlpha] = useState(0.6);

  const [hpcTitleMap, setHpcTitleMap] = useState(new Map());
  const [survivalMap, setSurvivalMap] = useState(new Map());

  const [adjacency, setAdjacency] = useState(null); // { pairEdgeCounts, tileNeighborPairs }
  const [adjSelection, setAdjSelection] = useState(null); // { a, b, tileSets }

  const [selectedTile, setSelectedTile] = useState(null); // { slide_tile, x_native, y_native, tile }

  // -- reset + load on slide change ---------------------------------------
  useEffect(() => {
    setSlideInfo(null);
    setTiles(null);
    setLoadError(null);
    setSelectedHpc(null);
    setInflF(null);
    setNecF(null);
    setMalF(null);
    setHeatHpc(null);
    setAdjacency(null);
    setAdjSelection(null);
    setSelectedTile(null);
    setHpcTitleMap(new Map());
    setSurvivalMap(new Map());

    if (!slideId) return;

    let cancelled = false;
    setLoading(true);

    Promise.all([api.getSlideInfo(slideId), api.getTilesMeta(slideId)])
      .then(([info, tilesMeta]) => {
        if (cancelled) return;
        const w0 = info.level_dimensions[0].width;
        const h0 = info.level_dimensions[0].height;
        setSlideInfo({ w0, h0, tileSizeNative: Number(info.tile_size_native) });
        setTiles(normalizeTiles(tilesMeta));
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e.message || String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [slideId]);

  // -- HPC title map for ids present on this slide -------------------------
  useEffect(() => {
    if (!tiles || tiles.length === 0) return;
    const ids = [...new Set(tiles.map((t) => t.hpc_id).filter((v) => v !== null))];
    if (ids.length === 0) {
      setHpcTitleMap(new Map());
      return;
    }
    let cancelled = false;
    getHpcTitleMapFor(ids).then((m) => {
      if (!cancelled) setHpcTitleMap(m);
    });
    return () => {
      cancelled = true;
    };
  }, [tiles]);

  // -- survival coefficients, only fetched when the mode needs them --------
  useEffect(() => {
    if (highlightMode !== "Survival Risk Heatmap" || !tiles || tiles.length === 0) return;
    const probIds = [...collectProbHpcIds(tiles)];
    if (probIds.length === 0) {
      setSurvivalMap(new Map());
      return;
    }
    let cancelled = false;
    getHpcSurvivalMapFor(probIds, 0.05).then((m) => {
      if (!cancelled) setSurvivalMap(m);
    });
    return () => {
      cancelled = true;
    };
  }, [highlightMode, tiles]);

  // Selecting Survival Risk Heatmap clears any HPC isolation, matching
  // show_wsi's `st.session_state[_k("selected_hpc")] = None` on entry.
  useEffect(() => {
    if (highlightMode === "Survival Risk Heatmap") setSelectedHpc(null);
  }, [highlightMode]);

  // -- adjacency data, fetched lazily when that mode is selected ------------
  useEffect(() => {
    if (highlightMode !== "Adjacency" || !slideId || adjacency) return;
    let cancelled = false;
    api
      .getAdjacency(slideId)
      .then((data) => {
        if (!cancelled) setAdjacency(parseAdjacency(data));
      })
      .catch(() => {
        if (!cancelled) setAdjacency({ pairEdgeCounts: [], tileNeighborPairs: new Map() });
      });
    return () => {
      cancelled = true;
    };
  }, [highlightMode, slideId, adjacency]);

  // -- tiles with hpc_title merged in (load_hpc_titles() left-join) --------
  const titledTiles = useMemo(() => {
    if (!tiles) return [];
    return tiles.map((t) => ({ ...t, hpc_title: hpcTitleMap.get(t.hpc_id) || "" }));
  }, [tiles, hpcTitleMap]);

  // -- survival risk score added to every tile (add_survival_risk_score) ---
  const { tiles: riskTiles, usableHpcs: usedSurvivalHpcs } = useMemo(() => {
    if (highlightMode !== "Survival Risk Heatmap" || titledTiles.length === 0) {
      return { tiles: titledTiles, usableHpcs: [] };
    }
    return addSurvivalRiskScore(titledTiles, survivalMap);
  }, [titledTiles, survivalMap, highlightMode]);

  // -- Heatmap mode's HPC dropdown options: present hpc_id AND has a p_hpc_ column
  const heatmapHpcOptions = useMemo(() => {
    if (!tiles || tiles.length === 0) return [];
    const present = new Set(tiles.map((t) => t.hpc_id).filter((v) => v !== null));
    const withProb = collectProbHpcIds(tiles);
    return [...present].filter((h) => withProb.has(h)).sort((a, b) => a - b);
  }, [tiles]);

  useEffect(() => {
    if (highlightMode !== "Heatmap") return;
    if (heatmapHpcOptions.length === 0) {
      setHeatHpc(null);
      return;
    }
    if (!heatmapHpcOptions.includes(heatHpc)) setHeatHpc(heatmapHpcOptions[0]);
  }, [highlightMode, heatmapHpcOptions, heatHpc]);

  // -- HPC-isolate filter (grid_df in the original) -------------------------
  const gridTiles = useMemo(() => {
    if (selectedHpc !== null && highlightMode === "HPC clusters") {
      return riskTiles.filter((t) => t.hpc_id === selectedHpc);
    }
    return riskTiles;
  }, [riskTiles, selectedHpc, highlightMode]);

  // Source dataframe for both the grid overlay and the OSD overlay builder —
  // matches the `grid_df if selected_hpc is not None and highlight_mode not
  // in (...) else df` expression used at both call sites in show_wsi.
  const overlaySourceTiles = useMemo(() => {
    if (selectedHpc !== null && !NON_ISOLATE_MODES.includes(highlightMode)) return gridTiles;
    return riskTiles;
  }, [gridTiles, riskTiles, selectedHpc, highlightMode]);

  // -- "Matched tiles: N" caption (filtered_df in the original) ------------
  const matchedCount = useMemo(() => {
    let f = riskTiles;
    if (selectedHpc !== null) f = f.filter((t) => t.hpc_id === selectedHpc);
    if (inflF !== null) f = f.filter((t) => String(t.inflammation ?? "").trim().toLowerCase() === inflF);
    if (necF !== null) f = f.filter((t) => String(t.necrosis ?? "").trim().toLowerCase() === necF);
    if (malF !== null) f = f.filter((t) => malignantLabelKey(t.malignant) === malF);
    return f.length;
  }, [riskTiles, selectedHpc, inflF, necF, malF]);

  // -- OSD overlay records (pyramid mode) -----------------------------------
  const osdOverlay = useMemo(() => {
    if (!showGrid || !slideInfo) return { records: [], total: 0, truncated: false };
    if (highlightMode === "Survival Risk Heatmap") {
      return buildSurvivalOsdOverlayRecords(overlaySourceTiles, slideInfo.tileSizeNative);
    }
    return buildOsdOverlayRecords(overlaySourceTiles, {
      highlightMode,
      inflF,
      necF,
      tileSizeNative: slideInfo.tileSizeNative,
      heatHpc,
      heatAlpha,
      adjTileSets: adjSelection?.tileSets,
    });
  }, [showGrid, slideInfo, highlightMode, overlaySourceTiles, inflF, necF, heatHpc, heatAlpha, adjSelection]);

  const selectedTileRectForPyramid = useMemo(() => {
    if (!selectedTile || !slideInfo) return null;
    return { x: selectedTile.x_native, y: selectedTile.y_native, w: slideInfo.tileSizeNative, h: slideInfo.tileSizeNative };
  }, [selectedTile, slideInfo]);

  if (!slideId) {
    return (
      <div className="viewer-root">
        <div className="viewer-placeholder">Select a slide to open the WSI viewer.</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="viewer-root">
        <div className="viewer-loading">Loading tile metadata for {slideId}…</div>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="viewer-root">
        <div className="viewer-error">Cannot reach tile server for slide {slideId}: {loadError}</div>
      </div>
    );
  }

  if (!slideInfo || !tiles) return null;

  if (tiles.length === 0) {
    return (
      <div className="viewer-root">
        <div className="viewer-warning">No tile data for slide {slideId}</div>
      </div>
    );
  }

  const thumbnailUrl = api.thumbnailUrl(slideId, THUMB_WIDTH, 85);
  const downsampleHint = { w0: slideInfo.w0 }; // divisor supplied once the img actually loads

  return (
    <div className="viewer-root">
      <div className="viewer-controls">
        <div className="viewer-radio-row">
          <span>Viewer mode</span>
          {["Pyramid zoom", "Click tile inspector"].map((m) => (
            <label className="viewer-radio-option" key={m}>
              <input type="radio" name="viewer-mode" checked={viewerMode === m} onChange={() => setViewerMode(m)} />
              {m}
            </label>
          ))}
        </div>

        <label className="viewer-checkbox-row">
          <input type="checkbox" checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} />
          Show tile grid overlays
        </label>

        <div className="viewer-radio-row">
          <span>Highlight mode</span>
          {HIGHLIGHT_MODES.map((m) => (
            <label className="viewer-radio-option" key={m}>
              <input type="radio" name="highlight-mode" checked={highlightMode === m} onChange={() => setHighlightMode(m)} />
              {m}
            </label>
          ))}
        </div>

        {highlightMode === "Heatmap" && (
          <HeatmapControls
            options={heatmapHpcOptions}
            heatHpc={heatHpc}
            onChangeHpc={setHeatHpc}
            heatAlpha={heatAlpha}
            onChangeAlpha={setHeatAlpha}
            hpcTitleMap={hpcTitleMap}
          />
        )}

        {highlightMode === "Survival Risk Heatmap" &&
          (usedSurvivalHpcs.length === 0 ? (
            <div className="viewer-warning">
              No survival-linked HPC probability columns found for this slide. This slide may not contain HPCs that also have
              survival data.
            </div>
          ) : (
            <div className="viewer-caption">
              Survival risk heatmap uses {usedSurvivalHpcs.length} HPCs that are both present in this WSI and available in survival
              analysis.
            </div>
          ))}

        {highlightMode === "Adjacency" && (
          <AdjacencyControls
            pairEdgeCounts={adjacency?.pairEdgeCounts || []}
            tileNeighborPairs={adjacency?.tileNeighborPairs || new Map()}
            onHighlight={(a, b, tileSets) => setAdjSelection({ a, b, tileSets })}
          />
        )}
      </div>

      {selectedHpc !== null && (
        <div>
          <div className="viewer-hpc-selected-banner" style={{ borderLeftColor: rgbToCss(colorForHpc(selectedHpc)) }}>
            <span style={{ fontWeight: 700 }}>HPC {selectedHpc}</span>
            {hpcTitleMap.get(selectedHpc) && <><br /><span style={{ opacity: 0.92 }}>{hpcTitleMap.get(selectedHpc)}</span></>}
          </div>
          <details open>
            <summary>HPC Biological Interpretation</summary>
            <HpcAnnotation hpcId={selectedHpc} />
          </details>
        </div>
      )}

      <div className="viewer-caption">Matched tiles: {matchedCount}</div>

      <div className="viewer-layout">
        <div className="viewer-main-col">
          {viewerMode === "Pyramid zoom" ? (
            <>
              <div className="viewer-caption">OpenSeadragon pyramid viewer with optional tile-grid overlay.</div>
              {osdOverlay.truncated && (
                <div className="viewer-caption">
                  Showing first 6000 of {osdOverlay.total} tiles in the overlay (capped for performance).
                </div>
              )}
              <PyramidViewer
                dziUrl={api.dziUrl(slideId)}
                overlayTiles={osdOverlay.records}
                selectedTileRect={selectedTileRectForPyramid}
                // The slide's whole tile list, not the filtered overlay set,
                // and not the 6,000-record cap the overlay draws under: what
                // a tile *is* does not depend on the legend filter, and a
                // pointer over tile 8,000 should still be answered.
                tileIndex={riskTiles}
                tileSizeNative={slideInfo.tileSizeNative}
                onSelectTile={setSelectedTile}
                height={780}
              />
              {selectedTile && selectedTile.tile && (
                <div className="viewer-tile-result">
                  <div className="viewer-info">
                    Tile selected: {String(selectedTile.tile.tiles || selectedTile.slide_tile)}
                  </div>
                  {/* Same panel the click inspector shows, so a tile picked in
                      either mode reads the same — and the selection itself is
                      shared state, so switching modes keeps it. */}
                  <TileInfo tile={selectedTile.tile} heatHpc={highlightMode === "Heatmap" ? heatHpc : null} />
                  <button type="button" className="viewer-btn" onClick={() => setSelectedTile(null)}>
                    Clear selection
                  </button>
                </div>
              )}
            </>
          ) : (
            <ClickInspectorViewer
              slideId={slideId}
              thumbnailUrl={thumbnailUrl}
              tiles={riskTiles}
              gridTiles={overlaySourceTiles}
              downsampleHint={downsampleHint}
              tileSizeNative={slideInfo.tileSizeNative}
              showGrid={showGrid}
              highlightMode={highlightMode}
              inflF={inflF}
              necF={necF}
              malF={malF}
              heatHpc={heatHpc}
              heatAlpha={heatAlpha}
              adjTileSets={adjSelection?.tileSets}
              selectedTile={selectedTile}
              onSelectTile={setSelectedTile}
              hpcTitleMap={hpcTitleMap}
            />
          )}
        </div>

        <div className="viewer-legend-col">
          <Legend
            highlightMode={highlightMode}
            tiles={riskTiles}
            selectedHpc={selectedHpc}
            onSelectHpc={setSelectedHpc}
            inflF={inflF}
            onSelectInfl={setInflF}
            necF={necF}
            onSelectNec={setNecF}
            malF={malF}
            onSelectMal={setMalF}
            heatHpc={heatHpc}
            usedSurvivalHpcs={usedSurvivalHpcs}
            adjTileSets={adjSelection?.tileSets}
            adjHpcA={adjSelection?.a}
            adjHpcB={adjSelection?.b}
          />
        </div>
      </div>
    </div>
  );
}

function HeatmapControls({ options, heatHpc, onChangeHpc, heatAlpha, onChangeAlpha, hpcTitleMap }) {
  if (options.length === 0) {
    return (
      <div className="viewer-warning">
        No heatmap cluster options found for this WSI. This slide may not have matching hpc_id values and p_hpc_* probability
        columns.
      </div>
    );
  }

  return (
    <div>
      <label className="viewer-caption" htmlFor="viewer-heat-hpc-select">
        Heatmap HPC clusters present in this WSI
      </label>
      <select
        id="viewer-heat-hpc-select"
        className="viewer-select"
        value={heatHpc ?? ""}
        onChange={(e) => onChangeHpc(Number(e.target.value))}
      >
        {options.map((hid) => {
          const title = hpcTitleMap.get(hid);
          return (
            <option key={hid} value={hid}>
              HPC {hid}
              {title ? `: ${title}` : ""}
            </option>
          );
        })}
      </select>
      <div className="viewer-caption">Showing heatmap options for {options.length} HPC clusters present on this WSI.</div>
      <div className="viewer-slider-row">
        <span>Heat intensity</span>
        <input
          type="range"
          min={0.1}
          max={1.0}
          step={0.05}
          value={heatAlpha}
          onChange={(e) => onChangeAlpha(Number(e.target.value))}
        />
        <span>{heatAlpha.toFixed(2)}</span>
      </div>
    </div>
  );
}
