import { useEffect, useMemo, useRef, useState } from "react";
import OpenSeadragon from "openseadragon";

import {
  buildTileLookup,
  GRID_HALO_COLOR,
  GRID_HALO_EXTRA,
  gridStrokeWidth,
  tileAtImagePoint,
} from "./overlayBuilders.js";

// Port of render_openseadragon_viewer (app_v28.py ~line 3863). The Streamlit
// version loads OpenSeadragon from a CDN `<script>` tag inside a
// components.html iframe and syncs an SVG overlay to the viewport on
// open/animation/animation-finish/resize/zoom/pan. Here we drive the
// `openseadragon` npm package directly from a useEffect — the sync math
// (imageToViewportRectangle + pixelFromPoint) is ported near-verbatim.
//
// `overlayTiles`: array of records with {x,y,w,h} in native slide pixels
// plus either `color` (grid-outline mode) or `fill`/`stroke` (heatmap/risk
// mode) — the same shape build_osd_overlay_records / build_survival_osd_overlay_records
// produce.
// `selectedTileRect`: optional {x,y,w,h} in native pixels — draws a
// yellow+lime double outline like the click-inspector's selected-tile
// highlight.
//
// `tileIndex` + `tileSizeNative` add hover and click to pyramid mode, which
// had neither: the overlay drew the grid and nothing could tell you which
// cell you were looking at. Three things make that work without a hit test
// over every tile on every mouse move:
//
//   * the tiles sit on a regular lattice — x_native = col * pitch, which is
//     what tile_coordinates records and what _tile_size_native() reads back
//     out of it — so the cell under a point is floor(x / pitch), and the
//     lookup is one Map get rather than a scan of 20,000 records;
//   * the highlight lives in its own <g>, so hovering redraws two rects
//     instead of rebuilding the whole grid (up to 6,000 nodes) per mouse move;
//   * hover state is only pushed into React when the cell *changes*, not on
//     every pixel of movement.
//
// The index is deliberately the slide's full tile list, not the filtered
// overlay set: hovering is a question about the slide ("what is this tile?"),
// and answering it only for tiles that happen to pass the current legend
// filter would make the answer depend on a control that has nothing to do
// with the question.
export default function PyramidViewer({
  dziUrl,
  overlayTiles = [],
  selectedTileRect = null,
  tileIndex = null,
  tileSizeNative = 0,
  onSelectTile = null,
  height = 780,
}) {
  const containerRef = useRef(null);
  const svgRef = useRef(null);
  const gridGroupRef = useRef(null);
  const highlightGroupRef = useRef(null);
  const viewerRef = useRef(null);
  const drawGridRef = useRef(null);
  const drawHighlightRef = useRef(null);
  const overlayTilesRef = useRef(overlayTiles);
  const selectedTileRectRef = useRef(selectedTileRect);
  const [failed, setFailed] = useState(false);
  const [hovered, setHovered] = useState(null); // the tile record under the pointer

  overlayTilesRef.current = overlayTiles;
  selectedTileRectRef.current = selectedTileRect;

  // Rebuilt only when the slide's tiles or its pitch change — not per render,
  // and certainly not per mouse move.
  const lookup = useMemo(() => buildTileLookup(tileIndex, tileSizeNative), [tileIndex, tileSizeNative]);

  const lookupRef = useRef(lookup);
  const pitchRef = useRef(tileSizeNative);
  const hoveredRef = useRef(null);
  const onSelectTileRef = useRef(onSelectTile);
  lookupRef.current = lookup;
  pitchRef.current = tileSizeNative;
  onSelectTileRef.current = onSelectTile;

  useEffect(() => {
    if (!dziUrl || !containerRef.current) return undefined;
    setFailed(false);

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
      tileSources: dziUrl,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      animationTime: 0.4,
      blendTime: 0.1,
      constrainDuringPan: true,
      visibilityRatio: 0.8,
      minZoomImageRatio: 0.8,
      maxZoomPixelRatio: 3.0,
      showRotationControl: false,
      gestureSettingsMouse: {
        // Off, where it used to be on: a single click now selects the tile
        // under the pointer, and the two cannot share the gesture. Double
        // click still zooms, which is the gesture people actually reach for
        // in a slide viewer, and the scroll wheel is untouched.
        clickToZoom: false,
        dblClickToZoom: true,
        dragToPan: true,
        scrollToZoom: true,
      },
    });
    viewerRef.current = viewer;

    // The rect's geometry in screen pixels, or null when it is entirely
    // outside the viewport.
    function screenRect(t) {
      const rectVp = viewer.viewport.imageToViewportRectangle(Number(t.x), Number(t.y), Number(t.w), Number(t.h));
      const p1 = viewer.viewport.pixelFromPoint(rectVp.getTopLeft(), true);
      const p2 = viewer.viewport.pixelFromPoint(rectVp.getBottomRight(), true);

      const x = Math.min(p1.x, p2.x);
      const y = Math.min(p1.y, p2.y);
      const w = Math.abs(p2.x - p1.x);
      const h = Math.abs(p2.y - p1.y);

      const container = viewer.container.getBoundingClientRect();
      if (x + w < 0 || y + h < 0 || x > container.width || y > container.height) return null;
      return { x, y, w, h };
    }

    function rectElement(box, { stroke, strokeWidth, fill, opacity }) {
      const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      r.setAttribute("x", box.x);
      r.setAttribute("y", box.y);
      r.setAttribute("width", Math.max(box.w, 1));
      r.setAttribute("height", Math.max(box.h, 1));
      r.setAttribute("fill", fill || "none");
      r.setAttribute("stroke", stroke);
      r.setAttribute("stroke-width", strokeWidth);
      r.setAttribute("opacity", opacity || "0.95");
      return r;
    }

    function appendRect(group, t) {
      const box = screenRect(t);
      if (!box) return;

      // A record carrying `color` is a grid outline; one carrying `fill` and
      // its own stroke_width is a heatmap or risk cell, whose hairline edge is
      // part of a colour scale and is left exactly as it was.
      const isGridOutline = !t.stroke_width && !!t.color;
      const strokeWidth = t.stroke_width || gridStrokeWidth(box.w);

      if (isGridOutline) {
        // Under the colour, wider by a hair, so the outline holds its edge on
        // pale tissue as well as on dark. Drawn as a separate rect rather than
        // an SVG filter: filters are re-rasterised per frame, and this runs on
        // every pan and zoom over thousands of rects.
        group.appendChild(rectElement(box, {
          stroke: GRID_HALO_COLOR,
          strokeWidth: strokeWidth + GRID_HALO_EXTRA,
          opacity: "1.0",
        }));
      }

      group.appendChild(rectElement(box, {
        stroke: t.stroke || t.color || "yellow",
        strokeWidth,
        fill: t.fill,
        opacity: t.opacity,
      }));
    }

    function sizeSvg() {
      const container = viewer.container.getBoundingClientRect();
      const svg = svgRef.current;
      if (svg) svg.setAttribute("viewBox", `0 0 ${container.width} ${container.height}`);
    }

    function drawTileOverlay() {
      if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) return;
      const group = gridGroupRef.current;
      if (!group) return;

      sizeSvg();
      group.innerHTML = "";
      for (const t of overlayTilesRef.current || []) {
        appendRect(group, t);
      }
      // The highlight is drawn into its own group, but it still has to follow
      // the viewport — so a pan or zoom redraws both.
      drawHighlight();
    }

    // Hover and selection only. Kept apart from the grid above because this
    // runs on every mouse move, and rebuilding up to 6,000 grid rects to move
    // one highlight would make pointing at a tile cost more than panning.
    function drawHighlight() {
      if (!viewer || !viewer.viewport || !viewer.world || viewer.world.getItemCount() === 0) return;
      const group = highlightGroupRef.current;
      if (!group) return;

      sizeSvg();
      group.innerHTML = "";

      const pitch = Number(pitchRef.current);
      const hoveredTile = hoveredRef.current;
      if (hoveredTile && Number.isFinite(pitch) && pitch > 0) {
        appendRect(group, {
          x: Number(hoveredTile.x_native),
          y: Number(hoveredTile.y_native),
          w: pitch,
          h: pitch,
          fill: "rgba(255,255,255,0.18)",
          stroke: "#ffffff",
          stroke_width: 2,
          opacity: "1.0",
        });
      }

      const sel = selectedTileRectRef.current;
      if (sel) {
        const pad = Number(sel.w || 0) * 0.06;
        appendRect(group, {
          x: sel.x - pad,
          y: sel.y - pad,
          w: sel.w + 2 * pad,
          h: sel.h + 2 * pad,
          fill: "none",
          stroke: "yellow",
          stroke_width: 6,
          opacity: "1.0",
        });
        appendRect(group, { x: sel.x, y: sel.y, w: sel.w, h: sel.h, fill: "none", stroke: "lime", stroke_width: 3, opacity: "1.0" });
      }
    }

    drawGridRef.current = drawTileOverlay;
    drawHighlightRef.current = drawHighlight;

    // The tile under a point in the viewer's own pixel space, or null where
    // the slide has no tile (background, or tissue below the threshold).
    function tileAtPixel(pixelX, pixelY) {
      if (!viewer.world || viewer.world.getItemCount() === 0) return null;
      const viewportPoint = viewer.viewport.pointFromPixel(new OpenSeadragon.Point(pixelX, pixelY), true);
      const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);
      return tileAtImagePoint(lookupRef.current, imagePoint.x, imagePoint.y, pitchRef.current);
    }

    function handleMove(event) {
      const bounds = viewer.container.getBoundingClientRect();
      const tile = tileAtPixel(event.clientX - bounds.left, event.clientY - bounds.top);
      if (tile === hoveredRef.current) return; // same cell — nothing to redraw
      hoveredRef.current = tile;
      // React state carries the readout; the highlight itself is drawn
      // straight into the SVG so it tracks the pointer without waiting for a
      // render.
      setHovered(tile);
      drawHighlight();
    }

    function handleLeave() {
      if (hoveredRef.current === null) return;
      hoveredRef.current = null;
      setHovered(null);
      drawHighlight();
    }

    viewer.container.addEventListener("mousemove", handleMove);
    viewer.container.addEventListener("mouseleave", handleLeave);

    // canvas-click rather than a DOM click: OpenSeadragon sets event.quick
    // false for a press that turned into a drag, which is the difference
    // between selecting a tile and finishing a pan on top of one.
    viewer.addHandler("canvas-click", (event) => {
      if (!event.quick || !onSelectTileRef.current) return;
      const tile = tileAtPixel(event.position.x, event.position.y);
      if (!tile) return;
      onSelectTileRef.current({
        slide_tile: String(tile.slide_tile || ""),
        x_native: Number(tile.x_native),
        y_native: Number(tile.y_native),
        tile,
      });
    });

    viewer.addHandler("open", drawTileOverlay);
    viewer.addHandler("animation", drawTileOverlay);
    viewer.addHandler("animation-finish", drawTileOverlay);
    viewer.addHandler("resize", drawTileOverlay);
    viewer.addHandler("zoom", drawTileOverlay);
    viewer.addHandler("pan", drawTileOverlay);

    viewer.addHandler("open-failed", (event) => {
      console.error("OpenSeadragon open failed", event);
      setFailed(true);
    });
    viewer.addHandler("tile-load-failed", (event) => {
      console.error("OpenSeadragon tile load failed", event);
    });

    return () => {
      drawGridRef.current = null;
      drawHighlightRef.current = null;
      hoveredRef.current = null;
      viewer.container.removeEventListener("mousemove", handleMove);
      viewer.container.removeEventListener("mouseleave", handleLeave);
      viewer.destroy();
      if (viewerRef.current === viewer) viewerRef.current = null;
    };
    // Only (re)create the viewer when the DZI source changes — overlay data
    // changes are handled by the effect below via the ref + a direct redraw.
  }, [dziUrl]);

  useEffect(() => {
    if (drawGridRef.current) drawGridRef.current();
  }, [overlayTiles]);

  useEffect(() => {
    if (drawHighlightRef.current) drawHighlightRef.current();
  }, [selectedTileRect]);

  const hoveredHpc = hovered && hovered.hpc_id !== null && hovered.hpc_id !== undefined ? hovered.hpc_id : null;

  return (
    <div>
      <div className="viewer-caption">OpenSeadragon DZI source: {dziUrl}</div>
      <div className="viewer-osd-wrap" style={{ height }}>
        <div ref={containerRef} className="viewer-osd-canvas" style={{ height }} />
        <svg ref={svgRef} className="viewer-osd-overlay-svg">
          <g ref={gridGroupRef} />
          <g ref={highlightGroupRef} />
        </svg>
        {hovered && (
          <div className="viewer-osd-hud">
            <span className="viewer-osd-hud-tile">{hovered.slide_tile}</span>
            <span className="viewer-osd-hud-hpc">{hoveredHpc === null ? "no HPC label" : `HPC ${hoveredHpc}`}</span>
          </div>
        )}
        {failed && (
          <div className="viewer-osd-failed">
            OpenSeadragon failed to open the DZI source. Try opening this directly:{" "}
            <a href={dziUrl} target="_blank" rel="noreferrer">
              {dziUrl}
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
