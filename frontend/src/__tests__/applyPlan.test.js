// Ports of app_v28.py's _plan_needs_slide and the slide-resolution half of
// ui_actions_v25.apply_plan_to_session — these decide "ask which slide" vs.
// "just open it", so a wrong answer either nags for a slide the plan already
// named or silently guesses one it didn't.
import { describe, expect, it } from "vitest";

import { planNeedsSlide, resolveActiveSlide, shouldOpenViewer } from "../components/chat/applyPlan.js";

function plan(overrides = {}) {
  return {
    intent: "general_query",
    entities: { tile: [], slide: [], hpc: [], sample: [] },
    flags: { malignant: null, non_malignant: null, survival: null },
    ui_actions: {
      open_slide_viewer: false,
      show_tile_preview: false,
      explore_hpc_slides: false,
      highlight_mode: null,
      selected_hpc: null,
    },
    ...overrides,
  };
}

describe("planNeedsSlide", () => {
  it("is false for a plain greeting with no slide-scoped action", () => {
    expect(planNeedsSlide(plan())).toBe(false);
  });

  it("is true when the plan wants to open the viewer", () => {
    expect(planNeedsSlide(plan({ ui_actions: { ...plan().ui_actions, open_slide_viewer: true } }))).toBe(true);
  });

  it("is true when an HPC is mentioned", () => {
    expect(planNeedsSlide(plan({ entities: { ...plan().entities, hpc: ["40"] } }))).toBe(true);
  });

  it("is true when selected_hpc is 0 (falsy but a real selection)", () => {
    // The Python original checks `is not None`, not truthiness — HPC 0 would
    // otherwise be indistinguishable from "no HPC selected".
    const p = plan({ ui_actions: { ...plan().ui_actions, selected_hpc: 0 } });
    expect(planNeedsSlide(p)).toBe(true);
  });

  it("is true for a malignant/non-malignant flag with no named entity", () => {
    expect(planNeedsSlide(plan({ flags: { ...plan().flags, malignant: true } }))).toBe(true);
  });
});

describe("resolveActiveSlide", () => {
  it("picks the first named slide, upper-cased", () => {
    const p = plan({ entities: { ...plan().entities, slide: ["tcga-55-7574-01z-00-dx1"] } });
    expect(resolveActiveSlide(p, "")).toBe("TCGA-55-7574-01Z-00-DX1");
  });

  it("keeps the current slide when the plan names none", () => {
    expect(resolveActiveSlide(plan(), "TCGA-EXISTING")).toBe("TCGA-EXISTING");
  });

  it("returns empty when neither the plan nor the current selection has one", () => {
    expect(resolveActiveSlide(plan(), "")).toBe("");
  });
});

describe("shouldOpenViewer", () => {
  it("is true when the plan explicitly asks to open the viewer", () => {
    expect(shouldOpenViewer(plan({ ui_actions: { ...plan().ui_actions, open_slide_viewer: true } }))).toBe(true);
  });

  it("is true for a non-chit-chat intent naming a slide", () => {
    const p = plan({ intent: "slide_query", entities: { ...plan().entities, slide: ["TCGA-X"] } });
    expect(shouldOpenViewer(p)).toBe(true);
  });

  it("is false for a greeting even if it somehow named a slide", () => {
    const p = plan({ intent: "greeting", entities: { ...plan().entities, slide: ["TCGA-X"] } });
    expect(shouldOpenViewer(p)).toBe(false);
  });

  it("is false for a plain general_query naming nothing", () => {
    expect(shouldOpenViewer(plan())).toBe(false);
  });
});
