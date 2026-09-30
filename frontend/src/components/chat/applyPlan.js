// Pure ports of the plan-interpretation logic app_v28.py runs after
// build_query_plan_v25() returns — the part that decides what a plan implies
// about which slide is active and whether the viewer should open.
//
// Deliberately NOT ported here: the deeper half of
// app/ui_actions_v25.py's apply_plan_to_session — setting highlight_mode,
// selected_hpc, heat_hpc, the malignant/inflammation/necrosis filters, and
// resolving an adjacency-pair intent into tile sets. That half reaches into
// the WSI viewer's own internal state (frontend/src/components/viewer/index.jsx),
// which is currently private to that component — wiring chat into it means
// lifting that state up (or introducing a shared store) first, a separate
// piece of work from "make chat work at all". See ChatPanel.jsx's own note.
// What IS ported below (which slide is active, whether to open the viewer)
// covers the single most common chat behaviour: "tell me about slide X" /
// "open TCGA-..." actually opens that slide.

// Port of app_v28.py's _plan_needs_slide.
export function planNeedsSlide(plan) {
  const ui = plan.ui_actions || {};
  const flags = plan.flags || {};
  const ents = plan.entities || {};
  return Boolean(
    ui.open_slide_viewer ||
      ui.highlight_mode ||
      ui.selected_hpc !== null && ui.selected_hpc !== undefined ||
      flags.malignant ||
      flags.non_malignant ||
      (ents.hpc && ents.hpc.length) ||
      (ents.tile && ents.tile.length)
  );
}

// Port of the slide-resolution half of ui_actions_v25.apply_plan_to_session:
// a slide named in the plan's entities becomes the active one; otherwise the
// currently active slide (if any) is left alone.
export function resolveActiveSlide(plan, currentSlideId) {
  const ents = plan.entities || {};
  if (ents.slide && ents.slide.length) {
    return String(ents.slide[0]).trim().toUpperCase();
  }
  return currentSlideId || "";
}

// Port of the "should the viewer open" half of the same function.
export function shouldOpenViewer(plan) {
  const ui = plan.ui_actions || {};
  const ents = plan.entities || {};
  if (ui.open_slide_viewer) return true;
  if (!["greeting", "help", "general_query"].includes(plan.intent)) {
    if ((ents.slide && ents.slide.length) || (ents.hpc && ents.hpc.length) || (ents.tile && ents.tile.length)) {
      return true;
    }
  }
  return false;
}
