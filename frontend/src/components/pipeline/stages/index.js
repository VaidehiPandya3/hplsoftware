// The step-key -> component mapping, as data.
//
// It used to be an object literal built inside RunProgress's render, closing
// over status/submissionId/refresh. That works, but nothing outside the
// component could see it — which is precisely why the JS port drifted from
// app_v28.py's _pipeline_steps without anything noticing: there was no way to
// assert that every key pipelineSteps() returns has something to render it.
//
// backend/tests/test_pipeline_steps.py makes exactly that assertion on the
// Python side, and CLAUDE.md explains why it has to: "a step key with no
// renderer is a KeyError on a screen nobody opens until a run reaches that
// stage." Same hazard here, so the mapping is a module-level constant and
// src/__tests__/pipelineSteps.test.js checks it.
//
// Every component receives the same props and ignores the ones it does not
// need, so this stays a plain lookup rather than six call signatures.
import AnorakStage from "./AnorakStage.jsx";
import AssignmentStage from "./AssignmentStage.jsx";
import ExtractionStage from "./ExtractionStage.jsx";
import KbLoadStage from "./KbLoadStage.jsx";
import PackagingStage from "./PackagingStage.jsx";
import RegistrationStage from "./RegistrationStage.jsx";
import TilingStage from "./TilingStage.jsx";

export const STAGE_RENDERERS = {
  tiling: TilingStage,
  packaging: PackagingStage,
  extraction: ExtractionStage,
  assignment: AssignmentStage,
  registration: RegistrationStage,
  kb_load: KbLoadStage,
  anorak: AnorakStage,
};
