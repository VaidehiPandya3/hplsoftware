// Shared formatting/classification helpers ported from app/app_v28.py.
// Keeping these here (rather than inline in each component) is what keeps
// TilingStage/PackagingStage/etc. from drifting on what an icon or a label
// means — see _STEP_ICON, _pipeline_steps, _error_detail etc. in app_v28.py.
import { api } from "../../api";

// -- Error formatting --------------------------------------------------
// Port of _error_detail(e): splits an HTTPError into (structured detail,
// message). Several endpoints raise a dict body (e.g. packaging's
// "tiling_incomplete") that the UI needs to branch on rather than just print.
export function errorDetail(e) {
  if (!(e instanceof api.ApiError)) {
    return { detail: null, message: String((e && e.message) || e) };
  }
  const body = e.body;
  const detail = body && typeof body === "object" ? body.detail : undefined;
  if (detail && typeof detail === "object") {
    return { detail, message: detail.message || JSON.stringify(detail) };
  }
  if (detail) {
    return { detail: null, message: String(detail) };
  }
  return { detail: null, message: e.message };
}

// Port of _http_detail(error): FastAPI's `detail` field itself, not the
// {"detail": ...} envelope around it, so a long message doesn't render with
// escaped newlines.
export function httpDetail(e) {
  if (!(e instanceof api.ApiError)) {
    return String((e && e.message) || e);
  }
  const body = e.body;
  if (body && typeof body === "object") {
    const detail = "detail" in body ? body.detail : body;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return e.message;
}

// -- Small formatters ----------------------------------------------------
export function humanBytes(n) {
  let step = Number(n || 0);
  const units = ["B", "KB", "MB", "GB", "TB"];
  for (let i = 0; i < units.length; i++) {
    const unit = units[i];
    if (step < 1024 || unit === "TB") {
      return unit === "B" ? `${step.toFixed(0)} B` : `${step.toFixed(1)} ${unit}`;
    }
    step /= 1024;
  }
  return `${step.toFixed(1)} TB`;
}

export function fmtInt(n) {
  return Number(n).toLocaleString();
}

function basename(p) {
  return String(p || "").split(/[\\/]/).pop();
}

// -- Job identity / labels ------------------------------------------------
// Port of _job_label(job).
export function jobLabel(job) {
  const jobIdField = job.job_id || "";
  const list = jobIdField.split(",").filter(Boolean);
  if (list.length > 1) return `${list.length} batches`;
  if (list.length) return list[0];
  return `submission ${(job.submission_id || "").slice(0, 8)}`;
}

// A run's state is written out as a word wherever it is shown; there is no
// icon for it (these were coloured-circle emoji, which the UI no longer uses).
export const RUN_STATE_ICONS = {};

export const STAGE_LABELS = {
  tiling: "Tiling",
  packaging: "Packaging",
  packaging_test: "Packaging (test)",
  extraction: "Feature extraction",
  extraction_test: "Feature extraction (test)",
  assignment: "Cluster assignment",
  assignment_test: "Cluster assignment (test)",
};

// Step state -> icon. The point (per app_v28.py) is that all 6 stages are
// visible at once with their state, not just whichever one is "current".
export const STEP_ICON = {
  done: "[Done]",
  running: "[Running]",
  action: "[Ready]", // ready for you to start
  attention: "[Needs attention]", // finished or stalled, needs a decision
  failed: "[Failed]",
  blocked: "[Waiting]", // can't start yet, an earlier step must finish
};

export const DELIVERABLE_ICON = {
  ready: "[Ready]",
  running: "[Packaging]",
  interrupted: "[Interrupted]",
};

// Must match IN_FLIGHT_SLURM_STATES in tile_server_v2_.py. It lacked
// CONFIGURING (nodes allocated, prologue still running — squeue reports it
// first now), so a job Slurm was starting read here as "did not finish" and got
// a Retry button. Stages whose submit guard matters (ANORAK) also get the
// server's own verdict in /status rather than trusting this copy.
export const SLURM_IN_FLIGHT = new Set([
  "PENDING",
  "RUNNING",
  "REQUEUED",
  "RESIZING",
  "SUSPENDED",
  "COMPLETING",
  "CONFIGURING",
]);

// Port of _describe_job_params(stage, params): the one line that tells two
// attempts at the same stage apart in the job-history list.
export function describeJobParams(stage, params) {
  if (!params) return "";
  const bits = [];
  if (stage === "tiling") {
    if (params.slides != null) bits.push(`${fmtInt(params.slides)} slides`);
    if (params.batches) bits.push(`${params.batches} batches`);
    if (params.failed_batches) bits.push(`${params.failed_batches} batch(es) failed to submit`);
  } else if (stage === "packaging" || stage === "packaging_test") {
    if (params.sample_size) bits.push(`${params.sample_size} slides`);
    if (params.scope) bits.push(params.scope === "tiled" ? "from everything tiled" : "from this run");
    if (params.pool_size) bits.push(`pool ${params.pool_size}`);
    if (params.random_seed != null) bits.push(`seed ${params.random_seed}`);
    if (params.slide_names && params.slide_names.length) bits.push(`${params.slide_names.length} named slides`);
    if (params.allow_incomplete) bits.push("allow-incomplete");
  } else if (stage === "extraction" || stage === "extraction_test") {
    if (params.checkpoint) bits.push(basename(params.checkpoint));
    if (params.h5_path) bits.push(`on ${basename(params.h5_path)}`);
  }
  return bits.join(" · ");
}

// -- Dataset-level helpers -------------------------------------------------
// Port of _dataset_key: both halves are needed because two runs can tile the
// same raw directory into different output folders, and two directories can
// share a folder name.
export function datasetKey(dataset) {
  return `${dataset.raw_dir || ""}::${dataset.dataset_name || ""}`;
}

export function datasetOverallState(dataset) {
  const states = (dataset.steps || []).map((s) => s.state);
  for (const state of ["failed", "attention", "action", "running"]) {
    if (states.includes(state)) return state;
  }
  return states.length && states.every((s) => s === "done") ? "done" : "blocked";
}

export function datasetOptionLabel(dataset) {
  const icon = STEP_ICON[datasetOverallState(dataset)] || "•";
  const bits = [`${icon} ${dataset.dataset_name || "?"}`];
  const total = dataset.total_slides;
  if (total) bits.push(`${fmtInt(total)} slides`);
  const firstUnfinished = (dataset.steps || []).find((s) => s.state !== "done");
  if (firstUnfinished) {
    const title = String(firstUnfinished.title || "").replace(/^\d+\.\s*/, "").toLowerCase();
    bits.push(`${title}: ${firstUnfinished.summary}`);
  } else {
    bits.push("complete");
  }
  return bits.join(" · ");
}

// An ANORAK run on its own (POST /anorak-runs).
export function isAnorakRun(run) {
  return run.status === "anorak_only" || run.run_kind === "anorak";
}

export function datasetRunLabel(run) {
  const state = String(run.slurm_state || run.status || "unknown").toLowerCase();
  const bits = [(run.submitted_at || "").slice(0, 16).replace("T", " ")];
  bits.push(isAnorakRun(run) ? "ANORAK" : String(run.job_id || "").startsWith("nf:") ? "HPL" : "HPL (before the pipeline)");
  const total = run.total_slides;
  if (total != null) bits.push(`${fmtInt(total)} slides${run.is_subset ? " (subset)" : ""}`);
  if (run.resumed_from_submission_id) bits.push("resume");
  bits.push(state);
  return bits.join(" · ");
}

// Port of _datasets_under_path: exact match only, deliberately not a prefix
// match (a parent directory is not "the same dataset" as everything under it).
export function datasetsUnderPath(datasets, path) {
  const wanted = String(path || "").trim().replace(/\/+$/, "");
  return (datasets || []).filter((d) => String(d.raw_dir || "").replace(/\/+$/, "") === wanted);
}

// -- Run-level pipeline (per submission) ----------------------------------
// Port of _full_packaging_scope_caption.
export function fullPackagingScopeCaption(status) {
  const total = status.total_slides;
  if (total == null) return "Package every tiled slide in this run into one .h5.";
  const scope = status.is_subset ? "in this run's subset manifest" : "in this run";
  return `Package all ${fmtInt(total)} slides ${scope} into one .h5.`;
}

// Port of _test_packaging_note: says whether a "Test on a subset" packaging
// job exists even when no full-dataset job does, so a run that only ever
// tried the subset option doesn't read as "packaging never touched".
export function testPackagingNote(status) {
  if (!status.test_h5_job_id) return "";
  if (status.test_h5_ready) return " · test subset: ready";
  const state = status.test_h5_slurm_state;
  if (SLURM_IN_FLIGHT.has(state)) return ` · test subset: running (${state})`;
  return ` · test subset: ${state || "interrupted"}`;
}

// Port of _dataset_job_display_stage: collapses the raw status into one
// short label used to detect a genuine state change (for the toast), without
// caring about every field.
export function displayStage(status) {
  const stage = status.status;
  if (["queued", "discovering", "error", "cancelled"].includes(stage)) return stage;
  if (stage === "submitted") {
    if (status.extraction_ready) return "features ready";
    if (status.extraction_job_id) return "extracting features";
    if (status.h5_ready) return "h5 ready";
    if (status.h5_job_id) return "packaging";
    if (status.tiling_complete) return "tiling done";
    return "tiling";
  }
  return stage || "unknown";
}

// Port of _pipeline_steps(status): classifies all 6 stages at once from a
// single /status response so the whole pipeline can be shown, not just
// whichever stage happens to be "current".
// Stages 1-4 of a pipeline run (POST /pipeline-runs) are one Nextflow run.
// Port of _pipeline_stage_states in app_v28.py: "done" still means the
// server's own validator accepted the stage's output; no stage has a button
// of its own — the pipeline starts each once the one before has verified.
export const PIPELINE_STAGES = ["tiling", "packaging", "extraction", "assignment"];
const PIPELINE_STAGE_NAME = {
  tiling: "tiling", packaging: "packaging", extraction: "feature extraction", assignment: "classification",
};
const PIPELINE_STAGE_READY = {
  packaging: "h5_ready", extraction: "extraction_ready", assignment: "assignment_ready",
};

export function pipelineStageVerified(status, stage) {
  const info = ((status.pipeline || {}).stages || {})[stage] || {};
  if (stage === "tiling") return info.state === "COMPLETED" && Boolean(status.tiling_complete);
  return Boolean(status[PIPELINE_STAGE_READY[stage]]);
}

export function pipelineStageStates(status, computed) {
  const stages = (status.pipeline || {}).stages || {};
  const out = [];
  let previousDone = true;
  PIPELINE_STAGES.forEach((stage, i) => {
    const slurm = (stages[stage] || {}).state;
    let result;
    if (pipelineStageVerified(status, stage)) {
      result = ["done", computed[i][1]];
    } else if (slurm === "COMPLETED") {
      // The task said done, the server's validator disagrees — never hidden.
      result = ["attention", "pipeline marked it done, but the output fails validation"];
    } else if (slurm === "RUNNING") {
      result = ["running", "running in the pipeline"];
    } else if (SLURM_IN_FLIGHT.has(slurm)) {
      result = ["blocked", previousDone ? "queued in the pipeline" : `waits for ${PIPELINE_STAGE_NAME[PIPELINE_STAGES[i - 1]]}`];
    } else if (slurm == null) {
      result = ["attention", "can't reach Slurm — state unknown"];
    } else if (previousDone) {
      result = ["failed", `pipeline stopped here (${slurm})`];
    } else {
      result = ["blocked", "not reached"];
    }
    out.push(result);
    previousDone = result[0] === "done";
  });
  return out;
}

export function pipelineSteps(status) {
  const stage = status.status;
  const total = status.total_slides;
  const succeeded = status.succeeded;

  // --- 1. Tiling ---------------------------------------------------------
  let tiling;
  if (stage === "queued" || stage === "discovering") {
    tiling = ["running", "discovering slides"];
  } else if (stage === "error") {
    tiling = ["failed", (status.error || "failed").slice(0, 70)];
  } else if (status.tiling_complete) {
    const notAttempted = status.not_yet_attempted || 0;
    if (notAttempted) {
      tiling = ["attention", `${succeeded}/${total} slides · ${notAttempted} never ran`];
    } else if (succeeded != null) {
      tiling = ["done", `${succeeded}/${total} slides have tiles`];
    } else {
      tiling = ["done", "complete"];
    }
  } else if (status.slurm_unreachable) {
    tiling = ["attention", "can't reach Slurm — state unknown"];
  } else {
    const counts = status.slurm_state_counts || {};
    const inFlight = Object.entries(counts).reduce((sum, [s, n]) => (SLURM_IN_FLIGHT.has(s) ? sum + n : sum), 0);
    tiling = ["running", inFlight ? `${fmtInt(inFlight)} task(s) in flight` : "in progress"];
  }

  // --- 2. Packaging --------------------------------------------------------
  const testNote = testPackagingNote(status);
  const h5State = status.h5_slurm_state;
  let packaging;
  if (status.h5_ready) {
    packaging = [
      "done",
      status.h5_legacy_tile_names ? ".h5 ready (tile names need migrating before KB load)" : ".h5 ready",
    ];
  } else if (status.h5_job_id) {
    if (SLURM_IN_FLIGHT.has(h5State)) {
      packaging = ["running", `running (${h5State})`];
    } else if (status.h5_packaging_active) {
      const written = humanBytes(status.h5_partial_bytes || 0);
      packaging = ["running", `writing (${written} so far)`];
    } else if (status.h5_invalid_reason) {
      packaging = ["attention", "finished but .h5 unusable"];
    } else {
      packaging = ["attention", `interrupted (${h5State || "no Slurm record"})`];
    }
  } else if (status.tiling_complete) {
    packaging = ["action", `ready to start${testNote}`];
  } else if (status.slurm_unreachable) {
    packaging = ["attention", `can't reach Slurm — tiling state unknown${testNote}`];
  } else {
    packaging = ["blocked", `waiting on tiling${testNote}`];
  }

  // --- 3. Feature extraction -----------------------------------------------
  const extState = status.extraction_slurm_state;
  const expectedRows = status.extraction_expected_rows;
  let extraction;
  if (status.extraction_ready) {
    extraction = ["done", expectedRows ? `features ready (${fmtInt(expectedRows)} tiles)` : "features ready"];
  } else if (status.extraction_job_id) {
    if (SLURM_IN_FLIGHT.has(extState)) {
      extraction = ["running", `running (${extState})`];
    } else if (status.extraction_invalid_reason) {
      const reason = status.extraction_invalid_reason || "";
      const shortfall = reason.match(/has ([\d,]+) embeddings but the input \.h5 has ([\d,]+) tiles/);
      if (shortfall) {
        const got = Number(shortfall[1].replace(/,/g, ""));
        const want = Number(shortfall[2].replace(/,/g, ""));
        extraction = [
          "attention",
          `incomplete — ${fmtInt(got)} of ${fmtInt(want)} tiles encoded (${((got / want) * 100).toFixed(0)}%)`,
        ];
      } else {
        extraction = ["attention", "finished but the output is incomplete"];
      }
    } else {
      extraction = ["attention", `did not finish (${extState || "no Slurm record"})`];
    }
  } else if (status.h5_ready) {
    extraction = ["action", "ready to start"];
  } else {
    extraction = ["blocked", "waiting on packaging"];
  }

  // --- 4. Cluster assignment ------------------------------------------------
  const asgState = status.assignment_slurm_state;
  let assignment;
  if (status.assignment_ready) {
    assignment = ["done", "clusters assigned"];
  } else if (status.assignment_job_id) {
    if (SLURM_IN_FLIGHT.has(asgState)) {
      assignment = ["running", `running (${asgState})`];
    } else if (status.assignment_invalid_reason) {
      assignment = ["attention", "finished but the output is incomplete"];
    } else {
      assignment = ["attention", `did not finish (${asgState || "no Slurm record"})`];
    }
  } else if (status.extraction_ready) {
    assignment = ["action", "ready to start"];
  } else {
    assignment = ["blocked", "waiting on feature extraction"];
  }

  // --- 5. Registration --------------------------------------------------------
  // Gated on h5_ready, not on the assignment: registration reads tile identity
  // out of the packaged .h5 and the raw slides and needs no cluster labels, so
  // making it wait for Stage 4 would keep Stage 6 blocked behind a step that
  // could have finished hours earlier.
  //
  // A run that completed before this step existed reports no registration_done,
  // and shows as "action" — which is correct. Those runs did not register.
  let registration;
  if (status.registration_done) {
    const rows = status.registration_rows;
    let summary = "registered";
    if (rows && typeof rows === "object" && Object.keys(rows).length) {
      summary = Object.entries(rows)
        .map(([t, n]) => `${fmtInt(n)} ${t.replace(/_/g, " ")}`)
        .join(", ");
    }
    registration = ["done", summary.slice(0, 70)];
  } else if (SLURM_IN_FLIGHT.has(status.registration_slurm_state)) {
    // A Slurm-backed write in flight. Without this the stage reads "ready to
    // register" while a job is actively writing, which invites a second one —
    // and Stage 5 writes identity rows for a whole cohort, so a second one is
    // not a no-op.
    registration = ["running", `running (${status.registration_slurm_state})`];
  } else if (status.registration_job_id) {
    // A job id with no in-flight state and no done flag: it ended without
    // recording a commit. The step has to say so rather than offering to
    // register, because the job's own log is the only place the refusal is.
    registration = ["attention", "a job ended without committing"];
  } else if (status.registration_ready) {
    registration = ["action", "ready to register"];
  } else {
    registration = ["blocked", "waiting on packaging"];
  }

  // --- 6. Knowledge Bank load ------------------------------------------------
  // Gated on registration as well as on the assignment, and that is the whole
  // point of the step above: load_hpc_assignments only UPDATEs, so without
  // identity rows its match rate is 0% and it refuses. Before this, that
  // refusal was the first sign anything was wrong, and it named a match rate
  // rather than a missing step.
  let kbLoad;
  if (status.kb_load_done) {
    const rows = status.kb_load_rows;
    kbLoad = ["done", rows != null ? `${fmtInt(rows)} tiles in the KB` : "loaded"];
  } else if (SLURM_IN_FLIGHT.has(status.kb_load_slurm_state)) {
    kbLoad = ["running", `running (${status.kb_load_slurm_state})`];
  } else if (status.kb_load_job_id) {
    kbLoad = ["attention", "a job ended without committing"];
  } else if (!status.assignment_ready) {
    kbLoad = ["blocked", "waiting on cluster classification"];
  } else if (!status.registration_done) {
    kbLoad = ["blocked", "waiting on registration"];
  } else {
    kbLoad = ["action", "ready to load"];
  }

  // --- 7. ANORAK growth-pattern grading -------------------------------------
  // Not gated on anything Stages 1-6 produce: ANORAK does its own tiling at its
  // own resolution and reads the raw slides, so it shares no artifact with
  // them. What it needs from them is the *slide list* — the cohort's tumour
  // slides, which come from the cluster composition Stage 6 loads. So this is
  // "action" as soon as there is something to grade, and the summary names
  // which list it is about to use rather than assuming one.
  let anorak;
  if (status.anorak_ready) {
    const scope = status.anorak_scope;
    const slides = status.anorak_slides;
    let detail = slides ? `${fmtInt(slides)} slides` : "complete";
    if (scope === "subset") detail += ` (random subset, seed ${status.anorak_seed})`;
    anorak = ["done", `growth patterns graded · ${detail}`];
  } else if (status.anorak_in_flight || SLURM_IN_FLIGHT.has(status.anorak_slurm_state)) {
    // The head job being alive is all this says. It submits a job per slide
    // per stage itself, so its own state carries no progress.
    anorak = ["running", `pipeline running (${status.anorak_slurm_state})`];
  } else if (status.anorak_job_id && status.anorak_state_unknown) {
    // Not "did not finish": Slurm could not be asked, so the head job may well
    // be alive — and the server refuses a new submission until it can.
    anorak = ["attention", "state unknown — Slurm unreachable"];
  } else if (status.anorak_job_id) {
    if (status.anorak_invalid_reason) {
      anorak = ["attention", "finished without a usable grading table"];
    } else {
      anorak = ["attention", `did not finish (${status.anorak_slurm_state || "no Slurm record"})`];
    }
  } else if (status.kb_load_done) {
    anorak = ["action", "ready to run"];
  } else {
    // Deliberately not "blocked": a slide list chosen some other way is a
    // perfectly good input, and blocking would hide the form that takes one.
    anorak = ["action", "ready to run (needs a tumour-slide list)"];
  }

  if (status.pipeline) {
    [tiling, packaging, extraction, assignment] = pipelineStageStates(status, [
      tiling, packaging, extraction, assignment,
    ]);
  }

  return [
    { key: "tiling", title: "1. Tiling", state: tiling[0], summary: tiling[1] },
    { key: "packaging", title: "2. Packaging (.h5)", state: packaging[0], summary: packaging[1] },
    { key: "extraction", title: "3. Feature extraction", state: extraction[0], summary: extraction[1] },
    { key: "assignment", title: "4. Cluster classification", state: assignment[0], summary: assignment[1] },
    { key: "registration", title: "5. Register in the Knowledge Bank", state: registration[0], summary: registration[1] },
    { key: "kb_load", title: "6. Knowledge Bank load", state: kbLoad[0], summary: kbLoad[1] },
    { key: "anorak", title: "7. Growth patterns (ANORAK)", state: anorak[0], summary: anorak[1] },
  ];
}

// Port of _report_resume_result: what a resume actually queued. Shared by
// the tiling step's own resume button and the dataset rollup's next-action
// button, so the tiling-settings warning below can't be forgotten in one of
// the two call sites — it guards against exactly the failure resuming is
// supposed to prevent (resumed slides tiled with different settings than the
// ones already on disk).
export function resumeResultMessage(result) {
  if (!result.resumed) {
    return { queued: false, alert: { type: "info", text: result.message || "Nothing missing." } };
  }
  const corrupt = result.corrupt_metadata_count || 0;
  const extra = corrupt ? ` (${corrupt} had corrupt metadata)` : "";
  const text =
    `Queued ${result.missing_slide_count} slides${extra} of ` +
    `${result.total_in_original_manifest} as new submission ${result.submission_id}.`;
  const params = result.tiling_params || {};
  let note = null;
  if (result.tiling_params_source === "original_run") {
    note = { type: "info", text: "Tiled with this run's own recorded settings." };
  } else if (Object.keys(params).length) {
    note = {
      type: "warning",
      text:
        "This run has no recorded tiling settings, so the resumed slides used current " +
        "defaults — check these match what the original run used, or they'll differ from " +
        "the tiles already on disk.",
    };
  }
  return { queued: true, alert: { type: "success", text }, note, params };
}

// Port of _find_existing_job_for_path: most recent *actionable* submission
// for this exact raw_dir, skipping cancelled/errored ones (dead ends).
export async function findExistingJobForPath(path) {
  const normalized = String(path || "").trim().replace(/\/+$/, "");
  if (!normalized) return null;
  let jobs;
  try {
    jobs = await api.listDatasetJobs();
  } catch {
    return null;
  }
  for (const job of jobs) {
    if (String(job.raw_dir || "").replace(/\/+$/, "") !== normalized) continue;
    if (job.status === "cancelled" || job.status === "error") continue;
    return job;
  }
  return null;
}
