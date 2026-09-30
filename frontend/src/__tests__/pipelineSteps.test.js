// The JS twin of backend/tests/test_pipeline_steps.py.
//
// That Python test exists because a step key with no renderer is a KeyError on
// a screen nobody opens until a run reaches that stage — it fails quietly, late,
// and only for the person whose cohort got that far. The JS port had no
// equivalent, and drifted: the Slurm in-flight branches for Stages 5 and 6 were
// missing, so a run whose registration job was actively writing displayed
// "ready to register", which invites a second write of a whole cohort's
// identity rows.
//
// So these assert the two things that drift: that every step key has a
// renderer, and that a Slurm-backed write in flight is never reported as ready.
import { describe, expect, it } from "vitest";

import { pipelineSteps, SLURM_IN_FLIGHT } from "../components/pipeline/utils.js";
import { STAGE_RENDERERS } from "../components/pipeline/stages/index.js";

const KEYS = ["tiling", "packaging", "extraction", "assignment", "registration", "kb_load", "anorak"];

/** A status where everything up to and including Stage 4 is finished, so
 * Stages 5 and 6 are the ones under test rather than blocked behind others. */
function readyForKb(extra = {}) {
  return {
    tiling_total_slides: 10,
    tiling_tiled_slides: 10,
    h5_ready: true,
    extraction_ready: true,
    assignment_ready: true,
    registration_ready: true,
    registration_done: true,
    ...extra,
  };
}

function step(status, key) {
  return pipelineSteps(status).find((s) => s.key === key);
}

describe("pipelineSteps", () => {
  it("returns all seven stages, in order, with the expected keys", () => {
    const steps = pipelineSteps({});
    expect(steps.map((s) => s.key)).toEqual(KEYS);
  });

  it("gives every step a renderer", () => {
    // The failure this mirrors from the Python side: a key with no renderer
    // throws only once a run reaches that stage.
    for (const s of pipelineSteps({})) {
      expect(STAGE_RENDERERS[s.key], `no renderer for step "${s.key}"`).toBeTruthy();
    }
  });

  it("gives every step a state and a summary", () => {
    for (const s of pipelineSteps({})) {
      expect(typeof s.state).toBe("string");
      expect(s.state.length).toBeGreaterThan(0);
      expect(typeof s.summary).toBe("string");
    }
  });

  describe("a Slurm-backed KB write in flight", () => {
    // The whole reason this file exists. Parameterised over every in-flight
    // state, because the bug was one missing branch and a test naming only
    // RUNNING would have passed while PENDING still read as ready.
    for (const slurmState of SLURM_IN_FLIGHT) {
      it(`registration is "running", not "action", while ${slurmState}`, () => {
        const s = step(
          readyForKb({ registration_done: false, registration_job_id: "123", registration_slurm_state: slurmState }),
          "registration"
        );
        expect(s.state).toBe("running");
        expect(s.summary).toContain(slurmState);
      });

      it(`kb_load is "running", not "action", while ${slurmState}`, () => {
        const s = step(
          readyForKb({ kb_load_job_id: "456", kb_load_slurm_state: slurmState }),
          "kb_load"
        );
        expect(s.state).toBe("running");
        expect(s.summary).toContain(slurmState);
      });
    }
  });

  it("reports a registration job that ended without committing", () => {
    const s = step(
      readyForKb({ registration_done: false, registration_job_id: "123", registration_slurm_state: "FAILED" }),
      "registration"
    );
    expect(s.state).toBe("attention");
  });

  it("reports a kb_load job that ended without committing", () => {
    const s = step(readyForKb({ kb_load_job_id: "456", kb_load_slurm_state: "FAILED" }), "kb_load");
    expect(s.state).toBe("attention");
  });

  it("still offers registration when no job has ever been submitted", () => {
    // The guards above must not swallow the ordinary case.
    const s = step(readyForKb({ registration_done: false }), "registration");
    expect(s.state).toBe("action");
  });

  it("still offers the KB load when no job has ever been submitted", () => {
    expect(step(readyForKb(), "kb_load").state).toBe("action");
  });

  it("prefers done over an in-flight state", () => {
    // A job id left over from an earlier attempt must not un-complete a stage
    // that has recorded a commit.
    const s = step(
      readyForKb({ registration_job_id: "123", registration_slurm_state: "RUNNING" }),
      "registration"
    );
    expect(s.state).toBe("done");
  });

  it("keeps the KB load blocked behind registration", () => {
    const s = step(readyForKb({ registration_done: false }), "kb_load");
    expect(s.state).toBe("blocked");
    expect(s.summary).toContain("registration");
  });

  describe("anorak (Stage 7)", () => {
    // Not gated on this pipeline's own artifacts — it needs only a slide list,
    // which can come from Stage 6 or from somewhere else entirely. So the
    // default state is "action", never "blocked".
    it("is never blocked, even from an empty status", () => {
      expect(step({}, "anorak").state).toBe("action");
    });

    it("offers a plain 'ready to run' once the KB load has completed", () => {
      const s = step(readyForKb({ kb_load_done: true }), "anorak");
      expect(s.state).toBe("action");
      expect(s.summary).toBe("ready to run");
    });

    it("still names the missing slide list before the KB load has completed", () => {
      const s = step(readyForKb(), "anorak");
      expect(s.state).toBe("action");
      expect(s.summary).toContain("tumour-slide list");
    });

    for (const slurmState of SLURM_IN_FLIGHT) {
      it(`is "running", not "action", while ${slurmState}`, () => {
        // The head job's own state carries no progress — it submits a job per
        // slide per stage itself — which is why the summary names the
        // pipeline rather than a percentage it does not have.
        const s = step(readyForKb({ anorak_job_id: "789", anorak_slurm_state: slurmState }), "anorak");
        expect(s.state).toBe("running");
        expect(s.summary).toContain(slurmState);
      });
    }

    it("reports a job that ended without a usable grading table", () => {
      const s = step(
        readyForKb({ anorak_job_id: "789", anorak_slurm_state: "FAILED", anorak_invalid_reason: "empty CSV" }),
        "anorak"
      );
      expect(s.state).toBe("attention");
      expect(s.summary).toContain("without a usable grading table");
    });

    it("reports a job that ended with no Slurm record at all", () => {
      const s = step(readyForKb({ anorak_job_id: "789", anorak_slurm_state: null }), "anorak");
      expect(s.state).toBe("attention");
      expect(s.summary).toContain("no Slurm record");
    });

    it("prefers done over an in-flight state", () => {
      const s = step(
        readyForKb({ anorak_ready: true, anorak_slides: 42, anorak_job_id: "789", anorak_slurm_state: "RUNNING" }),
        "anorak"
      );
      expect(s.state).toBe("done");
      expect(s.summary).toContain("42");
    });

    it("names the seed for a finished subset run", () => {
      const s = step(
        readyForKb({ anorak_ready: true, anorak_scope: "subset", anorak_slides: 10, anorak_seed: 7 }),
        "anorak"
      );
      expect(s.summary).toContain("subset");
      expect(s.summary).toContain("7");
    });
  });

  // A single uploaded slide is a one-slide run whose Stages 1 and 2 ran inside
  // the server rather than on Slurm, and it reaches this same stepper through
  // the upload panel. The status it arrives with carries the sentinel job ids
  // the server records for an in-process stage (see LOCAL_JOB_ID_PREFIX in
  // tile_server_v2_.py) — the point of these two is that nothing here treats
  // them as a special case, so an upload's next action is the real one.
  describe("an uploaded slide's run", () => {
    const uploaded = {
      status: "completed",
      total_slides: 1,
      succeeded: 1,
      tiling_complete: true,
      h5_job_id: "local:packaging",
      h5_slurm_state: "COMPLETED",
      h5_ready: true,
    };

    it("shows tiling and packaging as already done", () => {
      expect(step(uploaded, "tiling").state).toBe("done");
      expect(step(uploaded, "packaging").state).toBe("done");
    });

    it("offers feature extraction as the next action", () => {
      expect(step(uploaded, "extraction").state).toBe("action");
    });

    it("still blocks the Knowledge Bank load until it has been registered", () => {
      // The gate that matters most for an upload: the slide is already in
      // wsi_registry so it opens in the viewer, which makes "registered" look
      // done when none of its tiles are in the KB at all.
      const s = step(uploaded, "kb_load");
      expect(s.state).toBe("blocked");
    });
  });
});

// --- a pipeline run (POST /pipeline-runs) ----------------------------------
// Port of test_the_stepper_never_calls_an_unvalidated_stage_done in
// backend/tests/test_hpl_nf_pipeline.py: the task saying COMPLETED is not
// enough — "done" still needs the server's validator.
describe("pipeline run stages", () => {
  const run = (states, ready = {}) => ({
    status: "submitted",
    pipeline: { stages: Object.fromEntries(Object.entries(states).map(([k, v]) => [k, { state: v }])) },
    ...ready,
  });

  it("never calls a stage done that fails validation", () => {
    const steps = pipelineSteps(
      run(
        { tiling: "COMPLETED", packaging: "COMPLETED", extraction: "COMPLETED", assignment: "COMPLETED" },
        { tiling_complete: true, succeeded: 2, total_slides: 2, h5_ready: false, extraction_ready: true, assignment_ready: true },
      ),
    );
    const byKey = Object.fromEntries(steps.map((s) => [s.key, s]));
    expect(byKey.tiling.state).toBe("done");
    expect(byKey.packaging.state).toBe("attention");
    expect(byKey.packaging.summary).toMatch(/validation/);
  });

  it("shows where a stopped run stopped, and nothing after it as started", () => {
    const steps = pipelineSteps(
      run(
        { tiling: "COMPLETED", packaging: "COMPLETED", extraction: "FAILED", assignment: "FAILED" },
        { tiling_complete: true, succeeded: 2, total_slides: 2, h5_ready: true },
      ),
    );
    expect(steps.slice(0, 4).map((s) => s.state)).toEqual(["done", "done", "failed", "blocked"]);
  });

  it("queues later stages behind a running one", () => {
    const steps = pipelineSteps(run({ tiling: "RUNNING", packaging: "PENDING", extraction: "PENDING", assignment: "PENDING" }));
    expect(steps[0].state).toBe("running");
    expect(steps[1]).toMatchObject({ state: "blocked", summary: "waits for tiling" });
  });
});
