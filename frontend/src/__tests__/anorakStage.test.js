// The React side of backend/tests/test_anorak_server.py.
//
// Stage 7's form used to be reachable while the head job was alive: the form
// sent overwrite on every click on the grounds that "the in-flight check is the
// server's", and the server skipped that check whenever overwrite was set. The
// states that let the form show — Slurm unreachable, CONFIGURING — are asserted
// here against the real stepper and the real component, rendered to a string.
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "../api.js";
import AnorakStage from "../components/pipeline/stages/AnorakStage.jsx";
import { pipelineSteps, SLURM_IN_FLIGHT } from "../components/pipeline/utils.js";

function anorakStep(status) {
  return pipelineSteps(status).find((s) => s.key === "anorak");
}

function render(status) {
  return renderToStaticMarkup(
    createElement(AnorakStage, { status, submissionId: "sub1", onChanged: () => {} })
  );
}

describe("ANORAK stage state", () => {
  it("counts CONFIGURING as in flight, as the server does", () => {
    expect(SLURM_IN_FLIGHT.has("CONFIGURING")).toBe(true);
    const status = { anorak_job_id: "1111", anorak_slurm_state: "CONFIGURING" };
    expect(anorakStep(status).state).toBe("running");
    expect(render(status)).not.toContain("Retry ANORAK");
  });

  it("does not call an unknown state 'did not finish', or offer a retry over it", () => {
    const status = {
      anorak_job_id: "1111",
      anorak_slurm_state: null,
      anorak_state_unknown: true,
      anorak_submit_blocked: "Couldn't reach Slurm to confirm ANORAK job 1111 has stopped",
    };
    expect(anorakStep(status).summary).toContain("unknown");
    const html = render(status);
    expect(html).toContain("reach Slurm");
    expect(html).not.toContain("Retry ANORAK");
  });

  it("takes the server's in-flight verdict over its own copy of the states", () => {
    const status = { anorak_job_id: "1111", anorak_slurm_state: "SOME_NEW_STATE", anorak_in_flight: true };
    expect(anorakStep(status).state).toBe("running");
    expect(render(status)).not.toContain("Retry ANORAK");
  });

  it("still offers a retry once the head job has stopped", () => {
    const status = { anorak_job_id: "1111", anorak_slurm_state: "FAILED", anorak_state_unknown: false };
    expect(render(status)).toContain("Retry ANORAK");
  });

  it("shows a recorded submission error", () => {
    expect(render({ anorak_error: "sbatch failed (exit 1): invalid partition" })).toContain(
      "invalid partition"
    );
  });
});

describe("ANORAK stage: what a stopped run and a new one say", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("shows why a run stopped for good, from the supervisor's marker", () => {
    // A bare FAILED was all the stage said; the reason sat in nf_supervise.stop.
    const html = render({
      anorak_job_id: "1111,1112",
      anorak_slurm_state: "FAILED",
      anorak_state_unknown: false,
      anorak_stop_reason: "nextflow exited 1 on its own: a pipeline failure no restart can fix",
    });
    expect(html).toContain("stop marker");
    expect(html).toContain("a pipeline failure no restart can fix");
  });

  it("says a sample is required on every row, not optional", () => {
    // Blank samples used to be pooled into one tumour; the list is now refused.
    const html = render({});
    expect(html).not.toContain("if you want grades aggregated per tumour");
    expect(html).toContain("filled in on every row");
    expect(html).toContain("Head jobs");
  });

  it("submits a standby head job by default", async () => {
    // Without one, a head job that reaches its walltime ends the run.
    const calls = [];
    vi.stubGlobal("fetch", async (url, init) => {
      calls.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ selection: {} }), { status: 200 });
    });
    await api.startAnorak("sub1", { slidesCsv: "/x.csv" });
    expect(calls[0].chain).toBe(2);
  });
});

