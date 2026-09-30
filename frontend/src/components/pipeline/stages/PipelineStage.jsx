// Stages 1-4 of a pipeline run (POST /pipeline-runs) — port of
// _render_pipeline_overview, _render_pipeline_stage_step and
// _render_pipeline_resume in app_v28.py.
//
// Each stage still reports on its own and is still judged by the server's
// validator; what is gone is a button per stage. The pipeline starts each one
// once the one before has verified its output, and at the stage where a run
// stopped this shows why and offers Resume, the way AnorakStage does.
import { useState } from "react";
import { api } from "../../../api";
import { httpDetail, pipelineStageVerified, SLURM_IN_FLIGHT } from "../utils";
import { Alert, Button, Caption, CodeBlock, Expander, Field } from "../widgets";

export function PipelineOverview({ status }) {
  const pipeline = status.pipeline || {};
  const head = (pipeline.head_job_ids || []).join(", ") || "not submitted yet";
  const state = pipeline.head_state || "unknown";
  const selection = pipeline.selection || {};
  const settings = pipeline.settings || {};
  return (
    <div className="pipeline-overview">
      {pipeline.finished ? (
        <Alert type="success">
          Nextflow pipeline finished: Stages 1-4 are verified. Register and load into the Knowledge Bank below.
        </Alert>
      ) : SLURM_IN_FLIGHT.has(state) ? (
        <Alert type="info">
          Nextflow pipeline {state.toLowerCase()} — head job {head}. It submits a job per slide and per shard
          itself, so squeue shows many more.
        </Alert>
      ) : (
        <Alert type="warning">Nextflow pipeline stopped ({state}) — head job {head}.</Alert>
      )}
      {selection.scope === "subset" && (
        <Caption>
          Random subset: {selection.slides} of {selection.pool} slides, seed {selection.seed}. A test run — the
          full directory has not been processed.
        </Caption>
      )}
      {settings.checkpoint && (
        <Caption>
          Checkpoint {settings.checkpoint} · {settings.extraction_shards} GPU shard(s) on {settings.gpu_gres} ·
          reference {settings.reference} · vote {settings.vote} · {settings.assignment_shards} assignment
          shard(s) on {settings.device}
        </Caption>
      )}
      <Caption>Run directory (config, stage markers, nextflow.log, report): {pipeline.out_dir}</Caption>
    </div>
  );
}

function PipelineResume({ status, submissionId, onChanged }) {
  const pipeline = status.pipeline || {};
  const [chain, setChain] = useState(3);
  const [timeLimit, setTimeLimit] = useState("");
  const already = Boolean((pipeline.settings || {}).allow_incomplete);
  const [allowIncomplete, setAllowIncomplete] = useState(already);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  async function resume() {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.resumePipelineRun(submissionId, {
        chain: Number(chain),
        timeLimit: timeLimit.trim() || null,
        allowIncomplete: allowIncomplete && !already ? true : null,
      });
      setMessage({ type: "success", text: `Resumed — head job ${result.nf_job_id}.` });
      if (onChanged) onChanged();
    } catch (e) {
      setMessage({ type: "error", text: `Refused: ${httpDetail(e)}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {pipeline.stop_reason && (
        <>
          <Caption>Why it stopped (the supervisor&apos;s stop marker):</Caption>
          <CodeBlock>{pipeline.stop_reason}</CodeBlock>
        </>
      )}
      {pipeline.log_tail && (
        <Expander title="End of nextflow.log" defaultOpen>
          <CodeBlock>{pipeline.log_tail}</CodeBlock>
        </Expander>
      )}
      {pipeline.resumable && (
        <>
          <Caption>
            Resuming re-runs only what did not finish, with this run&apos;s own recorded settings; every stage is
            verified again before it counts.
          </Caption>
          <Field label="Head jobs">
            <input type="number" min={1} max={10} value={chain} onChange={(e) => setChain(e.target.value)} />
          </Field>
          <Field label="Head job walltime (optional)">
            <input type="text" value={timeLimit} onChange={(e) => setTimeLimit(e.target.value)} />
          </Field>
          <Field
            label="Package without slides that fail to tile"
            help="For a run that stopped because some slides cannot be tiled: they are left out of the .h5 and named on the tiling step. Everything already tiled is kept."
          >
            <input
              type="checkbox"
              checked={allowIncomplete}
              disabled={already}
              onChange={(e) => setAllowIncomplete(e.target.checked)}
            />
          </Field>
          <Button kind="primary" onClick={resume} disabled={busy}>
            {busy ? "Resuming…" : "Resume pipeline"}
          </Button>
        </>
      )}
      {message && <Alert type={message.type}>{message.text}</Alert>}
    </div>
  );
}

// An ANORAK run started on its own (Run ANORAK) — port of
// _render_anorak_run_step: what it graded, where its outputs are, and — if it
// stopped short — why, and Resume.
export function AnorakRunStage({ status, submissionId, onChanged }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  async function resume() {
    setBusy(true);
    setMessage(null);
    try {
      const result = await api.resumeAnorakRun(submissionId);
      setMessage({ type: "success", text: `Resumed — head job ${result.anorak_job_id}.` });
      if (onChanged) onChanged();
    } catch (e) {
      setMessage({ type: "error", text: `Refused: ${httpDetail(e)}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      {status.anorak_tumour_verified === false && (
        <Alert type="warning">
          Every slide in the directory was graded; tumour status was not checked, so non-tumour slides are in the
          grading table too.
        </Alert>
      )}
      {status.anorak_scope === "subset" ? (
        <Caption>
          Test run: {status.anorak_slides} slides sampled at random, seed {status.anorak_seed}.
        </Caption>
      ) : status.anorak_slides ? (
        <Caption>{Number(status.anorak_slides).toLocaleString()} slides.</Caption>
      ) : null}
      {status.anorak_error && <Alert type="error">{status.anorak_error}</Alert>}
      {status.anorak_ready ? (
        <>
          <Alert type="success">Growth-pattern grading complete:</Alert>
          <CodeBlock>{status.anorak_grades_csv}</CodeBlock>
        </>
      ) : status.anorak_in_flight ? (
        <Alert type="info">
          ANORAK running (head job {status.anorak_job_id}, {status.anorak_slurm_state}).
        </Alert>
      ) : status.anorak_job_id ? (
        <>
          <Alert type="warning">ANORAK stopped ({status.anorak_slurm_state || "no Slurm record"}).</Alert>
          {status.anorak_stop_reason && (
            <>
              <Caption>Why it stopped (the supervisor&apos;s stop marker):</Caption>
              <CodeBlock>{status.anorak_stop_reason}</CodeBlock>
            </>
          )}
          {status.anorak_invalid_reason && <Caption>Output not usable: {status.anorak_invalid_reason}</Caption>}
          {!status.anorak_submit_blocked && (
            <Button kind="primary" onClick={resume} disabled={busy}>
              {busy ? "Resuming…" : "Resume ANORAK"}
            </Button>
          )}
        </>
      ) : null}
      {message && <Alert type={message.type}>{message.text}</Alert>}
      {status.anorak_out_dir && (
        <>
          <Caption>Run directory (masks, proportions, nextflow.log, report):</Caption>
          <CodeBlock>{status.anorak_out_dir}</CodeBlock>
        </>
      )}
    </div>
  );
}

// Stages 1-4 of a run from before the pipeline: what it did, no buttons.
// Port of _render_legacy_stage_readonly.
export function LegacyStage({ stage, status }) {
  const [jobKey, pathKey, reasonKey] = {
    tiling: ["job_id", null, null],
    packaging: ["h5_job_id", "h5_output_path", "h5_invalid_reason"],
    extraction: ["extraction_job_id", "extraction_output_path", "extraction_invalid_reason"],
    assignment: ["assignment_job_id", "assignment_output_path", "assignment_invalid_reason"],
  }[stage];
  return (
    <div>
      {status[jobKey] && <Caption>Slurm job(s): {status[jobKey]}</Caption>}
      {stage === "tiling" && status.succeeded != null && (
        <Caption>
          {Number(status.succeeded).toLocaleString()} of {Number(status.total_slides || 0).toLocaleString()} slides
          have tiles.
        </Caption>
      )}
      {pathKey && status[pathKey] && (
        <>
          <Caption>Output:</Caption>
          <CodeBlock>{status[pathKey]}</CodeBlock>
        </>
      )}
      {reasonKey && status[reasonKey] && <Caption>Not usable: {status[reasonKey]}</Caption>}
      <Caption>
        Read-only: Stages 1-4 now run as one pipeline. Run the pipeline for this dataset to carry it on; it
        reuses whatever this run produced.
      </Caption>
    </div>
  );
}

const OUTPUT_KEY = {
  packaging: ["h5_output_path", "h5_invalid_reason", "tiles"],
  extraction: ["extraction_output_path", "extraction_invalid_reason", "embeddings"],
  assignment: ["assignment_output_path", "assignment_invalid_reason", "assignments"],
};

export default function PipelineStage({ stage, status, submissionId, state, onChanged }) {
  const pipeline = status.pipeline || {};
  const info = (pipeline.stages || {})[stage] || {};
  const done = info.done || {};

  let body;
  if (stage === "tiling") {
    const zeroTile = status.zero_tile_slides || done.zero_tile_slides || [];
    const excluded = done.excluded_slides || [];
    body = (
      <>
        {info.done && (
          <Caption>
            {Number(done.succeeded || 0).toLocaleString()} of{" "}
            {Number(done.slides || status.total_slides || 0).toLocaleString()} slides have tiles (every CSV
            checked against its summary).
          </Caption>
        )}
        {excluded.length > 0 && (
          <>
            <Alert type="warning">
              {done.excluded || excluded.length} slide(s) failed to tile and were left out of the .h5 (this run
              allows it). Their TILE task logs say why.
            </Alert>
            <CodeBlock>{excluded.join("\n")}</CodeBlock>
          </>
        )}
        {zeroTile.length > 0 && (
          <Alert type="warning">
            {zeroTile.length} slide(s) ran but saved zero tiles (no tissue above the minimum). Re-running
            won&apos;t change that.
          </Alert>
        )}
      </>
    );
  } else {
    const [pathKey, reasonKey, countKey] = OUTPUT_KEY[stage];
    body = (
      <>
        {status[pathKey] && (
          <>
            <Caption>Output:</Caption>
            <CodeBlock>{status[pathKey]}</CodeBlock>
          </>
        )}
        {done[countKey] != null && (
          <Caption>Verified by the pipeline: {Number(done[countKey]).toLocaleString()} rows.</Caption>
        )}
        {status[reasonKey] && (
          <Alert type="error">The server&apos;s check rejects this output: {status[reasonKey]}</Alert>
        )}
        {stage === "assignment" && status.assignment_vote && <Caption>Vote: {status.assignment_vote}</Caption>}
        {stage === "assignment" && pipelineStageVerified(status, stage) && (
          <Caption>Classification verified — registration and the Knowledge Bank load below are next.</Caption>
        )}
      </>
    );
  }

  return (
    <div>
      {body}
      {state === "failed" && <PipelineResume status={status} submissionId={submissionId} onChanged={onChanged} />}
    </div>
  );
}
