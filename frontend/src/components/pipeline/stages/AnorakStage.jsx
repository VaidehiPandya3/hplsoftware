// Port of _render_anorak_step() / _render_anorak_form() in app_v28.py — Stage 7.
//
// The one stage that runs as a Nextflow pipeline rather than a Slurm job of
// its own, which changes what there is to show. The job id here is a *head
// process* — it submits a job per slide per stage itself — so its Slurm state
// answers "is the pipeline alive" and nothing about how far through it is.
// Progress lives in the run's own output directory, which is why that path is
// shown whether the run is finished or still going.
//
// Never "blocked" (see pipelineSteps' anorak branch), so unlike the other
// stage components this one has no blocked-state early return.
import { useState } from "react";
import { api } from "../../../api";
import { httpDetail, SLURM_IN_FLIGHT } from "../utils";
import { Alert, Button, Caption, CodeBlock, Expander, Field, RadioGroup } from "../widgets";

const SCOPE_FULL = "Full slide list (production)";
const SCOPE_SUBSET = "Random subset (test)";

function AnorakForm({ status, submissionId, buttonLabel, defaultOpen, onChanged, overwrite = false }) {
  const [slidesCsv, setSlidesCsv] = useState("");
  const [scopeLabel, setScopeLabel] = useState(SCOPE_FULL);
  const [sampleSize, setSampleSize] = useState(10);
  const [seedText, setSeedText] = useState("");
  const [resume, setResume] = useState(true);
  const [timeLimit, setTimeLimit] = useState("");
  const [chain, setChain] = useState(2);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  const subset = scopeLabel === SCOPE_SUBSET;

  async function handleSubmit() {
    setMessage(null);
    if (!slidesCsv.trim()) {
      setMessage({ type: "error", text: "A slide list is required." });
      return;
    }
    setSubmitting(true);
    try {
      const seed = /^\d+$/.test(seedText.trim()) ? Number(seedText.trim()) : null;
      const result = await api.startAnorak(submissionId, {
        slidesCsv: slidesCsv.trim(),
        scope: subset ? "subset" : "full",
        sampleSize: subset ? Number(sampleSize) : null,
        seed,
        resume,
        // overwrite means "replace a finished grading table" and is set only
        // from the "Run again" form. The server refuses a live or unknown head
        // job whatever this says.
        overwrite,
        timeLimit: timeLimit.trim() || null,
        // Standbys resume the run before them, so without resume there is
        // nothing for one to continue — the server refuses the combination.
        chain: resume ? Math.max(1, Number(chain) || 1) : 1,
      });
      const selection = result.selection || {};
      const text =
        selection.scope === "subset"
          ? `Submitted: ${selection.slides} slides sampled at random from ${selection.pool}, ` +
            `seed ${selection.seed} (job ${result.anorak_job_id}).`
          : `Submitted: ${selection.slides} slides (job ${result.anorak_job_id}).`;
      setMessage({ type: "success", text, slideList: result.slide_list });
      if (onChanged) onChanged();
    } catch (e) {
      setMessage({
        type: "error",
        text: e instanceof api.ApiError ? httpDetail(e) : `Submission failed: ${e.message || e}`,
      });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Expander title="Run settings" defaultOpen={defaultOpen}>
      <Caption>
        ANORAK segments growth patterns in lung adenocarcinoma, so it runs only on slides that
        carry tumour. Point it at the output of <code>select_tumour_slides.py --out</code>, which
        writes tumour slides only, optionally filtered by <code>filter_slides_by_tile_count.py</code>{" "}
        for a floor on how much malignant tissue a verdict rests on. The run refuses the list before
        queueing anything if a row has a blank sample or is not marked as tumour.
      </Caption>

      {/* Deliberately NOT pre-filled with anorak_slide_list. That field holds the
          run's own copy of the list inside its output directory — pre-filling it
          would feed a re-run its own output, and a second submission would then
          truncate the file the first run's head job is reading. */}
      <Field
        label="Tumour-slide list (.csv)"
        help="Absolute path on the HPC filesystem. Needs a slide_id column and a samples column
              naming each slide's tumour, filled in on every row: grades are pooled by sample, and a
              blank one used to pool unrelated slides into one tumour. An is_tumour column, if
              present, must be true on every row. This is the source list — the run writes its own
              copy beside its outputs."
      >
        <input
          type="text"
          value={slidesCsv}
          placeholder="/path/to/tumour_slides.csv"
          onChange={(e) => setSlidesCsv(e.target.value)}
        />
      </Field>

      {status.anorak_slide_list && (
        <Caption>
          The previous attempt ran on this list — a copy, inside the run&apos;s own output
          directory. Point at the source it came from, not at this: <code>{status.anorak_slide_list}</code>
        </Caption>
      )}

      <RadioGroup
        label="Scope"
        name={`anorak-scope-${submissionId}`}
        options={[SCOPE_FULL, SCOPE_SUBSET]}
        value={scopeLabel}
        onChange={setScopeLabel}
        help="A subset runs every stage exactly as the full cohort does, on fewer slides — so a
              subset that works is evidence the full run will."
      />

      {subset && (
        <div className="pipeline-metrics-row">
          <Field
            label="Slides to sample"
            help="Sampled at random across the list rather than taken from the top — the first N
                  slides of a cohort are usually one or two patients, sharing a scanner, a batch
                  and a stain run."
          >
            <input
              type="number"
              min={1}
              step={1}
              value={sampleSize}
              onChange={(e) => setSampleSize(e.target.value)}
            />
          </Field>
          <Field
            label="Seed (optional)"
            help="Leave blank and one is chosen and recorded, so the sample can be asked for again
                  either way. On a retry of a subset run, blank repeats the previous sample — the
                  server refuses if it cannot."
          >
            <input type="text" value={seedText} onChange={(e) => setSeedText(e.target.value)} />
            {status.anorak_scope === "subset" && status.anorak_seed != null && (
              <Caption>Previous attempt: seed {status.anorak_seed}.</Caption>
            )}
          </Field>
        </div>
      )}

      <div className="pipeline-checkbox-row">
        <input
          type="checkbox"
          id={`anorak-resume-${submissionId}`}
          checked={resume}
          onChange={(e) => setResume(e.target.checked)}
        />
        <label htmlFor={`anorak-resume-${submissionId}`}>Continue the cached run</label>
      </div>
      <Caption>
        Nextflow re-runs only the tasks whose inputs changed. Uncheck to start from scratch — which
        for a full cohort is days.
      </Caption>

      <Field
        label="Head job walltime (optional)"
        help="Slurm format, e.g. 2-00:00:00 or 48:00:00. Leave blank for the server's default. Must
              be within the partition's MaxTime (sinfo -o '%P %l'); the head job has to outlive
              every job it submits, so pick the largest allowed."
      >
        <input
          type="text"
          value={timeLimit}
          placeholder="2-00:00:00"
          onChange={(e) => setTimeLimit(e.target.value)}
        />
      </Field>

      <Field
        label="Head jobs"
        help="The first head job plus standbys. A standby starts only if the one before it ended
              without finishing — it reached its walltime or ran out of watchdog restarts — and
              resumes it. A real failure or a scancel stops the whole chain. Needs “Continue the
              cached run”; without it one head job is submitted."
      >
        <input
          type="number"
          min={1}
          step={1}
          value={resume ? chain : 1}
          disabled={!resume}
          onChange={(e) => setChain(e.target.value)}
        />
      </Field>

      <Button kind="primary" onClick={handleSubmit} disabled={submitting}>
        {submitting ? "Submitting…" : buttonLabel}
      </Button>

      {message && <Alert type={message.type}>{message.text}</Alert>}
      {message?.slideList && (
        <>
          <Caption>The list this run was given, kept beside its outputs:</Caption>
          <CodeBlock>{message.slideList}</CodeBlock>
        </>
      )}
    </Expander>
  );
}

export default function AnorakStage({ status, submissionId, onChanged }) {
  if (status.anorak_ready) {
    const scope = status.anorak_scope;
    const slides = status.anorak_slides;
    return (
      <div>
        <Alert type="success">Growth pattern grading complete:</Alert>
        <CodeBlock>{status.anorak_grades_csv}</CodeBlock>
        {scope === "subset" ? (
          <Caption>
            Random subset: {slides} of {status.anorak_sample_size || slides} requested, seed{" "}
            {status.anorak_seed}. This is a test run — the full cohort has not been graded.
          </Caption>
        ) : (
          slides != null && <Caption>Full slide list: {slides} slides.</Caption>
        )}
        {status.anorak_out_dir && (
          <>
            <Caption>Per-slide masks, proportions and the Nextflow report:</Caption>
            <CodeBlock>{status.anorak_out_dir}</CodeBlock>
          </>
        )}
        <AnorakForm
          status={status}
          submissionId={submissionId}
          buttonLabel="Run again"
          defaultOpen={false}
          onChanged={onChanged}
          overwrite={true}
        />
      </div>
    );
  }

  const jobId = status.anorak_job_id;
  const anorakState = status.anorak_slurm_state;
  const inFlight =
    Boolean(jobId) && (Boolean(status.anorak_in_flight) || SLURM_IN_FLIGHT.has(anorakState));
  // The server's own refusal (in flight, or Slurm unreachable so it cannot
  // tell). Shown in place of a form it would refuse — with Slurm unreachable
  // the head job may still be running.
  const blocked = Boolean(jobId) && !inFlight && Boolean(status.anorak_submit_blocked);

  return (
    <div>
      {status.anorak_error && <Alert type="error">{status.anorak_error}</Alert>}
      {jobId && (
        <div>
          <Caption>Nextflow head job:</Caption>
          <CodeBlock>{jobId}</CodeBlock>
          {inFlight ? (
            <>
              <Alert type="info">
                Pipeline running (head job state: {anorakState}). It submits one job per slide per
                stage, so <code>squeue</code> shows many more jobs than this one.
              </Alert>
              {status.anorak_out_dir && (
                <>
                  <Caption>Live progress — Nextflow&apos;s own trace and report:</Caption>
                  <CodeBlock>{status.anorak_out_dir}/pipeline_info</CodeBlock>
                </>
              )}
            </>
          ) : blocked ? (
            <Alert type="warning">{status.anorak_submit_blocked}</Alert>
          ) : (
            <>
              <Alert type="warning">This attempt ended as: {anorakState || "no Slurm record"}</Alert>
              {/* Written by the head job's supervisor when the run ended for
                  good — the reason, and why no standby took over. */}
              {status.anorak_stop_reason && (
                <>
                  <Caption>Why it stopped (the supervisor&apos;s stop marker):</Caption>
                  <CodeBlock>{status.anorak_stop_reason}</CodeBlock>
                </>
              )}
              {status.anorak_invalid_reason && (
                <Alert type="error">
                  An output exists but is not a finished grading table: {status.anorak_invalid_reason}
                </Alert>
              )}
              {status.anorak_out_dir && (
                <>
                  <Caption>The head job&apos;s log is the first place to look:</Caption>
                  <CodeBlock>{status.anorak_out_dir}/nextflow.log</CodeBlock>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* While a head job is in flight or its state is unknown, no form — a
          second submission would rewrite the live run's slide list and resume
          into its cache. The server refuses it either way. */}
      {!inFlight && !blocked && (
        <AnorakForm
          status={status}
          submissionId={submissionId}
          buttonLabel={jobId ? "Retry ANORAK" : "Run ANORAK"}
          defaultOpen={true}
          onChanged={onChanged}
        />
      )}
    </div>
  );
}
