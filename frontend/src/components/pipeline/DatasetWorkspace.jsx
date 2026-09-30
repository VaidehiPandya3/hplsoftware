// Port of _render_dataset_workspace() and everything it renders in
// app_v28.py (~lines 364-732): the dataset-level rollup ("what has this path
// already had done to it, and what's the one next thing to do"), the
// deliverables list, and the per-dataset run picker that hands off to
// RunProgress.
//
// Path-first, matching the original: if a path is given, only datasets built
// from that exact raw_dir are shown; if it's empty, every known dataset is
// offered instead. Polls list_datasets() every 10s — in the original this
// entire tree (rollup + run picker + the picked run's full 5-stage view) is
// one @st.fragment(run_every="10s"), so all of it stays live together.
import { useState } from "react";
import { api } from "../../api";
import { usePolling } from "../../hooks/usePolling";
import {
  datasetKey,
  datasetOptionLabel,
  datasetRunLabel,
  datasetsUnderPath,
  isAnorakRun,
  DELIVERABLE_ICON,
  STEP_ICON,
} from "./utils";
import { Alert, Expander } from "./widgets";
import RunProgress from "./RunProgress";

// Port of _render_deliverables(): every .h5 this dataset has produced or
// attempted — the 3-step rollup names only "the" packaging output, which is
// lossy when a directory has several (a full run and per-experiment
// subsets), and re-making a multi-hour .h5 because it "wasn't listed" is
// the failure this avoids.
function Deliverables({ dataset }) {
  const deliverables = dataset.deliverables || [];
  if (!deliverables.length) return null;
  const trials = (dataset.tests || []).filter((t) => t.status === "ready");
  return (
    <div>
      <div className="pipeline-caption pipeline-caption-strong">
        <strong>.h5 files</strong>
      </div>
      {deliverables.map((item, i) => {
        const icon = DELIVERABLE_ICON[item.status] || "";
        const bits = [];
        const slides = item.total_slides;
        if (slides != null) bits.push(`${Number(slides).toLocaleString()} slides`);
        const others = (item.slides_seen || []).filter((s) => s !== slides);
        if (others.length) bits.push("also tried at " + others.map((s) => Number(s).toLocaleString()).join(", "));
        const submitted = (item.submitted_at || "").slice(0, 10);
        if (submitted) bits.push(submitted);
        if ((item.attempts || 1) > 1) bits.push(`${item.attempts} attempts`);
        if (item.status === "running") bits.push(`packaging now (${item.slurm_state})`);
        else if (item.status === "interrupted") bits.push("no file on disk");
        return (
          <div key={i} style={{ marginBottom: 4 }}>
            <div className="pipeline-caption pipeline-caption-strong">
              {icon} <code>{item.name}</code>
            </div>
            {bits.length > 0 && <div className="pipeline-caption">{bits.join(" · ")}</div>}
          </div>
        );
      })}
      {trials.length > 0 && (
        <div className="pipeline-caption">
          {trials.length} test .h5 also on disk — usable for a checkpoint trial without
          repackaging.
        </div>
      )}
    </div>
  );
}

// Port of _render_next_action(): where this dataset stands, in one line — no
// button. The per-stage actions it used to offer (resume tiling, start
// packaging, start extraction) are gone: Stages 1-4 are one pipeline run,
// started from the button above, which reuses every slide already tiled.
function NextAction({ dataset }) {
  const action = dataset.next_action || {};
  const kind = action.kind;
  if (kind === "complete" || kind === "none" || kind === "wait") {
    return (
      <div className="pipeline-caption">
        {action.label} — {action.detail || ""}
      </div>
    );
  }
  if (kind === "submit") return <div className="pipeline-caption">Nothing has been run for this dataset yet.</div>;
  if (!kind) return null;
  return (
    <div className="pipeline-caption">
      Earlier runs stopped at: {(action.label || "").replace(/ →$/, "")}. Run the pipeline above to carry this
      dataset through — it reuses every slide already tiled.
    </div>
  );
}

// Port of _render_dataset_rollup(): every step for this dataset, whether or
// not it's the active one — the aggregate across every run that has touched
// it (a resume forks a run in two, so this is the only level "finished vs.
// left" can be answered at).
function DatasetRollup({ dataset }) {
  const provenance = [];
  if (dataset.has_full_run) provenance.push("full-dataset run");
  if (dataset.has_subset_run) provenance.push("subset run(s)");
  if (dataset.run_count) provenance.push(`${dataset.run_count} run(s) total`);
  if (dataset.cancelled_run_count) provenance.push(`${dataset.cancelled_run_count} cancelled (hidden below)`);

  return (
    <div>
      <div className="pipeline-caption">
        <code>{dataset.raw_dir || ""}</code>
      </div>
      {provenance.length > 0 && <div className="pipeline-caption">{provenance.join(" · ")}</div>}

      {(dataset.steps || []).map((step, i) => (
        <div key={i} className="pipeline-caption pipeline-caption-strong">
          {STEP_ICON[step.state] || ""} <strong>{step.title}</strong> — {step.summary}
        </div>
      ))}

      {dataset.coverage_checked_at && (
        <div className="pipeline-caption">Tiling coverage checked from disk at {dataset.coverage_checked_at}.</div>
      )}

      <Deliverables dataset={dataset} />
      <NextAction dataset={dataset} />
    </div>
  );
}

// Port of _render_dataset_runs(): this dataset's runs (cancelled excluded —
// dead ends, nothing about them can be advanced), one shown at a time via a
// picker, handed off to RunProgress.
function DatasetRuns({ dataset, selectedRunId, onSelectRun }) {
  // Every run this dataset has had — HPL, ANORAK, runs from before the
  // pipeline, cancelled ones included — newest first.
  const runs = [...(dataset.runs || [])].reverse();
  if (!runs.length) return <div className="pipeline-caption">No runs for this dataset yet.</div>;

  const validSelection = runs.find((r) => r.submission_id === selectedRunId) ? selectedRunId : runs[0].submission_id;
  const chosenJob = runs.find((r) => r.submission_id === validSelection);

  return (
    <div>
      <div className="pipeline-field">
        <span className="pipeline-field-label">Run</span>
        <select value={validSelection} onChange={(e) => onSelectRun(e.target.value)}>
          {runs.map((r) => (
            <option key={r.submission_id} value={r.submission_id}>
              {datasetRunLabel(r)}
            </option>
          ))}
        </select>
      </div>
      <RunProgress submissionId={validSelection} job={chosenJob} />
    </div>
  );
}

export default function DatasetWorkspace({ path }) {
  const { data: payload, error, loading } = usePolling(() => api.listDatasets(), {
    intervalMs: 10000,
    deps: [],
  });

  const [coverageOverrides] = useState({}); // { [datasetKey]: {rollup, at} } — no longer set here
  const [pickedKey, setPickedKey] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);

  if (error) return <Alert type="warning">Could not load datasets: {error.message}</Alert>;
  if (loading && !payload) return <div className="pipeline-caption">Loading datasets…</div>;
  if (!payload) return null;

  const datasets = payload.datasets || [];

  let matches;
  if (path) {
    matches = datasetsUnderPath(datasets, path);
    if (!matches.length) {
      return (
        <Alert type="info">
          No runs recorded for this path yet. Run HPL or ANORAK above to start the first one.
        </Alert>
      );
    }
  } else if (datasets.length) {
    matches = datasets;
  } else {
    return (
      <div className="pipeline-caption">
        No dataset runs on record yet. Enter a path and submit one below and it will appear here
        with its progress.
      </div>
    );
  }

  // Port of _pick_dataset: skip the selector entirely when there's only one
  // option — a control with one setting is an extra click, not a choice.
  const byKey = {};
  matches.forEach((d) => {
    byKey[datasetKey(d)] = d;
  });
  const keys = Object.keys(byKey);
  const effectiveKey = byKey[pickedKey] ? pickedKey : keys[0];
  const rawDataset = byKey[effectiveKey];

  // Port of _apply_stored_coverage: overlay a previously fetched filesystem
  // coverage check on top of the live poll, but only for the tiling line —
  // packaging/extraction go stale within seconds so those stay live.
  const stored = coverageOverrides[effectiveKey];
  const dataset = stored
    ? {
        ...rawDataset,
        tiling: stored.rollup.tiling || rawDataset.tiling,
        steps: [stored.rollup.tiling || rawDataset.steps[0], ...rawDataset.steps.slice(1)],
        slides_tiled: stored.rollup.slides_tiled ?? rawDataset.slides_tiled,
        slides_untiled: stored.rollup.slides_untiled ?? rawDataset.slides_untiled,
        total_slides: stored.rollup.total_slides ?? rawDataset.total_slides,
        coverage_checked_at: stored.at,
      }
    : rawDataset;

  // The newest HPL run and the newest ANORAK run are what someone opening this
  // path is following; every run, those included, is in History.
  const runsHere = dataset.runs || [];
  const hplRuns = runsHere.filter((r) => String(r.job_id || "").startsWith("nf:"));
  const anorakRuns = runsHere.filter(isAnorakRun);
  const latestHpl = hplRuns.length ? hplRuns[hplRuns.length - 1] : null;
  const latestAnorak = anorakRuns.length ? anorakRuns[anorakRuns.length - 1] : null;

  return (
    <div>
      {payload.slurm_reachable === false ? (
        <Alert type="warning">
          Slurm isn&apos;t answering, so the states below come from what&apos;s on disk only. A
          job could be running right now without showing here.
        </Alert>
      ) : payload.slurm_states_complete === false ? (
        <div className="pipeline-caption">
          Older jobs were checked against the live queue only — open a run for its exact outcome.
        </div>
      ) : null}

      {keys.length > 1 && (
        <div className="pipeline-field">
          <span className="pipeline-field-label">Dataset</span>
          <select value={effectiveKey} onChange={(e) => setPickedKey(e.target.value)}>
            {keys.map((k) => (
              <option key={k} value={k}>
                {datasetOptionLabel(byKey[k])}
              </option>
            ))}
          </select>
          <span className="pipeline-field-help">
            Every run that tiled into the same folder from the same directory, rolled into one
            pipeline — including the extra runs each resume creates.
          </span>
        </div>
      )}

      {latestHpl && (
        <div>
          <div className="pipeline-caption pipeline-caption-strong">
            <strong>Latest HPL run</strong>
          </div>
          <RunProgress submissionId={latestHpl.submission_id} job={latestHpl} />
        </div>
      )}
      {latestAnorak && (
        <div>
          <div className="pipeline-caption pipeline-caption-strong">
            <strong>Latest ANORAK run</strong>
          </div>
          <RunProgress submissionId={latestAnorak.submission_id} job={latestAnorak} />
        </div>
      )}

      <Expander title="History">
        <DatasetRollup dataset={dataset} />
        <hr className="pipeline-divider" />
        <DatasetRuns dataset={dataset} selectedRunId={selectedRunId} onSelectRun={setSelectedRunId} />
      </Expander>
    </div>
  );
}
