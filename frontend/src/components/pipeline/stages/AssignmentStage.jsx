// Port of the cluster-assignment stage in app_v28.py: _render_assignment_step,
// _render_cohort_shift, _render_vote_picker and _render_assign_clusters_form
// (~lines 2332-2692).
import { useEffect, useState } from "react";
import { api } from "../../../api";
import { errorDetail, httpDetail, SLURM_IN_FLIGHT } from "../utils";
import { Alert, Button, CodeBlock, Expander, Field, Metric, RadioGroup } from "../widgets";

const SHIFT_STYLE = {
  consistent: { icon: "", type: "success" },
  notice: { icon: "", type: "warning" },
  alarm: { icon: "", type: "error" },
};

// "Is this cohort represented in the reference?" — a distinct question from
// "did Stage 4 work", not run automatically (cheap, but a result nobody
// asked for is a result nobody reads).
function CohortShift({ submissionId }) {
  const [readiness, setReadiness] = useState(undefined);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .cohortShiftReadiness(submissionId)
      .then((r) => {
        if (!cancelled) setReadiness(r);
      })
      .catch(() => {
        if (!cancelled) setReadiness(null);
      });
    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  async function handleCheck() {
    setChecking(true);
    setError(null);
    try {
      const r = await api.checkCohortShift(submissionId);
      setResult(r);
    } catch (e) {
      setError(httpDetail(e));
    } finally {
      setChecking(false);
    }
  }

  return (
    <Expander title="Is this cohort represented in the reference?">
      <div className="pipeline-caption">
        k-NN gives every tile its nearest cluster however far away that cluster is, and reports
        nothing. So a cohort from a different scanner or stain still produces a complete
        assignment and per-slide proportions that differ — differences that read as biology. This
        compares this dataset&apos;s distance-to-reference against the reference&apos;s own, which
        is the cheapest way to tell the two apart.
      </div>

      {readiness && !readiness.has_profile ? (
        <>
          <Alert type="warning">
            No reference baseline for the &apos;{readiness.vote_preset}&apos; vote yet. It is a
            one-off leave-one-out over the reference (~20 minutes) and is then reused by every
            dataset. Run this once on the cluster:
          </Alert>
          <CodeBlock>{readiness.build_profile_command || ""}</CodeBlock>
          <div className="pipeline-caption">
            It will be written to {readiness.profile_path}, where this page looks for it.
          </div>
        </>
      ) : readiness && !readiness.has_assignments ? (
        <Alert type="info">No assignment CSV for this run yet — finish Stage 4 first.</Alert>
      ) : (
        <>
          <Button onClick={handleCheck} disabled={checking}>
            {checking ? "Checking…" : "Check"}
          </Button>
          {error && <Alert type="error">{error}</Alert>}
          {result && (
            <div>
              {(() => {
                const style = SHIFT_STYLE[result.level] || { icon: "", type: "info" };
                return (
                  <Alert type={style.type}>
                    {style.icon} {String(result.level || "?").toUpperCase()} — {result.verdict || ""}
                  </Alert>
                );
              })()}
              <div className="pipeline-metrics-row">
                <Metric
                  label="Tiles beyond the reference's 99th percentile"
                  value={`${((result.beyond_envelope || 0) * 100).toFixed(1)}%`}
                  caption={
                    result.novelty_ratio != null ? `${result.novelty_ratio.toFixed(1)}x expected` : null
                  }
                />
                <Metric
                  label="Median tile sits at reference percentile"
                  value={result.median_percentile != null ? `${result.median_percentile.toFixed(0)}th` : "n/a"}
                />
                <Metric
                  label="Spread across slides"
                  value={result.slide_spread == null ? "n/a" : `${(result.slide_spread * 100).toFixed(0)} pts`}
                  caption="Wide means some slides carry it; narrow means it's the cohort itself."
                />
              </div>

              {(result.levels || []).length > 0 && result.cohort_distance && (
                <>
                  <div className="pipeline-caption">Distance to the reference, by quantile</div>
                  <div style={{ overflowX: "auto" }}>
                    <table className="pipeline-table">
                      <thead>
                        <tr>
                          <th>quantile</th>
                          <th>reference</th>
                          <th>this cohort</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.levels.map((lvl, i) => (
                          <tr key={lvl}>
                            <td>{(lvl * 100).toFixed(0)}%</td>
                            <td>{(result.reference_distance || [])[i]}</td>
                            <td>{(result.cohort_distance || [])[i]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}

              {(result.per_slide || []).length > 0 && (
                <>
                  <div className="pipeline-caption">
                    Worst {result.per_slide.length} of {result.n_slides || result.per_slide.length} slides by
                    share beyond the envelope
                  </div>
                  <div style={{ overflowX: "auto" }}>
                    <table className="pipeline-table">
                      <thead>
                        <tr>
                          {Object.keys(result.per_slide[0]).map((k) => (
                            <th key={k}>{k}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.per_slide.map((row, i) => (
                          <tr key={i}>
                            {Object.keys(result.per_slide[0]).map((k) => (
                              <td key={k}>{String(row[k])}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}

              <div className="pipeline-caption">Vote: {result.recorded_vote || result.vote_preset}</div>
              <div className="pipeline-caption">Reference profile: {result.profile_vote}</div>
            </div>
          )}
        </>
      )}
    </Expander>
  );
}

// Which vote Stage 4 should use — a preset list rather than raw number
// inputs, because a half-applied configuration (distance weighting on but
// the exponent left at 1, say) produces a complete CSV that isn't the
// measured configuration.
export function VotePicker({ submissionId, onChange }) {
  const [presets, setPresets] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [chosen, setChosen] = useState("");
  const [overrides, setOverrides] = useState({});
  const [overrideInputs, setOverrideInputs] = useState({});
  const [overrideError, setOverrideError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .votePresets()
      .then((served) => {
        if (cancelled) return;
        const list = served.presets || {};
        if (!Object.keys(list).length) {
          setPresets({});
          return;
        }
        const names = Object.keys(list).sort((a, b) => (list[b].accuracy || 0) - (list[a].accuracy || 0));
        const def = served.default && names.includes(served.default) ? served.default : names[0];
        setPresets(list);
        setChosen(def);
      })
      .catch((e) => setLoadError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, [submissionId]);

  useEffect(() => {
    onChange && onChange(chosen, overrides);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chosen, overrides]);

  if (loadError) {
    return <div className="pipeline-caption">Could not load vote presets ({loadError}); the server&apos;s default will be used.</div>;
  }
  if (!presets || !Object.keys(presets).length) return null;

  const names = Object.keys(presets).sort((a, b) => (presets[b].accuracy || 0) - (presets[a].accuracy || 0));
  const spec = presets[chosen] || presets[names[0]];
  const flags = spec.flags || {};
  const fieldDefs = [
    ["distance_power", "Distance exponent"],
    ["adaptive_margin", "Re-vote below this margin (0 = off)"],
    ["adaptive_k", "Re-vote at this k (0 = off)"],
  ].filter(([field]) => flags[field] != null);

  function updateOverride(field, entered) {
    setOverrideInputs((prev) => ({ ...prev, [field]: entered }));
    setOverrideError(null);
    if (!entered.trim()) {
      setOverrides((prev) => {
        const next = { ...prev };
        delete next[field];
        return next;
      });
      return;
    }
    const value = Number(entered);
    if (Number.isNaN(value)) {
      setOverrideError(`${field}: '${entered}' is not a number — ignoring it.`);
      return;
    }
    setOverrides((prev) => ({ ...prev, [field]: field === "adaptive_k" ? Math.trunc(value) : value }));
  }

  return (
    <div>
      <RadioGroup
        name={`assign-vote-${submissionId}`}
        label="Vote"
        options={names.map((n) => ({ value: n, label: presets[n].label || n }))}
        value={chosen}
        onChange={setChosen}
        help="How the k-NN vote is weighted. This changes the cluster ID of every tile, so two runs under different votes are not comparable and must not be mixed in the Knowledge Bank."
      />
      <div className="pipeline-caption">{spec.summary || ""}</div>
      <Expander title="What this means">
        <div className="pipeline-caption">{spec.why || ""}</div>
        {spec.accuracy != null && (
          <div className="pipeline-caption">
            {(spec.accuracy * 100).toFixed(2)}% leave-one-out on the production reference at
            200,000 tiles (seed 0). That measures recovery of the reference&apos;s own Leiden
            labels — not agreement with the original TCGA transfer, which is a separate check that
            can move the other way.
          </div>
        )}
      </Expander>

      {fieldDefs.length > 0 && (
        <Expander title="Override individual settings">
          <div className="pipeline-caption">
            Starts from the preset above and changes only what you touch. For comparing
            configurations — anything other than a preset as-is is not a measured setting.
          </div>
          {fieldDefs.map(([field, label]) => (
            <Field key={field} label={`${label} — preset uses ${flags[field]}`}>
              <input
                type="text"
                placeholder={String(flags[field])}
                value={overrideInputs[field] || ""}
                onChange={(e) => updateOverride(field, e.target.value)}
              />
            </Field>
          ))}
          {overrideError && <Alert type="error">{overrideError}</Alert>}
          {Object.keys(overrides).length > 0 && (
            <Alert type="warning">
              Overridden: {Object.entries(overrides).map(([k, v]) => `${k}=${v}`).join(", ")}. This
              is no longer the measured configuration.
            </Alert>
          )}
        </Expander>
      )}
    </div>
  );
}

function AssignClustersForm({ submissionId, buttonLabel = "Start cluster assignment", fullAvailable = true, onQueued }) {
  const [mode, setMode] = useState(fullAvailable ? "Full dataset" : "Test on a sample .h5");
  const [reference, setReference] = useState("");
  const [projections, setProjections] = useState("");
  const [votePreset, setVotePreset] = useState("");
  const [voteOverrides, setVoteOverrides] = useState({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);

  async function handleSubmit() {
    const ref = reference.trim() || null;
    setMessage(null);
    if (mode === "Test on a sample .h5" && !projections.trim()) {
      setMessage({ type: "error", text: "Enter the projections .h5 to assign." });
      return;
    }
    setBusy(true);
    try {
      let result;
      if (mode === "Test on a sample .h5") {
        result = await api.startTestClusterAssignment(submissionId, {
          projectionsH5: projections.trim(),
          reference: ref,
          votePreset,
          voteOverrides,
        });
        setMessage({ type: "success", text: "Test cluster assignment queued (not recorded against this run)." });
      } else {
        result = await api.startClusterAssignment(submissionId, {
          reference: ref,
          overwrite: true,
          votePreset,
          voteOverrides,
        });
        setMessage({ type: "success", text: "Cluster assignment job queued." });
      }
      if (result.vote) {
        setMessage((prev) => ({ ...prev, extra: `Vote: ${result.vote}` }));
      }
      onQueued && onQueued();
    } catch (e) {
      setMessage({ type: "error", text: `Failed to start cluster assignment: ${errorDetail(e).message}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <RadioGroup
        name={`assign-mode-${submissionId}`}
        label="Run"
        options={["Full dataset", "Test on a sample .h5"]}
        value={mode}
        onChange={setMode}
      />

      <VotePicker submissionId={submissionId} onChange={(name, overrides) => { setVotePreset(name); setVoteOverrides(overrides); }} />

      <Expander title="Options">
        <Field
          label="Reference .npz (blank = the server's configured reference)"
          help="The .npz built by build_hpc_reference.py — NOT the Leiden .h5ad it is built from. Leave blank unless you are deliberately comparing two references."
        >
          <input
            type="text"
            placeholder="hpc_reference_leiden_2p5_fold2.npz"
            value={reference}
            onChange={(e) => setReference(e.target.value)}
          />
        </Field>
      </Expander>

      {mode === "Test on a sample .h5" && (
        <Field
          label="Projections .h5 to assign"
          help="Typically what a test feature extraction wrote. Results are not recorded against this run."
        >
          <input
            type="text"
            placeholder="/path/to/results/.../hdf5_DS_he_train.h5"
            value={projections}
            onChange={(e) => setProjections(e.target.value)}
          />
        </Field>
      )}

      {mode === "Full dataset" && !fullAvailable ? (
        <Alert type="info">
          This run has no finished feature extraction, so there is no tracked projections file to
          assign. Either finish Stage 3, or switch to &quot;Test on a sample .h5&quot; above and
          give the path to a projections .h5 you already have.
        </Alert>
      ) : (
        <>
          <Button kind="primary" onClick={handleSubmit} disabled={busy}>
            {busy ? "Starting…" : buttonLabel}
          </Button>
          {message && (
            <>
              <Alert type={message.type}>{message.text}</Alert>
              {message.extra && <div className="pipeline-caption">{message.extra}</div>}
            </>
          )}
        </>
      )}
    </div>
  );
}

export default function AssignmentStage({ status, submissionId, state, onChanged }) {
  if (status.assignment_ready) {
    return (
      <div>
        <Alert type="success">Cluster assignments ready:</Alert>
        <CodeBlock>{status.assignment_output_path}</CodeBlock>
        {status.assignment_reference && <div className="pipeline-caption">Reference: {status.assignment_reference}</div>}
        {status.assignment_vote && <div className="pipeline-caption">Vote: {status.assignment_vote}</div>}
        <CohortShift submissionId={submissionId} />
      </div>
    );
  }

  // "blocked" only means this run's tracked extraction has no usable output.
  // Assigning an existing projections file by path doesn't depend on that,
  // so the form stays reachable rather than making an existing embeddings
  // file unreachable from the UI.
  if (state === "blocked") {
    return (
      <AssignClustersForm
        submissionId={submissionId}
        buttonLabel="Start cluster classification"
        fullAvailable={false}
        onQueued={onChanged}
      />
    );
  }

  const assignmentJobId = status.assignment_job_id;
  return (
    <div>
      {assignmentJobId && (
        <>
          <div className="pipeline-caption">Cluster assignment job:</div>
          <CodeBlock>{assignmentJobId}</CodeBlock>
          {SLURM_IN_FLIGHT.has(status.assignment_slurm_state) ? (
            <Alert type="info">Running (Slurm state: {status.assignment_slurm_state}).</Alert>
          ) : (
            <>
              <Alert type="warning">This attempt ended as: {status.assignment_slurm_state || "no Slurm record"}</Alert>
              {status.assignment_invalid_reason && (
                <Alert type="error">An output exists but is not usable: {status.assignment_invalid_reason}</Alert>
              )}
            </>
          )}
        </>
      )}
      {!(assignmentJobId && SLURM_IN_FLIGHT.has(status.assignment_slurm_state)) && (
        <AssignClustersForm
          submissionId={submissionId}
          buttonLabel={assignmentJobId ? "Retry cluster assignment" : "Start cluster assignment"}
          onQueued={onChanged}
        />
      )}
    </div>
  );
}
