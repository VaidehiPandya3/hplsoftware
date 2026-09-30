// Port of _render_kb_load_step() in app_v28.py (~lines 2695-2870).
//
// Mirrors load_hpc_assignments.py's own shape — dry-run preview, then an
// explicit commit — rather than one button, because this is the one stage
// that mutates the shared Knowledge Bank every other view reads from. It
// never auto-commits, and the commit button is disabled outright when the
// preview reports would_refuse_low_match — the UI must not allow committing
// when the CLI's own --commit guard would refuse it (95% match rate).
import { useState } from "react";
import { api } from "../../../api";
import { errorDetail, fmtInt } from "../utils";
import { Alert, Button, Caption, Expander, Field } from "../widgets";
import { KbJobState, WRITE_SERVER, WRITE_SLURM, WriteMode } from "../kbWrite.jsx";

export default function KbLoadStage({ status, submissionId, state }) {
  // "blocked" only means this run's own tracked assignment has no usable
  // output — loading from an explicit CSV path doesn't depend on that (it's
  // how output from Stage 4's "Test on a sample .h5" mode gets into the KB,
  // since that mode never records a path against any run).
  const blockedByDefault = state === "blocked";
  const [sourceChoice, setSourceChoice] = useState("This run's tracked assignment");
  const useManualPath = blockedByDefault || sourceChoice !== "This run's tracked assignment";

  const [manualCsvPath, setManualCsvPath] = useState("");
  const [minMargin, setMinMargin] = useState(0);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [report, setReport] = useState(null);

  const [cancerType, setCancerType] = useState("");
  const [allowUnknown, setAllowUnknown] = useState(false);
  const [skipProfiles, setSkipProfiles] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [commitMessage, setCommitMessage] = useState(null);
  const [writeMode, setWriteMode] = useState(WRITE_SLURM);
  const [jobInFlight, setJobInFlight] = useState(false);

  async function handlePreview() {
    setPreviewError(null);
    if (useManualPath && !manualCsvPath.trim()) {
      setPreviewError("Enter the assignment CSV path.");
      return;
    }
    setPreviewing(true);
    try {
      const r = await api.previewKbLoad(submissionId, {
        minMargin: Number(minMargin),
        csvPath: useManualPath ? manualCsvPath.trim() || null : null,
      });
      setReport(r);
      setCommitMessage(null);
    } catch (e) {
      setPreviewError(`Preview failed: ${errorDetail(e).message}`);
      setReport(null);
    } finally {
      setPreviewing(false);
    }
  }

  async function handleCommit() {
    setCommitMessage(null);
    setCommitting(true);
    const args = {
      cancerType: cancerType.trim() || null,
      allowUnknownClusters: allowUnknown,
      skipProfiles,
      minMargin: Number(minMargin),
      csvPath: useManualPath ? manualCsvPath.trim() || null : null,
    };
    try {
      if (writeMode === WRITE_SLURM) {
        // Queued, not written here. The job records kb_load_done itself, so
        // "done" still means committed — see submit_kb_write.py.
        const q = await api.submitKbLoad(submissionId, args);
        setCommitMessage({
          type: "success",
          text:
            `Queued as Slurm job ${q.kb_load_job_id}. It writes to ${q.database} ` +
            `whether or not this page stays open.` +
            (q.kb_load_log_path ? ` Log: ${q.kb_load_log_path}` : ""),
        });
        setReport(null);
        setCommitting(false);
        return;
      }
      const result = await api.commitKbLoad(submissionId, args);
      let msg = `Committed ${result.updated_rows.toLocaleString()} tile(s) to the Knowledge Bank.`;
      if (result.excluded_from_aggregates) {
        msg += ` ${result.excluded_from_aggregates.toLocaleString()} tile(s) below margin ${result.min_margin} were excluded from the aggregates.`;
      }
      if (result.recorded_on_run === false) {
        msg += " Not recorded against this run's own KB-load status, since the CSV wasn't this run's tracked output.";
      }
      setCommitMessage({ type: "success", text: msg });
      setReport(null);
    } catch (e) {
      setCommitMessage({ type: "error", text: `Load failed: ${errorDetail(e).message}` });
    } finally {
      setCommitting(false);
    }
  }

  return (
    <div>
      {status.kb_load_done && (
        <>
          <Alert type="success">
            Loaded into the Knowledge Bank
            {status.kb_load_rows != null ? `: ${status.kb_load_rows.toLocaleString()} tiles` : ""}
            {status.kb_load_reference ? ` (reference \`${status.kb_load_reference}\`)` : ""}
          </Alert>
          {status.kb_load_at && <div className="pipeline-caption">Last loaded: {status.kb_load_at}</div>}
          <div className="pipeline-caption">If Stage 4 has been re-run since, preview below and reload.</div>
        </>
      )}

      <KbJobState
        status={status}
        prefix="kb_load"
        label="Knowledge Bank load"
        onInFlight={setJobInFlight}
      />

      {!blockedByDefault && (
        <div className="pipeline-field">
          <span className="pipeline-field-label">Source</span>
          <div className="pipeline-radio-group">
            {["This run's tracked assignment", "A specific CSV path"].map((opt) => (
              <label className="pipeline-radio-option" key={opt}>
                <input
                  type="radio"
                  name={`kb-load-source-${submissionId}`}
                  checked={sourceChoice === opt}
                  onChange={() => setSourceChoice(opt)}
                />
                {opt}
              </label>
            ))}
          </div>
          <span className="pipeline-field-help">
            Use &quot;A specific CSV path&quot; for output from Stage 4&apos;s &quot;Test on a
            sample .h5&quot; mode — that mode never records a path against this run, so
            there&apos;s nothing tracked to load from automatically.
          </span>
        </div>
      )}

      {useManualPath && (
        <>
          {blockedByDefault && (
            <Alert type="info">
              This run has no tracked assignment yet. Load from a specific CSV instead —
              typically output from Stage 4&apos;s &quot;Test on a sample .h5&quot; mode — or
              finish Stage 4&apos;s &quot;Full dataset&quot; option first.
            </Alert>
          )}
          <Field
            label="Assignment CSV path"
            help="This still writes to the Knowledge Bank for real — it just isn't recorded against this run's own KB-load status, since the CSV may not be this run's tracked output."
          >
            <input
              type="text"
              placeholder="/path/to/DS_hpc_assignments.csv"
              value={manualCsvPath}
              onChange={(e) => setManualCsvPath(e.target.value)}
            />
          </Field>
        </>
      )}

      <Field
        label={`Exclude tiles below this vote_margin from the aggregates: ${minMargin}`}
        help="Leave-one-out validation against the reference: margin below 0.1 was 57% correct, 0.1-0.25 was 76%, 0.25+ was 92%+. tile_registry keeps every tile's own hpc_id and margin regardless of this — it only changes what counts toward the per-slide composition the chatbot and HPC panels read. 0 (default) excludes nothing."
      >
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={minMargin}
          onChange={(e) => setMinMargin(Number(e.target.value))}
        />
      </Field>

      <Button kind="primary" onClick={handlePreview} disabled={previewing}>
        {previewing ? "Checking…" : "Preview Knowledge Bank load"}
      </Button>
      {previewError && <Alert type="error">{previewError}</Alert>}

      {report && (
        <div>
          <div className="pipeline-caption pipeline-caption-strong">
            <strong>{report.rows.toLocaleString()}</strong> rows in the CSV · cluster column{" "}
            <code>{report.cluster_column}</code> · reference <code>{report.reference}</code>
          </div>
          <div className="pipeline-caption pipeline-caption-strong">
            Matched <strong>{report.matched.toLocaleString()}/{report.rows.toLocaleString()}</strong>{" "}
            ({(report.match_rate * 100).toFixed(1)}%) tiles in <code>tile_registry</code>
          </div>
          {(report.tile_names_normalized || 0) > 0 && (
            <Alert type="info">
              <code>.jpeg</code> was appended to {fmtInt(report.tile_names_normalized)} tile name(s)
              from this CSV so they match the Knowledge Bank&apos;s <code>18_15.jpeg</code> form —
              the match rate above depends on that correction. The CSV on disk still holds the short
              form.
            </Alert>
          )}

          {/* Only surfaced when some of it disagrees. A CSV carrying its own
              slide_tile column that matches the rebuilt key is the normal,
              healthy case and needs no notice; a partial disagreement means the
              CSV's own columns disagree with each other, which the match rate
              alone would not explain. */}
          {report.slide_tile_supplied && report.slide_tile_disagreed > 0 && (
            report.slide_tile_disagreed === report.rows ? (
              <Alert type="info">
                This CSV carries its own <code>slide_tile</code> column and every value differs from
                the join key rebuilt from <code>slides</code> + <code>tiles</code> (
                <code>{(report.slide_tile_example || [])[0]}</code> →{" "}
                <code>{(report.slide_tile_example || [])[1]}</code>) — the usual sign of a column
                written before the tile names were normalised. The rebuilt key is what joins, so
                this is expected and harmless.
              </Alert>
            ) : (
              <Alert type="warning">
                This CSV carries its own <code>slide_tile</code> column and{" "}
                {fmtInt(report.slide_tile_disagreed)} of {fmtInt(report.rows)} values differ from
                the key rebuilt from <code>slides</code> + <code>tiles</code> (
                <code>{(report.slide_tile_example || [])[0]}</code> →{" "}
                <code>{(report.slide_tile_example || [])[1]}</code>). A <em>partial</em>
                {" "}disagreement means the CSV&apos;s own columns disagree with each other — worth
                checking before loading, since the rebuilt key is what joins.
              </Alert>
            )
          )}

          {report.unmatched > 0 && (
            <Alert type="warning">
              {report.unmatched.toLocaleString()} unmatched, e.g. {JSON.stringify(report.unmatched_examples)}
            </Alert>
          )}
          {report.overwriting > 0 && (
            <Alert type="info">
              Will overwrite {report.overwriting.toLocaleString()} tile(s) that already carry a cluster ID
              {report.overwriting_other_reference
                ? ` (${report.overwriting_other_reference.toLocaleString()} from a different reference)`
                : ""}
            </Alert>
          )}
          {report.unwritable_cluster_ids > 0 && (
            <Alert type="error">
              {fmtInt(report.unwritable_cluster_ids)} cluster ID(s) are not whole numbers and{" "}
              <code>tile_registry.hpc_id</code> is <code>{report.cluster_column_type}</code>:{" "}
              {JSON.stringify(report.unwritable_cluster_examples)}. The load will refuse — the
              assignment CSV&apos;s cluster column has to be fixed. Shown here because the preview
              stages only the join key, so this would otherwise surface as a failed UPDATE after
              every row was copied.
            </Alert>
          )}
          {report.unknown_clusters && report.unknown_clusters.length > 0 && (
            <Alert type="warning">
              {report.unknown_clusters.length} cluster ID(s) have no <code>hpc_dictionary</code> row:{" "}
              {JSON.stringify(report.unknown_clusters.slice(0, 10))}. Those tiles would show a
              cluster with no pattern or malignancy annotation unless allowed below.
            </Alert>
          )}
          {report.low_margin > 0 && (
            <div className="pipeline-caption">{report.low_margin.toLocaleString()} tile(s) have vote_margin below 0.1</div>
          )}
          {/* What a confidence threshold would cost and buy, from this cohort's
              own margin distribution. This is the table the 0.25 threshold was
              picked off — without it the slider above is a number with no
              consequence attached. Open by default while no threshold is set,
              which is when the decision is still to be made. */}
          {(report.margin_tradeoff || []).length > 0 && (
            <Expander
              title="What a confidence threshold would cost and buy on this cohort"
              defaultOpen={report.min_margin === 0}
            >
              <Caption>
                Every tile&apos;s <code>vote_margin</code> is the winning cluster&apos;s share of
                the weighted k-NN vote minus the runner-up&apos;s — 1.0 means all 25 neighbours
                agreed, 0.0 a dead tie. The threshold above decides which tiles count toward the
                per-slide composition the chatbot and HPC panels read.{" "}
                <strong>
                  <code>tile_registry</code> keeps every tile either way
                </strong>
                , with its own cluster and margin.
              </Caption>
              <table className="pipeline-table">
                <thead>
                  <tr>
                    <th>min_margin</th>
                    <th>tiles kept</th>
                    <th>% kept</th>
                    <th>expected accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {report.margin_tradeoff.map((row) => (
                    <tr key={row.min_margin}>
                      <td>{row.min_margin.toFixed(2)}</td>
                      <td>{fmtInt(row.tiles_kept)}</td>
                      <td>{(row.share_kept * 100).toFixed(1)}%</td>
                      <td>{(row.expected_accuracy * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Caption>
                Expected accuracy is this cohort&apos;s own margin distribution weighted by
                leave-one-out accuracy measured on the reference (61.7% below 0.10, 81.2% to 0.25,
                95.5% to 0.50, 99.5% to 0.75, 100% above). That mapping was measured inside
                LATTICeA, so applying it here assumes a margin means the same thing on this
                cohort&apos;s scanner and stain — the one thing no accuracy number can settle
                without labels. Treat it as an estimate, and read <code>cohort_shift</code>{" "}
                alongside it.
              </Caption>
            </Expander>
          )}

          {report.min_margin > 0 && (
            <Alert type="info">
              At the {report.min_margin} threshold above,{" "}
              <strong>{report.excluded_from_aggregates.toLocaleString()}</strong> tile(s) would be
              excluded from the per-slide aggregates (tile_registry keeps them regardless).
            </Alert>
          )}
          {report.would_refuse_low_match && (
            <Alert type="error">
              Match rate {(report.match_rate * 100).toFixed(1)}% is below the required{" "}
              {(report.min_match_rate * 100).toFixed(0)}% — committing would be refused the same
              way the CLI refuses it.
            </Alert>
          )}

          <Expander title="Commit options">
            <Field
              label="Cancer type (optional — fills hpl_profile_summary.cancer_type)"
              help="Left blank leaves it unset rather than guessing, same as the CLI."
            >
              <input type="text" placeholder="LUAD" value={cancerType} onChange={(e) => setCancerType(e.target.value)} />
            </Field>
            <div className="pipeline-checkbox-row">
              <input
                type="checkbox"
                id={`kb-allow-unknown-${submissionId}`}
                checked={allowUnknown}
                disabled={!(report.unknown_clusters && report.unknown_clusters.length)}
                onChange={(e) => setAllowUnknown(e.target.checked)}
              />
              <label htmlFor={`kb-allow-unknown-${submissionId}`}>
                Load cluster IDs with no hpc_dictionary row anyway
              </label>
            </div>
            <div className="pipeline-checkbox-row">
              <input
                type="checkbox"
                id={`kb-skip-profiles-${submissionId}`}
                checked={skipProfiles}
                onChange={(e) => setSkipProfiles(e.target.checked)}
              />
              <label htmlFor={`kb-skip-profiles-${submissionId}`}>
                Skip refreshing the per-slide aggregates (tile_registry only)
              </label>
            </div>
            <div className="pipeline-caption">
              Leaves hpl_profile_proportion/summary disagreeing with the new tile_registry values.
              Off unless you have a specific reason.
            </div>
          </Expander>

          <WriteMode
            value={writeMode}
            onChange={setWriteMode}
            name={`kb-load-write-mode-${submissionId}`}
          />

          {jobInFlight && (
            <Caption>
              A Knowledge Bank load is already in flight for this run — starting another would write
              the same tiles twice, from possibly different CSVs.
            </Caption>
          )}

          <Button
            kind="primary"
            onClick={handleCommit}
            disabled={committing || jobInFlight || report.would_refuse_low_match}
          >
            {committing
              ? writeMode === WRITE_SERVER
                ? "Committing…"
                : "Queueing…"
              : "Commit to Knowledge Bank"}
          </Button>
        </div>
      )}

      {commitMessage && <Alert type={commitMessage.type}>{commitMessage.text}</Alert>}
    </div>
  );
}
