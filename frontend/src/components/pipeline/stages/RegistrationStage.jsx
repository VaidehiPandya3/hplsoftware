// Port of _render_registration_step() in app_v28.py.
//
// This step exists because Stage 6 could not work without it and said so only
// obliquely. load_hpc_assignments.py exclusively UPDATEs tile_registry, so a
// cohort that has never been in the Knowledge Bank has nothing to update and
// the load refuses at a 0% match rate — a number that reads like a naming bug
// and is in fact a missing step.
//
// Same preview-then-commit shape as KbLoadStage, and for the same reason: it
// writes to the shared Knowledge Bank, and its failure mode is a registration
// that succeeds against the wrong cohort. The commit button is disabled
// outright whenever the preview reports a state the CLI's own guard would
// refuse — this UI must never offer a write register_dataset.py would reject.
import { useEffect, useState } from "react";
import { api } from "../../../api";
import { errorDetail, fmtInt } from "../utils";
import { Alert, Button, Caption, Expander, Field, Metric, RadioGroup } from "../widgets";
import { KbJobState, WRITE_SERVER, WRITE_SLURM, WriteMode } from "../kbWrite.jsx";

const TYPE_IT = "Other — type it";
const SCOPE_FULL = "full";
const SCOPE_SUBSET = "subset";

export default function RegistrationStage({ status, submissionId, state, onChanged }) {
  const defaultId =
    status.registration_dataset_id || (status.dataset_name || "").toUpperCase();
  const recordedDatasetName = (status.dataset_name || "").trim();

  const [datasetId, setDatasetId] = useState(defaultId);
  const [slideMetadata, setSlideMetadata] = useState(false);
  const [writeDatasetConfig, setWriteDatasetConfig] = useState(true);
  const [replace, setReplace] = useState(false);

  // The tile folder Stage 1 wrote into. Mandatory, and asked for even when the
  // run recorded it, because Stage 5's whole coordinate half is read out of
  // <tile_dir>/<this name>/. It is NOT dataset_id above: that is the cohort key
  // the KB groups by, this is a directory on disk. Conflating the two is what
  // made the refusal read as "I already told you the dataset name".
  const [tileFolders, setTileFolders] = useState(null); // null = still loading
  const [folderChoice, setFolderChoice] = useState(recordedDatasetName);
  const [typedFolder, setTypedFolder] = useState(recordedDatasetName);

  const [scope, setScope] = useState(SCOPE_FULL);
  const [subsetText, setSubsetText] = useState("");

  const [writeMode, setWriteMode] = useState(WRITE_SLURM);

  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [report, setReport] = useState(null);

  const [committing, setCommitting] = useState(false);
  const [commitMessage, setCommitMessage] = useState(null);
  const [jobInFlight, setJobInFlight] = useState(false);

  useEffect(() => {
    let live = true;
    api
      .getTileDatasetNames()
      // Server unreachable or the route missing — fall back to typing it rather
      // than blocking the step on a convenience lookup.
      .catch(() => [])
      .then((names) => {
        if (!live) return;
        const list = names || [];
        setTileFolders(list);
        setFolderChoice(list.includes(recordedDatasetName) ? recordedDatasetName : list[0] || TYPE_IT);
      });
    return () => {
      live = false;
    };
  }, [recordedDatasetName]);

  // A picker rather than free text wherever possible: the names come off disk,
  // so a typo cannot silently select a folder that does not exist.
  const usingPicker = Array.isArray(tileFolders) && tileFolders.length > 0;
  const tileDatasetName = (
    !usingPicker || folderChoice === TYPE_IT ? typedFolder : folderChoice
  ).trim();

  const slideNames = subsetText
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  /** Everything both calls send. Sent on preview as well as commit because the
   * two have to read the same folder and the same scope — numbers previewed
   * against one and committed against another describe a cohort the write did
   * not touch. */
  function requestArgs() {
    return {
      datasetId: datasetId.trim(),
      tileDatasetName,
      scope,
      slideNames: scope === SCOPE_SUBSET ? slideNames : null,
      slideMetadata,
      writeDatasetConfig,
      replace,
    };
  }

  /** The two refusals that belong to the form rather than to the server. */
  function formProblem() {
    if (!datasetId.trim()) {
      return "Enter a dataset_id — every row written is scoped to it.";
    }
    if (!tileDatasetName) {
      return (
        "Choose the tile folder (dataset_name) — Stage 1's per-slide metadata " +
        "is read from processed_tiles/<that folder>/."
      );
    }
    if (scope === SCOPE_SUBSET && !slideNames.length) {
      return "Enter at least one slide ID or filename for subset registration.";
    }
    return null;
  }

  async function handlePreview() {
    setPreviewError(null);
    const problem = formProblem();
    if (problem) {
      setPreviewError(problem);
      return;
    }
    setPreviewing(true);
    try {
      setReport(await api.previewRegistration(submissionId, requestArgs()));
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
    const problem = formProblem();
    if (problem) {
      setCommitMessage({ type: "error", text: problem });
      return;
    }
    setCommitting(true);
    try {
      if (writeMode === WRITE_SLURM) {
        const r = await api.submitRegistration(submissionId, requestArgs());
        setCommitMessage({
          type: "success",
          text:
            `Queued as Slurm job ${r.registration_job_id}. It writes to ` +
            `${r.database} whether or not this page stays open.` +
            (r.registration_log_path ? ` Log: ${r.registration_log_path}` : ""),
        });
      } else {
        const r = await api.commitRegistration(submissionId, requestArgs());
        const written = r.written || {};
        setCommitMessage({
          type: "success",
          text:
            "Registered: " +
            Object.entries(written)
              .map(([t, n]) => `${t} +${fmtInt(n)}`)
              .join(", "),
        });
      }
      setReport(null);
      if (onChanged) onChanged();
    } catch (e) {
      setCommitMessage({ type: "error", text: `Registration refused: ${errorDetail(e).message}` });
    } finally {
      setCommitting(false);
    }
  }

  if (state === "blocked") {
    return (
      <Alert type="info">
        Registration reads tile identity out of the packaged .h5, so it needs Stage 2 to have
        finished. It does not need Stages 3 or 4 — as soon as the .h5 is ready this can run, and
        Stage 6 will be waiting only on the assignment.
      </Alert>
    );
  }

  // Every reason the commit must not be offered, computed by the server so the
  // rule lives next to the guard that enforces it rather than being
  // reimplemented once per frontend.
  const blocked =
    !!report &&
    (report.would_refuse_collision ||
      report.needs_replace ||
      (report.missing_tables || []).length > 0);

  const problemLists = [
    ["ambiguous_slides", "slide id(s) match more than one file — neither is registered"],
    ["slides_without_files", "slide(s) have no raw file; their tiles register, the slide will not open"],
    ["unreadable_slides", "slide(s) could not be opened for metadata"],
    ["conflicting_samples", "slide(s) carry more than one sample_id in the .h5"],
    ["missing_slides", "slide(s) have no usable Stage 1 metadata; no coordinates for their tiles"],
  ];

  const tileFolderHelp =
    "The folder under processed_tiles that Stage 1 wrote this run's per-slide " +
    "_tile_metadata.csv files into. Every tile's x/y comes from there, so a " +
    "wrong name does not fail — it registers tiles with no coordinates.";

  return (
    <div>
      {status.registration_done && (
        <Alert type="success">
          Registered in the Knowledge Bank
          {status.registration_dataset_id ? ` as ${status.registration_dataset_id}` : ""}
          {status.registration_at ? ` — last registered ${status.registration_at}` : ""}
        </Alert>
      )}

      <KbJobState
        status={status}
        prefix="registration"
        label="Registration"
        onInFlight={setJobInFlight}
      />

      <Field
        label="Knowledge Bank cohort (dataset_id)"
        help={
          "Every row this writes is scoped to this key, and a replace only ever touches its own. " +
          "It defaults to the run's dataset_name but is a different thing: dataset_name is the " +
          "folder of slides on scratch, this is the cohort the KB groups by."
        }
      >
        <input
          type="text"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
        />
      </Field>

      {usingPicker ? (
        <>
          <Field label="Tile folder (dataset_name)" help={tileFolderHelp}>
            <select value={folderChoice} onChange={(e) => setFolderChoice(e.target.value)}>
              {[...tileFolders, TYPE_IT].map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </Field>
          {folderChoice === TYPE_IT && (
            <Field label="Tile folder name">
              <input
                type="text"
                value={typedFolder}
                placeholder="TCGA"
                onChange={(e) => setTypedFolder(e.target.value)}
              />
            </Field>
          )}
        </>
      ) : (
        <Field label="Tile folder (dataset_name)" help={tileFolderHelp}>
          <input
            type="text"
            value={typedFolder}
            placeholder="TCGA"
            onChange={(e) => setTypedFolder(e.target.value)}
          />
        </Field>
      )}

      {!recordedDatasetName ? (
        <Caption>
          This run recorded no tile folder — it predates that column, or was tiled outside the
          submit flow — so the choice above is the only thing that says where its Stage 1 metadata
          is.
        </Caption>
      ) : (
        tileDatasetName &&
        tileDatasetName !== recordedDatasetName && (
          <Alert type="warning">
            This run recorded its tiles under <code>{recordedDatasetName}</code>, not{" "}
            <code>{tileDatasetName}</code>. Registration reads the folder selected above — check it
            before previewing.
          </Alert>
        )
      )}

      <RadioGroup
        label="Registration scope"
        name={`reg-scope-${submissionId}`}
        options={[
          { value: SCOPE_FULL, label: "Full packaged dataset" },
          { value: SCOPE_SUBSET, label: "Subset" },
        ]}
        value={scope}
        onChange={setScope}
        help="Register every slide in the packaged .h5, or only selected slides from it."
      />

      {scope === SCOPE_SUBSET && (
        <Field
          label="Slides to register (one per line)"
          help={
            "Enter slide IDs or filenames that already belong to the packaged .h5. Only those " +
            "slides and their tiles will be registered."
          }
        >
          <textarea
            rows={4}
            value={subsetText}
            placeholder={"SLIDE-001\nSLIDE-002.svs\nSLIDE-003"}
            onChange={(e) => setSubsetText(e.target.value)}
          />
        </Field>
      )}

      <div className="pipeline-checkbox-row">
        <label>
          <input
            type="checkbox"
            checked={slideMetadata}
            onChange={(e) => setSlideMetadata(e.target.checked)}
          />
          Also read slide headers into wsi_metadata
        </label>
        <span className="pipeline-field-help">
          Opens every slide file to record mpp, objective power and level dimensions. Minutes, not
          seconds, on a large cohort.
        </span>
      </div>

      <div className="pipeline-checkbox-row">
        <label>
          <input
            type="checkbox"
            checked={writeDatasetConfig}
            onChange={(e) => setWriteDatasetConfig(e.target.checked)}
          />
          Write dataset_config
        </label>
        <span className="pipeline-field-help">
          Records this cohort's target_mpp and tile size from the run's own tiling_params. Skipped
          automatically for a run that predates that column.
        </span>
      </div>

      <div className="pipeline-checkbox-row">
        <label>
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace this cohort's existing rows
        </label>
        <span className="pipeline-field-help">
          Required to re-register a dataset_id that already has rows. Only ever deletes WHERE
          dataset_id = the key above; a tile or slide claimed by a different cohort is refused
          outright, not reassigned.
        </span>
      </div>

      <Button onClick={handlePreview} disabled={previewing}>
        {previewing ? "Previewing…" : "Preview registration"}
      </Button>
      {previewError && <Alert type="error">{previewError}</Alert>}
      {commitMessage && <Alert type={commitMessage.type}>{commitMessage.text}</Alert>}

      {!report && !previewError && (
        <Caption>
          Preview first — this writes to the shared Knowledge Bank, so it never commits without
          showing you the numbers.
        </Caption>
      )}

      {report && (
        <div>
          <div className="pipeline-metrics-row">
            <Metric label="Slides" value={fmtInt(report.slides || 0)} />
            <Metric label="Tiles in .h5" value={fmtInt(report.tiles_in_h5 || 0)} />
            <Metric label="With coordinates" value={fmtInt(report.tiles_with_coordinates || 0)} />
            <Metric label="Slides registered" value={fmtInt(report.slides_registered || 0)} />
          </div>

          {/* A zero here is the tile-folder mistake, not a packaging one: the
              tiles register and Stage 6 loads, and every tile has no x/y. */}
          {report.tiles_in_h5 > 0 && !report.tiles_with_coordinates && (
            <Alert type="error">
              None of these tiles would get coordinates. That is almost always the tile folder above
              pointing somewhere Stage 1 did not write — the .h5 itself is fine. Check it before
              committing: registering without coordinates succeeds, and the viewer then has nothing
              to place.
            </Alert>
          )}

          {(report.tile_names_normalized || 0) > 0 && (
            <Alert type="info">
              <code>.jpeg</code> was appended to {fmtInt(report.tile_names_normalized)} tile name(s)
              so they match the Knowledge Bank's <code>18_15.jpeg</code> form. The .h5 on disk still
              holds the short form.
            </Alert>
          )}

          {!report.slides_registered && (
            <Alert type="warning">
              No wsi_registry rows would be written — the run's raw slide directory could not be
              read. The tiles would register and Stage 6 would load, and the viewer would still 404
              on every slide in this cohort, because it resolves slide paths from wsi_registry
              alone.
            </Alert>
          )}

          {problemLists.map(([field, label]) => {
            const items = report[field] || [];
            if (!items.length) return null;
            return (
              <Expander key={field} title={`${fmtInt(items.length)} ${label}`}>
                {items.slice(0, 200).map((item, i) => (
                  <div key={i} className="pipeline-code">
                    {item}
                  </div>
                ))}
                {items.length > 200 && (
                  <Caption>…and {fmtInt(items.length - 200)} more</Caption>
                )}
              </Expander>
            );
          })}

          {(report.missing_tables || []).length > 0 && (
            <Alert type="error">
              This database has no {report.missing_tables.join(", ")}. Run{" "}
              <code>psql … -f backend/migrate_kb_base_tables.sql</code> first — eight of the
              Knowledge Bank's tables had no CREATE TABLE in git until that file existed.
            </Alert>
          )}
          {report.would_refuse_collision && (
            <Alert type="error">
              Refusing: some of these tiles or slides already belong to a different dataset_id. Two
              cohorts cannot claim the same tile, and overwriting would repoint the viewer at
              another cohort's files. This needs investigating, not overwriting.
            </Alert>
          )}
          {report.needs_replace && (
            <Alert type="warning">
              This dataset_id already has rows. Tick “Replace this cohort's existing rows” and
              preview again to overwrite them.
            </Alert>
          )}

          <WriteMode
            value={writeMode}
            onChange={setWriteMode}
            name={`reg-write-mode-${submissionId}`}
          />

          {jobInFlight && (
            <Caption>
              A registration job is already in flight for this run — starting another would write
              the same cohort's identity rows twice.
            </Caption>
          )}

          <Button
            kind="primary"
            onClick={handleCommit}
            disabled={blocked || committing || jobInFlight}
          >
            {committing
              ? writeMode === WRITE_SERVER
                ? "Registering…"
                : "Queueing…"
              : scope === SCOPE_SUBSET
                ? "Register subset in the Knowledge Bank"
                : "Register full dataset in the Knowledge Bank"}
          </Button>
        </div>
      )}
    </div>
  );
}
