// Port of render_dataset_job_panel() plus the sidebar wiring at the bottom of
// app_v28.py (render_wsi_upload_panel() + render_dataset_job_panel()).
//
// PipelinePanel is the only export the rest of the app needs — it is fully
// self-contained: upload panel, the dataset path box, the 10s-polled
// dataset workspace, and — first, always open — the one-click "Run the
// pipeline" form (see _render_dataset_submit_section).
import { useEffect, useState } from "react";
import { findExistingJobForPath } from "./utils";
import { Expander } from "./widgets";
import UploadPanel from "./UploadPanel";
import DatasetWorkspace from "./DatasetWorkspace";
import NewRunForm from "./NewRunForm";
import "./pipeline.css";

// Run the pipeline (Stages 1-4) — one click, first in the panel. Port of
// _render_dataset_submit_section: it used to sit below the dataset view, and
// behind an opt-in expander once the path had runs, to guard against a second
// full re-tiling. A pipeline run cannot do that — slides already tiled on disk
// are skipped — and what it can collide with (another run's finished .h5) the
// server refuses, with the choice to move those outputs aside.
function SubmitSection({ datasetPath }) {
  const [existingJob, setExistingJob] = useState(undefined); // undefined = loading
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setExistingJob(undefined);
    findExistingJobForPath(datasetPath).then((job) => {
      if (!cancelled) setExistingJob(job);
    });
    return () => {
      cancelled = true;
    };
  }, [datasetPath, refreshTick]);

  return (
    <div className="pipeline-run-card">
      {datasetPath && existingJob && (
        <div className="pipeline-caption">
          This path already has runs (newest {existingJob.submitted_at || ""}), in History below. A new HPL run
          reuses every slide already tiled. If a run is still going, follow or resume it below instead of
          starting another.
        </div>
      )}
      <NewRunForm datasetPath={datasetPath} onSubmitted={() => setRefreshTick((t) => t + 1)} />
    </div>
  );
}

export default function PipelinePanel() {
  const [datasetPath, setDatasetPath] = useState("");

  return (
    <div className="pipeline-panel">
      <UploadPanel />

      <Expander title="Process a dataset" defaultOpen={false}>
        <div className="pipeline-field">
          <span className="pipeline-field-label">Dataset path</span>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="/mnt/cephfs-lts/long-term-scratch/users/vpandya/Radiogenomics"
          />
          <span className="pipeline-field-help">
            Full absolute path on the HPC filesystem. Enter one to see everything already done to
            it; leave empty to browse what&apos;s on record.
          </span>
        </div>

        {/* The one-click pipeline first, as ANORAK's button is first in its
            step; what has already been done to the dataset follows. */}
        <SubmitSection datasetPath={datasetPath.trim()} />

        <hr className="pipeline-divider" />

        <DatasetWorkspace path={datasetPath.trim()} />
      </Expander>
    </div>
  );
}
