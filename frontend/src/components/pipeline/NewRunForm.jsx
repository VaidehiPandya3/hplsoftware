// Port of _render_run_buttons() in app_v28.py: Run HPL, and below it Run
// ANORAK — each one click, each on its own.
//
// Nothing is asked for but the dataset path. HPL runs on the server's own
// settings (GET /pipeline-defaults), shown beside its button. ANORAK grades
// every slide in the directory unless a tumour-slide list is given. One shared
// option makes either a test run on a random, seeded subset.
import { useEffect, useState } from "react";
import { api } from "../../api";
import { httpDetail } from "./utils";
import { Alert, Button, Caption, Field } from "./widgets";

function useSubmit(submit, describe, onSubmitted) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState(null);
  async function run() {
    setBusy(true);
    setMessage(null);
    try {
      const result = await submit();
      setMessage({ type: "success", text: describe(result) });
      if (onSubmitted) onSubmitted(result);
    } catch (e) {
      setMessage({ type: "error", text: `Refused: ${httpDetail(e)}` });
    } finally {
      setBusy(false);
    }
  }
  return { busy, message, run };
}

export default function NewRunForm({ datasetPath, onSubmitted }) {
  const [defaults, setDefaults] = useState(null);
  const [defaultsError, setDefaultsError] = useState(null);
  const [subset, setSubset] = useState(false);
  const [sampleSize, setSampleSize] = useState(10);
  const [seed, setSeed] = useState("");
  const [slidesCsv, setSlidesCsv] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .getPipelineDefaults()
      .then((d) => !cancelled && setDefaults(d))
      .catch((e) => !cancelled && setDefaultsError(String(e.message || e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const path = (datasetPath || "").trim();
  const name = path.replace(/\/+$/, "").split("/").pop() || "?";
  const sample = subset ? Number(sampleSize) : null;
  const seedValue = subset && /^\d+$/.test(seed.trim()) ? Number(seed.trim()) : null;

  const hpl = useSubmit(
    () => api.startPipelineRun({ dataset_path: path, sample_size: sample, seed: seedValue }),
    (r) =>
      `HPL queued — run ${r.submission_id} for '${r.dataset_name}'. Its progress appears below.` +
      ((r.superseded || []).length ? ` Moved aside (nothing deleted): ${r.superseded.map((m) => m.from).join(", ")}.` : ""),
    onSubmitted,
  );
  const anorak = useSubmit(
    () =>
      api.startAnorakRun({
        datasetPath: path,
        slidesCsv: slidesCsv.trim() || null,
        sampleSize: sample,
        seed: seedValue,
      }),
    (r) =>
      `ANORAK queued — run ${r.submission_id}, ${(r.selection || {}).slides} slides. Its progress appears below.` +
      ((r.skipped_unsupported || []).length
        ? ` ${r.skipped_unsupported.length} .scn slide(s) left out: ANORAK cannot read that format.`
        : ""),
    onSubmitted,
  );

  if (!path) return <div className="pipeline-caption">Enter a dataset path above.</div>;

  return (
    <div>
      <Field
        label="Test on a random subset"
        help="Run on a random sample of the directory's slides instead of all of them — the same pipeline, fewer slides. The seed is recorded, so the same sample can be asked for again."
      >
        <input type="checkbox" checked={subset} onChange={(e) => setSubset(e.target.checked)} />
      </Field>
      {subset && (
        <>
          <Field label="Slides">
            <input type="number" min={1} value={sampleSize} onChange={(e) => setSampleSize(e.target.value)} />
          </Field>
          <Field label="Seed (optional)">
            <input type="text" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </Field>
        </>
      )}

      <h4 className="pipeline-subheading">Run HPL (tiling, packaging, feature extraction, classification)</h4>
      {defaults ? (
        <Caption>
          Tiles into <code>{defaults.tile_root}/{name}</code>, the .h5 into <code>{defaults.h5_root}/{name}</code> ·
          checkpoint <code>{defaults.checkpoint}</code> · reference <code>{defaults.reference}</code> ·{" "}
          {defaults.vote_preset} vote · {defaults.max_tiling} slides at a time · {defaults.min_tissue}% minimum
          tissue. Slides already tiled are reused; earlier outputs in the way are moved aside, never deleted.
        </Caption>
      ) : defaultsError ? (
        <Caption>Could not load HPL&apos;s settings ({defaultsError}); the server applies them anyway.</Caption>
      ) : null}
      <Button kind="primary" onClick={hpl.run} disabled={hpl.busy}>
        {hpl.busy ? "Checking and submitting…" : "Run HPL"}
      </Button>
      {hpl.message && <Alert type={hpl.message.type}>{hpl.message.text}</Alert>}

      <h4 className="pipeline-subheading">Run ANORAK (growth-pattern grading)</h4>
      <Field
        label="Tumour-slide list (optional)"
        help="select_tumour_slides.py's output, optionally filtered by filter_slides_by_tile_count.py. Blank grades every slide in the directory."
      >
        <input type="text" value={slidesCsv} onChange={(e) => setSlidesCsv(e.target.value)} />
      </Field>
      {!slidesCsv.trim() && (
        <Caption>
          Blank: every slide in the directory is graded, grouped into tumours by the part of the slide name before
          the first space. Tumour status is not checked, so non-tumour slides are graded too — give a tumour-slide
          list for a proper run.
        </Caption>
      )}
      <Button kind="primary" onClick={anorak.run} disabled={anorak.busy}>
        {anorak.busy ? "Checking and submitting…" : "Run ANORAK"}
      </Button>
      {anorak.message && <Alert type={anorak.message.type}>{anorak.message.text}</Alert>}
    </div>
  );
}
