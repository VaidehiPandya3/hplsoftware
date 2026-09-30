// The two pieces Stages 5 and 6 share, because both can write the Knowledge
// Bank either inside the request or as a Slurm job.
//
// Ports of app_v28.py's _render_write_mode (:2819) and _render_kb_job_state
// (:2780). Kept in their own module rather than in widgets.jsx: these two know
// what a KB write is, where the widgets there deliberately do not.
import { Alert, Caption, CodeBlock, RadioGroup } from "./widgets.jsx";
import { SLURM_IN_FLIGHT } from "./utils.js";

export const WRITE_SLURM = "slurm";
export const WRITE_SERVER = "server";

const MODE_OPTIONS = [
  { value: WRITE_SLURM, label: "On Slurm (survives closing this page)" },
  { value: WRITE_SERVER, label: "In the server (waits here)" },
];

/** Where the write runs. Defaults to Slurm, as the Streamlit radio does by
 * listing it first — the durable choice should not be the one you have to
 * remember to pick. */
export function WriteMode({ value, onChange, name }) {
  return (
    <RadioGroup
      label="Run the write"
      name={name}
      options={MODE_OPTIONS}
      value={value}
      onChange={onChange}
      help={
        "Slurm is the durable one: the job keeps writing if you close the " +
        "browser or the tile server dies, and records the outcome on the run " +
        "itself. Running it in the server keeps the numbers in front of you, " +
        "but a killed server takes the write with it. Both run exactly the " +
        "same code with the same guards."
      }
    />
  );
}

/**
 * A Slurm-backed KB write's state. Returns null when there is nothing to show.
 *
 * Stages 5 and 6 can run either in the server or as a Slurm job. The job is the
 * durable one — it outlives this page and the server — so its state has to be
 * visible here, or a queued write looks exactly like nothing having happened,
 * which is the confusion this whole feature exists to end.
 *
 * `inFlight` is reported back through onInFlight so the caller can disable its
 * own submit button; the Streamlit version returns a bool from the render
 * function, which a component cannot do.
 */
export function KbJobState({ status, prefix, label, onInFlight }) {
  const jobId = status?.[`${prefix}_job_id`];
  const done = status?.[`${prefix}_done`];
  const jobState = status?.[`${prefix}_slurm_state`];
  const logPath = status?.[`${prefix}_log_path`];
  const error = status?.[`${prefix}_error`];

  const inFlight = Boolean(jobId) && !done && SLURM_IN_FLIGHT.has(jobState);
  if (onInFlight) onInFlight(inFlight);

  if (!jobId || done) return null;

  if (inFlight) {
    return (
      <div>
        <Caption>{label} job:</Caption>
        <CodeBlock>{jobId}</CodeBlock>
        <Alert type="info">
          Running on Slurm (state: {jobState}). This continues whether or not
          this page is open, and whether or not the tile server is running. The
          step turns green when the job records that it committed.
        </Alert>
        {logPath && <Caption>Log: {logPath}</Caption>}
      </div>
    );
  }

  // Not in flight and not done: it ended without committing. The job's own
  // error is worth more than the Slurm state, since every refusal this stage
  // makes is a sentence rather than an exit code.
  return (
    <div>
      <Caption>{label} job:</Caption>
      <CodeBlock>{jobId}</CodeBlock>
      <Alert type="warning">
        The last {label.toLowerCase()} job ended as:{" "}
        {jobState || "no Slurm record"}, without recording a commit.
      </Alert>
      {error && <Alert type="error">{error}</Alert>}
      {logPath && (
        <Caption>
          Log: {logPath} — a refusal inside the job is printed there in full.
        </Caption>
      )}
    </div>
  );
}
