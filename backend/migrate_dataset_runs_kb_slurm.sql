-- ============================================================
-- Stages 5 and 6 submitted to Slurm instead of run inside the request.
-- Run once on the HPCC PostgreSQL instance, after
-- migrate_dataset_runs_registration.sql and migrate_dataset_runs_kb_load.sql.
--
-- Both stages used to run in-process inside the server: /register and /kb-load
-- did the whole write during the HTTP request. That survived the UI closing
-- (FastAPI runs a sync endpoint in a threadpool and uvicorn does not cancel it
-- on client disconnect) but not the server process going away, which is what
-- actually happened — a killed server took an hours-long write with it and left
-- nothing on the run to say whether it had committed.
--
-- These columns give each stage the job_id/state pair every Slurm-backed stage
-- already has, so the work outlives both the browser and the server. The
-- existing registration_done / kb_load_done columns keep their meaning exactly:
-- the *job* sets them when it finishes, so "done" still means committed, never
-- "submitted".
--
-- *_submitted_at is what distinguishes "queued and still waiting" from "never
-- started" when Slurm has no record of the id — a job can age out of accounting
-- retention, and without a submission time there is nothing to date it against.
-- *_log_path is recorded because a job that fails inside the container leaves
-- its only explanation there, and hunting for it by job id afterwards is the
-- kind of detail nobody has when they need it.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_submitted_at TIMESTAMPTZ;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_log_path TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS registration_error TEXT;

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_submitted_at TIMESTAMPTZ;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_log_path TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS kb_load_error TEXT;
