-- ============================================================
-- Stage 7: ANORAK growth-pattern grading, run as a Nextflow pipeline.
-- Run once on the HPCC PostgreSQL instance, after
-- migrate_dataset_runs_kb_slurm.sql.
--
-- One Slurm job per run, and it is a head process rather than the work: it
-- submits a job per slide per stage itself. So anorak_job_id is the id to poll
-- for "is the pipeline alive", and it says nothing about how far through the
-- cohort it is — that is what the output directory and Nextflow's own trace
-- file are for, and anorak_out_dir is where both live.
--
-- anorak_scope / anorak_sample_size / anorak_seed record how the slide list was
-- chosen. A subset run is a *random* sample, and without the seed beside the
-- result a test run cannot be repeated and cannot be reconciled with a later
-- full run that disagreed with it. anorak_slide_list is the list as submitted,
-- written into the run's own directory, because the filtered CSV it came from
-- can be regenerated at a different threshold afterwards.
--
-- There is no anorak_done boolean. Unlike Stages 5 and 6 this stage commits
-- nothing to the Knowledge Bank — it writes files — so "done" is answerable by
-- reading the grading table, and a flag set separately from it could disagree
-- with it. See _validate_anorak_output in the tile server.
-- ============================================================

ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_job_id TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_submitted_at TIMESTAMPTZ;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_out_dir TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_slide_list TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_scope TEXT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_sample_size INTEGER;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_seed BIGINT;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_slides INTEGER;
ALTER TABLE slurm_dataset_runs ADD COLUMN IF NOT EXISTS anorak_error TEXT;
