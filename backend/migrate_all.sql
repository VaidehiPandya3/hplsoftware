-- ============================================================
-- Every schema migration, in dependency order. Safe to re-run.
--
--   psql -h <socket> -d hpl_kb -f backend/migrate_all.sql
--
-- Why this exists. The migrations were applied to the live database by hand,
-- one at a time, and only four of the fifteen were ever committed. So a fresh
-- checkout got a *partial* schema history — which is worse than none, because
-- it looks complete. Concretely: load_hpc_assignments.py joins on
-- UPPER(TRIM(...)) because migrate_indexes.sql normalised those columns, and
-- migrate_indexes.sql was not in git. The code was tracked; the thing that
-- makes the code correct was not.
--
-- Why a plain include list and not a migration framework. Every one of these is
-- idempotent, so "which have been applied" is a question nobody has to answer:
--
--   * almost all statements are ADD COLUMN / CREATE INDEX IF NOT EXISTS;
--   * the one bare ADD COLUMN sits inside a DO $$ ... IF NOT EXISTS block
--     (migrate_indexes.sql, tile_registry.slide_tile);
--   * the one ADD PRIMARY KEY is preceded by DROP CONSTRAINT IF EXISTS
--     (migrate_dataset_runs_async.sql);
--   * the backfills are UPDATE ... SET x = UPPER(TRIM(x)), and
--     UPPER(TRIM(UPPER(TRIM(x)))) = UPPER(TRIM(x)).
--
-- So running this against an up-to-date database is a no-op, and running it
-- against a fresh one builds the whole schema. A version table would add a
-- second thing that can be wrong about what the database contains.
--
-- Adding a migration: append it below in dependency order, keep it idempotent,
-- and state its prerequisite in its own header comment the way the existing
-- ones do — the order here was reconstructed from those comments.
-- ============================================================

\set ON_ERROR_STOP on

-- First, because everything below indexes, alters or normalises these. Eight of
-- the live database's seventeen tables had no CREATE TABLE anywhere in this
-- repository, so this file's claim above — that it builds the whole schema —
-- was false: a fresh database stopped at migrate_indexes.sql's
-- `UPDATE tile_coordinates`, blaming an index migration for a missing table.
-- Still missing after this: the four hpc_* cluster reference tables and the
-- slide_metadata view. See the header of migrate_kb_base_tables.sql.
\echo '== base tables that nothing else creates =='
\ir migrate_kb_base_tables.sql

\echo '== slurm_dataset_runs: the table, then its identity change =='
\ir migrate_dataset_runs.sql
-- async replaces the primary key (job_id -> submission_id), so nothing that
-- adds columns should run before it.
\ir migrate_dataset_runs_async.sql

\echo '== per-stage tracking columns on slurm_dataset_runs =='
\ir migrate_dataset_runs_h5.sql
\ir migrate_dataset_runs_manual_stages.sql
\ir migrate_dataset_runs_dataset_name.sql
\ir migrate_dataset_runs_tiling_params.sql
\ir migrate_dataset_runs_test_packaging.sql
\ir migrate_dataset_runs_lineage.sql

\echo '== the append-only attempt history =='
\ir migrate_dataset_run_jobs.sql

\echo '== stages 4 and 5 =='
\ir migrate_dataset_runs_assignment.sql
\ir migrate_dataset_runs_kb_load.sql
\ir migrate_dataset_runs_assignment_vote.sql
-- Registration creates the identity rows Stage 5's UPDATE needs, so it is
-- tracked next to Stage 5 even though it runs before it.
\ir migrate_dataset_runs_registration.sql
-- Which KB each of those two stages wrote to. Run tracking stays in
-- production, so without this a run says it registered and not where.
\ir migrate_dataset_runs_kb_target.sql
-- Both of those stages now submit a Slurm job instead of writing inside the
-- request, so each needs the job_id/state pair the other stages have.
\ir migrate_dataset_runs_kb_slurm.sql

\echo '== slide processing status =='
\ir migrate_processing_status.sql

\echo '== tile_registry confidence columns =='
\ir migrate_tile_registry_confidence.sql

-- Last on purpose. It normalises casing across tile_coordinates,
-- tile_registry, wsi_registry and hpl_profile_*, and builds the indexes — both
-- of which want the final shape of those tables.
\echo '== casing normalisation and indexes =='
\ir migrate_indexes.sql

\echo '== done. Run ANALYZE; to refresh planner statistics. =='
