-- ============================================================
-- PostgreSQL Index Migration for HPC Tile Server
-- Run once on the HPCC PostgreSQL instance to speed up queries.
--
-- The normalising UPDATEs below are guarded with `<> UPPER(TRIM(...))` rather
-- than only `IS NOT NULL`. Without that guard each one rewrites every row of
-- its table on every run, even when every value is already normalised — and
-- migrate_all.sql exists precisely so this file gets re-run. On the live
-- database that is ~360 MB of tile_registry and tile_coordinates rewritten to
-- change nothing, doubling both tables until the next VACUUM. With the guard a
-- re-run touches only rows that actually differ, which after the first run is
-- none.
-- ============================================================

-- 1. tile_coordinates — the most queried table
--    Normalise slides/slide_tile to uppercase once, then index plain columns.
UPDATE tile_coordinates SET slides = UPPER(TRIM(slides))
    WHERE slides IS NOT NULL AND slides <> UPPER(TRIM(slides));
UPDATE tile_coordinates SET slide_tile = UPPER(TRIM(slide_tile))
    WHERE slide_tile IS NOT NULL AND slide_tile <> UPPER(TRIM(slide_tile));

CREATE INDEX IF NOT EXISTS idx_tc_slides      ON tile_coordinates(slides);
CREATE INDEX IF NOT EXISTS idx_tc_slide_tile  ON tile_coordinates(slide_tile);
CREATE INDEX IF NOT EXISTS idx_tc_x_y_native  ON tile_coordinates(x_native, y_native);

-- 2. tile_registry — joined on slide_tile and queried by slides/hpc_id
--    Add slide_tile column if it doesn't exist yet.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tile_registry' AND column_name = 'slide_tile'
    ) THEN
        ALTER TABLE tile_registry ADD COLUMN slide_tile VARCHAR(250);
        UPDATE tile_registry SET slide_tile = UPPER(TRIM(slides || '_' || tiles));
    END IF;
END$$;

UPDATE tile_registry SET slides = UPPER(TRIM(slides))
    WHERE slides IS NOT NULL AND slides <> UPPER(TRIM(slides));
UPDATE tile_registry SET slide_tile = UPPER(TRIM(slide_tile))
    WHERE slide_tile IS NOT NULL AND slide_tile <> UPPER(TRIM(slide_tile));

CREATE INDEX IF NOT EXISTS idx_tr_slide_tile ON tile_registry(slide_tile);
CREATE INDEX IF NOT EXISTS idx_tr_slides     ON tile_registry(slides);
CREATE INDEX IF NOT EXISTS idx_tr_hpc_id     ON tile_registry(hpc_id);

-- 3. wsi_registry — looked up by slide_id on every request
UPDATE wsi_registry SET slide_id = UPPER(TRIM(slide_id))
    WHERE slide_id IS NOT NULL AND slide_id <> UPPER(TRIM(slide_id));
CREATE UNIQUE INDEX IF NOT EXISTS idx_wsi_slide_id ON wsi_registry(slide_id);

-- 4. hpl_profile_proportion — queried per slide
UPDATE hpl_profile_proportion SET slides = UPPER(TRIM(slides))
    WHERE slides IS NOT NULL AND slides <> UPPER(TRIM(slides));
CREATE INDEX IF NOT EXISTS idx_hpp_slides ON hpl_profile_proportion(slides);
CREATE INDEX IF NOT EXISTS idx_hpp_hpc_id ON hpl_profile_proportion(hpc_id);

-- 5. hpl_profile_summary — queried per slide and sample
UPDATE hpl_profile_summary SET slides = UPPER(TRIM(slides))
    WHERE slides IS NOT NULL AND slides <> UPPER(TRIM(slides));
UPDATE hpl_profile_summary SET samples = UPPER(TRIM(samples))
    WHERE samples IS NOT NULL AND samples <> UPPER(TRIM(samples));
CREATE INDEX IF NOT EXISTS idx_hps_slides  ON hpl_profile_summary(slides);
CREATE INDEX IF NOT EXISTS idx_hps_samples ON hpl_profile_summary(samples);

-- 6/7. hpc_dictionary and hpc_survival_analysis.
-- Guarded by table rather than stated flat, because these are two of the four
-- reference tables with no CREATE TABLE anywhere in git (see
-- migrate_kb_base_tables.sql's header). CREATE INDEX IF NOT EXISTS still raises
-- when the TABLE is missing, so on an empty database these two lines were the
-- last thing stopping migrate_all.sql from running to completion — found by
-- executing it, which nothing had done before 2026-08-26.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['hpc_dictionary', 'hpc_survival_analysis'] LOOP
        IF EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = current_schema() AND table_name = t) THEN
            EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (hpc_id)',
                           'idx_' || CASE t WHEN 'hpc_dictionary' THEN 'hd' ELSE 'hsa' END
                           || '_hpc_id', t);
        ELSE
            RAISE NOTICE '% absent; its index was not created.', t;
        END IF;
    END LOOP;
END$$;

-- 8. The expressions the previews actually filter on.
--
-- Every index above is on the bare column, and every lookup in
-- load_hpc_assignments.py and register_dataset.py filters on a *function* of it
-- — `WHERE UPPER(slide_tile) IN :tiles`, `WHERE UPPER(TRIM(slides)) IN :slides`.
-- Postgres cannot use a plain b-tree for that, so those queries were sequential
-- scans: Stage 6's preview chunks 18.5M keys 10,000 at a time, which is ~1,850
-- full scans of an 18.5M-row table for one dry run. That is the whole reason a
-- preview that writes nothing took longer than the write.
--
-- The normalising UPDATEs above mean UPPER(TRIM(x)) = x for every row they
-- touched, so these indexes are redundant *in content* — and load-bearing in
-- planning, because the query says UPPER() and the planner matches expressions,
-- not values. Indexing the expression rather than dropping UPPER() from the
-- query keeps rows that were written outside this pipeline (the original
-- notebooks) matching exactly as they do today.
CREATE INDEX IF NOT EXISTS idx_tr_slide_tile_upper
    ON tile_registry (UPPER(slide_tile));
CREATE INDEX IF NOT EXISTS idx_tc_slide_tile_upper
    ON tile_coordinates (UPPER(slide_tile));
-- The per-slide variants, used by the delete/replace paths and by Stage 6's
-- foreign-cohort check.
CREATE INDEX IF NOT EXISTS idx_tr_slides_upper
    ON tile_registry (UPPER(TRIM(slides)));
CREATE INDEX IF NOT EXISTS idx_tc_slides_upper
    ON tile_coordinates (UPPER(TRIM(slides)));
CREATE INDEX IF NOT EXISTS idx_wr_slide_id_upper
    ON wsi_registry (UPPER(slide_id));
-- slide_hpc_membership is deleted per slide on every Stage 6 refresh, with
-- `WHERE UPPER(TRIM(slide_id)) IN :slides`. Nothing normalises this table's
-- slide_id — the UPDATEs above cover tile_coordinates, tile_registry,
-- wsi_registry and the profiles, not this one — so the TRIM in that query is
-- load-bearing, and the index has to be on the same expression rather than the
-- query being simplified to match a plainer index. Its primary key leads with
-- the bare column, which this expression cannot use.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = current_schema()
                 AND table_name = 'slide_hpc_membership') THEN
        CREATE INDEX IF NOT EXISTS idx_shm_slide_id_upper
            ON slide_hpc_membership (UPPER(TRIM(slide_id)));
    ELSE
        RAISE NOTICE 'slide_hpc_membership absent; its index was not created.';
    END IF;
END$$;
-- wsi_metadata is checked by the same collision guard. Guarded by existence:
-- it is one of the tables migrate_kb_base_tables.sql adds, and CREATE INDEX
-- still raises when the table is absent.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = current_schema()
                 AND table_name = 'wsi_metadata') THEN
        CREATE INDEX IF NOT EXISTS idx_wm_slide_id_upper
            ON wsi_metadata (UPPER(slide_id));
    ELSE
        RAISE NOTICE 'wsi_metadata absent; its index was not created.';
    END IF;
END$$;

-- Done. Run `ANALYZE;` after to refresh planner statistics.
ANALYZE;
