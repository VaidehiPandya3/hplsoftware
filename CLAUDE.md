# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline and web UI for assigning **HPL histomorphological phenotype clusters (HPCs)** to H&E
whole-slide-image tiles, running on the Beatson HPC (Slurm). Whole slides are tiled, packaged into
HDF5, encoded by a frozen self-supervised model, classified by k-NN against a reference, and loaded
into a Postgres "Knowledge Bank" that the UI and chatbot read.

## Commands

```bash
# Tests. Every suite also runs standalone (no pytest), which is how they run on the cluster.
python3 -m pytest backend/tests -q
python3 backend/tests/test_extraction_guards.py      # standalone mode
python3 -m pytest backend/tests/test_kb_load.py::test_join_key_matches_the_registrys_format

# Services (both must restart to pick up changes; they are separate processes)
cd backend && python tile_server_v2_.py              # FastAPI on :8000
streamlit run app/app_v28.py                         # UI, talks to localhost:8000

# One-time cluster setup
python backend/submit_feature_extraction.py --bootstrap-extras   # installs container extras
python backend/build_hpc_reference.py --h5ad <leiden .h5ad> --out hpc_reference_leiden_2p5_fold2.npz
```

Dependencies: `backend/requirements.txt`. No lint or build step.

**Pick the highest version number.** `tile_server_v2_.py` (4.7k lines) is current despite the name —
`tile_server_v3.py`/`v4.py` are smaller, older experiments. `app/app_v28.py` is the current UI; v3–v27
are history. Confirm with `ls -lt` before editing.

## Architecture

Six stages, surfaced as numbered steps in the UI sidebar. Stage state lives in Postgres
(`slurm_dataset_runs`), and the UI gates each stage on the previous one having produced a *valid*
output, not merely having run.

```
1 Tiling            submit_mask_tile_slurm.py     → tiles + per-slide _tile_metadata.csv ON DISK
2 Packaging         make_hpl_hdf5.py              → one gzip HDF5 per dataset
3 Feature extract   submit_feature_extraction.py  → projections .h5 (GPU, Singularity)
4 Classification    submit_cluster_assignment.py  → assignments CSV (CPU, faiss k-NN)
5 Registration      register_dataset.py           → wsi_registry, wsi_metadata, dataset_config,
                                                    tile_coordinates, tile_registry (no hpc_id yet)
6 KB load           load_hpc_assignments.py       → tile_registry.hpc_id + per-slide aggregates
```

**Stage 1 does not write `tile_coordinates`.** It writes a per-slide `_tile_metadata.csv` to disk;
nothing puts those rows in Postgres until Stage 5. This file said otherwise until 2026-08-26 and the
mistake is worth naming, because it is what made the gap below invisible for so long.

Those name where each stage's *logic* lives; how it reaches Slurm is not uniform. Stages 3 and 4 have
dedicated `submit_*.py` modules that build and run their own `sbatch`. Stage 1 also has a shell
wrapper, `submit_dataset_tiling.sh`, which expects a `mask_and_tile_array.sbatch` alongside it that is
not in this repo — it lives on the cluster. Stage 2's submission is
driven by the server's `/package` endpoint, and `make_hpl_hdf5.package_slides_to_h5` is additionally
imported and called in-process on the single-slide upload path (see the one-slide run below). Stages 5 and 6 can run **either**
way: `register_dataset.py`'s and `load_hpc_assignments.py`'s functions run in-process inside
`tile_server_v2_.py` (`/register-preview`, `/register`, `/kb-load-preview`, `/kb-load`), and the same
two CLIs are submitted to Slurm by `submit_kb_write.py` via `/register-submit` and `/kb-load-submit`
(`migrate_dataset_runs_kb_slurm.sql` adds their job_id/state pair). The in-process path was never
lost to a closed browser — FastAPI runs a sync endpoint in a threadpool and uvicorn does not cancel it
on client disconnect — but it dies with the server, which is why the Slurm path exists. `registration_done`
and `kb_load_done` mean *committed* on both paths: on the Slurm path the **job** sets them, through
`--record-run` and `run_record.py`, because the job is the only process that knows.

**A single uploaded slide is a one-slide run, not a separate pipeline.**
`/upload-slide` creates its own `slurm_dataset_runs` row (`_start_upload_run`) and
`_run_postupload_pipeline` records Stages 1 and 2 against it as they finish in this process, so the
same UI stepper and the same Stage 3-6 endpoints carry an upload the rest of the way. Three things
make that work, and each is the kind that fails silently if changed back:

- **The sentinel job id.** Every later stage gates on "which Slurm state is this stage in", and an
  in-process stage has no job. It records `local:<stage>` instead (`LOCAL_JOB_ID_PREFIX`), which
  `_get_slurm_job_state` reads as COMPLETED — written only *after* the stage returned, and still
  subject to `_job_output_ready`'s validator, which opens the `.h5`. Every Slurm query helper strips
  these before asking, because sacct and squeue reject a whole call for one id they do not know: a
  sentinel reaching them would report "can't reach Slurm" for every real run listed beside it.
- **One cohort per uploaded slide,** `UPLOADED_<SLIDE_ID>` (`upload_dataset_name`), used as the
  `dataset_id`, the tile folder and the run's `dataset_name` at once. `register_dataset.commit()`
  scopes `--replace` to a `dataset_id` and DELETEs what it finds there, so the old shared `UPLOADED`
  bucket would have meant registering the second uploaded slide either refused or deleted the first.
  The prefix is what still keeps a bulk dataset folder from colliding with a slide_id, and it is why
  `_is_upload_dataset_id` is a prefix test — rows written before this carry the bare `UPLOADED`.
  Registering an upload always needs **Replace**, because the upload already wrote that slide's
  `wsi_registry` row so the viewer could open it immediately; the delete is scoped to that one slide.
- **The saved filename,** `{slide_id}_{uuid}_{original}`. Registration finds each raw slide through
  `slide_naming.slide_id_from_raw_path()`, and a name that does not parse is not an error — it
  registers a cohort with no `wsi_registry` row, and the viewer 404s on a slide sitting on disk.

`backend/tests/test_upload_pipeline.py` pins all three.

**New runs are one click: Stages 1-4 as one Nextflow run, like ANORAK.** The dataset panel in both
UIs takes one input — the dataset path — and two buttons, **Run HPL** and below it **Run ANORAK**,
either usable on its own; one shared "Test on a random subset" option (slide count + recorded seed)
applies to both. Below them: the latest HPL run, the latest ANORAK run, and a collapsed **History**
with every run (cancelled included). The panel uses no emoji — states are words (`[Done]`). The
HPL request carries only the path; everything else is the server's (`GET /pipeline-defaults`): tile
folder = the directory's name, `HPL_CHECKPOINT`, `HPC_REFERENCE_PATH`, the tuned vote,
`HPL_NF_MAX_TILING` (50) slides at a time, and earlier complete outputs at the run's paths *moved*
into `superseded-<stamp>/` beside them (never deleted; refused while any recorded job may be
writing them). Runs from before the pipeline are shown read-only under "Earlier runs"; their
per-stage buttons are gone. Upload runs keep their Stage 3/4 buttons — an uploaded slide does not
go through the pipeline — and Stages 5-7 keep theirs everywhere. `POST /pipeline-runs`
submits `hpl-nf/` through `submit_hpl_nf.py`: one
supervised head job (`head_sbatch_command()`, chain of standbys, and the watchdog at
`hpl-nf/tools/nf_supervise.sh`, a symlink to ANORAK's that submission refuses if it has drifted),
whose tasks run the per-stage wrappers in `hpl-nf/bin/` — which import and call the *same* tiler,
packager and container command builders from `backend/`, so there is one copy of the `--cleanenv`
rules. `hpl-nf/` is laid out like `anorak-nf/` (README, `bin/`, `conf/`, `tools/preflight.py`,
`tools/test_workflow_*.py`), and it and `backend/` must reach the cluster together. Every
refusal the per-stage submitters make is made up front by `resolve_run()`, before the queue. Stages 5
and 6 stay manual, with their dry runs. How the rest of the server sees such a run:

- Its four stage job-id columns hold `nf:<submission_id>:<stage>` sentinels (`hpl_nf_state.py`),
  stripped from every Slurm query exactly like `local:`, and resolved from the run directory
  (`HPL_NF_RESULTS_ROOT/<submission_id>`): COMPLETED **only** on `stages/<stage>.done.json`, which the
  stage's last task writes after the server's own validator (`stage_outputs.py`) accepts the output,
  bound to the previous stage's row count. A head job that exited 0 without a marker reads FAILED.
  Output paths are recorded at submission, so `_job_output_ready` still has the last word.
- No migration: the sentinel in `job_id` *is* the record that a run is a pipeline run.
- The per-stage endpoints refuse a pipeline run (`_refuse_if_pipeline_run`); recovery is
  `/pipeline-resume`, and each task skips an output that already validates — which is only safe
  because `refuse_foreign_outputs()` stops a fresh run from starting over another run's outputs.
- A shard retried after a crash clears its own half-written part before the encoder sees it —
  the encoder treats any existing output as done and crashes on it, and the plan step clears only
  before the *first* attempt. An unreadable slide stops the run unless it was started or resumed
  with `allow_incomplete`, in which case the tiling gate leaves it out of
  `manifest.packaged.txt` (what packaging reads) and names it in the marker. Resume reuses the
  Nextflow params recorded in `run_config.json` (`resume_params`), re-choosing only the GPU type.
- **Run ANORAK** (`POST /anorak-runs`) is its own run: a `slurm_dataset_runs` row whose status is
  `anorak_only` for life (it is the run's kind — cancel and errors do not overwrite it), submitted
  through Stage 7's own `_submit_anorak`. With no tumour-slide list it grades every slide ANORAK can
  read (`slide_list_from_directory`: slide_id = file stem, samples by HPL's
  `sample_from_slide_id`, `is_tumour = unverified`); only that server-built list may pass
  `tumour_verified=False`, and a list that says a slide is *not* tumour is refused regardless.
  `/anorak-resume` repeats a run from its own `slide_list.selection.json`.
- `hpl-nf/tools/test_workflow_*.py` make each per-stage guard fail and run real Nextflow in
  `-stub` mode; `backend/tests/test_hpl_nf_pipeline.py` covers the sentinels and the stepper.

**A Slurm-backed KB write needs Postgres reachable from a compute node,** and nothing about the
server's own connection tells you whether it is. `DB_HOST=127.0.0.1` means the compute node itself,
and a unix socket path is local to the database's machine however shared the filesystem is — so
`submit_kb_write.resolve_job_db_host()` refuses both at submit time and demands `HPL_JOB_DB_HOST`.
Check it once with `python backend/submit_kb_write.py --check-db` (or `GET /kb-job-db-check`), which
sruns a one-second TCP probe from an actual compute node.

**Stage 5 exists because Stage 6 only ever `UPDATE`s.** `load_hpc_assignments.load()` sets
`tile_registry.hpc_id` on rows that must already be there, so for a cohort that has never touched the
KB its match rate is 0% by construction and it refuses — a number that reads like a slide-naming bug
and is in fact a missing step. Registration is gated on Stage 2, not Stage 4: it reads tile identity
out of the packaged `.h5` and the raw slides and needs no cluster labels.

Adding a stage means touching four places: the submitter (or, for a non-Slurm stage like 5 or 6, the
in-process functions it calls), endpoints in `tile_server_v2_.py`, methods in `app/api_client.py`, and a
step entry + render function in `app/app_v28.py`. `backend/tests/test_pipeline_steps.py` checks the
last two agree — a step key with no renderer is a `KeyError` on a screen nobody opens until a run
reaches that stage. Stage 4 is the cleanest Slurm-backed model to copy —
see `_pipeline_steps` and `_render_assignment_step`. Stage 6 (`_render_kb_load_step`) is the model for a
stage that writes straight to the KB: it never auto-commits — a `/kb-load-preview` dry run has to be
pulled up in the UI first, and `/kb-load` enforces the exact same guards (95% match rate, unknown
cluster IDs) the CLI's `--commit` does, because it calls the same functions rather than reimplementing
them.

### Two repositories

`HPL-LATTICeA/` is a **git subtree** of `K-Rakovic/HPL-LATTICeA` (upstream `aec5145`, remote `hpl`),
holding Kai's frozen encoder. Our changes to it are ordinary commits **and** are mirrored in
`backend/patches/hpl-encode-io.patch` — because the HPC has its own separate clone of that repo at
`$HPL_REPO_DIR`, which the subtree does not update. Changing the encoder means updating both.
`backend/tests/test_encode_loop_equivalence.py` fails if a `git subtree pull` reverts the patch.

### The container

Stages 3 and 4 run inside an NGC TensorFlow 1.15 image (`tensorflow-23.03-tf1-py3.sif`, CUDA 12, Python
3.8) because the host conda env cannot drive a Hopper GPU. The image lacks packages the code needs
(`scikit-image`, `faiss-cpu`), and `$HOME` is not writable inside it, so they live in a bound
directory on scratch (`HPL_CONTAINER_EXTRAS`) placed on `PYTHONPATH`. **The package list lives in
`submit_feature_extraction.py`, so that file and `--bootstrap-extras` must always move to the cluster
together** — a stale copy silently installs the wrong set.

### The Knowledge Bank

Postgres `hpl_kb`, 26 relations — 17 tables, 1 view, 8 sequences. Column-level definitions are in
`backend/kb_live_schema_2026-08-26.txt`, transcribed from `\d` against the live database; that capture
is the only complete record of this schema, and `backend/migrate_kb_base_tables.sql` was written from
it. `backend/migrate_all.sql` builds 13 of the 17 tables from an empty database — verified by running
it against a real PostgreSQL — and stops at the four `hpc_*` reference tables, whose `\d` has never
been captured. **`schema.sql` in the repo root is a stale 2025-10-23 `pg_dump`** that declares
`tile_registry.hpc_id` as `varchar(100)` with an `id` primary key and no `slide_tile`, `dataset_id`, or
confidence columns. Do not build a database from it.

Three things matter for classification results:

- `tile_registry` — per-tile `hpc_id` plus confidence columns. Joined via
  `slide_tile`, which is `"<slides>_<tiles>"` upper-cased (`TCGA-55-7574-01Z-00-DX1_18_15.JPEG`).
- `hpl_profile_proportion` / `hpl_profile_summary` — per-slide aggregates **derived** from the same
  assignments. These are what the chatbot and HPC panels read, *not* `tile_registry`. Writing tiles
  without refreshing these leaves the UI internally inconsistent with nothing to signal it.
- `hpc_dictionary` and the `hpc_*_details` tables describe the 71 clusters themselves. Reference data;
  never derived from an assignment.

Which tables a run actually fills, and which nothing fills — audited 2026-08-26, full evidence in
`KB_TABLE_COVERAGE_2026-08-26.md`:

| filled by a run | never filled by anything |
|---|---|
| `tile_coordinates`, `tile_registry`, `wsi_registry`, `wsi_metadata`, `dataset_config` (Stage 5) · `hpl_profile_*`, `slide_hpc_membership` (Stage 6) · `slurm_dataset_run*` (the server) | `tile_hpc_heatmap` — **read live** at server startup, nothing writes it · `h_latent_vectors` (4.4 GB) · `tile_hpc_heatmap_old` |

`tile_hpc_heatmap` is the one gap with a live reader still open. `_load_heatmap_probs()`
(`tile_server_v2_.py:1332`) reads the whole 149 MB table into memory at startup and merges its 71
`p_hpc_*` columns into `/slide/{id}/tiles_meta`. Nothing in this repository has ever written it, and
the k-NN classifier does not produce a 71-class distribution to write — it produces a top-1 label and
a vote margin. Filling it is a modelling decision, not plumbing.

**A grep will not find every reader.** `app/hpc_chat_handlers_v23.py:334` enumerates the whole
database with `insp.get_table_names()`, keeps every table carrying an `hpc_id` or `dominant_hpc`
column — skipping only `hpc_dictionary` and `h_latent_vectors` — and renders up to five matching
rows straight to the user. So any such table is answered out of the chatbot without ever being
named. That is how `slide_hpc_membership` looked unreferenced while serving stale rows for cohorts
nobody was asking about; Stage 6 now refreshes it alongside the aggregates. Before concluding a
table is dead, check whether it has an `hpc_id` column.

`h_latent_vectors` is the one table with genuinely no live reader — it is in that skip list by name,
and nothing else touches it. Leave it alone rather than "completing" it.

## Invariants that cost time to rediscover

**Feature extraction is read-bound, not GPU-bound.** Each tile is its own gzip chunk, so a batch read
is one decompression per tile: ~1.4k tiles/s, below what any available GPU encodes. Consequences:
raising `--batch-size` buys almost nothing; h5py serialises HDF5 calls on a global lock so reader
threads do not parallelise decode (separate processes do — hence `--shards`); and falling back from an
H200 to an H100/A100 costs almost nothing in wall clock.

**`--cleanenv` means the container cannot read Slurm's variables.** Stages 3 and 4 run their
work inside `singularity exec --cleanenv`, which wipes the environment before the inner shell
starts — so a thread count written as `${SLURM_CPUS_PER_TASK:-1}` and expanded in there always
took the fallback, and every cluster assignment ran on one core while Slurm held 16. Nothing
failed and nothing warned: a 2.5M-row reference at 127 dims came to 49 tiles/s, and 18.5M tiles
took three days instead of hours. Thread counts are baked in at submit time now
(`_build_assignment_command(threads=...)`), which means the number and the sbatch's
`--cpus-per-task` live in different strings and must be changed together —
`test_assign_streaming.py` pins both.

The same door let a second one through: `SLURM_ARRAY_TASK_ID`, which the shard preamble used
to index its bounds arrays *inside* the container. Under `set -u` that aborted every array task
the moment the import check finished — stdout ending at "container packages: ok", the array
purged, and the merge left on `DependencyNeverSatisfied` with nothing in the log but the place it
stopped. The bounds are resolved outside the container now and passed in via
`SINGULARITYENV_`/`APPTAINERENV_`, which is the documented route through `--cleanenv`, and the
`set -u` check stays out there where an unset index really does mean a sharded command was
submitted as a plain job. `submit_feature_extraction.py` had the same pattern (`shard_preamble` inside `inner`) until
2026-09-28; it now resolves the range outside the container the same way.

And a third: `CUDA_VISIBLE_DEVICES`, which is how Slurm tells each `--gres=gpu:1` task which
physical card is its own. Stripped, every task on a node sees all of them and
`index_cpu_to_gpu(res, 0, ...)` puts them all on GPU 0 — N shards contending for one device
while the rest idle, failing at no point. Passed through the same way, defaulting to 0 so a
hand-run job outside Slurm still works.

Assume nothing the job needs from the submitting environment survives, and note that a string
assertion cannot catch this class — every broken version looked correct. The tests run the
generated shell against a stand-in that strips the environment the way `--cleanenv` does.

**GPU type names are cluster-specific and matching is exact.** This cluster has `nvidia_h200`,
`nvidia_h100_80gb_hbm3`, `nvidia_h100_pcie`, `nvidia_a100_80gb_pcie`. A preference list naming types
that do not exist is inert, not approximate — it silently falls through to queueing for the first one.
Check with `sinfo -p gpu -o "%N %G %t %D"`; note each type appears twice per node line.

**`--centering query` couples every tile to every other one.** It mirrors `sc.tl.ingest` by subtracting
the mean over *all* queries, so chunking or sharding the assignment changes the labels unless the mean
is computed once and shared (`--precompute-mean` / `--query-mean`). `project()` therefore refuses to
derive a mean from the chunk it was handed. Sharding without the shared mean is rejected outright.

**Stage 4 resumes; Stage 3 does not.** A preempted or requeued assignment task used to restart
its whole range. Chunks are checkpoints now: each is written under a `.tmp` name and renamed, so a
chunk file exists only if it is whole, and `<out>.chunks/` survives a kill while the `.partial`
does not. Resume is automatic and unconditional — a requeued job re-runs the identical command
line, so anything needing a flag would never happen on the attempt that needed it. The chunk
directory carries a manifest of everything that could change a label (reference, vote, centering,
chunk size, rep_key, row range, query mean) and **refuses** to resume across a difference, because
that is two computations concatenated into one CSV with nothing to say which rows came from where.
Device is deliberately excluded, so a preempted GPU shard can finish on a CPU; the manifest records
which devices contributed. The equivalence test is the point: a killed-and-resumed run must be
byte-identical to an uninterrupted one. Note the summary statistics are read back from the
assembled CSV after a resume, since the in-memory arrays only cover the chunks that attempt
computed.

**The encoder has no resume and mishandles its own leftovers.** It creates the output with `mode='w'`
before encoding, and its "output already exists" path crashes on an unbound local. So any interrupted
attempt makes every retry fail in seconds with an error pointing nowhere near the cause — which is why
`validate_extraction_output()` exists and stale outputs are cleared before resubmission.

**Short tile names are repaired on read, not refused — except when mixed.** A `.h5` or
assignments CSV packaged before make_hpl_hdf5.py stored the suffix holds `18_15`, which joins
nothing in a KB keyed on `..._18_15.JPEG`. `register_dataset.py` and `load_hpc_assignments.py`
append `.jpeg` themselves (`slide_naming.normalize_tile_names`) and report the count, because
`auto_tile_from_mask.py:150` writes every tile as `{col}_{row}.jpeg`, which makes the mapping a
bijection rather than a guess. Both sides must be normalised — the `.h5` *and* Stage 1's
metadata CSVs — or the refusal just becomes `tiles_with_coordinates: 0`. A **mixed** file (some
names suffixed, some not) is still refused: that is a resume that straddled the fix, the two
sides are indistinguishable by name, and appending would attach correct cluster IDs to the wrong
tiles. Note `tiles_missing_suffix()` cannot see that case — it samples 100 names and needs them
all short — which is why `tile_name_verdict()` reads every name. `migrate_tile_names.py` still
exists and is still the only fix for the artifacts on disk, and for mixed.

**Stage 6 writes with one statement, and the scratch table it joins is not KB
state.** The load used to send one `UPDATE ... WHERE UPPER(slide_tile) = :tile`
per tile through `executemany`, and the preview used to send 1,850 expanding-`IN`
queries of 10,000 keys each. At 18.5M tiles that is ~20M statement executions,
preceded by a Python list of 18.5M six-key dicts. Both are one statement now,
joined against a scratch table (`kb_stage.py`) that `COPY`s the CSV in on
several connections at once. The parallelism is only sound because the scratch
table is *not* Knowledge Bank state — a half-staged table is discarded and
remade, never repaired — which is why staging happens outside the write's
transaction and the write itself is still one transaction covering the tiles and
both aggregates. `stage_frame()` verifies the staged row count against the frame
before anything joins against it, because a scratch table short by a slice would
update a subset of the cohort and report success.

Three consequences worth knowing before touching it. A **duplicate
`slide_tile`** is now refused at read time: the old per-row writer applied
duplicates in file order and let the last win, whereas `UPDATE ... FROM` picks an
arbitrary match, so which one survived would be a property of the query plan.
An **empty `vote_margin`** is refused for the same reason — it used to be written
as `float('nan')` and `COPY`'s CSV format reads an empty field as NULL, so the two
writers disagreed on exactly those rows. And `--rebuild-hpc-index` is **off by
default**: dropping `idx_tr_hpc_id` for the duration is less total work, because
an indexed `hpc_id` means none of these updates can be HOT, but `DROP INDEX`
holds an ACCESS EXCLUSIVE lock on `tile_registry` for the whole transaction, so
the viewer and the chatbot block until the load commits.

`migrate_indexes.sql` §8 is still what makes any of this fast — the join
predicate is `UPPER(tile_registry.slide_tile)`, and without
`idx_tr_slide_tile_upper` the planner has only a sequential scan. And the write
leaves a dead row version per updated tile, roughly doubling `tile_registry` on
disk, which is why `VACUUM ANALYZE` runs afterwards (outside the transaction,
where it must be, and never fatally — the rows are committed by then).

**Tile pitch is a property of the slide, not of the deployment.** Tiles are tessellated at 1.8
µm/px, so their stride in native pixels is `round(224 * 1.8 / native_mpp)` — 1600 at 0.252 µm/px,
1734 at 0.2325, 804 on a 20x scan. Only 520 of the 1,598 slides in `wsi_metadata` are 0.252. The
tile server published that one number as `tile_size_native` for every slide, and every overlay in
both UIs sizes its rectangles from it, so the grid was drawn undersized on the 468 finer slides
(a visible gutter, up to 10.1% of the pitch) and at roughly twice the tile on the 610 coarser
ones. `_tile_size_native()` resolves it per slide now, preferring the pitch read back off
`tile_coordinates` (`x_native = col * pitch`, so the rows carry the stride Stage 1 actually used,
whatever defaults were in force) over re-deriving it from mpp, and `/slide/{id}/info` names which
in `tile_size_native_source`. `native_tile_px()` in `auto_tile_from_mask.py` is the single
definition — the tiler calls it and the server imports it, because a box has to land on the same
integer as the tile under it. The same constant was the grid `_compute_adjacency` indexed on,
where a wrong pitch does not collide cells but *skips* them: at 1734 every twelfth column is
empty and the two tiles either side of it stop being neighbours. It reads `col`/`row` now, which
the tiler already wrote. The pyramid viewer's hover/click hit test is the third reader of the
same number — it answers "which tile is under the pointer" as `floor(point / pitch)` against a
`col_row` map (`buildTileLookup`/`tileAtImagePoint`, and `build_osd_tile_index` on the Streamlit
side), so the wrong pitch there names the tile next door with no sign that it has.

**The ANORAK head job can go idle with nothing failing, so it runs Nextflow under a watchdog.**
Stage 7 (`submit_anorak_nf.py`) submits one head job running `nextflow run`, which submits a job
per slide per step itself. Nextflow notices finished tasks on a single "Task monitor" thread that
reads each task's `.exitcode` off CephFS. On 2026-09-23 one read, landing while another node was
still writing that file, blocked in the CephFS kernel client (`wait_woken`) and never returned:
every job after it finished in Slurm uncollected, all `queueSize` slots stayed "busy", and 7,161
tasks sat unsubmitted for five hours while the submitter thread kept the log growing. No Nextflow
setting can time out a read stuck below the JVM. `anorak-nf/tools/nf_supervise.sh` counts
`[Task monitor]` log lines — the monitor logs a summary every `executor.dumpInterval` while
anything runs — and after 30 min of silence stops Nextflow, `scancel`s this run's jobs (matched
by work directory) and restarts it with `-resume`. Watch the monitor's lines, not the log's size
or mtime: those looked alive the whole time. To see a live head job's threads,
`srun --jobid=<head> --overlap jstack <java pid>` (the JVM runs as `-jar nextflow-*-one.jar`, so
`pgrep -f nextflow.cli.Launcher` finds nothing). `test_anorak_watchdog.py` runs the real script.

The watchdog's first real restart (job 1250456, 2026-09-23) died in one second: the submitter
already passes `-resume`, the supervisor appended a second, and Nextflow refuses a repeated option.
Every test had built its own command and used a fake `nextflow` more permissive than the real one.
So the fakes now refuse what Nextflow refuses, a test restarts the command `build_nextflow_command()`
actually builds, and the same audit found three more of the kind: Nextflow rotates the `-log` file at
startup (the old count armed the watchdog before this attempt's monitor existed), a TERM near the
time limit waited forever on a wedged Nextflow, and `--dependency=afternotok` woke every chain
standby on any fast failure, including `scancel`. A final outcome now writes `nf_supervise.stop` in
the outdir, which every later head job reads and exits on; the submitter clears it on a fresh submit.

**ANORAK exit 65 is a refusal; everything else is retried.** Every `bin/` wrapper refuses through
`anorak_common.refuse()`, and `nextflow.config` finishes the run on 65 alone. The old list retried
only kill signals and finished on `Integer.MAX_VALUE` (a `.exitcode` Nextflow could not read —
routine on CephFS) and on 1. **Nextflow wipes the container's environment** (`env - ... singularity
exec`), so `CUDA_VISIBLE_DEVICES` crosses only via `singularity.envWhitelist` — the `--cleanenv`
lesson above, again. **Cache keys include code:** TILE_SLIDE, PREDICT_GP and SS1_STITCH take a digest
of their `bin/` scripts, the upstream AIgrading files they import, the checkpoint and the image,
because `python3 ${projectDir}/bin/x.py` is not hashed and Nextflow keys a container by path, not
bytes; so editing heavy-step code mid-cohort re-runs the cohort. `meta` is `[id, slide_name]` only —
the sample joins at SLIDE_PROPORTIONS, so relabelling a tumour re-grades without re-tiling.
`tools/test_workflow_cache.py` runs real Nextflow to pin each of these.

**Reference `.npz` keys are `reference, components, codes, categories, n_neighbors, meta` (+ optional
`mean`).** There is no `labels` key. `build_hpc_reference.save()` is the authority;
`test_reference_keys_match_the_builder` round-trips through it so readers cannot drift.

**`--device` defaults to `auto`, and `auto` never lowers the correctness bar.** The submitter
resolves it before any node exists, from the one observable it has — whether the GPU extras were
bootstrapped (`resolve_device()`) — and prints the choice with its reason. The job resolves its own
`auto` against the GPU actually in front of it and prints the fallback. `gpu` refuses rather than
falling back; `cpu` never tries. What no mode does is accept a GPU index that disagrees with the CPU
one: that means a broken build, which would produce a complete CSV of wrong cluster IDs whatever
asked for it. A `--gres=gpu:1` on a partition advertising no GPUs is also refused at submit time,
because it otherwise pends forever as `ReqNodeNotAvail` and reads like a busy queue.

**The GPU search is the same exact scan, and it is verified rather than trusted.** `--device gpu`
moves the flat index to GPU 0 (`GpuIndexFlat` compares every query against every reference vector,
exactly as `IndexFlat` does), which is why it is admissible where faiss-ivf was not. It needs a
*separate* extras directory — `HPL_CONTAINER_EXTRAS_GPU`, populated by
`submit_feature_extraction.py --bootstrap-extras-gpu` — because faiss-cpu and GPU faiss both import
as `faiss` and PYTHONPATH cannot hold both. `Searcher._verify_matches_cpu` searches a sample of the
reference against both indexes at startup and refuses on disagreement: a build can expose
`StandardGpuResources`, report `get_num_gpus() == 1`, accept `index_cpu_to_gpu`, and still be a stub
that returns well-formed nonsense. Measured on a working build, CPU and GPU agree 100% on the nearest
neighbour and ~99.99% across k=25, with near-ties reordering inside the list. `--device gpu` also
means the GPU partition, so pair it with `--shards`: a shard is the checkpoint that makes preemption
cost one task rather than the run.

**k-NN search is faiss-only, exact, with no backend choice.** `Searcher` in `assign_hpc_clusters.py`
requires `faiss` and always builds an exact flat index (`IndexFlatL2`) — there is no numpy fallback and
no approximate (`faiss-ivf`) option anymore. An approximate index was tried and measured against the
real reference: it agreed on the nearest neighbour only 33% of the time, for no speed gain at this
reference size, which would have put an approximation inside the one number this pipeline is judged on.
`faiss-cpu` is in `backend/requirements.txt` and in the container's `CONTAINER_EXTRAS`
(`submit_feature_extraction.py`), so it's expected to always be present; if it's missing, that's a setup
defect to fix, not something to route around.

## The failure mode this codebase is written against

Almost nothing here fails by crashing. A wrong reference, a naming mismatch, a shard with its own mean,
a half-merged output, a stale aggregate — each produces a file or table of the right shape and dtype
with no missing values, which passes every completeness check while every cluster ID is attached to the
wrong tile. This is why guards refuse before the queue rather than inside the job, why validators check
row counts against their source rather than believing the parts, why outputs are written under a
temporary name and renamed only when whole, and why `load_hpc_assignments.py` is dry-run by default and
refuses below a 95% match rate.

Two things follow for anyone extending this. Prefer a loud refusal at submit time over a plausible
result later. And when adding a test, make it prove the guard can *fail* — several bugs here were found
by checking that a validator could come out bad, not that it came out good.

## Validation

```bash
# Does k-NN recover the reference's own labels? Needs only the .npz.
python backend/validate_reference.py --reference hpc_reference_leiden_2p5_fold2.npz

# Full end-to-end: reproduce Kai's TCGA cluster transfer. Needs TCGA projections.
python backend/submit_cluster_assignment.py --projections-h5 <TCGA .h5> --out /tmp/check.csv \
  --validate-against TCGA_LUAD_5x_he_train_filtered_leiden_2p5__fold2.csv
```

The first covers the search, the vote and whether the clusters are k-NN-separable. It does **not**
cover projecting raw embeddings into the reference space or the centering choice — only the second
does. Below 99% agreement on the second is a defect, not drift.
