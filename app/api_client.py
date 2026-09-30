"""
API client for the HPC Tile Server.

Streamlit imports this instead of calling OpenSlide / psycopg2 directly.
Every image method returns a PIL Image; every metadata method returns a
dict or DataFrame. A local disk cache avoids repeated network fetches.

Usage in Streamlit:
    from api_client import TileServerClient
    client = TileServerClient("http://localhost:8000")
    thumb = client.get_thumbnail("TCGA-55-7574-01Z-00-DX1")
"""

import io
from typing import Optional

import pandas as pd
import requests
from PIL import Image

from local_cache import LocalImageCache


#: Read timeout for the four Knowledge Bank endpoints, which run their whole
#: job in-process inside the request rather than handing it to Slurm. On a real
#: cohort that is minutes to hours: registration opens the packaged .h5, reads a
#: per-slide metadata CSV for every slide, and queries the KB for collisions,
#: and with "Also read slide headers" ticked it opens every slide file as well.
#: The 30s default turned that into a read timeout that looks like a failure
#: while the server is still working — and, worse, on /register and /kb-load the
#: work carries on and commits after the client has given up, so the UI reports
#: an error over rows that are now in the KB. A day is a deliberate ceiling
#: rather than an estimate: nothing here should come close, and the cost of
#: setting it too low is a false failure on a real write, while the cost of
#: setting it too high is only that a genuinely wedged request has to be killed
#: by restarting the client. Note requests treats this as a *read* timeout —
#: time waiting for bytes, not total duration — and these endpoints send nothing
#: until they finish, so for them the two are the same thing.
KB_REQUEST_TIMEOUT = 24 * 60 * 60


class TileServerClient:
    #: Knowledge Bank the server should read and write. "production" is hpl_kb,
    #: "test" is hpl_kb_test.
    PRODUCTION = "production"
    TEST = "test"

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30,
                 kb_target: str = "production"):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache = LocalImageCache()
        self._session = requests.Session()
        # Held on the client rather than passed to each method. Roughly a dozen
        # endpoints honour it and app_v28 calls them from far more places than
        # that; threading an argument through every call site is how one of
        # them ends up reading production while the rest read test, which is
        # the failure this whole feature exists to avoid.
        self.kb_target = kb_target

    def set_kb_target(self, kb_target: str) -> None:
        """Point this client at a different Knowledge Bank.

        Clears the image cache: it is keyed by slide_id, and a slide_id only
        means one thing within a single KB. Without this, switching to test
        would show production's cached tiles for any id present in both.
        """
        if kb_target != self.kb_target:
            self.kb_target = kb_target
            try:
                self.cache.clear()
            except Exception:
                # A cache that will not clear is a stale-image problem, not a
                # reason to refuse the switch.
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None, stream: bool = False, timeout: int | None = None):
        url = f"{self.base_url}{path}"
        # Sent on every GET. Endpoints that do not declare it ignore it — the
        # pipeline and run-tracking routes are production-only by design — so
        # this cannot make one of them read the wrong database, and it removes
        # the need to remember which reads are KB reads.
        params = {**(params or {}), "kb_target": self.kb_target}
        r = self._session.get(url, params=params, timeout=timeout or self.timeout, stream=stream)
        r.raise_for_status()
        return r

    def _get_json(self, path: str, params: dict | None = None, timeout: int | None = None) -> dict | list:
        return self._get(path, params, timeout=timeout).json()

    def _post_json(self, path: str, json_body: dict, timeout: int | None = None,
                   params: dict | None = None) -> dict | list:
        r = self._session.post(
            f"{self.base_url}{path}", json=json_body, params=params,
            timeout=timeout or self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _get_image(self, path: str, params: dict | None = None,
                   cache_key: tuple | None = None) -> Image.Image:
        if cache_key:
            cached = self.cache.get_image(*cache_key)
            if cached:
                return cached

        r = self._get(path, params, stream=True)
        data = r.content
        img = Image.open(io.BytesIO(data)).convert("RGB")

        if cache_key:
            self.cache.put_bytes(data, *cache_key)
        return img

    # ------------------------------------------------------------------
    # Health / slide list
    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._get_json("/health")

    def list_slides(self) -> list[str]:
        return self._get_json("/slides")["slides"]

    # ------------------------------------------------------------------
    # Upload endpoints
    # ------------------------------------------------------------------

    def upload_slide(self, uploaded_file, slide_id: str | None = None,
                      confirm_overwrite: bool = False) -> dict:
        """Upload a WSI file to the FastAPI tile server.

        The server queues tissue masking + tiling as a background task right
        after saving the file; poll get_processing_status() for progress.

        Raises requests.HTTPError on any non-2xx response, same as every
        other method here — including a 409 when slide_id already exists
        from a previous upload. That 409's body is a structured dict
        ({"error": "slide_id_exists", ...} or {"error": "slide_id_conflict",
        ...}), not a plain message, specifically so a caller can catch
        requests.HTTPError, inspect e.response.json()["detail"]["error"],
        and — only for "slide_id_exists" — offer the user a confirmation
        before retrying with confirm_overwrite=True. Silently retrying here
        would defeat that; this method never overwrites without the caller
        explicitly asking it to.
        """
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                getattr(uploaded_file, "type", None) or "application/octet-stream",
            )
        }
        data = {
            "slide_id": (slide_id or "").strip(),
            "confirm_overwrite": "true" if confirm_overwrite else "false",
        }

        r = self._session.post(
            f"{self.base_url}/upload-slide",
            files=files,
            data=data,
            timeout=max(self.timeout, 600),
        )
        r.raise_for_status()
        return r.json()

    def get_processing_status(self, slide_id: str) -> dict:
        """Background mask/tiling status for a slide: queued/masking/tiling/done/error."""
        return self._get_json(f"/slide/{slide_id}/processing-status")

    # ------------------------------------------------------------------
    # Dataset-wide Slurm jobs
    # ------------------------------------------------------------------

    def get_dataset_roots(self) -> list[str]:
        """Top-level directories under long-term-scratch, for reference only."""
        return self._get_json("/dataset-roots")["datasets"]

    def get_tile_dataset_names(self) -> list[str]:
        """Existing dataset folders under processed_tiles (e.g. TCGA,
        Radiogenomics), so the UI can offer reusing one instead of everyone
        retyping the name by hand."""
        return self._get_json("/tile-dataset-names")["dataset_names"]

    def submit_dataset_job(
        self,
        dataset_path: str,
        max_concurrent: int = 10,
        min_tissue: float | None = 30.0,
        sample_size: int | None = None,
        slide_names: list[str] | None = None,
        partition: str | None = None,
        notify_email: str | None = None,
        dataset_name: str | None = None,
        tiling_params: dict | None = None,
    ) -> dict:
        """Queues the job and returns immediately with a submission_id to
        poll via get_dataset_job_status — slide discovery + sbatch submission
        happen server-side in the background, not within this request.

        dataset_name is the folder tiles land in under processed_tiles (e.g.
        "TCGA" or "Radiogenomics"). Leave it None to fall back to
        dataset_path's own folder name.

        tiling_params reproduces an earlier run's tiling exactly (as returned by
        that run's status under the same key). Pass min_tissue=None alongside
        it: the server treats an explicitly-sent min_tissue as an override of
        the block, and it cannot distinguish a deliberate 30.0 from this
        argument's default unless the key is absent from the request entirely.
        """
        body = {
            "dataset_path": dataset_path,
            "max_concurrent": max_concurrent,
            "sample_size": sample_size,
            "slide_names": slide_names,
            "partition": partition,
            "notify_email": notify_email,
            "dataset_name": dataset_name,
        }
        # Omitted rather than sent as null, so it stays out of the server's
        # model_fields_set and leaves tiling_params authoritative.
        if min_tissue is not None:
            body["min_tissue"] = min_tissue
        if tiling_params:
            body["tiling_params"] = tiling_params
        return self._post_json("/dataset-jobs", body)

    def get_pipeline_defaults(self) -> dict:
        """The settings a one-click run uses — all the server's own."""
        return self._get_json("/pipeline-defaults")

    def start_pipeline_run(
        self,
        dataset_path: str,
        checkpoint: str | None = None,
        *,
        dataset_name: str | None = None,
        max_concurrent: int | None = None,
        min_tissue: float | None = None,
        sample_size: int | None = None,
        slide_names: list[str] | None = None,
        seed: int | None = None,
        partition: str | None = None,
        notify_email: str | None = None,
        model: str = "BarlowTwins_3",
        extraction_shards: int = 1,
        reference: str | None = None,
        vote_preset: str | None = None,
        vote_overrides: dict | None = None,
        assignment_shards: int = 1,
        device: str = "auto",
        chain: int | None = None,
        time_limit: str | None = None,
        allow_incomplete: bool = False,
        move_existing_outputs: bool = True,
    ) -> dict:
        """One click: Stages 1-4 as a single Nextflow run (POST /pipeline-runs).

        Every input a later stage used to ask for at its own button is sent
        here, once, and the server refuses before queueing anything if one is
        wrong — a checkpoint typo is a 400 now, not a failed GPU task after
        hours of tiling. Returns a submission_id polled through
        get_dataset_job_status like any run; its `pipeline` block says where
        each stage is. Registration and the KB load stay manual.
        """
        body = {
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "max_concurrent": max_concurrent,
            "sample_size": sample_size,
            "slide_names": slide_names,
            "seed": seed,
            "partition": partition,
            "notify_email": notify_email,
            "checkpoint": checkpoint,
            "model": model,
            "extraction_shards": extraction_shards,
            "reference": reference,
            "assignment_shards": assignment_shards,
            "device": device,
            "chain": chain,
            "time_limit": time_limit,
            "allow_incomplete": allow_incomplete,
            "move_existing_outputs": move_existing_outputs,
        }
        if min_tissue is not None:
            body["min_tissue"] = min_tissue
        if vote_preset:
            body["vote_preset"] = vote_preset
        body.update({k: v for k, v in (vote_overrides or {}).items() if v is not None})
        # Unset means "the server's default": dropped rather than sent as null,
        # so a one-click call carries only the path.
        body = {k: v for k, v in body.items() if v is not None}
        return self._post_json("/pipeline-runs", body)

    def start_anorak_run(self, dataset_path: str, *, slides_csv: str | None = None,
                         sample_size: int | None = None, seed: int | None = None) -> dict:
        """One click: ANORAK on its own over a dataset path (POST /anorak-runs).

        Without slides_csv every slide in the directory is graded, grouped into
        tumours by HPL's slide-id rule, and recorded as tumour-unverified. With
        one (select_tumour_slides.py's output) it is checked as Stage 7 checks
        it. sample_size makes it a test run on a random, seeded subset.
        """
        body = {"dataset_path": dataset_path, "slides_csv": slides_csv,
                "sample_size": sample_size, "seed": seed}
        return self._post_json("/anorak-runs", {k: v for k, v in body.items() if v is not None},
                               timeout=300)

    def resume_anorak_run(self, submission_id: str) -> dict:
        """Resubmit a stopped ANORAK run exactly as it was, with -resume."""
        return self._post_json(f"/dataset-jobs/{submission_id}/anorak-resume", {}, timeout=300)

    def resume_pipeline_run(self, submission_id: str, chain: int | None = None,
                            time_limit: str | None = None,
                            allow_incomplete: bool | None = None) -> dict:
        """Resubmit a stopped pipeline run with -resume: re-runs only what did
        not finish, with the run's own recorded settings."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/pipeline-resume",
            {"chain": chain, "time_limit": time_limit,
             "allow_incomplete": allow_incomplete},
        )

    def check_pipeline_submit(self, partition: str | None = None) -> dict:
        """Can a compute node run sbatch? The head job submits every task."""
        return self._get_json("/pipeline-submit-check",
                              params={"partition": partition} if partition else None,
                              timeout=180)

    def list_dataset_jobs(self, with_state: bool = False) -> list[dict]:
        """Recent dataset runs, newest first.

        with_state=True adds "stage" and "slurm_state" (running / pending /
        complete / failed / no record / unknown) per run. The server resolves
        the whole list in one sacct call, so this costs one cluster round-trip
        for the list rather than one per row — but it is a cluster round-trip,
        hence the longer timeout.
        """
        if not with_state:
            return self._get_json("/dataset-jobs")
        return self._get_json("/dataset-jobs", params={"with_state": "true"},
                              timeout=60)

    def list_datasets(self) -> dict:
        """Every dataset, with all of its runs rolled up into one pipeline state.

        The dataset-level counterpart to list_dataset_jobs(), which lists runs.
        Worth having both because a resume does not continue a run — it starts a
        new one — so a dataset that took three resumes to tile is four rows in
        the job list, none of which can say whether the dataset is finished.
        Here they arrive as one entry with one set of three step states and one
        next_action naming the single thing left to do.

        Pollable: the server answers it with two queries and a bounded Slurm
        lookup — squeue for every job, sacct only for recent ones and only
        within a time budget. Longer timeout than a plain query because even
        bounded, that is a cluster round-trip.

        Returns the whole payload rather than just the list, because
        slurm_states_complete belongs to the response as a whole: when it is
        False some jobs were answered by the live queue alone, so a recent
        failure can be showing as merely "no longer queued". Callers that
        display these states must pass that on.
        """
        return self._get_json("/datasets", timeout=60)

    def get_dataset_coverage(self, dataset_name: str, raw_dir: str) -> dict:
        """One dataset's rollup with the authoritative on-disk tiling count.

        Separate from list_datasets() because of what it costs: the server
        stats two files per slide across the whole directory over cephfs, which
        is minutes on a 14,000-slide dataset. Call it when someone opens a
        dataset or asks "how much is actually tiled", never on the poll.

        This is the only thing that can confirm tiling is genuinely finished.
        Without it the rollup reports what Slurm says about the runs' own
        manifests, and a directory tiled only by subset runs can have every job
        COMPLETED while most of it sits untouched.
        """
        return self._get_json(
            "/datasets",
            params={
                "dataset_name": dataset_name,
                "raw_dir": raw_dir,
                "coverage": "true",
            },
            timeout=300,
        )["datasets"]

    def get_dataset_job_history(self, submission_id: str) -> dict:
        """Every Slurm job this run has submitted, all stages, newest first.

        Separate call from get_dataset_job_status because it costs an sacct
        round-trip and answers a different question: status says what the run
        can do next, this says what it has already tried — including repackaging
        and repeated test runs, which the status fields overwrite.
        """
        return self._get_json(f"/dataset-jobs/{submission_id}/jobs", timeout=60)

    def get_dataset_job_status(self, submission_id: str) -> dict:
        # Longer than the default timeout — this endpoint runs sacct
        # against the cluster's accounting DB, which can take a while for
        # a large dataset's array jobs even after batching it into one
        # call server-side (see tile_server_v2_'s _get_slurm_array_state_counts).
        return self._get_json(f"/dataset-jobs/{submission_id}/status", timeout=60)

    def resume_dataset_job(self, submission_id: str) -> dict:
        """Finds whatever slides from this run never got tiled (checked
        against the filesystem) and queues a new submission for just those."""
        return self._post_json(f"/dataset-jobs/{submission_id}/resume", {})

    def cancel_dataset_job(self, submission_id: str) -> dict:
        """scancels every Slurm job for this run (all tiling batches, the
        packaging job, and the feature-extraction job, if any) and marks it
        cancelled."""
        return self._post_json(f"/dataset-jobs/{submission_id}/cancel", {})

    def get_tiled_coverage(self, submission_id: str) -> dict:
        """How much of this run's raw directory has tiles on disk right now.

        Separate from get_dataset_job_status() on purpose — it stats two files
        per slide over the network filesystem, so it must not ride along on the
        10s status poll. Call it when the user is deciding what to package.
        """
        return self._get_json(f"/dataset-jobs/{submission_id}/tiled-coverage")

    def start_packaging_job(
        self,
        submission_id: str,
        allow_incomplete: bool = False,
        scope: str = "run",
        resume: bool | None = None,
    ) -> dict:
        """User-triggered: package this run's tiles into a .h5 now. Only
        valid once tiling has been submitted; the server still adds its own
        Slurm --dependency on the tiling jobs as a safety net.

        Also resumes: if a previous attempt left a .partial and its
        checkpoint on disk, the packaging job picks up from there rather
        than re-decoding everything. Call get_packaging_progress() first to
        tell the user which of the two is about to happen.

        allow_incomplete=False means the server refuses (400
        "tiling_incomplete") when any tiling task failed, because the
        packaging job's Slurm dependency is afterok and could never be
        satisfied. Pass True to package the dataset with those slides
        missing — a deliberate choice, hence not the default.
        """
        params = {
            "allow_incomplete": "true" if allow_incomplete else "false",
            # "run" packages this run's own manifest; "tiled" packages every
            # slide in the raw directory that has tiles on disk right now,
            # whichever run produced them.
            "scope": scope,
        }
        # Omitted when None so the server keeps its "continue a checkpoint if
        # one exists" fallback for non-interactive callers. The UI always sends
        # True or False, because resuming silently is the behaviour this
        # parameter exists to remove.
        if resume is not None:
            params["resume"] = "true" if resume else "false"
        return self._post_json(
            f"/dataset-jobs/{submission_id}/package", {}, params=params,
        )

    def get_packaging_progress(self, submission_id: str, exact: bool = True) -> dict:
        """How far a packaging attempt has got: state, tiles_done / tiles_total,
        bytes written, time since the last write, and whether it's resumable.

        Not part of get_dataset_job_status() on purpose — with exact=True it
        counts lines in a checkpoint file that can be hundreds of MB, so that
        form is meant to be called when the user opens the packaging step, not
        on every poll.

        exact=False is the pollable form: tiles_done is estimated from the
        .partial's size (one stat, since every tile occupies an identical number
        of bytes) and comes back flagged as tiles_done_is_estimate. Use it for
        live progress; use exact=True when the number drives a decision, such as
        telling the user how much a resume would skip.
        """
        return self._get_json(
            f"/dataset-jobs/{submission_id}/packaging-progress",
            params={"exact": "true" if exact else "false"},
            timeout=60,
        )

    def start_feature_extraction(
        self,
        submission_id: str,
        checkpoint: str,
        model: str = "BarlowTwins_3",
        marker: str = "he",
    ) -> dict:
        """User-triggered: run the packaged .h5 through Kai's frozen
        self-supervised encoder. Only valid once packaging's .h5 is ready."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/extract-features",
            {"checkpoint": checkpoint, "model": model, "marker": marker},
        )

    def vote_presets(self) -> dict:
        """The named vote configurations Stage 4 offers, from the server.

        Not a local constant: a preset is seven numbers and an accuracy figure,
        and a copy of those in the UI is a copy that drifts. Showing "97.27%"
        next to something that is no longer that configuration would be exactly
        the kind of confident wrongness the pipeline guards against elsewhere.
        """
        return self._get_json("/vote-presets")

    def start_cluster_assignment(
        self,
        submission_id: str,
        reference: str | None = None,
        k: int | None = None,
        overwrite: bool = False,
        vote_preset: str | None = None,
        vote_overrides: dict | None = None,
    ) -> dict:
        """User-triggered: assign HPL cluster IDs to this run's embeddings by
        k-NN vote against the reference. Only valid once extraction's
        projections .h5 validates.

        vote_preset names which measured configuration to use; the server
        defaults to the tuned one. vote_overrides adjusts individual settings on
        top of it — omitted keys keep the preset's value, so a partial override
        cannot quietly reduce the vote to something never measured.
        """
        body = {"reference": reference, "k": k, "overwrite": overwrite}
        if vote_preset:
            body["vote_preset"] = vote_preset
        body.update(vote_overrides or {})
        return self._post_json(
            f"/dataset-jobs/{submission_id}/assign-clusters", body,
        )

    def start_anorak(
        self,
        submission_id: str,
        slides_csv: str,
        scope: str = "full",
        sample_size: int | None = None,
        seed: int | None = None,
        resume: bool = True,
        overwrite: bool = False,
        time_limit: str | None = None,
        chain: int = 2,
    ) -> dict:
        """User-triggered: run the ANORAK Nextflow pipeline over a slide list.

        scope="subset" samples `sample_size` slides at random rather than
        taking the first N — the first N of a cohort sorted by slide id is
        usually one or two patients, sharing a scanner, a batch and a stain
        run, which is the least informative way to spend a test. A seed is
        recorded whether or not one is given, so the sample can be asked for
        again and a later disagreement has something to point at.

        resume continues the run's cached Nextflow work directory, which is
        what makes a resubmission after a fixed container re-run only what
        failed.

        overwrite only permits replacing a run that already has a valid
        grading table. The server refuses while the previous head job is in
        flight, or while Slurm cannot say whether it is, whatever this is set
        to — so a caller cannot opt out of that check, and should not try.

        chain is the number of head jobs: the first plus chain-1 standbys that
        resume it if it reaches its walltime or runs out of watchdog restarts.
        2 by default here, where the server's own default is 1 for old clients.
        """
        return self._post_json(
            f"/dataset-jobs/{submission_id}/anorak",
            {
                "slides_csv": slides_csv,
                "scope": scope,
                "sample_size": sample_size,
                "seed": seed,
                "resume": resume,
                "overwrite": overwrite,
                "time_limit": time_limit,
                "chain": chain,
            },
        )

    def check_anorak_submit(self, partition: str | None = None) -> dict:
        """Whether a compute node can reach the Slurm controller.

        Worth once per cluster before the first ANORAK run: the Nextflow head
        job submits every task itself, and on a cluster where compute nodes
        cannot submit it waits out its time limit having done nothing.
        """
        return self._get_json(
            "/anorak-submit-check",
            params={"partition": partition} if partition else None,
        )

    def start_test_cluster_assignment(
        self,
        submission_id: str,
        projections_h5: str,
        reference: str | None = None,
        k: int | None = None,
        vote_preset: str | None = None,
        vote_overrides: dict | None = None,
    ) -> dict:
        """Assign clusters for an arbitrary projections .h5 without recording
        it against the run — so a test attempt can never make the run look
        further along than it is."""
        body = {"projections_h5": projections_h5, "reference": reference, "k": k}
        if vote_preset:
            body["vote_preset"] = vote_preset
        body.update(vote_overrides or {})
        return self._post_json(
            f"/dataset-jobs/{submission_id}/assign-clusters-test", body,
        )

    def cohort_shift_readiness(self, submission_id: str) -> dict:
        """Can the cohort check run for this run yet, and what is missing?

        Separate GET so the UI can state the prerequisite before anyone clicks.
        The reference profile is a one-off ~20 minute job reused by every
        dataset, so "not built yet" is a setup step, not an error.
        """
        return self._get_json(
            f"/dataset-jobs/{submission_id}/cohort-shift-readiness")

    def check_cohort_shift(self, submission_id: str, csv_path: str | None = None,
                          top_slides: int = 10) -> dict:
        """Is this dataset's tissue represented in the reference at all?

        Read-only and cheap: two columns of the assignment CSV against
        precomputed reference quantiles. Separate from the Stage 5 preview on
        purpose — that one asks whether the load will be clean, this one asks
        whether the cluster IDs mean anything for this cohort, and a clean
        preview on a shifted cohort is exactly the combination that looks fine
        and is not.
        """
        return self._post_json(
            f"/dataset-jobs/{submission_id}/cohort-shift",
            {"csv_path": csv_path, "top_slides": top_slides},
        )

    def preview_kb_load(self, submission_id: str, min_margin: float = 0.0,
                        csv_path: str | None = None) -> dict:
        """Stage 5 dry run: what loading an assignment CSV into the
        Knowledge Bank would do. Read-only, safe to call as often as the UI
        wants — mirrors load_hpc_assignments.py's default --dry-run posture.
        min_margin previews how many tiles compute_profiles() would exclude
        from the per-slide aggregates at that vote_margin threshold. csv_path
        overrides this run's tracked assignment output — needed for output
        from Stage 4's "Test on a sample .h5" mode, which has no tracked
        path of its own."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/kb-load-preview",
            {"min_margin": min_margin, "csv_path": csv_path,
             "kb_target": self.kb_target},
            timeout=KB_REQUEST_TIMEOUT,
        )

    def commit_kb_load(
        self,
        submission_id: str,
        cancer_type: str | None = None,
        allow_unknown_clusters: bool = False,
        skip_profiles: bool = False,
        min_margin: float = 0.0,
        csv_path: str | None = None,
    ) -> dict:
        """User-triggered: write cluster assignments into tile_registry plus
        the per-slide aggregates, after the same guards load_hpc_assignments.py
        --commit enforces on the CLI. min_margin excludes tiles below that
        vote_margin from the aggregates only — tile_registry keeps every
        tile's own hpc_id and margin regardless. csv_path loads from an
        explicit path instead of this run's tracked output (see
        preview_kb_load) — this still writes to the KB for real, it just
        skips marking this particular run as having loaded it."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/kb-load",
            {
                "cancer_type": cancer_type,
                "allow_unknown_clusters": allow_unknown_clusters,
                "skip_profiles": skip_profiles,
                "csv_path": csv_path,
                "min_margin": min_margin,
                "kb_target": self.kb_target,
            },
            timeout=KB_REQUEST_TIMEOUT,
        )

    def preview_registration(self, submission_id: str,
                         dataset_id: str | None = None,
                         tile_dataset_name: str | None = None,
                         dataset_name: str | None = None,
                         raw_dir: str | None = None,
                         tile_dir: str | None = None,
                         h5_path: str | None = None,
                         scope: str = "full",
                         slide_names: list[str] | None = None,
                         slide_metadata: bool = False,
                         write_dataset_config: bool = True,
                         replace: bool = False) -> dict:
        """Dry run of the registration step: what identity rows creating this
        cohort would write, without writing them.

        Registration is what makes Stage 5 possible at all. Stage 5 only ever
        UPDATEs tile_registry.hpc_id, so for a cohort that has never touched
        the KB its match rate is 0% by construction — there is nothing to
        update. This creates wsi_registry, wsi_metadata, dataset_config,
        tile_coordinates and tile_registry from what Stages 1–2 wrote to disk.

        Every path it needs is already on the run record, so nothing but the
        cohort key and the tile folder is passed: dataset_id defaults to the
        run's own dataset_name, upper-cased, and tile_dataset_name to the
        recorded dataset_name itself — which a run predating that column, or one
        tiled by hand, does not have, so the UI asks for it outright rather than
        letting registration refuse."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/register-preview",
            {
            "dataset_id": dataset_id,
            "tile_dataset_name": tile_dataset_name,
            "kb_target": self.kb_target,
            # None means "take it from the run record", which is what every run
            # driven through this UI holds. Supplied only for runs predating a
            # column, or work done on the cluster before the pipeline existed.
            "dataset_name": dataset_name,
            "raw_dir": raw_dir,
            "tile_dir": tile_dir,
            "h5_path": h5_path,
            "scope": scope,
            "slide_names": slide_names,
            "slide_metadata": slide_metadata,
            "write_dataset_config": write_dataset_config,
            "replace": replace,
            },
            timeout=KB_REQUEST_TIMEOUT,
        )

    def commit_registration(self, submission_id: str,
                        dataset_id: str | None = None,
                        tile_dataset_name: str | None = None,
                        dataset_name: str | None = None,
                        raw_dir: str | None = None,
                        tile_dir: str | None = None,
                        h5_path: str | None = None,
                        scope: str = "full",
                        slide_names: list[str] | None = None,
                        slide_metadata: bool = False,
                        write_dataset_config: bool = True,
                        replace: bool = False) -> dict:
        """User-triggered: create this cohort's identity rows in the KB, in one
        transaction, after the same guards register_dataset.py --commit
        enforces on the CLI — a slide or tile already claimed by a different
        dataset_id is refused rather than reassigned, and an existing
        registration needs replace=True.

        slide_metadata additionally reads each slide's OpenSlide header into
        wsi_metadata. It opens every file, so it costs minutes on a large
        cohort and is off by default."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/register",
            {
            "dataset_id": dataset_id,
            # The tile folder Stage 1 wrote into. Sent on both calls because
            # the two have to read the same folder — a preview against one and
            # a commit against another would report numbers from a cohort it
            # did not write.
            "tile_dataset_name": tile_dataset_name,
            "kb_target": self.kb_target,
            # None means "take it from the run record", which is what every run
            # driven through this UI holds. Supplied only for runs predating a
            # column, or work done on the cluster before the pipeline existed.
            "dataset_name": dataset_name,
            "raw_dir": raw_dir,
            "tile_dir": tile_dir,
            "h5_path": h5_path,
            # scope and slide_names were accepted by this method and then left
            # out of the body, so a subset previewed as three slides committed
            # as the whole dataset — silently, because registering more than
            # you meant to still succeeds.
            "scope": scope,
            "slide_names": slide_names,
            "slide_metadata": slide_metadata,
            "write_dataset_config": write_dataset_config,
            "replace": replace,
            },
            timeout=KB_REQUEST_TIMEOUT,
        )


    def submit_registration(self, submission_id: str,
                            dataset_id: str | None = None,
                            tile_dataset_name: str | None = None,
                            raw_dir: str | None = None,
                            tile_dir: str | None = None,
                            h5_path: str | None = None,
                            scope: str = "full",
                            slide_names: list[str] | None = None,
                            slide_metadata: bool = False,
                            write_dataset_config: bool = True,
                            replace: bool = False) -> dict:
        """Queue Stage 5 on Slurm instead of writing inside the request.

        Returns as soon as sbatch has taken the job, so the write outlives this
        client and the server both. Poll get_dataset_job_status() for
        registration_slurm_state and registration_done — the job sets done
        itself, so it still means committed.

        Same arguments as commit_registration, and the job runs the same
        functions with the same guards; the difference is only where."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/register-submit",
            {
                "dataset_id": dataset_id,
                "tile_dataset_name": tile_dataset_name,
                "kb_target": self.kb_target,
                # The same overrides as the in-server path; the server resolves
                # both through one function, so they read the same files.
                "raw_dir": raw_dir,
                "tile_dir": tile_dir,
                "h5_path": h5_path,
                "scope": scope,
                "slide_names": slide_names,
                "slide_metadata": slide_metadata,
                "write_dataset_config": write_dataset_config,
                "replace": replace,
            },
        )

    def submit_kb_load(self, submission_id: str,
                       cancer_type: str | None = None,
                       allow_unknown_clusters: bool = False,
                       skip_profiles: bool = False,
                       min_margin: float = 0.0,
                       csv_path: str | None = None) -> dict:
        """Queue Stage 6 on Slurm. See submit_registration."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/kb-load-submit",
            {
                "cancer_type": cancer_type,
                "allow_unknown_clusters": allow_unknown_clusters,
                "skip_profiles": skip_profiles,
                "csv_path": csv_path,
                "min_margin": min_margin,
                "kb_target": self.kb_target,
            },
        )

    def check_kb_job_db(self) -> dict:
        """Can a compute node reach *and use* Postgres? Queues a one-second srun,
        so it is slow — minutes if the queue is busy — and is only worth calling
        when a Slurm-backed KB write has been refused or has failed to connect.

        Read "usable", not "reachable": the socket opening proves only that
        something is listening, and a job that connects and then fails
        authentication has already cost the queue time this call exists to save.
        Probes whichever Knowledge Bank this client is pointed at, since the test
        database not existing on that host is one of the answers."""
        return self._get_json("/kb-job-db-check",
                              params={"kb_target": self.kb_target}, timeout=300)

    def start_test_packaging(
        self,
        submission_id: str,
        sample_size: int | None = None,
        slide_names: list[str] | None = None,
        random_seed: int | None = None,
        scope: str = "run",
    ) -> dict:
        """Package a chosen subset of slides into a separate test .h5 — for
        trying packaging (and a checkpoint) on a sample before committing to
        the full multi-hour run. Not tracked against the run's own packaging
        state, so it can be run any number of times without affecting (or
        being blocked by) the real one.

        scope="run" draws from this run's own manifest; "tiled" draws from
        every slide with tiles on disk, which is the only way to test a
        sample larger than the run itself.

        Leave random_seed None for a fresh random draw each time. Pass the
        seed echoed back in the response to reproduce a specific draw — and
        note that re-sending the same explicit seed and count is treated as
        the same attempt, so it will be refused as a duplicate rather than
        packaged twice.
        """
        return self._post_json(
            f"/dataset-jobs/{submission_id}/package-test",
            {
                "sample_size": sample_size,
                "slide_names": slide_names,
                "random_seed": random_seed,
                "scope": scope,
            },
        )

    def get_test_packaging_status(self, submission_id: str, job_id: str, output_path: str) -> dict:
        return self._get_json(
            f"/dataset-jobs/{submission_id}/package-test-status",
            params={"job_id": job_id, "output_path": output_path},
        )

    def start_test_feature_extraction(
        self,
        submission_id: str,
        h5_path: str,
        checkpoint: str,
        model: str = "BarlowTwins_3",
        marker: str = "he",
    ) -> dict:
        """Run feature extraction against an arbitrary .h5 (typically a
        test-sample .h5 from start_test_packaging) instead of this run's
        own tracked one — validate a checkpoint on a small sample first."""
        return self._post_json(
            f"/dataset-jobs/{submission_id}/extract-features-test",
            {"h5_path": h5_path, "checkpoint": checkpoint, "model": model, "marker": marker},
        )

    def get_test_feature_extraction_status(self, submission_id: str, job_id: str, output_path: str) -> dict:
        return self._get_json(
            f"/dataset-jobs/{submission_id}/extract-features-test-status",
            params={"job_id": job_id, "output_path": output_path},
        )

    # ------------------------------------------------------------------
    # Slide-level endpoints
    # ------------------------------------------------------------------

    def get_slide_info(self, slide_id: str) -> dict:
        return self._get_json(f"/slide/{slide_id}/info")

    def get_thumbnail(self, slide_id: str, max_width: int = 3000,
                      quality: int = 85) -> Image.Image:
        return self._get_image(
            f"/slide/{slide_id}/thumbnail",
            params={"max_width": max_width, "quality": quality},
            cache_key=("thumb", slide_id, max_width),
        )

    def get_tile(self, slide_id: str, level: int, x: int, y: int,
                 w: int = 256, h: int = 256, quality: int = 85) -> Image.Image:
        return self._get_image(
            f"/slide/{slide_id}/tile",
            params={"level": level, "x": x, "y": y, "w": w, "h": h, "quality": quality},
            cache_key=("tile", slide_id, level, x, y, w, h),
        )

    def get_region(self, slide_id: str, x: int, y: int,
                   w: int, h: int, level: int = 0,
                   quality: int = 85) -> Image.Image:
        return self._get_image(
            f"/slide/{slide_id}/region",
            params={"x": x, "y": y, "w": w, "h": h, "level": level, "quality": quality},
            cache_key=("region", slide_id, level, x, y, w, h),
        )

    # ------------------------------------------------------------------
    # Tile metadata
    # ------------------------------------------------------------------

    def get_tiles_meta(self, slide_id: str) -> pd.DataFrame:
        """All tile coords + HPC labels + heatmap probs for a slide."""
        records = self._get_json(f"/slide/{slide_id}/tiles_meta")
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def get_adjacency(self, slide_id: str) -> dict:
        return self._get_json(f"/slide/{slide_id}/adjacency")

    # ------------------------------------------------------------------
    # HPC endpoints
    # ------------------------------------------------------------------

    def get_hpc_info(self, hpc_id: int) -> dict:
        return self._get_json(f"/hpc/{hpc_id}/info")

    def get_hpc_survival(self, hpc_id: int) -> dict:
        return self._get_json(f"/hpc/{hpc_id}/survival")

    # ------------------------------------------------------------------
    # H5 tile image by slide_tile key
    # ------------------------------------------------------------------

    def get_tile_image(self, slide_tile: str, quality: int = 85) -> Image.Image:
        return self._get_image(
            f"/tile_image/{slide_tile}",
            params={"quality": quality},
            cache_key=("h5tile", slide_tile),
        )

    # ------------------------------------------------------------------
    # Full query (Phase 2: move NL pipeline to server)
    # ------------------------------------------------------------------

    def query(self, query_text: str, slide_id: Optional[str] = None) -> dict:
        r = self._session.post(
            f"{self.base_url}/query",
            json={"query": query_text, "slide_id": slide_id,
                  "kb_target": self.kb_target},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()
