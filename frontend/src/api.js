// JS port of app/api_client.py's TileServerClient.
// Every method maps 1:1 to a method there — see that file for the "why"
// behind each endpoint's shape (timeouts, dry-run semantics, etc.).

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

// A Knowledge Bank write can run for hours on a full cohort, and so can its
// preview: Stage 6 previews 18.5M keys against tile_registry. The default 30s
// here meant the browser gave up long before the server did, and the write kept
// going invisibly — the same failure api_client.py's KB_REQUEST_TIMEOUT exists
// to prevent. Matches that constant exactly.
const KB_REQUEST_TIMEOUT = 24 * 60 * 60 * 1000;

// Which Knowledge Bank the server should read and write. Held here rather than
// passed to each method, for the reason api_client.py gives: roughly a dozen
// endpoints honour it and the app calls them from far more places than that, so
// threading an argument through every call site is how one of them ends up
// reading production while the rest read test — the exact failure this feature
// exists to avoid.
export const KB_PRODUCTION = "production";
export const KB_TEST = "test";
let kbTarget = KB_PRODUCTION;

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, { method = "GET", params, json, timeoutMs = 30000, ...rest } = {}) {
  const url = new URL(BASE_URL + path);
  // Sent on every GET, mirroring api_client.py's _get. Endpoints that do not
  // declare it ignore it — the pipeline and run-tracking routes are
  // production-only by design — so this cannot make one of them read the wrong
  // database, and it removes the need to remember which reads are KB reads.
  // POSTs carry it in the body instead, again as api_client.py does.
  const merged = method === "GET" ? { ...(params || {}), kb_target: kbTarget } : params;
  if (merged) {
    for (const [k, v] of Object.entries(merged)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    }
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method,
      headers: json !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: json !== undefined ? JSON.stringify(json) : undefined,
      signal: controller.signal,
      ...rest,
    });
    if (!res.ok) {
      let body = null;
      try {
        body = await res.json();
      } catch {
        /* not JSON */
      }
      throw new ApiError(`${method} ${path} failed: ${res.status}`, res.status, body);
    }
    return res;
  } finally {
    clearTimeout(timer);
  }
}

async function getJson(path, opts) {
  return (await request(path, opts)).json();
}

async function postJson(path, jsonBody, opts) {
  return (await request(path, { method: "POST", json: jsonBody, ...opts })).json();
}

function imageUrl(path, params) {
  const url = new URL(BASE_URL + path);
  // Image endpoints go through _get in api_client.py, so they receive
  // kb_target there too. Including it here matters for a second reason the
  // Python client handles differently: set_kb_target() clears its image cache
  // because a slide_id only means one thing within a single KB. The browser's
  // cache cannot be cleared from here, so the target is part of the URL and a
  // switch simply misses the old entries.
  const merged = { ...(params || {}), kb_target: kbTarget };
  {
    for (const [k, v] of Object.entries(merged)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    }
  }
  return url.toString();
}

export const api = {
  baseUrl: BASE_URL,
  ApiError,

  // -- Knowledge Bank target ------------------------------------------
  getKbTarget: () => kbTarget,
  setKbTarget(target) {
    kbTarget = target === KB_TEST ? KB_TEST : KB_PRODUCTION;
    return kbTarget;
  },

  // -- Health / slide list --------------------------------------------
  health: () => getJson("/health"),
  listSlides: async () => (await getJson("/slides")).slides,

  // -- Upload -----------------------------------------------------------
  async uploadSlide(file, slideId, confirmOverwrite = false) {
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("slide_id", (slideId || "").trim());
    form.append("confirm_overwrite", confirmOverwrite ? "true" : "false");
    return postFormData("/upload-slide", form, { timeoutMs: 600000 });
  },
  getProcessingStatus: (slideId) => getJson(`/slide/${slideId}/processing-status`),

  // -- Dataset-wide Slurm jobs -------------------------------------------
  getDatasetRoots: async () => (await getJson("/dataset-roots")).datasets,
  getTileDatasetNames: async () => (await getJson("/tile-dataset-names")).dataset_names,

  submitDatasetJob: (body) => postJson("/dataset-jobs", cleanBody(body)),

  // One click: Stages 1-4 as a single Nextflow run (POST /pipeline-runs).
  // Every input a later stage used to ask for at its own button is sent here,
  // once; the server refuses before queueing anything if one is wrong. Polled
  // through getDatasetJobStatus like any run — its `pipeline` block says where
  // each stage is. Registration and the KB load stay manual.
  // One click: ANORAK on its own over a dataset path. Without slidesCsv every
  // slide in the directory is graded, recorded as tumour-unverified.
  startAnorakRun: ({ datasetPath, slidesCsv = null, sampleSize = null, seed = null }) =>
    postJson(
      "/anorak-runs",
      cleanBody({ dataset_path: datasetPath, slides_csv: slidesCsv, sample_size: sampleSize, seed }),
      { timeoutMs: 300000 },
    ),
  // Resubmit a stopped ANORAK run exactly as it was, with -resume.
  resumeAnorakRun: (submissionId) =>
    postJson(`/dataset-jobs/${submissionId}/anorak-resume`, {}, { timeoutMs: 300000 }),

  // The settings a one-click run uses — all the server's own.
  getPipelineDefaults: () => getJson("/pipeline-defaults"),
  startPipelineRun: (body) => postJson("/pipeline-runs", cleanBody(body), { timeoutMs: 120000 }),
  // Resubmit a stopped pipeline run with -resume: re-runs only what did not
  // finish, with the run's own recorded settings.
  resumePipelineRun: (submissionId, { chain = null, timeLimit = null, allowIncomplete = null } = {}) =>
    postJson(`/dataset-jobs/${submissionId}/pipeline-resume`, {
      chain,
      time_limit: timeLimit,
      allow_incomplete: allowIncomplete,
    }),
  checkPipelineSubmit: (partition = null) =>
    getJson("/pipeline-submit-check", { params: partition ? { partition } : undefined, timeoutMs: 180000 }),

  listDatasetJobs: (withState = false) =>
    withState
      ? getJson("/dataset-jobs", { params: { with_state: "true" }, timeoutMs: 60000 })
      : getJson("/dataset-jobs"),

  listDatasets: () => getJson("/datasets", { timeoutMs: 60000 }),

  getDatasetCoverage: async (datasetName, rawDir) =>
    (
      await getJson("/datasets", {
        params: { dataset_name: datasetName, raw_dir: rawDir, coverage: "true" },
        timeoutMs: 300000,
      })
    ).datasets,

  getDatasetJobHistory: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/jobs`, { timeoutMs: 60000 }),

  getDatasetJobStatus: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/status`, { timeoutMs: 60000 }),

  resumeDatasetJob: (submissionId) => postJson(`/dataset-jobs/${submissionId}/resume`, {}),
  cancelDatasetJob: (submissionId) => postJson(`/dataset-jobs/${submissionId}/cancel`, {}),

  getTiledCoverage: (submissionId) => getJson(`/dataset-jobs/${submissionId}/tiled-coverage`),

  startPackagingJob: (submissionId, { allowIncomplete = false, scope = "run", resume } = {}) => {
    const params = { allow_incomplete: allowIncomplete ? "true" : "false", scope };
    if (resume !== undefined && resume !== null) params.resume = resume ? "true" : "false";
    return postJson(`/dataset-jobs/${submissionId}/package`, {}, { params });
  },

  getPackagingProgress: (submissionId, exact = true) =>
    getJson(`/dataset-jobs/${submissionId}/packaging-progress`, {
      params: { exact: exact ? "true" : "false" },
      timeoutMs: 60000,
    }),

  startFeatureExtraction: (submissionId, { checkpoint, model = "BarlowTwins_3", marker = "he" }) =>
    postJson(`/dataset-jobs/${submissionId}/extract-features`, { checkpoint, model, marker }),

  votePresets: () => getJson("/vote-presets"),

  startClusterAssignment: (
    submissionId,
    { reference = null, k = null, overwrite = false, votePreset = null, voteOverrides = {} } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/assign-clusters`, {
      reference,
      k,
      overwrite,
      ...(votePreset ? { vote_preset: votePreset } : {}),
      ...voteOverrides,
    }),

  startTestClusterAssignment: (
    submissionId,
    { projectionsH5, reference = null, k = null, votePreset = null, voteOverrides = {} }
  ) =>
    postJson(`/dataset-jobs/${submissionId}/assign-clusters-test`, {
      projections_h5: projectionsH5,
      reference,
      k,
      ...(votePreset ? { vote_preset: votePreset } : {}),
      ...voteOverrides,
    }),

  cohortShiftReadiness: (submissionId) =>
    getJson(`/dataset-jobs/${submissionId}/cohort-shift-readiness`),

  checkCohortShift: (submissionId, { csvPath = null, topSlides = 10 } = {}) =>
    postJson(`/dataset-jobs/${submissionId}/cohort-shift`, { csv_path: csvPath, top_slides: topSlides }),

  // Stage 7: ANORAK growth-pattern grading. Gated on nothing this pipeline
  // produces — see startAnorak's own note — so it can be submitted whenever a
  // tumour-slide list exists.
  //
  // scope="subset" samples sampleSize slides at random rather than taking the
  // first N — the first N of a cohort sorted by slide id is usually one or two
  // patients, sharing a scanner, a batch and a stain run. resume continues the
  // run's cached Nextflow work directory, which is what makes a resubmission
  // after a fixed container re-run only what failed. chain is the number of
  // head jobs — the first plus standbys that resume it if it hits its walltime
  // or runs out of watchdog restarts; 2 here, 1 on the server for old clients.
  startAnorak: (
    submissionId,
    {
      slidesCsv,
      scope = "full",
      sampleSize = null,
      seed = null,
      resume = true,
      overwrite = false,
      timeLimit = null,
      chain = 2,
    } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/anorak`, {
      slides_csv: slidesCsv,
      scope,
      sample_size: sampleSize,
      seed,
      resume,
      overwrite,
      time_limit: timeLimit,
      chain,
    }),

  // Whether a compute node can reach the Slurm controller. Worth once per
  // cluster before the first ANORAK run: the Nextflow head job submits every
  // task itself, and on a cluster where compute nodes cannot submit it waits
  // out its time limit having done nothing.
  checkAnorakSubmit: (partition = null) =>
    getJson("/anorak-submit-check", partition ? { params: { partition } } : undefined),

  // Registration — the identity rows Stage 6's UPDATE needs to exist.
  //
  // tile_dataset_name is the folder under processed_tiles that Stage 1 wrote
  // this run's per-slide _tile_metadata.csv files into, and it is NOT
  // dataset_id: that is the cohort key the KB groups by, this is a directory on
  // disk. Every tile's x/y is read from there, so a wrong or absent value does
  // not fail — it registers tiles with no coordinates. It is sent on both
  // preview and commit because the two have to read the same folder; numbers
  // previewed against one and committed against another describe a cohort the
  // write did not touch.
  //
  // scope/slide_names were accepted by api_client.py's methods and then left
  // out of the body once, so a subset previewed as three slides committed as
  // the whole dataset — silently, because registering more than you meant to
  // still succeeds.
  previewRegistration: (
    submissionId,
    {
      datasetId = null,
      tileDatasetName = null,
      scope = "full",
      slideNames = null,
      slideMetadata = false,
      writeDatasetConfig = true,
      replace = false,
    } = {}
  ) =>
    postJson(
      `/dataset-jobs/${submissionId}/register-preview`,
      {
        dataset_id: datasetId,
        tile_dataset_name: tileDatasetName,
        kb_target: kbTarget,
        scope,
        slide_names: slideNames,
        slide_metadata: slideMetadata,
        write_dataset_config: writeDatasetConfig,
        replace,
      },
      { timeoutMs: KB_REQUEST_TIMEOUT }
    ),

  commitRegistration: (
    submissionId,
    {
      datasetId = null,
      tileDatasetName = null,
      scope = "full",
      slideNames = null,
      slideMetadata = false,
      writeDatasetConfig = true,
      replace = false,
    } = {}
  ) =>
    postJson(
      `/dataset-jobs/${submissionId}/register`,
      {
        dataset_id: datasetId,
        tile_dataset_name: tileDatasetName,
        kb_target: kbTarget,
        scope,
        slide_names: slideNames,
        slide_metadata: slideMetadata,
        write_dataset_config: writeDatasetConfig,
        replace,
      },
      { timeoutMs: KB_REQUEST_TIMEOUT }
    ),

  // Queue Stage 5 on Slurm instead of writing inside the request. Returns as
  // soon as sbatch has taken the job, so the write outlives this tab and the
  // server both — poll getDatasetJobStatus for registration_slurm_state and
  // registration_done. The job sets done itself, so it still means committed.
  // Same arguments and the same guards as commitRegistration; only the where
  // differs. No long timeout: this call only waits for sbatch.
  submitRegistration: (
    submissionId,
    {
      datasetId = null,
      tileDatasetName = null,
      scope = "full",
      slideNames = null,
      slideMetadata = false,
      writeDatasetConfig = true,
      replace = false,
    } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/register-submit`, {
      dataset_id: datasetId,
      tile_dataset_name: tileDatasetName,
      kb_target: kbTarget,
      scope,
      slide_names: slideNames,
      slide_metadata: slideMetadata,
      write_dataset_config: writeDatasetConfig,
      replace,
    }),

  previewKbLoad: (submissionId, { minMargin = 0.0, csvPath = null } = {}) =>
    postJson(
      `/dataset-jobs/${submissionId}/kb-load-preview`,
      { min_margin: minMargin, csv_path: csvPath, kb_target: kbTarget },
      { timeoutMs: KB_REQUEST_TIMEOUT }
    ),

  commitKbLoad: (
    submissionId,
    {
      cancerType = null,
      allowUnknownClusters = false,
      skipProfiles = false,
      minMargin = 0.0,
      csvPath = null,
    } = {}
  ) =>
    postJson(
      `/dataset-jobs/${submissionId}/kb-load`,
      {
        cancer_type: cancerType,
        allow_unknown_clusters: allowUnknownClusters,
        skip_profiles: skipProfiles,
        csv_path: csvPath,
        min_margin: minMargin,
        kb_target: kbTarget,
      },
      { timeoutMs: KB_REQUEST_TIMEOUT }
    ),

  // Queue Stage 6 on Slurm. See submitRegistration.
  submitKbLoad: (
    submissionId,
    {
      cancerType = null,
      allowUnknownClusters = false,
      skipProfiles = false,
      minMargin = 0.0,
      csvPath = null,
    } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/kb-load-submit`, {
      cancer_type: cancerType,
      allow_unknown_clusters: allowUnknownClusters,
      skip_profiles: skipProfiles,
      csv_path: csvPath,
      min_margin: minMargin,
      kb_target: kbTarget,
    }),

  // Can a compute node reach Postgres? Queues a one-second srun, so it is slow
  // — minutes if the queue is busy — and is only worth calling when a
  // Slurm-backed KB write has been refused or has failed to connect.
  checkKbJobDb: () => getJson("/kb-job-db-check", { timeoutMs: 300000 }),

  startTestPackaging: (
    submissionId,
    { sampleSize = null, slideNames = null, randomSeed = null, scope = "run" } = {}
  ) =>
    postJson(`/dataset-jobs/${submissionId}/package-test`, {
      sample_size: sampleSize,
      slide_names: slideNames,
      random_seed: randomSeed,
      scope,
    }),

  getTestPackagingStatus: (submissionId, jobId, outputPath) =>
    getJson(`/dataset-jobs/${submissionId}/package-test-status`, {
      params: { job_id: jobId, output_path: outputPath },
    }),

  startTestFeatureExtraction: (
    submissionId,
    { h5Path, checkpoint, model = "BarlowTwins_3", marker = "he" }
  ) =>
    postJson(`/dataset-jobs/${submissionId}/extract-features-test`, {
      h5_path: h5Path,
      checkpoint,
      model,
      marker,
    }),

  getTestFeatureExtractionStatus: (submissionId, jobId, outputPath) =>
    getJson(`/dataset-jobs/${submissionId}/extract-features-test-status`, {
      params: { job_id: jobId, output_path: outputPath },
    }),

  // -- Slide-level --------------------------------------------------------
  getSlideInfo: (slideId) => getJson(`/slide/${slideId}/info`),
  thumbnailUrl: (slideId, maxWidth = 3000, quality = 85) =>
    imageUrl(`/slide/${slideId}/thumbnail`, { max_width: maxWidth, quality }),
  tileUrl: (slideId, level, x, y, w = 256, h = 256, quality = 85) =>
    imageUrl(`/slide/${slideId}/tile`, { level, x, y, w, h, quality }),
  regionUrl: (slideId, x, y, w, h, level = 0, quality = 85) =>
    imageUrl(`/slide/${slideId}/region`, { x, y, w, h, level, quality }),

  // -- Tile metadata --------------------------------------------------------
  getTilesMeta: (slideId) => getJson(`/slide/${slideId}/tiles_meta`),
  getAdjacency: (slideId) => getJson(`/slide/${slideId}/adjacency`),

  // -- HPC -------------------------------------------------------------------
  getHpcInfo: (hpcId) => getJson(`/hpc/${hpcId}/info`),
  getHpcSurvival: (hpcId) => getJson(`/hpc/${hpcId}/survival`),

  // -- H5 tile image by slide_tile key ---------------------------------------
  tileImageUrl: (slideTile, quality = 85) => imageUrl(`/tile_image/${slideTile}`, { quality }),

  // Full NL query pipeline (Phase 2 — see tile_server_v2_.py's /query
  // docstring). `history` is the last few {role, content} chat turns (used
  // for follow-up questions); `sessionContext` mirrors app_v28.py's
  // active_slide/viewer_open/selected_hpc/highlight_mode dict, so "this
  // slide"/"that HPC" can resolve against whatever this client currently has
  // open — the server has no session of its own to read that from.
  query: (queryText, slideId = null, { history = null, sessionContext = null } = {}) =>
    postJson("/query", {
      query: queryText,
      slide_id: slideId,
      kb_target: kbTarget,
      history,
      session_context: sessionContext,
    }),

  // -- DZI (OpenSeadragon reads this directly, but exposed for convenience) --
  // OpenSeadragon fetches this itself, so none of the parameters get() and
  // post() attach come with it — kb_target has to be on the URL or the viewer
  // resolves every slide against production. It goes on the ".dzi" URL
  // specifically: DziTileSource matches /\.(dzi|xml|js)\?/ and copies the query
  // onto each ..._files/{level}/{col}_{row}.jpeg it builds, so this is also
  // what carries the target to the tiles. The id is encoded because ours hold
  // spaces and colons.
  dziUrl: (slideId) =>
    `${BASE_URL}/dzi/${encodeURIComponent(slideId)}.dzi?kb_target=${encodeURIComponent(kbTarget)}`,
};

async function postFormData(path, form, { timeoutMs = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(BASE_URL + path, { method: "POST", body: form, signal: controller.signal });
    if (!res.ok) {
      let body = null;
      try {
        body = await res.json();
      } catch {
        /* not JSON */
      }
      throw new ApiError(`POST ${path} failed: ${res.status}`, res.status, body);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

// Drop null/undefined keys so the server doesn't get an explicit null where
// omission has different meaning (see submit_dataset_job's min_tissue note
// in api_client.py).
function cleanBody(body) {
  const out = {};
  for (const [k, v] of Object.entries(body)) {
    if (v !== undefined && v !== null) out[k] = v;
  }
  return out;
}
