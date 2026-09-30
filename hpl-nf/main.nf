#!/usr/bin/env nextflow

/*
 * HPL Stages 1-4 as one pipeline: tile -> package -> encode -> classify.
 *
 * What the UI's "Run pipeline" button submits. It replaces clicking through
 * four stages that each waited on the last, and it is deliberately NOT a
 * reimplementation of them: every task runs a wrapper in bin/ (one per stage,
 * see bin/hpl_common.py), which calls the same tiler, packager and container
 * command builders from backend/ that the per-stage Slurm submitters use. The science and the --cleanenv workarounds
 * live in one place; this file only says what runs after what.
 *
 * Stages 5 and 6 (Knowledge Bank registration and load) are not here, on
 * purpose. Both write the shared Knowledge Bank, and Stage 6 never commits
 * without a human looking at a dry run first (CLAUDE.md). The pipeline stops
 * at a validated assignments CSV and the UI takes it from there.
 *
 * Two properties every step keeps, and why:
 *
 *   A stage is done only when its last task has validated the output with
 *   the check the server's own gate uses, and written stages/<stage>.done.json
 *   (see backend/hpl_nf_state.py). Exit status alone never counts — this
 *   codebase's failure mode is a complete-looking file of the wrong length.
 *
 *   Outputs live where the per-stage path puts them (processed_tiles/,
 *   model_input/, the HPL repo's results/), not in Nextflow's work/ directory.
 *   Registration, the viewer and the KB load read them there, and a run
 *   started from the UI has to leave the same artifacts as one clicked
 *   through by hand. The cost is that tasks write outside their work
 *   directory, so each step checks whether its output already exists and
 *   validates before doing anything — which is what makes -resume, and a
 *   retried task after a cache miss, safe.
 *
 * Every value a task needs comes from run_config.json (params.config), written
 * by backend/submit_hpl_nf.py at submission. Nothing is read from the
 * environment a task inherits.
 */

nextflow.enable.dsl = 2

params.config      = null   // <outdir>/run_config.json, written by submit_hpl_nf.py
params.manifest    = null   // one raw slide path per line; the cohort
params.outdir      = null   // this run's directory: config, markers, logs, reports
params.python      = null   // the interpreter the server runs under, absolute
params.backend_dir = null   // the repository's backend/, which the bin/ wrappers import

// One shell word for any string. Slide paths on this cluster carry spaces and
// colons ("BB232181 A3-1 - 2023-09-06 22.15.28.ndpi"), and a val is not
// escaped by Nextflow the way a staged path is.
def shq(value) {
    "'" + value.toString().replace("'", "'\\''") + "'"
}

// `python <interpreter> ${projectDir}/bin/x.py` rather than relying on Nextflow
// putting bin/ on PATH with the exec bit intact — the same choice, for the same
// reason, as anorak-nf: this pipeline reaches the cluster by file copy, and a
// copy that drops the exec bit turns every task into exit 126 after the queue.
// params.python, not python3: the wrappers import the tiler and h5py through
// backend/, which need the interpreter the tile server itself runs under.
def task_cmd(script, mode) {
    "${shq(params.python)} ${shq("${projectDir}/bin/${script}")} ${mode} --config ${shq(params.config)}"
}

// Stubs write the stage markers through the real module, so a -stub run
// exercises the wiring AND the files the server reads, without a slide or a GPU.
def stub_mark(stage) {
    """${shq(params.python)} -c 'import sys; sys.path.insert(0, sys.argv[1]); import hpl_nf_state as s; s.mark_started(sys.argv[2], sys.argv[3]); s.mark_done(sys.argv[2], sys.argv[3], {"stub": True})' ${shq(params.backend_dir)} ${shq(params.outdir)} ${stage}"""
}

// --- Stage 1: tiling ---------------------------------------------------------

process TILE {
    tag   { new File(slide).name }
    label 'hpl_tiling'
    // The throttle the per-stage form called "max concurrent tasks": how many
    // slides tile at once, whatever the queue size allows.
    maxForks params.max_tiling_forks

    input:
    val slide

    output:
    val slide

    script:
    """
    ${task_cmd('hpl_tile.py', 'tile')} --slide ${shq(slide)}
    """

    stub:
    """
    echo "stub tile ${slide}"
    """
}

process TILING_GATE {
    label 'hpl_light'

    input:
    val slides

    output:
    val true

    script:
    """
    ${task_cmd('hpl_tile.py', 'gate')}
    """

    stub:
    """
    ${stub_mark('tiling')}
    """
}

// --- Stage 2: packaging --------------------------------------------------------

process PACKAGE {
    label 'hpl_package'

    input:
    val ready

    output:
    val true

    script:
    // --processes from task.cpus so the process pool matches the allocation,
    // the same pairing submit_packaging_job makes with --cpus-per-task.
    """
    ${task_cmd('hpl_package.py', 'package')} --processes ${task.cpus}
    """

    stub:
    """
    ${stub_mark('packaging')}
    """
}

// --- Stage 3: feature extraction ---------------------------------------------

process EXTRACT_PLAN {
    label 'hpl_light'

    input:
    val ready

    output:
    path 'ranges.txt'

    script:
    """
    ${task_cmd('hpl_extract.py', 'plan')} --ranges-out ranges.txt
    """

    stub:
    def shards = params.stub_extract_shards ?: 1
    """
    ${shq(params.python)} -c 'import sys; n = int(sys.argv[1]); print("skip" if n <= 0 else "all" if n == 1 else "\\n".join(f"{i * 100} {(i + 1) * 100}" for i in range(n)))' ${shards} > ranges.txt
    """
}

process EXTRACT_SHARD {
    tag   { rows }
    label 'hpl_gpu'

    input:
    val rows

    output:
    val rows

    script:
    """
    ${task_cmd('hpl_extract.py', 'shard')} --range ${shq(rows)}
    """

    stub:
    """
    echo "stub extract ${rows}"
    """
}

process EXTRACT_FINISH {
    label 'hpl_light'

    input:
    val shards

    output:
    val true

    script:
    """
    ${task_cmd('hpl_extract.py', 'finish')}
    """

    stub:
    """
    ${stub_mark('extraction')}
    """
}

// --- Stage 4: cluster assignment ---------------------------------------------

process ASSIGN_PLAN {
    // Computes the shared query mean when sharding: one streamed pass over the
    // projections, which is why it has cores and memory of its own.
    label 'hpl_mean'

    input:
    val ready

    output:
    path 'ranges.txt'

    script:
    """
    ${task_cmd('hpl_assign.py', 'plan')} --ranges-out ranges.txt
    """

    stub:
    def shards = params.stub_assign_shards ?: 1
    """
    ${shq(params.python)} -c 'import sys; n = int(sys.argv[1]); print("skip" if n <= 0 else "all" if n == 1 else "\\n".join(f"{i * 100} {(i + 1) * 100}" for i in range(n)))' ${shards} > ranges.txt
    """
}

process ASSIGN_SHARD {
    tag   { rows }
    label 'hpl_assign'

    input:
    val rows

    output:
    val rows

    script:
    // --threads is task.cpus: the container cannot read SLURM_CPUS_PER_TASK
    // (--cleanenv), and Nextflow sets --cpus-per-task from this same number,
    // so the two cannot drift the way they once did.
    """
    ${task_cmd('hpl_assign.py', 'shard')} --range ${shq(rows)} --threads ${task.cpus}
    """

    stub:
    """
    echo "stub assign ${rows}"
    """
}

process ASSIGN_FINISH {
    label 'hpl_light'

    input:
    val shards

    output:
    val true

    script:
    """
    ${task_cmd('hpl_assign.py', 'finish')}
    """

    stub:
    """
    ${stub_mark('assignment')}
    """
}

// --- workflow ----------------------------------------------------------------

workflow {
    // Every one of these is answerable before a single task is queued.
    ['config', 'manifest', 'outdir', 'python', 'backend_dir'].each { name ->
        if (!params[name]) error "Set --${name}. This pipeline is launched by backend/submit_hpl_nf.py, which sets all five."
    }
    if (!file(params.config).exists()) error "No run config at ${params.config}"
    if (!file("${params.backend_dir}/hpl_nf_state.py").exists()) {
        error "No hpl_nf_state.py under ${params.backend_dir}. The backend/ copy on the cluster predates this pipeline — copy it again."
    }
    def manifest = file(params.manifest)
    if (!manifest.exists() || manifest.size() == 0) {
        error "The slide manifest ${params.manifest} is missing or empty — the run has no slides."
    }

    slides = Channel.fromPath(params.manifest)
        .splitText()
        .map { it.trim() }
        .filter { it }

    // Each stage waits on the whole of the one before: collect() holds the
    // next step back until every slide (or shard) has finished, and a failed
    // task ends the run before anything downstream starts — afterok, which is
    // what the per-stage packaging job always used.
    TILE(slides)
    TILING_GATE(TILE.out.collect())
    PACKAGE(TILING_GATE.out)

    // "skip" is a plan saying the output already exists and validates. It is
    // filtered out here rather than handed to a shard, so a finished stage
    // does not queue behind every GPU job to print "nothing to do" — and
    // ifEmpty keeps FINISH running anyway, since it is the step that
    // re-validates the output and writes the stage's done marker.
    EXTRACT_PLAN(PACKAGE.out)
    EXTRACT_SHARD(EXTRACT_PLAN.out.splitText().map { it.trim() }.filter { it && it != 'skip' })
    EXTRACT_FINISH(EXTRACT_SHARD.out.collect().ifEmpty(['skip']))

    ASSIGN_PLAN(EXTRACT_FINISH.out)
    ASSIGN_SHARD(ASSIGN_PLAN.out.splitText().map { it.trim() }.filter { it && it != 'skip' })
    ASSIGN_FINISH(ASSIGN_SHARD.out.collect().ifEmpty(['skip']))
}
