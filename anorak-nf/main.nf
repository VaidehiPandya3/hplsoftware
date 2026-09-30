#!/usr/bin/env nextflow

/*
 * ANORAK growth-pattern segmentation and IASLC grading.
 *
 * Pan et al., "The artificial intelligence-based model ANORAK improves
 * histopathological grading of lung adenocarcinoma", Nature Cancer 5, 347-363
 * (2024).  Model and reference implementation: github.com/xi11/AIgrading.
 *
 * This pipeline calls that implementation unchanged. What it adds is the
 * things a 7,000-slide cohort needs and a single-process script does not: one
 * task per slide, a check between every step that the previous step's output
 * is whole rather than merely present, and a slide list so only the slides
 * worth segmenting are segmented.
 *
 * The one substantive deviation is how a slide is selected. Upstream picks its
 * slide with `sorted(glob(dir/pattern))[nfile]` — an integer index into a
 * directory listing, evaluated separately in prediction, stitching and
 * post-processing. Here every task is given exactly one slide and addresses it
 * by name, so the index cannot drift between steps. See bin/anorak_common.py.
 *
 * Note the processes run bin/ scripts as `python3 ${projectDir}/bin/x.py`
 * rather than relying on Nextflow putting bin/ on PATH and the scripts being
 * executable. On this deployment the pipeline reaches the cluster by file copy,
 * and a copy that drops the exec bit turns every task into exit 126,
 * "Permission denied" — after the queue, once per slide. Invoking the
 * interpreter explicitly cannot fail that way, and costs nothing.
 */

nextflow.enable.dsl = 2

// --- inputs ---------------------------------------------------------------

params.slides_csv    = null   // filtered slide list; the cohort is its slide_id column
params.raw_dir       = null   // directory of whole-slide images, searched recursively
params.outdir        = 'results'
params.anorak_dir    = null   // clone of github.com/xi11/AIgrading
params.slide_column  = 'slide_id'
params.sample_column = 'samples'   // the tumour a slide belongs to; grading aggregates over it
params.tumour_column = 'is_tumour' // if present, every row must be true; see resolveSlides

// Grade what arrived and list what did not, instead of refusing a short
// table. Implied by -process.errorStrategy=ignore, which is what makes
// slides go missing; see the workflow block.
params.allow_missing_slides = false

// Any new value re-checks every published slide against its task output and
// repairs what differs; see PUBLISH_SLIDE.
params.republish = ''

// Upstream's flag, and NOT the output resolution: the effective output is
// exactly twice this for any scanner, so 0.22 gives 0.44 um/px (x20), the
// resolution the model was trained and published at. Changing it changes the
// magnification the model sees.
params.output_mpp    = 0.22

params.patch_size    = 768
params.patch_stride  = 192
params.colour_norm   = true   // the model was trained on Reinhard-normalised input

// --- resolving the slide list --------------------------------------------

/*
 * Map every slide id in the list to a file under raw_dir, or stop.
 *
 * Deliberately eager and deliberately total. A cohort that resolves 7,000 of
 * 7,221 slides runs to completion, publishes a grading table, and is short a
 * fifth of its tumours with nothing in the output to say which — so an
 * unresolved id is fatal here rather than a warning. Ambiguity is fatal for
 * the same reason: two files answering to one id means one of them wins by
 * directory order.
 */
def resolveSlides(rows, rawDir, slideColumn, sampleColumn, tumourColumn) {
    // Extensions upstream's save_cws.single_file_run dispatches on. It returns
    // silently for anything else, so an unlisted slide would be skipped rather
    // than reported.
    def SUPPORTED = ['.svs', '.ndpi', '.mrxs', '.tif', '.tiff', '.png', '.qptiff']

    def index = [:].withDefault { [] }
    file(rawDir).eachFileRecurse { entry ->
        if (entry.isFile() && SUPPORTED.any { entry.name.toLowerCase().endsWith(it) }) {
            index[entry.name] << entry
            index[entry.baseName] << entry
        }
    }
    if (index.isEmpty()) {
        error "No slides with a supported extension under ${rawDir} (looked for ${SUPPORTED.join(', ')})"
    }

    def resolved = []
    def missing = []
    def ambiguous = [:]
    def unsampled = []
    def notTumour = []
    rows.each { row ->
        def slideId = row[slideColumn]?.toString()?.trim()
        if (!slideId) return
        // A slide with no sample is refused rather than defaulted: grading
        // pools slides by sample, so every blank one used to land in a single
        // tumour called '' and be graded as one — a well-formed row, pooling
        // unrelated patients, at the very end of the run.
        if (!row[sampleColumn]?.toString()?.trim()) {
            unsampled << slideId
            return
        }
        // select_tumour_slides.py writes its non-tumour rows too unless given
        // --tumour-only, and nothing downstream reads is_tumour, so a list
        // straight from it grades a cohort's normal slides as tumours. A value
        // this does not recognise as true is refused rather than guessed at.
        if (tumourColumn && row.containsKey(tumourColumn)
                && !(row[tumourColumn]?.toString()?.trim()?.toLowerCase() in ['true', '1', 'yes', 't', 'y'])) {
            notTumour << "${slideId} (${tumourColumn}='${row[tumourColumn]}')"
            return
        }
        def matches = (index[slideId] ?: []).unique { it.toRealPath().toString() }
        if (matches.isEmpty()) {
            missing << slideId
        }
        else if (matches.size() > 1) {
            ambiguous[slideId] = matches
        }
        else {
            // meta is part of every task's cache key, so it carries only what
            // the heavy steps read: the id (their tag and published name) and
            // the file name (their directory inside cws). The sample rides
            // separately and is joined in at SLIDE_PROPORTIONS, the first step
            // that reads it — it used to be in here, and relabelling one
            // tumour in the slide list re-tiled and re-segmented its slides.
            resolved << tuple(
                [ id: slideId, slide_name: matches[0].name ],
                matches[0],
                row[sampleColumn].toString().trim()
            )
        }
    }

    if (unsampled) {
        error """\
            |${unsampled.size()} of ${rows.size()} slides have a blank '${sampleColumn}':
            |${unsampled.take(10).collect { "    ${it}" }.join('\n')}${unsampled.size() > 10 ? "\n    ... and ${unsampled.size() - 10} more" : ''}
            |Grading pools slides by that column, so these would be graded together as one
            |tumour that does not exist. Fill them in, or drop the rows.
            |""".stripMargin()
    }
    if (notTumour) {
        error """\
            |${notTumour.size()} of ${rows.size()} slides are not marked as tumour:
            |${notTumour.take(10).collect { "    ${it}" }.join('\n')}${notTumour.size() > 10 ? "\n    ... and ${notTumour.size() - 10} more" : ''}
            |ANORAK grades adenocarcinoma, and a normal slide pooled into a tumour shifts
            |its proportions. Drop these rows (select_tumour_slides.py --tumour-only), or
            |pass --tumour_column '' if that column means something else in this list.
            |""".stripMargin()
    }

    if (missing) {
        error """\
            |${missing.size()} of ${rows.size()} slides in ${slideColumn} have no file under ${rawDir}:
            |${missing.take(10).collect { "    ${it}" }.join('\n')}${missing.size() > 10 ? "\n    ... and ${missing.size() - 10} more" : ''}
            |Refusing rather than grading the ${resolved.size()} that resolved.
            |""".stripMargin()
    }
    if (ambiguous) {
        error """\
            |${ambiguous.size()} slide ids match more than one file:
            |${ambiguous.take(5).collect { id, paths -> "    ${id}:\n" + paths.collect { "      - ${it}" }.join('\n') }.join('\n')}
            |""".stripMargin()
    }

    def duplicates = resolved.countBy { it[0].slide_name }.findAll { name, n -> n > 1 }
    if (duplicates) {
        error "These slide files are listed more than once and would process into the same output: ${duplicates.keySet().take(5)}"
    }

    // SLIDE_PROPORTIONS names its output after safeName(id), and TUMOUR_GRADE
    // stages every one of them into one directory — so 'A B' and 'A_B', both
    // 'A_B.counts.csv', used to run the whole cohort and then fail on
    // "input file name collision" at the last step, again on every resume.
    // Refused here instead of renamed, because the naming is in cached tasks'
    // script text and changing it would re-run them.
    def collisions = resolved.groupBy { safeName(it[0].id) }.findAll { name, group -> group.size() > 1 }
    if (collisions) {
        error """\
            |${collisions.size()} output names are shared by more than one slide id once
            |characters outside [A-Za-z0-9._-] are replaced by '_':
            |${collisions.take(5).collect { name, group -> "    ${name}.counts.csv <- " + group.collect { "'${it[0].id}'" }.join(', ') }.join('\n')}
            |Rename one of each pair in the slide list and on disk.
            |""".stripMargin()
    }
    return resolved
}

/*
 * The name SLIDE_PROPORTIONS gives a slide's counts file. The process writes
 * the same expression out rather than calling this, and must: its script text
 * is part of every cached task's key.
 */
def safeName(id) {
    return id.replaceAll(/[^A-Za-z0-9._-]/, '_')
}

/*
 * Why a checkpoint is not something generate_gp can load, or null if it is.
 *
 * The Groovy copy of bin/anorak_common.py's checkpoint_problem, which is the
 * definition; tools/test_checkpoint_predicate.py runs both against the same
 * fixtures. Keras load_model takes an HDF5 file or a SavedModel directory, and
 * convert_anorak_model.py --in-place leaves the second at the first's name —
 * which is why this used to be a bare exists() here and an is_file() in the
 * task, and every GPU task refused a checkpoint the launch had accepted.
 */
def checkpointProblem(path) {
    def f = file(path)
    if (f.isDirectory()) {
        return file("${path}/saved_model.pb").isFile() ? null
            : "${path} is a directory but not a SavedModel (no saved_model.pb in it) — a conversion that did not finish?"
    }
    if (!f.isFile()) return "no model checkpoint at ${path}"
    // HDF5's signature, at byte 0 or at 512 and every doubling after it, up
    // to the end of the file — as anorak_common does. This used to stop at
    // 1 MiB, so a checkpoint with a larger user block loaded in the task and
    // was refused here; eight bytes per doubling is ~30 reads for any file.
    def signature = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]
    def size = f.size()
    def offsets = ([0L] + (0..53).collect { 512L << it }).findAll { it + signature.size() <= size }
    def found = java.nio.channels.FileChannel.open(f).withCloseable { channel ->
        offsets.any { at ->
            def buf = readAt(channel, at, signature.size())
            buf.remaining() == signature.size() &&
                (0..<signature.size()).every { i -> (buf.get(i) & 0xff) == signature[i] }
        }
    }
    return found ? null
        : "${path} is a file of ${size} bytes with no HDF5 signature, so it is not a Keras checkpoint — an interrupted or failed download?"
}

/*
 * What a task's output depends on that Nextflow does not see, as a string
 * handed to it as a `val` input so that a change to any of it re-runs the task.
 *
 * Nextflow keys a task on its script *source* — the text as written in this
 * file — its inputs and its container's name. `python3 ${projectDir}/bin/x.py`
 * is none of those, and nor is the AIgrading clone every heavy step imports,
 * the checkpoint the GPU step loads, or the bytes of an image rebuilt at the
 * same path (verified on 26.04.6: a new path re-runs, new content does not).
 * Editing any of them used to change what a task does and not its key, so
 * -resume kept every output of the old version and ran the new one only on
 * slides not yet done — a cohort segmented by two versions of the code, with
 * nothing in the output to say which slides got which. That is the failure
 * CLAUDE.md is written against, and for the heavy steps it was the documented
 * behaviour until 2026-09-25, when the cohort was restarted from nothing and a
 * re-run on a real change became cheaper than not knowing.
 *
 * Everything here is read once per launch, in the head job, so none of it may
 * read a whole large file: see sampledDigest.
 */
def md5Hex(bytes) {
    return java.security.MessageDigest.getInstance('MD5').digest(bytes).encodeHex().toString()
}

/*
 * `length` bytes of an open file from `at`, or fewer at its end. A ByteBuffer
 * rather than an array because the strict-syntax parser takes no `new T[n]`.
 */
def readAt(channel, long at, int length) {
    def buf = java.nio.ByteBuffer.allocate(length)
    // A positional read may return short; `find` is the loop, since the
    // strict-syntax parser takes no `while`. It stops when full or at EOF.
    (0..<1024).find { !buf.hasRemaining() || channel.read(buf, at + buf.position()) < 0 }
    buf.flip()
    return buf
}

/*
 * size + MD5 of a file, or for one over 4 MiB, of its first, middle and last
 * MiB. The checkpoint is hundreds of MB and the inference image ~9 GB, and a
 * whole read of either on every launch is exactly the head-job CephFS reading
 * that hung the run of 2026-09-23. Not mtime: a copy or an rsync without -t
 * changes it, and that would re-segment 7,000 slides for bytes that did not
 * change. Two different builds of a model or an image agreeing in size and in
 * all three samples is not a thing that happens by accident.
 */
def sampledDigest(f) {
    def size = f.size()
    def chunk = 1048576L
    if (size <= 4 * chunk) return "${size}:${md5Hex(f.bytes)}"
    def md = java.security.MessageDigest.getInstance('MD5')
    java.nio.channels.FileChannel.open(f).withCloseable { channel ->
        [0L, size.intdiv(2) - chunk.intdiv(2), size - chunk].each { at ->
            md.update(readAt(channel, at as long, chunk as int))
        }
    }
    return "${size}:${md.digest().encodeHex()}"
}

/*
 * The upstream code one step imports: every .py under the AIgrading
 * directories its wrapper puts on sys.path, plus a colour-normalisation target
 * image kept beside them. __pycache__ is left out because importing the code
 * writes it, and a key that changed because the code ran would re-run
 * everything on every resume.
 */
def upstreamDigest(sub) {
    def root = file("${params.anorak_dir}/${sub}")
    if (!root.isDirectory()) {
        error "No ${sub}/ in ${params.anorak_dir}. The wrappers import upstream code from it; is --anorak_dir a clone of github.com/xi11/AIgrading?"
    }
    def entries = []
    root.eachFileRecurse { f ->
        def rel = root.relativize(f).toString()
        if (f.isFile() && !('__pycache__' in rel.tokenize('/'))
                && (f.name.endsWith('.py') || f.name ==~ /(?i)target.*\.(jpe?g|png|tiff?)/)) {
            entries << "${rel}:${sampledDigest(f)}"
        }
    }
    if (!entries.any { it.contains('.py:') }) error "No .py files under ${root}; the step importing it cannot run."
    return "${sub}:${md5Hex(entries.sort().join('\n').bytes)}"
}

/*
 * The checkpoint, in either form checkpointProblem accepts: an HDF5 file, or a
 * SavedModel directory, which is its graph plus its variables/ shards.
 */
def checkpointIdentity(path) {
    def f = file(path)
    if (!f.isDirectory()) return "checkpoint:${sampledDigest(f)}"
    def parts = ["saved_model.pb:${sampledDigest(file("${path}/saved_model.pb"))}"]
    def vars = file("${path}/variables")
    if (vars.isDirectory()) {
        vars.eachFile { v -> if (v.isFile()) parts << "variables/${v.name}:${sampledDigest(v)}" }
    }
    return "checkpoint:${md5Hex(parts.sort().join('\n').bytes)}"
}

/*
 * A local image's content (Nextflow already keys on its name), and the GPU
 * extras by listing: its *.dist-info names carry package and version, so a
 * re-bootstrap to the same versions changes nothing and a different cv2 does.
 */
def imageIdentity(image) {
    if (!image) return 'image:none'
    def f = file(image.toString())
    return f.isFile() ? "image:${sampledDigest(f)}" : "image:${image}"
}

def extrasIdentity(dir) {
    if (!dir || !file(dir.toString()).isDirectory()) return 'extras:none'
    return "extras:${md5Hex(file(dir.toString()).list().sort().join('\n').bytes)}"
}

/*
 * The bin/ scripts a step runs, each by content. anorak_common.py belongs in
 * every list: each wrapper imports it, and its checks decide what a step
 * accepts.
 */
def codeDigest(names) {
    return names.collect { "${it}:${file("${projectDir}/bin/${it}").text.md5()}" }.join(' ')
}

// --- processes ------------------------------------------------------------

process TILE_SLIDE {
    tag   "${meta.id}"
    label 'process_tiling'
    // No publishDir: PUBLISH_SLIDE publishes the tiles, from a compute node.
    // publishDir ran in the head job, and on the run of 2026-09-16 that is
    // what the head job was doing when it died — 72 lines of "Waiting for file
    // transfers to complete (1 files)" and then "Timed out while waiting to
    // publish outputs" in its .err, the tiling itself long finished. Even as
    // hard links it could leave a slide's directory partial (the head killed
    // mid-slide) or stale (a re-tiled slide keeps its old links), because with
    // overwrite:false an existing directory was skipped, and overwrite:true
    // re-links every cached slide on every resume. See bin/publish_tree.py.
    input:
    tuple val(meta), path(slide)
    val code   // bin/ + generating_tile/ + image; see codeDigest and friends

    output:
    tuple val(meta), path('cws_tiling'), emit: cws

    script:
    // NOT quoted, and that is the whole point. Nextflow escapes a staged
    // `path` when it interpolates it, so a slide called
    // `BB232181 A3-1 - 2023-09-06 22.15.28.ndpi` arrives here already written
    // as `BB232181\\ A3-1\\ ...`. Wrapping that in double quotes does not
    // undo the escaping — bash keeps a backslash literal inside double
    // quotes — so the filename handed to openslide contained backslashes and
    // matched nothing on disk. It fails as
    // `OpenSlideUnsupportedFormatError: Unsupported or missing image file`,
    // which reads as an unreadable slide rather than a missing one, and it
    // fails for every slide in a cohort whose names carry spaces. Unquoted,
    // Nextflow's own escaping is what makes it one shell word.
    // `cws`, `masks` and `final_mask` below are quoted because they are names
    // this pipeline chose ('cws_tiling', 'gp_masks', 'ss1_final'), and
    // `meta.slide_name` is a val, which Nextflow does not escape.
    """
    python3 ${projectDir}/bin/anorak_tile.py \\
        --slide ${slide} \\
        --anorak-dir "${params.anorak_dir}" \\
        --out-dir cws_tiling \\
        --output-mpp ${params.output_mpp}
    """

    stub:
    """
    mkdir -p "cws_tiling/${meta.slide_name}"
    for i in 0 1 2 3; do touch "cws_tiling/${meta.slide_name}/Da\$i.jpg"; done
    touch "cws_tiling/${meta.slide_name}/param.p" "cws_tiling/${meta.slide_name}/Ss1.jpg" "cws_tiling/${meta.slide_name}/FinalScan.ini"
    """
}

process PREDICT_GP {
    tag   "${meta.id}"
    label 'process_gpu'

    input:
    tuple val(meta), path(cws)
    val code   // bin/ + inference_slide/ + checkpoint + image + extras

    output:
    tuple val(meta), path('gp_masks'), emit: masks

    script:
    def normFlag = params.colour_norm ? '' : '--no-colour-norm'
    // Exported here, inside the script, and therefore inside the container —
    // not in a beforeScript, which Nextflow runs on the host before the
    // container is entered, and not left to Singularity's environment
    // pass-through, which a --cleanenv anywhere in the chain would erase.
    // CLAUDE.md's rule for the HPL container is the same one: assume nothing
    // the job needs from the submitting environment survives.
    //
    // The image is NGC TensorFlow, which has TensorFlow and CUDA and does not
    // have cv2, so upstream's `import cv2` in inference_slide/predict_gp.py
    // ended every GPU task in seconds. `${params.gpu_extras}` is a directory
    // of the packages upstream needs and the image lacks, built by
    // tools/bootstrap_gpu_extras.sh.
    // `:+` rather than a bare colon: on an unset PYTHONPATH the plain form
    // leaves a trailing separator, and an empty PYTHONPATH component means
    // the current directory, which here is the task work directory.
    def extras = params.gpu_extras
        ? "export PYTHONPATH=\"${params.gpu_extras}\${PYTHONPATH:+:\$PYTHONPATH}\""
        : "true"
    """
    ${extras}

    python3 ${projectDir}/bin/anorak_predict.py \\
        --cws-dir "${cws}" \\
        --slide-name "${meta.slide_name}" \\
        --anorak-dir "${params.anorak_dir}" \\
        --out-dir gp_masks \\
        --patch-size ${params.patch_size} \\
        --patch-stride ${params.patch_stride} \\
        ${normFlag}
    """

    stub:
    """
    mkdir -p "gp_masks/${meta.slide_name}"
    for i in 0 1 2 3; do touch "gp_masks/${meta.slide_name}/Da\$i.png"; done
    """
}

process SS1_STITCH {
    tag   "${meta.id}"
    label 'process_stitch'
    // No publishDir, for TILE_SLIDE's reason: PUBLISH_SLIDE publishes the
    // mask. (The saveAs that used to be here did at least publish it — the
    // pattern before that, 'ss1_final/*', matched nothing — but a re-stitched
    // slide kept its old mask under overwrite:false.)
    input:
    tuple val(meta), path(cws), path(masks)
    val code   // bin/ + inference_slide/ + image

    output:
    tuple val(meta), path('ss1_final'), emit: final_mask

    script:
    """
    python3 ${projectDir}/bin/anorak_stitch.py \\
        --cws-dir "${cws}" \\
        --mask-dir "${masks}" \\
        --slide-name "${meta.slide_name}" \\
        --anorak-dir "${params.anorak_dir}" \\
        --ss1-dir ss1 \\
        --ss1-final-dir ss1_final
    """

    stub:
    """
    mkdir -p ss1_final
    touch "ss1_final/${meta.slide_name}_Ss1.png"
    """
}

/*
 * Publish one slide's tiles and mask into outdir, complete-or-absent and
 * current — see bin/publish_tree.py for how, and TILE_SLIDE for why this is
 * not a publishDir.
 *
 * A task on a compute node rather than work in the head job, and cached like
 * any other: its key is the two directories' work paths, which are new
 * whenever either was re-made, so a resume that re-tiled nothing runs none of
 * these and links nothing. One task for both, after the stitch, so a cohort
 * costs one of these per slide rather than two; the tiles of a slide that
 * never stitched are therefore not published, which keeps what is published a
 * set that belongs together.
 *
 * `republish` is the repair handle. A published directory damaged by hand
 * after its task finished is not something a cache can see; any new value
 * (--republish 2026-10-01) re-runs every one of these, and each checks its
 * slide by inode and re-links only a slide that differs.
 */
process PUBLISH_SLIDE {
    tag   "${meta.id}"
    label 'process_publish'

    input:
    tuple val(meta), path(cws), path(final_mask)
    val outdir      // absolute; the task runs in its work directory
    val code        // bin/publish_tree.py + anorak_common.py
    val republish

    script:
    """
    python3 ${projectDir}/bin/publish_tree.py \
        --src "${cws}" --root "${outdir}/cws_tiling" --name "${meta.id}"
    python3 ${projectDir}/bin/publish_tree.py \
        --src "${final_mask}" --root "${outdir}/ss1_final" --name "${meta.id}"
    """
}

process SLIDE_PROPORTIONS {
    tag   "${meta.id}"
    label 'process_light'

    input:
    tuple val(meta), path(final_mask)
    val code   // codeDigest(): re-run on an edit to the scripts below

    output:
    path "*.counts.csv", emit: counts

    script:
    // The output name is derived from the slide id, not the slide filename,
    // so a cohort whose files carry spaces and colons still produces names
    // that survive a shell and a spreadsheet.
    def safe = meta.id.replaceAll(/[^A-Za-z0-9._-]/, '_')
    """
    python3 ${projectDir}/bin/slide_proportions.py \\
        --mask "${final_mask}/${meta.slide_name}_Ss1.png" \\
        --slide-id "${meta.id}" \\
        --sample "${meta.sample}" \\
        --out "${safe}.counts.csv"
    """

    stub:
    def safe = meta.id.replaceAll(/[^A-Za-z0-9._-]/, '_')
    """
    echo 'slide_id,sample,pattern_pixels,lepidic_px,papillary_px,acinar_px,cribriform_px,micropapillary_px,solid_px' > "${safe}.counts.csv"
    echo '${meta.id},${meta.sample},600,100,100,100,100,100,100' >> "${safe}.counts.csv"
    """
}

process TUMOUR_GRADE {
    label 'process_light'
    publishDir "${params.outdir}", mode: 'copy'
    // Content, not path+mtime: `expected` is rewritten by every launch, and
    // with the default hash that alone would re-run this on every resume.
    cache 'deep'

    input:
    path counts
    path expected
    val code          // codeDigest(): re-run on an edit to the scripts below
    val allowMissing

    output:
    path 'anorak_slide_proportions.csv', emit: slides
    path 'anorak_tumour_grades.csv',     emit: tumours
    path 'anorak_missing_slides.csv',    emit: missing

    script:
    """
    python3 ${projectDir}/bin/tumour_grade.py \\
        --slide-counts ${counts} \\
        --expected ${expected} \\
        --out-slides anorak_slide_proportions.csv \\
        --out-tumours anorak_tumour_grades.csv \\
        --out-missing anorak_missing_slides.csv \\
        ${allowMissing ? '--allow-missing' : ''}
    """

    stub:
    """
    touch anorak_slide_proportions.csv anorak_tumour_grades.csv anorak_missing_slides.csv
    """
}

// --- workflow -------------------------------------------------------------

workflow {
    // Checked before a single task is queued. A cohort this size is days of
    // cluster time, and every one of these is answerable in milliseconds.
    if (!params.slides_csv) error 'Set --slides_csv to the filtered slide list.'
    if (!params.raw_dir)    error 'Set --raw_dir to the directory of whole-slide images.'
    if (!params.anorak_dir) error 'Set --anorak_dir to a clone of github.com/xi11/AIgrading.'
    if (!file(params.slides_csv).exists()) error "No such file: ${params.slides_csv}"
    if (!file(params.raw_dir).isDirectory()) error "Not a directory: ${params.raw_dir}"

    def checkpoint = "${params.anorak_dir}/models/AIgrading_anorak.h5"
    def checkpointIssue = checkpointProblem(checkpoint)
    if (checkpointIssue) {
        error """\
            |${checkpointIssue}.
            |The checkpoint is at https://zenodo.org/records/15272883 and belongs at that
            |path — predict_gp.generate_gp() builds it from its own location and cannot
            |be pointed elsewhere.
            |""".stripMargin()
    }

    // Checked before splitCsv rather than after. Nextflow's own failure for an
    // empty file is "Missing 'header' in CSV file", which reads as a malformed
    // header and sends you looking at the columns of a file that has none.
    def slide_list = file(params.slides_csv)
    if (!slide_list.exists()) error "No slide list at ${params.slides_csv}"
    if (slide_list.size() == 0) {
        error """
            |The slide list ${params.slides_csv} is empty — zero bytes, not even a header.
            |
            |Where it came from decides what to do. A list written by
            |submit_anorak_nf.py sits beside the run's outputs with a
            |slide_list.selection.json next to it, saying how many slides were
            |chosen and from which source CSV. If that count is non-zero the
            |write was interrupted, so re-submit. If the source CSV is itself
            |empty, the filter that produced it matched nothing.
            |""".stripMargin()
    }

    def rows = slide_list.splitCsv(header: true)
    if (!rows) error "${params.slides_csv} has a header but no slides."
    if (!rows[0].containsKey(params.slide_column)) {
        error "${params.slides_csv} has no '${params.slide_column}' column; found ${rows[0].keySet().join(', ')}"
    }
    // Grading is by tumour, and the sample column is the only thing that says
    // which tumour a slide is. Without it every slide used to be given sample
    // '' and the cohort graded as one tumour.
    if (!rows[0].containsKey(params.sample_column)) {
        error "${params.slides_csv} has no '${params.sample_column}' column (--sample_column); found ${rows[0].keySet().join(', ')}"
    }

    def resolved = resolveSlides(rows, params.raw_dir, params.slide_column, params.sample_column, params.tumour_column)
    log.info "ANORAK: ${resolved.size()} slides, tiles at ${2 * params.output_mpp} um/px"

    // What TUMOUR_GRADE reconciles its counts against. collect() hands it
    // whatever arrived; this is what should have. Written in the work
    // directory, not outdir, because it is an input rather than a result.
    def expected = file("${workflow.workDir}/anorak_expected_slides.csv")
    expected.parent.mkdirs()
    expected.text = (['slide_id,sample'] + resolved.collect { meta, slide, sample ->
        [meta.id, sample].collect { '"' + it.replace('"', '""') + '"' }.join(',')
    }).join('\n') + '\n'

    // -process.errorStrategy=ignore is the documented best-effort sweep, and
    // under it a failed slide simply never reaches TUMOUR_GRADE. That mode
    // gets a table plus a list of what is missing; every other mode gets a
    // refusal, because there a missing slide means something dropped it.
    def strategy = workflow.session.config.navigate('process.errorStrategy')
    def allowMissing = params.allow_missing_slides as boolean || (strategy instanceof CharSequence && strategy.toString() == 'ignore')
    if (allowMissing) log.warn "ANORAK: missing slides will be listed in anorak_missing_slides.csv rather than refused"

    slides_ch = Channel.fromList(resolved.collect { meta, slide, sample -> tuple(meta, slide) })
    def sampleOf = resolved.collectEntries { meta, slide, sample -> [(meta.id): sample] }

    // Each heavy step's code identity (see codeDigest and the functions above
    // it): its wrapper, the upstream directory that wrapper puts on sys.path,
    // and the image it runs in. PREDICT_GP adds the checkpoint and the extras.
    def common  = ['anorak_common.py']
    def tileImg = imageIdentity(params.tiling_container)
    def infer   = upstreamDigest('inference_slide')
    def tileCode    = [codeDigest(['anorak_tile.py'] + common), upstreamDigest('generating_tile'), tileImg].join(' ')
    def predictCode = [codeDigest(['anorak_predict.py'] + common), infer, checkpointIdentity(checkpoint),
                       imageIdentity(params.gpu_container), extrasIdentity(params.gpu_extras)].join(' ')
    def stitchCode  = [codeDigest(['anorak_stitch.py'] + common), infer, tileImg].join(' ')

    TILE_SLIDE(slides_ch, tileCode)
    PREDICT_GP(TILE_SLIDE.out.cws, predictCode)

    // Rejoined on meta rather than carried through PREDICT_GP, so a whole
    // tiled slide is not re-emitted through a channel to be handed back.
    // failOnMismatch: an unmatched slide is one of the two tasks having failed
    // in a way that let the run carry on, which only the ignore sweep permits.
    SS1_STITCH(TILE_SLIDE.out.cws.join(PREDICT_GP.out.masks,
                                      failOnDuplicate: true, failOnMismatch: !allowMissing),
               stitchCode)

    PUBLISH_SLIDE(TILE_SLIDE.out.cws.join(SS1_STITCH.out.final_mask,
                                         failOnDuplicate: true, failOnMismatch: !allowMissing),
                  file(params.outdir).toAbsolutePath().toString(),
                  codeDigest(['publish_tree.py'] + common),
                  params.republish.toString())

    // The sample joins here, the first step that reads it. A lookup rather
    // than a channel join: every slide has one, and a join would have to be
    // told that the slides which failed upstream are allowed to be missing.
    SLIDE_PROPORTIONS(SS1_STITCH.out.final_mask.map { meta, mask -> tuple(meta + [sample: sampleOf[meta.id]], mask) },
                      codeDigest(['slide_proportions.py'] + common))
    TUMOUR_GRADE(SLIDE_PROPORTIONS.out.counts.collect(),
                 expected,
                 codeDigest(['tumour_grade.py'] + common),
                 allowMissing)
}
