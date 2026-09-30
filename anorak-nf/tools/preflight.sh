#!/usr/bin/env bash
# Everything that can be checked before the ANORAK pipeline queues anything.
#
# Each check here exists because it failed once, on the cluster, one at a time,
# after a wait. They are all cheap and they are all answerable in advance:
#
#   a missing AIgrading clone, or a checkpoint
#     the GPU task cannot load                 -> check 3
#   extras that drifted from their lock        -> check 4
#   a slide list with blank samples, normal
#     slides, or ids the run would refuse      -> checks 5, 5b
#   .mrxs slides the run cannot open           -> check 5c
#   an outdir PUBLISH_SLIDE cannot hard-link
#     into from the work directory            -> check 5d
#   a head-job walltime, or any label's
#     largest retry, over what its partition
#     has; a GPU type the cluster does not have -> check 6
#   more jobs than the submit limit allows     -> check 6b
#   GPU tasks that are not isolated            -> check 6c, 9c
#   a path that lists on the login node and
#     does not exist inside the container      -> check 8   (the symlink trap)
#   a container whose TensorFlow cannot load
#     the checkpoint, or cannot see the GPU    -> check 9
#   an inference image with TensorFlow but no
#     cv2, i.e. no upstream imports at all     -> check 9b
#   a GPU task that sees every card on the
#     node rather than the one it was given    -> check 9c
#   a tiling image with no openslide, or a
#     slide with no mpp in its header          -> check 10
#   node-local scratch smaller than a slide    -> check 11
#
# Every value compared is read from the EFFECTIVE config
# (`nextflow config -profile beatson -flat`), never restated here: a preflight
# that checks its own idea of the settings passes on a config nobody loads.
# Checks 8-11 run on a real compute node, inside the real containers, launched
# the way Nextflow launches a task, because that is the only place the answers
# live.
#
#   ./preflight.sh --slides-csv <list.csv> --raw-dir <slides/> --outdir <out/>
#                  [--work-dir <work/>] [--chain N]
#
# Exit status: 0 every check ran and passed; 1 at least one check failed;
# 2 nothing failed but some checks were skipped. 2 is not a pass — a skipped
# check is a question nobody answered. Run it after changing a container, a
# config, or the cluster.
#
# Deliberately no `set -e`: a failed check is counted and the rest still run,
# so one run of this lists every problem rather than the first. Every check
# reports through ok/bad/skip, and the exit status is decided from the count.

set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANORAK_DIR="${ANORAK_REPO_DIR:-}"
SLIDES_CSV=""
RAW_DIR=""
GPU_PARTITION_OVERRIDE="${ANORAK_GPU_PARTITION:-}"
CPU_PARTITION_OVERRIDE="${ANORAK_CPU_PARTITION:-}"
HEAD_PARTITION="${ANORAK_HEAD_PARTITION:-}"
HEAD_TIME_LIMIT="${ANORAK_HEAD_TIME_LIMIT:-2-00:00:00}"
SLIDE_COLUMN=""
SAMPLE_COLUMN=""
CHAIN=""
QUEUE_SIZE=""
OUT_DIR=""
WORK_DIR=""
# Per-tile output sizes for the scratch estimate (check 11), in MB, as a
# (low, high) range. A 2000x2000 RGB JPEG of H&E at upstream's quality lands at
# roughly 0.5-1 MB; a 2000x2000 colour mask PNG of seven flat colours at
# roughly 0.05-0.25 MB. Assumptions, stated in the output, overridable here.
TILE_MB_LOW="${ANORAK_TILE_MB_LOW:-0.5}";  TILE_MB_HIGH="${ANORAK_TILE_MB_HIGH:-1.0}"
MASK_MB_LOW="${ANORAK_MASK_MB_LOW:-0.05}"; MASK_MB_HIGH="${ANORAK_MASK_MB_HIGH:-0.25}"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "$1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slides-csv)     need_value "$@"; SLIDES_CSV="$2"; shift 2 ;;
        --raw-dir)        need_value "$@"; RAW_DIR="$2"; shift 2 ;;
        --anorak-dir)     need_value "$@"; ANORAK_DIR="$2"; shift 2 ;;
        --gpu-partition)  need_value "$@"; GPU_PARTITION_OVERRIDE="$2"; shift 2 ;;
        --cpu-partition)  need_value "$@"; CPU_PARTITION_OVERRIDE="$2"; shift 2 ;;
        --head-partition) need_value "$@"; HEAD_PARTITION="$2"; shift 2 ;;
        --slide-column)   need_value "$@"; SLIDE_COLUMN="$2"; shift 2 ;;
        --sample-column)  need_value "$@"; SAMPLE_COLUMN="$2"; shift 2 ;;
        --chain)          need_value "$@"; CHAIN="$2"; shift 2 ;;
        --queue-size)     need_value "$@"; QUEUE_SIZE="$2"; shift 2 ;;
        --outdir)         need_value "$@"; OUT_DIR="$2"; shift 2 ;;
        --work-dir)       need_value "$@"; WORK_DIR="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
for pair in "--chain:$CHAIN" "--queue-size:$QUEUE_SIZE"; do
    value="${pair#*:}"
    if [[ -n "$value" && ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "${pair%%:*} must be a positive integer, got '$value'" >&2; exit 2
    fi
done

PASS=0; FAIL=0; SKIP=0; WARN=0
SKIPPED=""
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; WARN=$((WARN+1)); }
skip() { printf '  --    %s\n' "$*"; SKIP=$((SKIP+1)); SKIPPED="${SKIPPED}    ${CURRENT_STEP%%.*}: $*"$'\n'; }
CURRENT_STEP=""
step() { CURRENT_STEP="$*"; printf '\n%s\n' "$*"; }
indent() { sed 's/^/        /'; }

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/anorak_preflight.XXXXXX")" || { echo "cannot make a temporary directory" >&2; exit 2; }
trap 'rm -rf "$TMPD"' EXIT

# --- the effective configuration --------------------------------------------
#
# Read once, as `nextflow config -profile beatson -flat`: one `key = value` line
# per setting, after profiles, includes and params are resolved — the values
# the run will use. The flat form rather than the nested one because a nested
# name like `queue` or `memory` appears once per label, and matching it by
# name alone (as this script used to) reads whichever label happens to come
# first. Queried through the small parser below, which is Python 3.6-clean
# because it runs on whatever python3 the login node has.
CFG_FILE="$TMPD/config.flat"
CFG_SOURCE=""
read -r -d '' CFG_PY <<'PY' || true
import json, re, sys
from pathlib import Path

KEY = re.compile(r"^((?:[^\s=']|'[^']*')+) = (.*)$")
SEL = re.compile(r"^process\.'?withLabel:\s*([^']+?)'?\.(\w+)$")
MB = {"kb": 1.0 / 1024, "mb": 1.0, "gb": 1024.0, "tb": 1024.0 ** 2}
SECONDS = {"ms": 0.001, "s": 1, "sec": 1, "second": 1, "seconds": 1,
           "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
           "h": 3600, "hour": 3600, "hours": 3600,
           "d": 86400, "day": 86400, "days": 86400}

def load(path):
    entries, last = {}, None
    for line in Path(path).read_text().splitlines():
        m = KEY.match(line)
        if m:
            last = m.group(1)
            entries[last] = m.group(2)
        elif last is not None:
            entries[last] += "\n" + line   # a multi-line closure
    return entries

def decode(raw):
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "null":
        return None
    if raw in ("true", "false"):
        return raw == "true"
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return re.sub(r"\\(.)", r"\1", raw[1:-1])
    if raw.startswith("[") and raw.endswith("]"):
        items = re.findall(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[^,\s][^,]*", raw[1:-1])
        return [decode(i) for i in items]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d*", raw):
        return float(raw)
    return raw

def show(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(show(v) for v in value)
    return str(value)

def label_raw(entries, label, attr):
    # withLabel selectors are regexes over the label name; the last matching
    # one in file order wins, and process.<attr> is the default beneath them.
    found = None
    for key, raw in entries.items():
        m = SEL.match(key)
        if m and m.group(2) == attr:
            pattern = m.group(1).strip()
            negate = pattern.startswith("!")
            if (re.fullmatch(pattern.lstrip("!"), label) is not None) != negate:
                found = raw
    return found if found is not None else entries.get("process." + attr)

def size_mb(text):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\.?\s*([KMGT]B)\s*", str(text), re.I)
    return float(m.group(1)) * MB[m.group(2).lower()] if m else None

def duration_s(text):
    text = str(text)
    unit = r"(ms|seconds?|sec|s|minutes?|mins?|min|m|hours?|h|days?|d)"
    if not re.fullmatch(r"(\s*\d+(?:\.\d+)?\s*\.?\s*" + unit + r"\s*)+", text):
        return None
    return sum(float(n) * SECONDS[u] for n, u in
               re.findall(r"(\d+(?:\.\d+)?)\s*\.?\s*" + unit, text))

def request(entries, label, kind, attempts):
    """The largest `kind` any attempt of `label` asks for: min(base x attempt,
    ceiling), evaluated from the closure text nextflow.config writes."""
    convert = size_mb if kind == "memory" else duration_s
    units = r"(KB|MB|GB|TB)" if kind == "memory" else r"(ms|s|sec|min|m|h|d)"
    raw = label_raw(entries, label, kind)
    value = decode(raw)
    if value is None:
        return None, "not set (Slurm's default applies)"
    text = str(value)
    if not text.lstrip().startswith("{"):
        v = convert(text)
        return (v, text) if v is not None else (None, "cannot read " + text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*\.\s*" + units + r"\b(\s*\*\s*task\.attempt)?", text)
    if not m:
        return None, "cannot evaluate " + " ".join(text.split())
    base = convert(m.group(1) + m.group(2))
    scaled = base * attempts if m.group(3) else base
    how = "%s%s" % (m.group(1) + "." + m.group(2), " x %d attempts" % attempts if m.group(3) else "")
    c = re.search(r"params\.(max_\w+)", text)
    if c:
        ceiling = decode(entries.get("params." + c.group(1)))
        if ceiling is not None:
            cv = convert(str(ceiling))
            if cv is None:
                return None, "cannot read params.%s = %r" % (c.group(1), ceiling)
            how += ", capped at params.%s = %s" % (c.group(1), ceiling)
            scaled = min(scaled, cv)
    return scaled, how

def fallback(pipeline_dir):
    """Literal settings read from the config text, for when nextflow cannot
    run. Expressions it cannot evaluate are reported, not guessed at."""
    out, unevaluated = {}, []
    for name in ("nextflow.config", "conf/beatson.config"):
        path = Path(pipeline_dir) / name
        if not path.is_file():
            continue
        text = re.sub(r"/\*.*?\*/", "", path.read_text(), flags=re.S)
        stack, pending = [], None
        for line in text.splitlines():
            line = re.sub(r"(^|\s)//.*$", "", line).strip()
            if not line:
                continue
            if pending is not None:
                pending[1] += "\n" + line
                if pending[1].count("{") <= pending[1].count("}"):
                    out[pending[0]] = pending[1]; pending = None
                continue
            if line == "}":
                stack and stack.pop(); continue
            if line.endswith("{") and "=" not in line:
                block = line[:-1].strip()
                m = re.match(r"withLabel\s*:\s*['\"]?([^'\"]+?)['\"]?$", block)
                stack.append("'withLabel:%s'" % m.group(1) if m else block)
                continue
            if stack and stack[0] == "profiles":
                continue
            m = re.match(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", line)
            if not m:
                continue
            key = ".".join(stack + [m.group(1)])
            if m.group(2).startswith("{") and m.group(2).count("{") > m.group(2).count("}"):
                pending = [key, m.group(2)]; continue
            out[key] = m.group(2)
    for key, raw in list(out.items()):
        s = raw.strip()
        ref = re.fullmatch(r"params\.(\w+)", s)
        if ref:
            out[key] = out.get("params." + ref.group(1), "null")
        elif s.startswith("{"):
            continue   # a closure: kept as text, which is what `request` reads
        elif "${" in s or (s[:1] not in "'\"[" and decode(s) == s):
            unevaluated.append(key); out[key] = "null"
    for key, raw in out.items():
        print("%s = %s" % (key, raw))
    if unevaluated:
        print("could not evaluate without nextflow: " + ", ".join(unevaluated), file=sys.stderr)

def main(argv):
    command = argv[1]
    if command == "fallback":
        return fallback(argv[2])
    if command == "labels":   # the labels main.nf's processes carry
        text = Path(argv[2]).read_text()
        seen = []
        for label in re.findall(r"^\s*label\s+['\"]([^'\"]+)['\"]", text, re.M):
            if label not in seen:
                seen.append(label)
        print("\n".join(seen)); return 0
    entries = load(argv[2])
    if command == "get":
        print(show(decode(entries.get(argv[3])))); return 0
    if command == "label":
        print(show(decode(label_raw(entries, argv[3], argv[4])))); return 0
    if command == "request":
        value, how = request(entries, argv[3], argv[4], int(argv[5]))
        print("%s|%s" % ("" if value is None else int(round(value)), how)); return 0
    return 2

if __name__ == "__main__":
    if sys.argv[1] == "size_mb":   # no config file for this one
        v = size_mb(sys.argv[2]); print("" if v is None else int(round(v))); sys.exit(0)
    sys.exit(main(sys.argv) or 0)
PY
cfg()       { python3 -c "$CFG_PY" get "$CFG_FILE" "$1"; }
cfg_label() { python3 -c "$CFG_PY" label "$CFG_FILE" "$1" "$2"; }
cfg_request() { python3 -c "$CFG_PY" request "$CFG_FILE" "$1" "$2" "$3"; }
unset_or_null() { [[ -z "$1" || "$1" == "null" ]]; }

# --- host ------------------------------------------------------------------

step "1-2. Tooling and pipeline"
if command -v nextflow >/dev/null; then ok "nextflow: $(nextflow -v 2>&1 | head -1)"
else bad "nextflow is not on PATH. If it arrives via modules, set HPL_NEXTFLOW_PRELUDE (e.g. 'module load nextflow') so the head job loads it too."; fi
if command -v java >/dev/null; then ok "java: $(java -version 2>&1 | head -1)"
else bad "java is not on PATH; nextflow needs it"; fi
# bin/anorak_common.py and scan_slide_headers.py use `from __future__ import
# annotations`, which is 3.7+. A login node's system python3 can be older, and
# then check 3 would report a broken checkpoint when the checkpoint is fine.
if python3 -c 'import sys; sys.exit(sys.version_info < (3, 7))' 2>/dev/null; then
    ok "python3: $(python3 -V 2>&1)"
else bad "python3 on PATH is $(python3 -V 2>&1 || echo missing); this script needs 3.7+ — activate the cluster conda env first"; fi
[[ -f "$PIPELINE_DIR/main.nf" ]] && ok "pipeline: $PIPELINE_DIR" || bad "no main.nf under $PIPELINE_DIR"

step "Effective configuration"
if command -v nextflow >/dev/null \
        && (cd "$PIPELINE_DIR" && nextflow config -profile beatson -flat .) > "$CFG_FILE" 2> "$TMPD/config.err" \
        && grep -q ' = ' "$CFG_FILE"; then
    CFG_SOURCE="nextflow config -profile beatson -flat"
    ok "read from \`$CFG_SOURCE\` ($(grep -c '^[^ ]* = ' "$CFG_FILE") settings)"
else
    if command -v nextflow >/dev/null; then
        bad "\`nextflow config -profile beatson\` failed, so the pipeline would not start either:"
        indent < "$TMPD/config.err" | tail -15
    fi
    # Falls back to reading literal settings out of the two config files, so
    # the rest of this still says something. Anything computed (runOptions is
    # built from bind_roots) cannot be read that way, and is named.
    python3 -c "$CFG_PY" fallback "$PIPELINE_DIR" > "$CFG_FILE" 2> "$TMPD/fallback.err"
    CFG_SOURCE="the config files' text (nextflow unavailable)"
    warn "values below are read from $CFG_SOURCE — only literals; an override in a profile or on the command line is invisible to this"
    [[ -s "$TMPD/fallback.err" ]] && warn "$(cat "$TMPD/fallback.err")"
fi

LABELS=()
while IFS= read -r label; do [[ -n "$label" ]] && LABELS+=("$label"); done \
    < <(python3 -c "$CFG_PY" labels "$PIPELINE_DIR/main.nf" 2>/dev/null)
PARAM_CPU_PARTITION="$(cfg params.cpu_partition)"
PARAM_GPU_PARTITION="$(cfg params.gpu_partition)"
# The queue a label's tasks go to, with --cpu-partition/--gpu-partition
# applied the way the matching --cpu_partition/--gpu_partition on the
# nextflow command line would apply them: to the labels that read that param.
label_queue() {
    local q; q="$(cfg_label "$1" queue)"
    if [[ -n "$CPU_PARTITION_OVERRIDE" && "$q" == "$PARAM_CPU_PARTITION" ]]; then q="$CPU_PARTITION_OVERRIDE"
    elif [[ -n "$GPU_PARTITION_OVERRIDE" && "$q" == "$PARAM_GPU_PARTITION" ]]; then q="$GPU_PARTITION_OVERRIDE"; fi
    printf '%s\n' "$q"
}
# Labels that ask Slurm for a GPU, read from their clusterOptions rather than
# assumed from a name.
is_gpu_label() { [[ "$(cfg_label "$1" clusterOptions)" == *gres*gpu* || -n "$(cfg_label "$1" accelerator)" ]]; }
GPU_LABEL=""; CPU_LABEL=""
for label in ${LABELS[@]+"${LABELS[@]}"}; do
    if is_gpu_label "$label"; then [[ -z "$GPU_LABEL" ]] && GPU_LABEL="$label"
    else [[ -z "$CPU_LABEL" ]] && CPU_LABEL="$label"; fi
done
GPU_PARTITION="${GPU_PARTITION_OVERRIDE:-${GPU_LABEL:+$(label_queue "$GPU_LABEL")}}"
GPU_PARTITION="${GPU_PARTITION:-${PARAM_GPU_PARTITION:-gpu}}"
CPU_PARTITION="${CPU_PARTITION_OVERRIDE:-${CPU_LABEL:+$(label_queue "$CPU_LABEL")}}"
CPU_PARTITION="${CPU_PARTITION:-${PARAM_CPU_PARTITION:-compute}}"
if (( ${#LABELS[@]} )); then
    ok "labels in main.nf: ${LABELS[*]} (GPU: ${GPU_LABEL:-none}, queue $GPU_PARTITION; CPU queue $CPU_PARTITION)"
else bad "found no process labels in $PIPELINE_DIR/main.nf, so no per-label check below can run"; fi

step "3. AIgrading clone and checkpoint"
CHECKPOINT=""
if [[ -z "$ANORAK_DIR" ]]; then
    bad "set ANORAK_REPO_DIR or pass --anorak-dir"
else
    CHECKPOINT="$ANORAK_DIR/models/AIgrading_anorak.h5"
    [[ -d "$ANORAK_DIR" ]] && ok "clone: $ANORAK_DIR" || bad "no clone at $ANORAK_DIR"
    [[ -f "$ANORAK_DIR/generating_tile/save_cws.py" && -f "$ANORAK_DIR/inference_slide/predict_gp.py" ]] \
        && ok "generating_tile/ and inference_slide/ present" \
        || bad "$ANORAK_DIR does not look like github.com/xi11/AIgrading"
    # bin/anorak_common.py's checkpoint_problem, the one test main.nf and the
    # GPU task also apply. A bare -e here once passed a SavedModel directory
    # that every GPU task then refused.
    if verdict="$(python3 "$PIPELINE_DIR/bin/anorak_common.py" checkpoint "$CHECKPOINT" 2>&1)"; then
        ok "checkpoint loadable in form ($(du -sh "$CHECKPOINT" 2>/dev/null | cut -f1))"
    else
        bad "$verdict — https://zenodo.org/records/15272883"
    fi
fi

step "4. Containers"
GPU_IMAGE="$(cfg params.gpu_container)"
TILING_IMAGE="$(cfg params.tiling_container)"
NATIVE_BIN="$(cfg params.native_env_bin)"
GPU_EXTRAS="$(cfg params.gpu_extras)"
BINDS="$(cfg singularity.runOptions)"
WHITELIST="$(cfg singularity.envWhitelist)"
# Split the way Nextflow splits runOptions — on whitespace — but into an array,
# so a later unquoted expansion cannot also glob it.
read -r -a BIND_ARGS <<< "$BINDS"
SING="$(command -v singularity || command -v apptainer || true)"

# gpu_container is required: nothing outside a CUDA-12 image can drive these
# cards. tiling_container being unset is a *choice* — the CPU steps run in the
# cluster conda env — so it is not a failure, it just moves the question to
# "are those imports actually there", which check 10 answers by importing them.
if unset_or_null "$GPU_IMAGE"; then
    bad "gpu_container is unset in conf/beatson.config"; GPU_IMAGE=""
elif [[ -r "$GPU_IMAGE" ]]; then ok "gpu_container: $GPU_IMAGE"
else bad "gpu_container is set to $GPU_IMAGE, which is not readable"; fi

if unset_or_null "$GPU_EXTRAS"; then
    skip "gpu_extras unset — the inference image is assumed to have every module upstream imports (check 9b tests that)"
    GPU_EXTRAS=""
elif [[ -d "$GPU_EXTRAS" ]]; then
    ok "gpu_extras: $GPU_EXTRAS"
    # The lock bootstrap_gpu_extras.sh writes, against what is in the
    # directory now. The directory is shared and pip --target overwrites in
    # place, so a re-run of the bootstrap months later can change what every
    # later GPU task imports with nothing in the run to say so.
    LOCK="$GPU_EXTRAS/anorak_extras.lock"
    lock_image="$(sed -n 's/^# image: //p' "$LOCK" 2>/dev/null | head -1)"
    if [[ ! -f "$LOCK" ]]; then
        bad "no $LOCK — re-run tools/bootstrap_gpu_extras.sh, which writes it"
    elif [[ -n "$GPU_IMAGE" && "$lock_image" != "$GPU_IMAGE" ]]; then
        # What the extras hold is decided by what the image lacks, so a lock
        # made against another image describes the wrong gap.
        bad "$LOCK was built against '${lock_image:-an unrecorded image}', not gpu_container $GPU_IMAGE — re-run tools/bootstrap_gpu_extras.sh --refresh"
    elif [[ -z "$SING" || -z "$GPU_IMAGE" ]]; then
        skip "singularity/apptainer or the image is not available here, so the extras lock is not compared"
    elif "$SING" exec --cleanenv ${BIND_ARGS[@]+"${BIND_ARGS[@]}"} "$GPU_IMAGE" \
            python3 -m pip freeze --path "$GPU_EXTRAS" > "$TMPD/freeze.raw" 2> "$TMPD/freeze.err"; then
        LC_ALL=C sort "$TMPD/freeze.raw" > "$TMPD/freeze.now"
        grep -v '^#' "$LOCK" | LC_ALL=C sort > "$TMPD/freeze.lock"
        if cmp -s "$TMPD/freeze.now" "$TMPD/freeze.lock"; then
            ok "gpu_extras matches its lock ($(grep -c . "$TMPD/freeze.lock") packages: $(paste -sd' ' "$TMPD/freeze.lock"))"
        else
            bad "gpu_extras has drifted from $LOCK (< lock, > directory now):"
            diff "$TMPD/freeze.lock" "$TMPD/freeze.now" | grep '^[<>]' | indent
        fi
    else
        bad "could not list the packages in $GPU_EXTRAS from inside $GPU_IMAGE:"
        indent < "$TMPD/freeze.err" | tail -5
    fi
else bad "gpu_extras is $GPU_EXTRAS, which does not exist — build it with tools/bootstrap_gpu_extras.sh"; fi

if unset_or_null "$TILING_IMAGE"; then
    TILING_IMAGE=""
    ok "tiling_container unset — CPU steps run natively (checked in 10)"
    if ! unset_or_null "$NATIVE_BIN"; then
        [[ -x "$NATIVE_BIN/python3" ]] && ok "native_env_bin: $NATIVE_BIN" \
            || bad "native_env_bin is $NATIVE_BIN but there is no python3 in it"
    else
        # Not fatal, but worth saying: the tasks would then use whatever PATH
        # they inherit, and a Slurm job's environment comes from whatever
        # submitted it rather than from the shell reading this.
        NATIVE_BIN=""
        skip "native_env_bin unset — tasks will use the inherited PATH"
    fi
elif [[ -r "$TILING_IMAGE" ]]; then ok "tiling_container: $TILING_IMAGE"
else bad "tiling_container is set to $TILING_IMAGE, which is not readable"; fi
unset_or_null "$NATIVE_BIN" && NATIVE_BIN=""

step "5. Slide list and slides"
RESULTS_ROOT="${ANORAK_RESULTS_ROOT:-}"
[[ -z "$SLIDE_COLUMN"  ]] && SLIDE_COLUMN="$(cfg params.slide_column)"
[[ -z "$SAMPLE_COLUMN" ]] && SAMPLE_COLUMN="$(cfg params.sample_column)"
TUMOUR_COLUMN="$(cfg params.tumour_column)"
SLIDE_COLUMN="${SLIDE_COLUMN:-slide_id}"; SAMPLE_COLUMN="${SAMPLE_COLUMN:-samples}"
if [[ -n "$SLIDES_CSV" ]]; then
    if [[ ! -f "$SLIDES_CSV" ]]; then
        bad "no slide list at $SLIDES_CSV"
    else
        # Parsed as CSV, by column name. This used to grep the header for the
        # substring "slide_id", which passes a column called "slide_ids",
        # ignores --slide_column, and never looked at the samples column that
        # grading pools by — a blank one of which graded every such slide as
        # one tumour. The rules are main.nf's; step 5b runs main.nf itself.
        if report="$(python3 - "$SLIDES_CSV" "$SLIDE_COLUMN" "$SAMPLE_COLUMN" "$TUMOUR_COLUMN" 2>"$TMPD/list.err" <<'PY'
import csv, re, sys
path, slide_col, sample_col, tumour_col = sys.argv[1:5]
with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows, header = list(reader), reader.fieldnames or []
problems = []
for col, flag in ((slide_col, "--slide_column"), (sample_col, "--sample_column")):
    if col not in header:
        problems.append("no '%s' column (%s); header is %s" % (col, flag, header))
if not problems:
    cell = lambda row, col: (row.get(col) or "").strip()
    listed = [(cell(r, slide_col), r) for r in rows if cell(r, slide_col)]
    if not listed:
        problems.append("no slides: every '%s' is blank" % slide_col)
    blank = [i for i, r in listed if not cell(r, sample_col)]
    if blank:
        problems.append("%d slides with a blank '%s', e.g. %s" % (len(blank), sample_col, blank[:3]))
    if tumour_col and tumour_col in header:
        normal = [i for i, r in listed if cell(r, tumour_col).lower() not in ("true", "1", "yes", "t", "y")]
        if normal:
            problems.append("%d slides not marked '%s' true, e.g. %s" % (len(normal), tumour_col, normal[:3]))
    names = {}
    for i, _ in listed:
        names.setdefault(re.sub(r"[^A-Za-z0-9._-]", "_", i), set()).add(i)
    clash = [sorted(v) for v in names.values() if len(v) > 1]
    if clash:
        problems.append("%d groups of ids share an output name, e.g. %s" % (len(clash), clash[:2]))
    print("%d slides, %d samples" % (len(listed), len({cell(r, sample_col) for _, r in listed})))
print("\n".join(problems), file=sys.stderr)
sys.exit(1 if problems else 0)
PY
)"; then
            ok "slide list: $SLIDES_CSV — $report; '$SLIDE_COLUMN' and '$SAMPLE_COLUMN' present, no blank samples${TUMOUR_COLUMN:+, every '$TUMOUR_COLUMN' true or absent}"
        else
            [[ -s "$TMPD/list.err" ]] || echo "the slide list could not be read" > "$TMPD/list.err"
            while IFS= read -r line; do [[ -n "$line" ]] && bad "$SLIDES_CSV: $line"; done < "$TMPD/list.err"
        fi
        # A list living under the results root is a previous run's own copy,
        # not a source. Feeding one back in makes the input mutable by the
        # process consuming it: a second submission truncates the file the
        # first run's head job is reading, and Nextflow reports a missing CSV
        # header from a file that looks fine by the time anyone looks at it.
        # submit_anorak_nf.py refuses this outright; caught here it costs a
        # sentence instead of a queue slot.
        if [[ -n "$RESULTS_ROOT" && "$(readlink -f "$SLIDES_CSV")" == "$(readlink -f "$RESULTS_ROOT")"/* ]]; then
            bad "$SLIDES_CSV is inside ANORAK_RESULTS_ROOT — that is a previous run's own copy of a list, not a source. Point at the list the cohort came from."
        elif [[ -n "$RESULTS_ROOT" ]]; then
            ok "the list is outside the results root, so it is a source not an output"
        fi
    fi
else skip "no --slides-csv given"; fi
if [[ -n "$RAW_DIR" ]]; then
    [[ -d "$RAW_DIR" ]] && ok "raw slides: $RAW_DIR" || bad "no raw directory at $RAW_DIR"
else skip "no --raw-dir given"; fi

step "5b. The workflow's own launch checks (nextflow -preview)"
# Everything main.nf refuses before queueing — every id resolving to exactly
# one file, the checkpoint, the columns, the samples — asked of main.nf itself
# rather than restated here. -preview evaluates the workflow body and submits
# nothing. In a throwaway directory, so it leaves no .nextflow/ history or
# work directory behind for a later -resume to find.
if [[ -n "$SLIDES_CSV" && -n "$RAW_DIR" && -n "$ANORAK_DIR" ]] && command -v nextflow >/dev/null; then
    scratch="$TMPD/preview"; mkdir -p "$scratch"
    if (cd "$scratch" && nextflow -q run "$PIPELINE_DIR" -profile beatson -preview \
            --slides_csv "$SLIDES_CSV" --raw_dir "$RAW_DIR" --anorak_dir "$ANORAK_DIR" \
            --slide_column "$SLIDE_COLUMN" --sample_column "$SAMPLE_COLUMN" \
            --outdir "$scratch/results" -work-dir "$scratch/work") > "$scratch/out" 2>&1; then
        ok "main.nf accepts this list, raw directory and checkpoint"
    else
        bad "main.nf refuses this launch:"; indent < "$scratch/out" | head -30
    fi
else skip "needs --slides-csv, --raw-dir, --anorak-dir and nextflow"; fi

step "5c. .mrxs slides"
# MIRAX is a file plus a directory of the same stem holding the pixel data, and
# a task is handed the file alone — staged into its work directory as a
# symlink, beside which that directory does not exist. Whether main.nf lets
# them in at all is read from main.nf; which of the listed slides are .mrxs is
# resolved by main.nf's own rules (scan_slide_headers.py --resolve-only).
MRXS_OK=""
if [[ -n "$SLIDES_CSV" && -n "$RAW_DIR" && -f "$SLIDES_CSV" && -d "$RAW_DIR" ]]; then
    python3 "$PIPELINE_DIR/tools/scan_slide_headers.py" --resolve-only \
        --slides-csv "$SLIDES_CSV" --raw-dir "$RAW_DIR" --slide-column "$SLIDE_COLUMN" \
        > "$TMPD/resolve.out" 2>&1
    summary="$(sed -n 's/^RESOLVE_SUMMARY //p' "$TMPD/resolve.out" | tail -1)"
    stance="$(python3 - "$PIPELINE_DIR/main.nf" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
lines = [re.sub(r"(^|\s)//.*$", "", l) for l in code.splitlines()]
listed = re.search(r"def\s+SUPPORTED\s*=\s*\[([^\]]*)\]", code)
if not listed or ".mrxs" not in listed.group(1):
    print("unsupported: main.nf's SUPPORTED list does not include .mrxs"); sys.exit()
other = [(n, l.strip()) for n, l in enumerate(lines, 1)
         if "mrxs" in l.lower() and not re.search(r"def\s+SUPPORTED", l)]
if other:
    print("refused: main.nf special-cases .mrxs at line %d: %s" % other[0]); sys.exit()
print("supported: .mrxs is in main.nf's SUPPORTED list")
PY
)"
    if [[ -z "$summary" ]]; then
        bad "could not resolve the slide list against $RAW_DIR:"; indent < "$TMPD/resolve.out" | tail -10
    else
        read -r n_mrxs n_nodata examples <<< "$(python3 -c '
import json, sys
s = json.loads(sys.argv[1])
print(len(s["mrxs"]), len(s["mrxs_without_data_dir"]), ",".join((s["mrxs_without_data_dir"] or s["mrxs"])[:3]) or "-")' "$summary")"
        if (( n_mrxs == 0 )); then
            ok "no .mrxs among the listed slides ($stance)"
        elif [[ "$stance" != supported:* ]]; then
            bad "$n_mrxs listed slides are .mrxs (e.g. $examples), and ${stance#*: } — drop them from the list or convert them"
        elif (( n_nodata > 0 )); then
            bad "$n_nodata of $n_mrxs .mrxs slides have no data directory (<stem>/Slidedat.ini) beside them, e.g. $examples — openslide cannot open them anywhere"
        else
            MRXS_OK=1
            # The tiler opens the slide by its real path (bin/anorak_tile.py
            # resolves the staged symlink), so the data directory beside it is
            # found; upstream was always handed the real directory.
            ok "$n_mrxs listed slides are .mrxs (e.g. $examples), each with its data directory beside it"
        fi
    fi
else skip "needs --slides-csv and --raw-dir"; fi

step "5d. outdir and work directory on one filesystem"
# PUBLISH_SLIDE hard-links every tile from the work directory into outdir, and
# a hard link cannot cross filesystems: across two, every slide's publish job
# refuses (exit 65), and that finishes the run at its first stitched slide.
# The submitter always puts work/ inside the outdir; a launch by hand need not.
# Compared by the mount df reports for the nearest existing ancestor, after
# resolving symlinks, because /hpc-home/.../long-term-scratch is a symlink
# into /mnt/cephfs-lts and the two names are one filesystem.
mount_of() {
    local p
    p=$(python3 -c 'import os, sys
p = os.path.realpath(sys.argv[1])
while not os.path.exists(p):
    p = os.path.dirname(p)
print(p)' "$1") || return 1
    df -P "$p" 2>&1 | awk 'NR == 2 { print $NF } NR > 2 { exit }'
}
if [[ -n "$OUT_DIR" ]]; then
    work="${WORK_DIR:-${OUT_DIR%/}/work}"
    out_mount=$(mount_of "$OUT_DIR"); work_mount=$(mount_of "$work")
    if [[ -z "$out_mount" || -z "$work_mount" ]]; then
        bad "could not tell which filesystem $OUT_DIR ('${out_mount:-?}') or $work ('${work_mount:-?}') is on"
    elif [[ "$out_mount" != "$work_mount" ]]; then
        bad "outdir $OUT_DIR is on $out_mount but the work directory $work is on $work_mount; PUBLISH_SLIDE hard-links between them and will refuse every slide"
    else
        ok "outdir and work directory are both on $out_mount"
    fi
else skip "needs --outdir (the work directory defaults to <outdir>/work, as the submitter sets it)"; fi

# A Slurm time (D-HH:MM:SS, D-HH, HH:MM:SS, MM:SS, MM) in seconds, or empty.
slurm_seconds() {
    local t="$1" d=0 h=0 m=0 s=0 a b c
    case "$t" in infinite|UNLIMITED|unlimited) echo 999999999; return ;; esac
    [[ "$t" =~ ^[0-9]+-[0-9]+(:[0-9]+){0,2}$ || "$t" =~ ^[0-9]+(:[0-9]+){0,2}$ ]] || { echo ""; return; }
    if [[ "$t" == *-* ]]; then
        d="${t%%-*}"; IFS=: read -r a b c <<< "${t#*-}"
        h="${a:-0}"; m="${b:-0}"; s="${c:-0}"
    else
        IFS=: read -r a b c <<< "$t"
        if [[ -n "${c:-}" ]]; then h=$a; m=$b; s=$c
        elif [[ -n "${b:-}" ]]; then m=$a; s=$b
        else m=$a; fi
    fi
    echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}
hms() { local s="$1"; printf '%dd%02dh%02dm' $((s/86400)) $((s%86400/3600)) $((s%3600/60)); }
part_maxtime() { sinfo -h -p "$1" -o "%l" 2>/dev/null | head -1; }
part_max()     { sinfo -h -p "$1" -o "$2" 2>/dev/null | tr -d '+' | sort -n | tail -1; }
# GPU gres tokens a partition advertises, one "type count" per line; the type
# is empty for an untyped gres. sinfo repeats a type on a node line (and
# across lines) so these are de-duplicated rather than counted.
part_gpu_types() {
    sinfo -h -p "$1" -o "%G" 2>/dev/null | grep -oE 'gpu(:[A-Za-z0-9_.-]+)?:[0-9]+' \
        | sed -E 's/^gpu:([^:]+):[0-9]+$/\1/; s/^gpu:[0-9]+$//' | sort -u
}

step "6. Partition limits against every label's largest request"
DEFAULT_PARTITION=""
if ! command -v sinfo >/dev/null; then
    skip "sinfo not available here"
else
    DEFAULT_PARTITION="$(sinfo -h -o '%P' 2>/dev/null | sed -n 's/\*$//p' | head -1)"
    # The head job is submitted with no --partition (submit_anorak_nf.py), so
    # it lands on the cluster default; it is the one job sbatch refuses
    # outright rather than a task failing mid-run.
    HEAD_PARTITION="${HEAD_PARTITION:-${DEFAULT_PARTITION:-$CPU_PARTITION}}"
    head_cap="$(part_maxtime "$HEAD_PARTITION")"
    head_s="$(slurm_seconds "$HEAD_TIME_LIMIT")"
    if [[ -z "$head_cap" ]]; then bad "sinfo knows no partition '$HEAD_PARTITION' (head job)"
    elif [[ -z "$head_s" ]]; then bad "ANORAK_HEAD_TIME_LIMIT '$HEAD_TIME_LIMIT' is not a Slurm time"
    elif (( head_s <= $(slurm_seconds "$head_cap") )); then
        ok "head job $HEAD_TIME_LIMIT fits $HEAD_PARTITION (MaxTime $head_cap)"
    else
        bad "head job asks $HEAD_TIME_LIMIT but $HEAD_PARTITION MaxTime is $head_cap — sbatch will refuse it outright. Set ANORAK_HEAD_TIME_LIMIT."
    fi

    # Every request is min(base x attempt, max_*) — see nextflow.config — and
    # evaluated here per label from the resolved config, for the last attempt
    # any label can reach. This used to compare a hard-coded 24h and 36h, then
    # the max_* ceilings alone: the first said nothing about memory or the
    # stitch step, and the second failed a ceiling no attempt ever reaches.
    for label in ${LABELS[@]+"${LABELS[@]}"}; do
        q="$(label_queue "$label")"; q="${q:-$DEFAULT_PARTITION}"
        retries="$(cfg_label "$label" maxRetries)"
        attempts=$(( ${retries:-1} + 1 ))
        cap="$(part_maxtime "$q")"
        if [[ -z "$cap" ]]; then bad "$label: sinfo knows no partition '${q:-<none>}'"; continue; fi
        IFS='|' read -r t_s t_how <<< "$(cfg_request "$label" time "$attempts")"
        if [[ -z "$t_s" && "$t_how" == "not set"* ]]; then
            ok "$label: time $t_how"
        elif [[ -z "$t_s" ]]; then
            skip "$label: time — $t_how; check by hand against $q MaxTime $cap"
        elif (( t_s <= $(slurm_seconds "$cap") )); then
            ok "$label on $q: attempt $attempts asks at most $(hms "$t_s") ($t_how), within MaxTime $cap"
        else
            bad "$label on $q: attempt $attempts asks $(hms "$t_s") ($t_how), over MaxTime $cap — that retry pends or is refused. Lower the ceiling in conf/beatson.config"
        fi
        node_mb="$(part_max "$q" '%m')"
        IFS='|' read -r m_mb m_how <<< "$(cfg_request "$label" memory "$attempts")"
        if [[ -z "$m_mb" && "$m_how" == "not set"* ]]; then
            ok "$label: memory $m_how"
        elif [[ -z "$m_mb" ]]; then
            skip "$label: memory — $m_how; check by hand against $q's nodes"
        elif [[ -z "$node_mb" ]]; then
            bad "$label: sinfo reports no node memory for $q"
        elif (( m_mb <= node_mb )); then
            ok "$label on $q: attempt $attempts asks at most ${m_mb} MB ($m_how), within its largest node (${node_mb} MB)"
        else
            bad "$label on $q: attempt $attempts asks ${m_mb} MB ($m_how), more than any $q node has (${node_mb} MB) — it pends forever while the head job looks healthy"
        fi
        cpus="$(cfg_label "$label" cpus)"; node_cpus="$(part_max "$q" '%c')"
        if [[ -n "$cpus" && -n "$node_cpus" && "$cpus" =~ ^[0-9]+$ ]] && (( cpus > node_cpus )); then
            bad "$label asks $cpus cpus; the largest $q node has $node_cpus"
        fi
        if is_gpu_label "$label"; then
            types="$(part_gpu_types "$q")"
            gpu_type="$(cfg params.gpu_type)"
            if ! sinfo -h -p "$q" -o "%G" 2>/dev/null | grep -q 'gpu'; then
                bad "$label asks for --gres=gpu but $q advertises no GPUs — it would pend forever as ReqNodeNotAvail"
            elif [[ -n "$gpu_type" ]] && ! grep -qxF -- "$gpu_type" <<< "$types"; then
                bad "gpu_type '$gpu_type' is not a GPU type on $q (it has: $(grep . <<< "$types" | paste -sd, -)). Type names are exact; a wrong one is inert and the job queues forever"
            else
                ok "$label on $q: GPUs advertised (${gpu_type:-any type}; types: $(grep . <<< "$types" | paste -sd, - || echo untyped))"
            fi
        fi
    done
fi

step "6b. Submit limits against what this run keeps queued"
# Nextflow keeps up to queueSize task jobs in Slurm at once, and a --chain of N
# is N head jobs — one running, N-1 pending on a dependency, all of which count
# against a submit limit. An sbatch refused for QOSMaxSubmitJobPerUserLimit
# fails that task, and on its third refusal ends the run.
SUBMITTER="$PIPELINE_DIR/../backend/submit_anorak_nf.py"
if [[ -z "$CHAIN" ]]; then
    CHAIN="$(sed -nE 's/.*"--chain", *type=int, *default=([0-9]+).*/\1/p' "$SUBMITTER" 2>/dev/null | head -1)"
    chain_from="submit_anorak_nf.py's --chain default"
    [[ -z "$CHAIN" ]] && { CHAIN=1; chain_from="assumed; backend/submit_anorak_nf.py is not beside the pipeline — pass --chain"; }
else chain_from="--chain"; fi
if [[ -z "$QUEUE_SIZE" ]]; then
    QUEUE_SIZE="$(cfg 'executor.$slurm.queueSize')"; qs_from="executor.\$slurm.queueSize"
    [[ -z "$QUEUE_SIZE" ]] && { QUEUE_SIZE="$(cfg executor.queueSize)"; qs_from="executor.queueSize"; }
    [[ -z "$QUEUE_SIZE" ]] && { QUEUE_SIZE=100; qs_from="Nextflow's default, nothing set"; }
else qs_from="--queue-size"; fi
if ! [[ "$QUEUE_SIZE" =~ ^[0-9]+$ ]]; then
    skip "queueSize is '$QUEUE_SIZE' ($qs_from), not a number this can compare — pass --queue-size"
elif ! command -v sacctmgr >/dev/null; then
    skip "sacctmgr not available here"
else
    user="${USER:-$(id -un)}"
    need=$(( QUEUE_SIZE + CHAIN ))
    echo "        queueSize $QUEUE_SIZE ($qs_from) + $CHAIN head job(s) ($chain_from) = $need"
    defacct="$(sacctmgr -n -P show user "$user" format=DefaultAccount 2>/dev/null | head -1)"
    if ! sacctmgr -n -P show assoc where user="$user" \
            format=Account,Partition,MaxSubmitJobs,MaxJobs,QOS,DefaultQOS > "$TMPD/assoc" 2> "$TMPD/assoc.err"; then
        skip "sacctmgr show assoc failed, so the submit limit is unknown: $(head -2 "$TMPD/assoc.err" | paste -sd' ' -)"
    elif ! grep -q . "$TMPD/assoc"; then
        skip "sacctmgr lists no association for $user, so the submit limit is unknown"
    else
        # Rows for the account jobs will be charged to; all rows if that is
        # unknown, which can only make the limit found stricter.
        grep -q "^${defacct}|" "$TMPD/assoc" 2>/dev/null && [[ -n "$defacct" ]] \
            && grep "^${defacct}|" "$TMPD/assoc" > "$TMPD/assoc.use" || cp "$TMPD/assoc" "$TMPD/assoc.use"
        QOS_SCOPES=()   # "qos|partition-or-*" pairs to check
        while IFS='|' read -r acct part msub mjobs qos defqos; do
            if [[ -n "$msub" ]]; then
                if (( need <= msub )); then ok "association $acct${part:+/$part}: MaxSubmitJobs $msub >= $need"
                else bad "association $acct${part:+/$part}: MaxSubmitJobs is $msub, below the $need jobs this run keeps submitted — lower executor.queueSize in conf/beatson.config (or --chain)"; fi
            else ok "association $acct${part:+/$part}: MaxSubmitJobs unlimited"; fi
            [[ -n "$mjobs" ]] && (( mjobs < need )) && warn "association $acct: MaxJobs $mjobs — at most that many run at once; the rest pend, which is slow but not a failure"
            # The QOS a job gets when it names none: the association's default,
            # else 'normal' where it is allowed, else the first one listed.
            use="$defqos"
            if [[ -z "$use" ]]; then
                if [[ ",$qos," == *",normal,"* ]]; then use=normal; else use="${qos%%,*}"; fi
            fi
            [[ -n "$use" ]] && QOS_SCOPES+=("$use|*")
        done < "$TMPD/assoc.use"
        # A partition's own QOS applies to every job on that partition, on
        # top of the job's QOS; its per-user limit sees only those jobs.
        if command -v scontrol >/dev/null; then
            for part in $(for l in ${LABELS[@]+"${LABELS[@]}"}; do label_queue "$l"; done; echo "$HEAD_PARTITION"); do
                pq="$(scontrol show partition "$part" -o 2>/dev/null | grep -oE '(^| )QoS=[^ ]+' | sed 's/.*QoS=//')"
                [[ -n "$pq" && "$pq" != "N/A" ]] && QOS_SCOPES+=("$pq|$part")
            done
        fi
        while IFS= read -r scope; do
            [[ -z "$scope" ]] && continue
            name="${scope%%|*}"; part="${scope#*|}"
            if [[ "$part" == "*" ]]; then kind="the jobs' QOS"; else kind="partition $part's QOS"; fi
            if [[ "$part" == "*" ]]; then n=$need; what="all $need"
            else
                n=0
                for l in ${LABELS[@]+"${LABELS[@]}"}; do [[ "$(label_queue "$l")" == "$part" ]] && { n=$QUEUE_SIZE; break; }; done
                [[ "$HEAD_PARTITION" == "$part" ]] && n=$(( n + CHAIN ))
                what="the $n on $part"
            fi
            row="$(sacctmgr -n -P show qos "$name" format=Name,MaxSubmitPU,MaxJobsPU 2>"$TMPD/qos.err" | head -1)"
            if [[ -z "$row" ]]; then skip "QOS $name: sacctmgr returned nothing ($(head -1 "$TMPD/qos.err"))"; continue; fi
            IFS='|' read -r _ qsub qjobs <<< "$row"
            if [[ -z "$qsub" ]]; then ok "QOS $name ($kind): MaxSubmitPU unlimited"
            elif (( n <= qsub )); then ok "QOS $name ($kind): MaxSubmitPU $qsub >= $what"
            else bad "QOS $name ($kind): MaxSubmitPU is $qsub, below $what this run keeps submitted — lower executor.queueSize in conf/beatson.config (or --chain)"; fi
            [[ -n "$qjobs" ]] && (( qjobs < n )) && warn "QOS $name: MaxJobsPU $qjobs — at most that many run at once; the rest pend"
        done < <(printf '%s\n' ${QOS_SCOPES[@]+"${QOS_SCOPES[@]}"} | sort -u)
        current="$(squeue -h -u "$user" -o '%i' 2>/dev/null | grep -c . || true)"
        (( current > 0 )) && warn "$user already has $current jobs in the queue; they count against the same limits"
    fi
fi

step "6c. GPU isolation"
# Informational, then one hard check. Where Slurm constrains devices with
# cgroups, a task physically cannot touch another task's card; where it does
# not, CUDA_VISIBLE_DEVICES is the only thing keeping two PREDICT_GP on one node
# apart, and it reaches the container only if envWhitelist names it.
if command -v scontrol >/dev/null && scontrol show config > "$TMPD/scontrol" 2>/dev/null; then
    proctrack="$(sed -nE 's/^ProctrackType[[:space:]]*=[[:space:]]*//p' "$TMPD/scontrol" | head -1)"
    taskplugin="$(sed -nE 's/^TaskPlugin[[:space:]]*=[[:space:]]*//p' "$TMPD/scontrol" | head -1)"
    constrain="$(sed -nE 's/^[[:space:]]*ConstrainDevices[[:space:]]*=[[:space:]]*//p' "$TMPD/scontrol" | head -1)"
    from="scontrol show config"
    if [[ -z "$constrain" ]]; then
        conf_dir="$(sed -nE 's/^SLURM_CONF[[:space:]]*=[[:space:]]*//p' "$TMPD/scontrol" | head -1)"
        for f in "${conf_dir%/*}/cgroup.conf" /etc/slurm/cgroup.conf; do
            if [[ -r "$f" ]]; then
                constrain="$(sed -nE 's/^[[:space:]]*ConstrainDevices[[:space:]]*=[[:space:]]*([A-Za-z]+).*/\1/p' "$f" | tail -1)"
                from="$f"; constrain="${constrain:-no (not set)}"; break
            fi
        done
    fi
    echo "        ProctrackType=${proctrack:-?} TaskPlugin=${taskplugin:-?} ConstrainDevices=${constrain:-unknown} (from $from)"
    if [[ "$(tr '[:upper:]' '[:lower:]' <<< "$constrain")" == yes* ]]; then
        ok "Slurm confines each job to its own GPUs (ConstrainDevices=yes)"
    else
        warn "Slurm does not confine devices (ConstrainDevices=${constrain:-unknown}) — every task on a node can open every card, so CUDA_VISIBLE_DEVICES is the only isolation"
    fi
else skip "scontrol not available here, so device confinement is not reported"; fi
if tr ',' '\n' <<< "$WHITELIST" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -qx CUDA_VISIBLE_DEVICES; then
    ok "singularity.envWhitelist carries CUDA_VISIBLE_DEVICES ('$WHITELIST')"
else
    bad "singularity.envWhitelist is '${WHITELIST}', without CUDA_VISIBLE_DEVICES — Nextflow's env - launch drops it and every GPU task on a node computes on GPU 0"
fi

step "7. Can a compute node submit jobs?"
# The Nextflow head job submits every task itself. Where a compute node cannot
# reach the controller it starts, submits nothing, and waits out its limit.
# Asked of the partition the head job runs on, and with srun's own errors kept:
# an srun that never started is not a node that cannot submit.
if command -v srun >/dev/null; then
    out="$(srun ${HEAD_PARTITION:+-p "$HEAD_PARTITION"} -t 1 -n 1 \
            bash -c 'echo PROBE_STARTED; command -v sbatch >/dev/null && squeue --version >/dev/null && echo SUBMIT_OK' 2>&1)"
    if grep -q '^SUBMIT_OK$' <<< "$out"; then ok "a compute node on ${HEAD_PARTITION:-the default partition} can run sbatch and squeue"
    elif ! grep -q '^PROBE_STARTED$' <<< "$out"; then
        bad "srun itself failed, so this says nothing about the node:"; indent <<< "$out" | tail -5
    else bad "a compute node could not run sbatch/squeue — the head job would submit nothing and wait:"; indent <<< "$out" | tail -5; fi
else skip "srun not available here"; fi

# --- inside the containers, on a real node ---------------------------------
#
# A containerised task, launched the way Nextflow launches one: `env -` with
# only PATH, TMP/TMPDIR and singularity.envWhitelist passed (as
# SINGULARITYENV_<name>), then `singularity exec <runOptions> <image>`, then
# the task script — which for PREDICT_GP starts by prepending gpu_extras to
# whatever PYTHONPATH the image sets. Every on-node check goes through this one
# launcher, so none of them can pass on an environment no task gets: the old
# checks added --nv themselves (passing a config whose runOptions lacked it),
# and let the host environment into the container.
#   args: image extras whitelist n_binds binds... -- command...
read -r -d '' TASK_LAUNCH <<'SH' || true
image=$1 extras=$2 whitelist=$3 nbind=$4; shift 4
binds=(); while [ "$nbind" -gt 0 ]; do binds+=("$1"); shift; nbind=$((nbind - 1)); done
[ "$1" = "--" ] && shift
envs=()
for name in $(printf '%s' "$whitelist" | tr ',' ' '); do
    if [ -n "${!name+x}" ]; then envs+=("SINGULARITYENV_$name=${!name}"); fi
done
echo "launch: on ${SLURMD_NODENAME:-$(hostname)}, host CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}, passing [${envs[*]}]" >&2
exec env - PATH="$PATH" ${TMP:+"SINGULARITYENV_TMP=$TMP"} ${TMPDIR:+"SINGULARITYENV_TMPDIR=$TMPDIR"} "${envs[@]}" \
    singularity exec "${binds[@]}" "$image" /bin/bash -c '
        if [ -n "$1" ]; then export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"; fi
        shift; cd / && exec "$@"' _ "$extras" "$@"
SH
# srun <srun options...> -- <image> <extras> <command...>
task_srun() {
    local opts=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do opts+=("$1"); shift; done
    shift
    local image="$1" extras="$2"; shift 2
    srun ${opts[@]+"${opts[@]}"} bash -c "$TASK_LAUNCH" _ "$image" "$extras" "$WHITELIST" \
        "${#BIND_ARGS[@]}" ${BIND_ARGS[@]+"${BIND_ARGS[@]}"} -- "$@"
}
GPU_SRUN=(-p "$GPU_PARTITION" --gres=gpu:1)

if ! command -v srun >/dev/null; then
    step "8-11. On-node checks"
    skip "need srun; run this from the cluster"
else

step "8. Paths visible INSIDE the container"
# The one that looks impossible from outside. /hpc-home/.../long-term-scratch is
# a symlink into /mnt/cephfs-lts, and a symlink inside a container resolves
# against the container's filesystem — so binding one side leaves a directory
# that lists fine from the login node and does not exist inside the job.
#
# One srun per image, every path handed over as an argument rather than pasted
# into a command string, and srun's own output kept. This used to be one srun
# per path with 2>/dev/null, so an srun that never ran — no allocation, a bad
# partition, singularity not on the node's PATH — was reported as "invisible
# inside the container". Each image is asked for what its own tasks open.
probe_paths() {   # label image extras srun-opts... -- paths...
    local what="$1" image="$2" extras="$3"; shift 3
    local opts=(); while [[ "$1" != "--" ]]; do opts+=("$1"); shift; done; shift
    local out
    out="$(task_srun "${opts[@]}" -t 5 -- "$image" "$extras" \
        sh -c 'for p in "$@"; do [ -e "$p" ] || echo "MISSING $p"; done; echo PROBE_RAN' _ "$@" 2>&1)"
    if ! grep -q '^PROBE_RAN$' <<< "$out"; then
        bad "$what: the probe did not run inside $image, so this says nothing about paths:"
        indent <<< "$out" | tail -10
    elif grep -q '^MISSING ' <<< "$out"; then
        bad "$what: invisible inside the container: $(sed -n 's/^MISSING //p' <<< "$out" | paste -sd, -) — check runOptions binds BOTH the symlink and its target (realpath)"
    else ok "$what: all $# paths resolve inside $image"; fi
}
if [[ -z "$GPU_IMAGE" ]]; then
    skip "no gpu_container set"
else
    gpu_paths=("$PIPELINE_DIR")
    [[ -n "$ANORAK_DIR" ]] && gpu_paths+=("$ANORAK_DIR" "$CHECKPOINT")
    [[ -n "$GPU_EXTRAS" ]] && gpu_paths+=("$GPU_EXTRAS")
    probe_paths "inference ($GPU_LABEL)" "$GPU_IMAGE" "" "${GPU_SRUN[@]}" -- "${gpu_paths[@]}"
fi
if [[ -n "$TILING_IMAGE" ]]; then
    tile_paths=("$PIPELINE_DIR")
    [[ -n "$ANORAK_DIR" ]] && tile_paths+=("$ANORAK_DIR")
    [[ -n "$RAW_DIR" ]] && tile_paths+=("$RAW_DIR")
    probe_paths "tiling" "$TILING_IMAGE" "" -p "$CPU_PARTITION" -- "${tile_paths[@]}"
fi

step "9. GPU and checkpoint, inside the inference container"
if [[ -z "$GPU_IMAGE" || -z "$ANORAK_DIR" ]]; then
    skip "need gpu_container and --anorak-dir"
elif task_srun "${GPU_SRUN[@]}" -t 20 -- "$GPU_IMAGE" "$GPU_EXTRAS" \
        python3 "$PIPELINE_DIR/tools/check_anorak_model.py" --checkpoint "$CHECKPOINT" --require-gpu; then
    ok "checkpoint loads, predicts, and a GPU is visible — with gpu_extras on PYTHONPATH, as the task has it"
else bad "see the output above (check_anorak_model.py, or srun/singularity if it never started)"; fi

step "9b. Upstream's inference imports, inside that container"
# Check 9 loads the checkpoint, which needs TensorFlow and nothing else, and
# passed while every GPU task was dying on `import cv2`. Importing the module
# upstream actually imports is the only form of this check that covers its
# whole list — a probe for cv2 alone would pass and say nothing about the next
# missing one. The clone's path is an argument, not pasted into the source.
if [[ -z "$GPU_IMAGE" || -z "$ANORAK_DIR" ]]; then
    skip "need gpu_container and --anorak-dir"
elif task_srun "${GPU_SRUN[@]}" -t 10 -- "$GPU_IMAGE" "$GPU_EXTRAS" \
        python3 -c 'import sys; sys.path.insert(0, sys.argv[1] + "/inference_slide"); import predict_gp' "$ANORAK_DIR"; then
    ok "predict_gp imports inside the inference container"
else
    bad "predict_gp does not import inside $GPU_IMAGE (module named above). Add it to PINNED in tools/bootstrap_gpu_extras.sh and re-run that script."
fi

step "9c. One GPU per task, launched the way Nextflow launches it"
# Without CUDA_VISIBLE_DEVICES in envWhitelist every GPU task on a node saw
# every card and computed on GPU 0. Run on a real --gres=gpu:1 allocation
# through the launcher above, so it fails if the whitelist loses the name.
if [[ -z "$GPU_IMAGE" ]]; then
    skip "no gpu_container set"
else
    gpu_out="$(task_srun "${GPU_SRUN[@]}" -t 10 -- "$GPU_IMAGE" "$GPU_EXTRAS" python3 -c '
import os, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("container CUDA_VISIBLE_DEVICES=%s" % os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("GPUS_VISIBLE=%d" % len(gpus))' 2>&1)"
    grep -E 'CUDA_VISIBLE_DEVICES|GPUS_VISIBLE' <<< "$gpu_out" | indent
    visible="$(sed -n 's/^GPUS_VISIBLE=//p' <<< "$gpu_out")"
    if [[ "$visible" == 1 ]]; then ok "a --gres=gpu:1 task sees exactly one GPU inside the container"
    elif [[ -z "$visible" ]]; then bad "the probe did not run:"; tail -5 <<< "$gpu_out" | indent
    else bad "a --gres=gpu:1 task sees $visible GPUs — CUDA_VISIBLE_DEVICES is not reaching the container (singularity.envWhitelist is '${WHITELIST}')"; fi
fi

step "10. openslide and real slides, where tiling will actually run"
SCAN_SUMMARY=""
if [[ -z "$RAW_DIR" ]]; then
    skip "need --raw-dir"
else
    # Opens actual slides and reads the two header fields tiling refuses
    # without, rather than only importing openslide: a cohort whose headers
    # carry no mpp fails every task, and it is answerable here.
    #
    # With a slide list, every listed slide, through scan_slide_headers.py —
    # which resolves ids by main.nf's rules and asks anorak_tile.py's own
    # function for the tile count, at the run's output_mpp. This used to open
    # only the first file under --raw-dir, by lowercase extension, with the
    # directory pasted into the Python source, so one quote in a path was a
    # syntax error.
    output_mpp="$(cfg params.output_mpp)"
    if [[ -n "$SLIDES_CSV" ]]; then
        cmd=(python3 "$PIPELINE_DIR/tools/scan_slide_headers.py" --slides-csv "$SLIDES_CSV"
             --raw-dir "$RAW_DIR" --slide-column "$SLIDE_COLUMN" --output-mpp "${output_mpp:-0.22}"
             --machine-summary)
        what="every listed slide reports objective and mpp"
    else
        cmd=(python3 -c '
import os, sys, openslide
ext = (".ndpi", ".svs", ".mrxs", ".tif", ".tiff", ".png", ".qptiff")
found = sorted(os.path.join(d, f) for d, _, fs in os.walk(sys.argv[1]) for f in fs if f.lower().endswith(ext))
if not found:
    sys.exit("no slides found under %s" % sys.argv[1])
s = openslide.OpenSlide(found[0])
mpp = s.properties.get(openslide.PROPERTY_NAME_MPP_X)
obj = s.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
print("%s: objective %s, mpp %s" % (os.path.basename(found[0]), obj, mpp))
if not mpp or not obj:
    sys.exit("this slide reports no mpp/objective; tiling would silently change resolution")
' "$RAW_DIR")
        what="the first slide reports objective and mpp (pass --slides-csv to check them all)"
    fi
    # Run exactly the way the task will: in the container if one is set,
    # otherwise natively with native_env_bin on PATH (the beforeScript's
    # export). Checking the other one would prove nothing about the one that
    # runs.
    if [[ -n "$TILING_IMAGE" ]]; then
        where="inside $TILING_IMAGE"
        task_srun -p "$CPU_PARTITION" -t 60 -- "$TILING_IMAGE" "" "${cmd[@]}" 2>&1 | tee "$TMPD/scan.out"
    else
        where="natively${NATIVE_BIN:+ with $NATIVE_BIN on PATH}"
        srun -p "$CPU_PARTITION" -t 60 bash -c 'if [ -n "$1" ]; then export PATH="$1:$PATH"; fi; shift; exec "$@"' \
            _ "$NATIVE_BIN" "${cmd[@]}" 2>&1 | tee "$TMPD/scan.out"
    fi
    rc=${PIPESTATUS[0]}
    SCAN_SUMMARY="$(sed -n 's/^SCAN_SUMMARY //p' "$TMPD/scan.out" | tail -1)"
    if (( rc == 0 )); then ok "openslide works $where, and $what"
    else bad "see above ($where)"; fi
fi

step "11. Node-local scratch against the largest slide"
# `scratch = true` runs a task in a directory on the node's own disk ($TMPDIR,
# else /tmp) and copies the outputs back at the end, so a slide's whole output
# must fit there first — alongside every other task the node is running.
# Asked of a real node on each partition concerned, against the slide sizes
# check 10 read off the headers.
ESTIMATE_PY='
import json, sys
s = json.loads(sys.argv[1]); avail_mb = int(sys.argv[2]) / 1024.0
low, high, per_node = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
big = s["largest_tiles"]
print("%d %d %d %d %s" % (big * low, big * high, per_node * s["mean_tiles"] * high, avail_mb, s["largest_slide"]))'
scratch_of() { local s; s="$(cfg_label "$1" scratch)"; [[ -z "$s" || "$s" == false ]] && return 1; printf '%s\n' "$s"; }
proc_label() {   # the label main.nf gives a process
    python3 -c 'import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"process\s+" + sys.argv[2] + r"\s*\{(.*?)\n\}", text, re.S)
l = m and re.search(r"^\s*label\s+[\x27\"]([^\x27\"]+)", m.group(1), re.M)
print(l.group(1) if l else "")' "$PIPELINE_DIR/main.nf" "$1"
}
TILE_LABEL="$(proc_label TILE_SLIDE)"; PREDICT_LABEL="$(proc_label PREDICT_GP)"
any_scratch=""
for label in ${LABELS[@]+"${LABELS[@]}"}; do
    scratch_of "$label" >/dev/null || continue
    any_scratch=1
    if [[ "$label" == "$TILE_LABEL" ]]; then low=$TILE_MB_LOW high=$TILE_MB_HIGH what="2000x2000 JPEG tiles at ${TILE_MB_LOW}-${TILE_MB_HIGH} MB each"
    elif [[ "$label" == "$PREDICT_LABEL" ]]; then low=$MASK_MB_LOW high=$MASK_MB_HIGH what="mask PNGs at ${MASK_MB_LOW}-${MASK_MB_HIGH} MB per tile"
    else ok "$label: scratch = $(scratch_of "$label"); writes one mask or table per slide, not estimated"; continue; fi
    if [[ -z "$SCAN_SUMMARY" ]]; then skip "$label: scratch is on, but the slide sizes need check 10's scan of the whole list (--slides-csv)"; continue; fi
    q="$(label_queue "$label")"; opts=(-p "$q"); is_gpu_label "$label" && opts+=(--gres=gpu:1)
    df_out="$(srun "${opts[@]}" -t 5 bash -c '
        d="$1"
        case "$d" in true) d="${TMPDIR:-/tmp}" ;; \$*) d="$(eval echo "$d")" ;; esac
        echo "SCRATCH_NODE=${SLURMD_NODENAME:-$(hostname)}"; echo "SCRATCH_DIR=$d"
        df -Pk "$d" | awk "NR == 2 { print \"SCRATCH_AVAIL_KB=\" \$4 }"' _ "$(scratch_of "$label")" 2>&1)"
    node="$(sed -n 's/^SCRATCH_NODE=//p' <<< "$df_out")"; dir="$(sed -n 's/^SCRATCH_DIR=//p' <<< "$df_out")"
    avail="$(sed -n 's/^SCRATCH_AVAIL_KB=//p' <<< "$df_out")"
    if [[ -z "$avail" ]]; then bad "$label: could not read scratch free space on a $q node:"; indent <<< "$df_out" | tail -5; continue; fi
    # How many of these tasks one node runs at once: its cores over the task's
    # cpus, its memory over the task's first-attempt memory, its GPUs.
    read -r n_cpu n_mem n_gres <<< "$(sinfo -h -n "$node" -o '%c %m %G' 2>/dev/null | head -1 | tr -d '+')"
    t_cpu="$(cfg_label "$label" cpus)"; t_mem="$(cfg_request "$label" memory 1)"; t_mem="${t_mem%%|*}"
    per_node=$(( ${n_cpu:-1} / ${t_cpu:-1} ))
    [[ -n "$t_mem" && -n "${n_mem:-}" ]] && (( n_mem / t_mem < per_node )) && per_node=$(( n_mem / t_mem ))
    if is_gpu_label "$label"; then
        n_gpu="$(grep -oE 'gpu(:[A-Za-z0-9_.-]+)?:[0-9]+' <<< "${n_gres:-}" | head -1 | sed 's/.*://')"
        (( ${n_gpu:-1} < per_node )) && per_node=${n_gpu:-1}
    fi
    (( per_node < 1 )) && per_node=1
    read -r need_low need_high packed avail_mb largest <<< "$(python3 -c "$ESTIMATE_PY" "$SCAN_SUMMARY" "$avail" "$low" "$high" "$per_node")"
    echo "        $node:$dir has ${avail_mb} MB free; largest slide $largest needs ${need_low}-${need_high} MB ($what); ~$per_node such tasks per node need ~${packed} MB"
    if (( need_low > avail_mb )); then
        bad "$label: the largest slide's output (${need_low} MB even at the low estimate) does not fit the ${avail_mb} MB free in scratch on $node — set scratch = false for $label, or point it at a larger disk"
    elif (( need_high > avail_mb )); then
        warn "$label: the largest slide's output may not fit scratch on $node (${need_high} MB at the high estimate, ${avail_mb} MB free)"
    elif (( packed > avail_mb )); then
        warn "$label: one slide fits, but $per_node concurrent tasks on one node (~${packed} MB) may not (${avail_mb} MB free on $node)"
    else ok "$label: scratch on $q has room — ${avail_mb} MB free on $node against ${need_high} MB for the largest slide"; fi
done
[[ -z "$any_scratch" ]] && ok "no label uses scratch"

fi   # srun available

step "Summary"
printf '  %d ok, %d failed, %d warnings, %d skipped\n' "$PASS" "$FAIL" "$WARN" "$SKIP"
if (( FAIL > 0 )); then
    printf '\nFix the failures above before running the pipeline. Every one of them\n'
    printf 'costs more to discover after the queue than before it.\n'
    exit 1
fi
if (( SKIP > 0 )); then
    printf '\nNothing failed, but these did not run, so they are not known to pass:\n%s' "$SKIPPED"
    printf 'Supply what they need (or run from a cluster login node) and run this again.\n'
    exit 2
fi
printf '\nReady. Run a subset first:\n'
printf '  nextflow run %q -profile beatson --slides_csv <list> --raw_dir <dir> --anorak_dir %q --outdir <out>\n' \
    "$PIPELINE_DIR" "${ANORAK_DIR:-<anorak-dir>}"
