#!/usr/bin/env bash
# Build the directory of Python packages the inference image does not have.
#
# The image is NGC TensorFlow 23.03 (TF 2.11, CUDA 12.1, Python 3.8). It has
# TensorFlow and numpy, and not cv2 — so `import cv2` at the top of upstream's
# inference_slide/predict_gp.py ends every GPU task in seconds. $HOME is not
# writable inside it, so the packages go in a bound directory on scratch and
# reach the interpreter through PYTHONPATH (params.gpu_extras). This is the
# same arrangement as HPL_CONTAINER_EXTRAS in submit_feature_extraction.py.
#
#   ./bootstrap_gpu_extras.sh [--extras-dir DIR] [--image SIF] [--anorak-dir DIR]
#                             [--lock FILE | --refresh]
#
# Every package is installed at an exact version (PINNED below), and the set a
# verified build ended up with is written to <extras-dir>/anorak_extras.lock.
# The directory is shared by every GPU task of every run and pip --target
# overwrites in place, so an unpinned re-run months later would change what a
# half-finished cohort's remaining slides import, silently. So once a lock
# exists a re-run installs exactly it; --lock FILE installs another (to rebuild
# a lost directory from a copy); --refresh re-resolves from PINNED and writes a
# new one. A lock records the image it was built against and is refused for
# any other, because what the directory must hold is the image's gap.
# preflight.sh check 4 fails if the directory no longer matches its lock.
#
# Two things it deliberately does NOT do:
#
#   - install a package the image already has. It probes each import inside
#     the image first. PYTHONPATH sits ahead of site-packages, so an extras
#     copy of numpy would shadow the one TensorFlow was built against, and
#     that breaks at a depth nobody wants to debug.
#
#   - resolve dependencies. Every install is --no-deps, so pip cannot pull a
#     numpy or an h5py in behind a package that merely mentions it. A module
#     that is still missing after that is named by the verification below, in
#     one second, instead of on the GPU queue — and is refused, not fetched,
#     unless PINNED gives it an exact version.
#
# Run it from the login node: pip needs the internet and this needs no GPU.
# Exit status: 0 verified and locked; 1 did not verify (no lock written);
# 2 bad arguments or environment.

set -uo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTRAS_DIR=""
IMAGE=""
ANORAK_DIR="${ANORAK_REPO_DIR:-}"
LOCK_IN=""
REFRESH=0

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "$1 needs a value" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extras-dir) need_value "$@"; EXTRAS_DIR="$2"; shift 2 ;;
        --image)      need_value "$@"; IMAGE="$2"; shift 2 ;;
        --anorak-dir) need_value "$@"; ANORAK_DIR="$2"; shift 2 ;;
        --lock)       need_value "$@"; LOCK_IN="$2"; shift 2 ;;
        --refresh)    REFRESH=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ -n "$LOCK_IN" && $REFRESH == 1 ]]; then
    echo "--lock and --refresh contradict each other: one installs a lock, the other replaces it" >&2; exit 2
fi

# What a GPU task imports that an NGC TensorFlow image may lack, at exact
# versions. Only what runs in that container: bin/anorak_predict.py (argparse,
# os, sys, pathlib, tensorflow) and upstream's inference_slide/predict_gp.py
# (os, numpy, cv2, PIL, math, glob, tensorflow). scikit-image, scipy, pandas,
# tifffile and matplotlib used to be listed too, from upstream's
# requirments.txt; they are imported only by stitching and grading, which run
# natively in the conda env, and every extra package here is one more thing
# on PYTHONPATH ahead of the image's own.
#
# Chosen for Python 3.8 manylinux x86_64 (Ubuntu 20.04, glibc 2.31) against
# the image's numpy 1.x, each checked on PyPI to have that wheel:
#   opencv-python-headless 4.8.1.78 — cp37-abi3 manylinux2014 wheel, numpy
#     >= 1.17.3 on 3.8, and from before 4.10 moved its builds to numpy 2 —
#     with --no-deps nothing else would notice a mismatch. Headless on purpose: plain opencv-python links libGL,
#     which an NGC image does not carry, so `import cv2` would then fail on a
#     missing shared library instead of a missing module.
#   pillow 10.4.0 — the last release supporting 3.8; no required deps.
# import name:pip requirement
PINNED=(
    "cv2:opencv-python-headless==4.8.1.78"
    "PIL:pillow==10.4.0"
)
pinned_for() {   # the requirement for an import name, or nothing
    local entry
    for entry in "${PINNED[@]}"; do
        [[ "${entry%%:*}" == "$1" ]] && { printf '%s\n' "${entry#*:}"; return 0; }
    done
    return 1
}

# The container runtime, under whichever name this cluster installs it. Checked
# by name rather than assumed: where it arrives via modules and none is loaded,
# every probe below fails identically to "the image does not have this module".
SINGULARITY=""
for candidate in singularity apptainer; do
    if command -v "$candidate" >/dev/null; then SINGULARITY="$candidate"; break; fi
done
if [[ -z "$SINGULARITY" ]]; then
    echo "Neither singularity nor apptainer is on PATH." >&2
    echo "If it arrives via modules, load it first (e.g. 'module load singularity')." >&2
    exit 2
fi

# The image, the extras path and the binds, read out of the EFFECTIVE config
# (the flat form, where a name cannot match the wrong block), so this cannot
# bootstrap a directory the run will not load, or probe an image through binds
# the run does not use.
if ! command -v nextflow >/dev/null; then
    echo "nextflow is not on PATH, so conf/beatson.config cannot be resolved (the" >&2
    echo "image, the extras directory and runOptions' binds). Load it first, e.g." >&2
    echo "\$HPL_NEXTFLOW_PRELUDE." >&2
    exit 2
fi
CONFIG="$(cd "$PIPELINE_DIR" && nextflow config -profile beatson -flat . 2>&1)" || {
    echo "nextflow config -profile beatson failed:" >&2; printf '%s\n' "$CONFIG" | tail -15 >&2; exit 2; }
config_value() {   # a string setting from the flat config; empty for null
    printf '%s\n' "$CONFIG" | sed -n "s/^$1 = '\\(.*\\)'\$/\\1/p" | head -1
}
[[ -z "$IMAGE"      ]] && IMAGE="$(config_value 'params\.gpu_container')"
[[ -z "$EXTRAS_DIR" ]] && EXTRAS_DIR="$(config_value 'params\.gpu_extras')"
read -r -a BINDS <<< "$(config_value 'singularity\.runOptions')"

if [[ -z "$IMAGE" ]]; then
    echo "No gpu_container in conf/beatson.config and none given with --image." >&2; exit 2
fi
if [[ ! -r "$IMAGE" ]]; then echo "Cannot read image: $IMAGE" >&2; exit 2; fi
if [[ -z "$EXTRAS_DIR" ]]; then
    echo "gpu_extras is unset in conf/beatson.config. Set it, or pass --extras-dir." >&2; exit 2
fi
mkdir -p "$EXTRAS_DIR" || { echo "Cannot create $EXTRAS_DIR" >&2; exit 2; }
echo "runtime: $SINGULARITY ($(command -v "$SINGULARITY"))"
echo "image:   $IMAGE"
echo "extras:  $EXTRAS_DIR"
echo "binds:   ${BINDS[*]:-(none)}"

in_image() {   # importable with NOTHING on PYTHONPATH, i.e. the image itself
    "$SINGULARITY" exec --cleanenv ${BINDS[@]+"${BINDS[@]}"} "$IMAGE" python3 -c 'import importlib, sys; importlib.import_module(sys.argv[1])' "$1" 2>/dev/null
}
with_extras() { # run python the way a task will see it; $1 is the code, the rest its argv
    # Composed exactly as PREDICT_GP's script composes it in main.nf — the
    # extras *prepended* to whatever PYTHONPATH the image itself sets — under
    # --cleanenv, which is what Nextflow's `env - ... singularity exec` launch
    # amounts to, and from /, not from wherever this was run, so the current
    # directory cannot lend a module a task will not have. This used to set
    # SINGULARITYENV_PYTHONPATH, which replaces the image's own PYTHONPATH
    # rather than extending it, and let the host's environment through: a
    # verification of an interpreter no task gets.
    "$SINGULARITY" exec --cleanenv ${BINDS[@]+"${BINDS[@]}"} "$IMAGE" bash -c \
        'export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"; shift; cd / && exec python3 -c "$@"' \
        _ "$EXTRAS_DIR" "$@" 2>&1
}
pip_into_extras() {
    "$SINGULARITY" exec --cleanenv ${BINDS[@]+"${BINDS[@]}"} "$IMAGE" \
        python3 -m pip install --quiet --no-cache-dir --no-deps \
        --target "$EXTRAS_DIR" --upgrade "$@"
}
freeze_extras() {
    "$SINGULARITY" exec --cleanenv ${BINDS[@]+"${BINDS[@]}"} "$IMAGE" \
        python3 -m pip freeze --path "$EXTRAS_DIR"
}

# Before any probe: does python run in this image at all? Otherwise a runtime
# that cannot start the image (a missing bind, a bad image) makes every probe
# below read "module missing", and this installs everything and then reports
# that none of it imports.
if ! err="$("$SINGULARITY" exec --cleanenv ${BINDS[@]+"${BINDS[@]}"} "$IMAGE" python3 -c 'import sys; print(sys.version.split()[0])' 2>&1)"; then
    echo "python3 does not run inside $IMAGE:" >&2; printf '%s\n' "$err" | tail -5 >&2; exit 2
fi
echo "python:  $(printf '%s\n' "$err" | tail -1) (in the image)"

# Install whatever a module needs to import, one missing name at a time, and
# only from PINNED.
#
# A name the image ALREADY has stops this rather than being installed: the
# import would not have failed on it, so its appearing here means something
# other than a missing package, and putting a second copy on PYTHONPATH ahead
# of the image's own is how you shadow the numpy TensorFlow was built against.
# A name PINNED does not know stops it too — this used to pip-install whatever
# the error named, at whatever version was newest, which is exactly the
# unpinned drift the lock exists to prevent. $2, when given, is a module that
# must never be installed: upstream's own module comes from sys.path, not
# PyPI, so a wrong --anorak-dir would otherwise fetch whatever shares its name.
resolve_imports() {   # $1 label, $2 never-install name, then the python code and its argv
    local label="$1" protected="$2" round=0 err missing req; shift 2
    local added=""
    while (( round < 8 )); do
        if err="$(with_extras "$@")"; then
            [[ -n "$added" ]] && printf '        pulled in:%s\n' "$added"
            return 0
        fi
        missing="$(printf '%s\n' "$err" \
            | sed -n "s/^ModuleNotFoundError: No module named '\\([A-Za-z0-9_]*\\).*/\\1/p" | tail -1)"
        if [[ -z "$missing" ]]; then
            printf '%s\n' "$err" | tail -3 | sed 's/^/        /'
            return 1
        fi
        if [[ -n "$protected" && "$missing" == "$protected" ]]; then
            printf '        %s itself is not importable. It is reached through sys.path,\n' "$missing"
            printf '        not from PyPI, so check --anorak-dir rather than installing it.\n'
            return 1
        fi
        if in_image "$missing"; then
            printf '        %s is in the image already, so this is not a missing package:\n' "$missing"
            printf '%s\n' "$err" | tail -3 | sed 's/^/        /'
            return 1
        fi
        if ! req="$(pinned_for "$missing")"; then
            printf '        needs %s, which the image lacks and PINNED does not list. Add it\n' "$missing"
            printf '        to PINNED in this script with an exact version (for Python 3.8,\n'
            printf '        against the image'"'"'s numpy) and re-run with --refresh.\n'
            return 1
        fi
        if ! pip_into_extras "$req"; then
            printf '        could not install %s (for %s)\n' "$req" "$missing"
            return 1
        fi
        added="$added $req"
        round=$((round + 1))
    done
    printf '        %s: still unresolved after 8 rounds; last error:\n        %s\n' \
        "$label" "$(printf '%s\n' "$err" | tail -1)"
    return 1
}

LOCK="$EXTRAS_DIR/anorak_extras.lock"
[[ -z "$LOCK_IN" && $REFRESH == 0 && -f "$LOCK" ]] && LOCK_IN="$LOCK"

if [[ -n "$LOCK_IN" ]]; then
    # Exactly the locked set, and nothing resolved: the lock is what a
    # verified build contained, so this reproduces it rather than re-deciding.
    [[ -r "$LOCK_IN" ]] || { echo "Cannot read lock: $LOCK_IN" >&2; exit 2; }
    lock_image="$(sed -n 's/^# image: //p' "$LOCK_IN" | head -1)"
    if [[ "$lock_image" != "$IMAGE" ]]; then
        echo "$LOCK_IN was built against '${lock_image:-an unrecorded image}', not $IMAGE." >&2
        echo "What the extras must hold is that image's gap; re-run with --refresh" >&2
        echo "(into a new --extras-dir if a run is still importing from this one)." >&2
        exit 2
    fi
    grep -v '^#' "$LOCK_IN" | grep . > "$EXTRAS_DIR/.lock_install.$$" || true
    echo
    echo "Installing the $(grep -c . "$EXTRAS_DIR/.lock_install.$$") pinned package(s) in $LOCK_IN with --no-deps"
    echo "(--refresh to re-resolve from PINNED instead)"
    if [[ -s "$EXTRAS_DIR/.lock_install.$$" ]] && ! pip_into_extras -r "$EXTRAS_DIR/.lock_install.$$"; then
        rm -f "$EXTRAS_DIR/.lock_install.$$"
        echo "pip failed. It needs the internet; run this on the login node." >&2
        exit 1
    fi
    rm -f "$EXTRAS_DIR/.lock_install.$$"
else
    if (( REFRESH )) && [[ -n "$(ls -A "$EXTRAS_DIR" 2>/dev/null)" ]]; then
        echo
        echo "note: $EXTRAS_DIR is not empty. --refresh installs over it in place, and"
        echo "      a GPU task running now imports from it; a new --extras-dir is safer."
    fi
    echo
    echo "Probing the image"
    WANTED=()
    for entry in "${PINNED[@]}"; do
        mod="${entry%%:*}"; req="${entry#*:}"
        if in_image "$mod"; then printf '  have     %s\n' "$mod"
        else printf '  missing  %-12s -> %s\n' "$mod" "$req"; WANTED+=("$req"); fi
    done
    if (( ${#WANTED[@]} )); then
        echo
        echo "Installing ${#WANTED[@]} package(s) with --no-deps: ${WANTED[*]}"
        pip_into_extras "${WANTED[@]}" || {
            echo "pip failed. It needs the internet; run this on the login node." >&2
            exit 1
        }
    else
        echo
        echo "Nothing missing — the image already has all of them."
    fi
fi

# --- verification ----------------------------------------------------------
#
# The point of the whole script, and the check that was missing when every GPU
# task died on cv2. Importing the packages proves the packages; importing
# upstream's own module proves the thing the job actually does, including any
# dependency this list does not know about.

echo
echo "Verifying each import as a task will see it"
BAD=0
for entry in "${PINNED[@]}"; do
    mod="${entry%%:*}"
    if resolve_imports "$mod" "" 'import importlib, sys; importlib.import_module(sys.argv[1])' "$mod"; then
        printf '  \033[32mok\033[0m    %s\n' "$mod"
    else printf '  \033[31mFAIL\033[0m  %s\n' "$mod"; BAD=1; fi
done

if [[ -n "$ANORAK_DIR" && -d "$ANORAK_DIR/inference_slide" ]]; then
    echo
    echo "Importing upstream's predict_gp"
    # Through resolve_imports too, so a module upstream needs that PINNED
    # knows is installed here rather than discovered on the GPU queue. The
    # clone's path is an argument, not pasted into the source, so a quote in
    # it is just a character.
    if resolve_imports predict_gp predict_gp \
            'import sys; sys.path.insert(0, sys.argv[1] + "/inference_slide"); import predict_gp' \
            "$ANORAK_DIR"; then
        echo "  predict_gp imports clean"
    else
        echo "  predict_gp does not import (reason above)." >&2
        BAD=1
    fi
else
    echo
    echo "  (skipped importing predict_gp: pass --anorak-dir to include it,"
    echo "   which is the only check that covers upstream's whole import list)"
    echo "  Not locking without it: a lock is a statement that the directory works."
    BAD=1
fi

# What the directory holds now, against what was asked for. Installing a lock
# must reproduce it exactly — anything else in the directory would be absorbed
# into the next lock without anyone deciding it. Packages outside PINNED from
# an earlier (unpinned) build are named: they sit ahead of the image on every
# task's PYTHONPATH.
if freeze_extras 2>/dev/null | LC_ALL=C sort > "$EXTRAS_DIR/.freeze.$$"; then
    if [[ -n "$LOCK_IN" ]] && ! diff <(grep -v '^#' "$LOCK_IN" | grep . | LC_ALL=C sort) \
            "$EXTRAS_DIR/.freeze.$$" > "$EXTRAS_DIR/.diff.$$"; then
        echo
        echo "The directory does not match $LOCK_IN after installing it (< lock, > directory):" >&2
        grep '^[<>]' "$EXTRAS_DIR/.diff.$$" | sed 's/^/  /' >&2
        echo "Rebuild into an empty --extras-dir with --lock $LOCK_IN, or --refresh." >&2
        BAD=1
    fi
    pinned_names=" $(for e in "${PINNED[@]}"; do r="${e#*:}"; printf '%s ' "$(tr '[:upper:]_' '[:lower:]-' <<< "${r%%==*}")"; done)"
    while IFS= read -r line; do
        name="$(tr '[:upper:]_' '[:lower:]-' <<< "${line%%==*}")"
        [[ -n "$name" && "$pinned_names" != *" $name "* ]] \
            && echo "  note: $line is in $EXTRAS_DIR but not in PINNED — left by an earlier build?"
    done < "$EXTRAS_DIR/.freeze.$$"
else
    echo "could not list $EXTRAS_DIR with pip freeze inside the image" >&2
    BAD=1
fi
rm -f "$EXTRAS_DIR/.freeze.$$" "$EXTRAS_DIR/.diff.$$"

# --- the lock ----------------------------------------------------------------
#
# Written only after everything above imported, so a lock always describes a
# directory that worked. Written to a temporary name in the same directory and
# renamed, so a reader (preflight, or a later bootstrap) sees the old lock or
# the new one, never half of one — rename is atomic within a filesystem, and a
# temporary name elsewhere (/tmp) would make it a copy.
if (( BAD == 0 )); then
    tmp_lock="$LOCK.tmp.$$"
    if freeze_extras > "$tmp_lock.freeze" 2> "$tmp_lock.err" \
            && {
                echo "# anorak gpu_extras lock — written by tools/bootstrap_gpu_extras.sh"
                echo "# image: $IMAGE"
                echo "# written: $(date -u +%Y-%m-%dT%H:%M:%SZ) by ${USER:-?}"
                LC_ALL=C sort "$tmp_lock.freeze"
            } > "$tmp_lock" \
            && mv -f "$tmp_lock" "$LOCK"; then
        echo
        echo "Lock: $LOCK ($(grep -vc '^#' "$LOCK") packages)"
        grep -v '^#' "$LOCK" | sed 's/^/  /'
    else
        echo "could not write $LOCK:" >&2; tail -5 "$tmp_lock.err" >&2
        BAD=1
    fi
    rm -f "$tmp_lock" "$tmp_lock.freeze" "$tmp_lock.err"
else
    echo
    echo "Not writing a lock: the directory did not verify."
fi

exit "$BAD"
