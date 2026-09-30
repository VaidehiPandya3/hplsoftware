#!/usr/bin/env bash
#
# Run a Nextflow head process, and restart it with -resume if its task monitor
# goes silent.
#
#     nf_supervise.sh --log <nextflow.log> --work-dir <work> [options] -- nextflow ...
#
# Why this exists. Nextflow has exactly one thread that notices a task has
# finished (the "Task monitor"), and it does so by reading each task's
# .exitcode off the shared filesystem. On 2026-09-23 one of those reads, of a
# file another node was writing at that very moment, blocked inside the CephFS
# kernel client and never returned. Every job the run submitted after that
# finished in Slurm and was never collected, all 50 queue slots stayed "busy",
# and 7,161 tasks sat unsubmitted for five hours — with no error, no exit, and
# a log still being written by the submitter thread. The read is stuck in the
# kernel, below anything Nextflow can put a timeout on, so the only remedy is
# the one applied here: notice, stop the process, start it again. A fresh open
# of the same file works.
#
# How silence is measured. The monitor thread logs every task it collects and,
# every executor.dumpInterval (5 min, set in conf/beatson.config) while any task
# is running, a "tasks to be completed" summary. So a live monitor puts a
# "[Task monitor]" line in the log at least every five minutes, and a stuck one
# puts none. The watchdog counts those lines, and arms only after it has seen
# the count RISE in the current attempt — before that the run may still be
# resolving its slide list (main.nf walks the whole raw_dir on CephFS before
# any process exists), and silence means nothing.
#
# Rise, not change. Nextflow rotates its -log at startup (nextflow.log becomes
# nextflow.log.1), so the count read just after launch is the previous
# attempt's, and it then drops to zero. Arming on "different" armed on that
# drop, before this attempt's monitor existed, and a slide-list walk longer
# than --stall-seconds was killed as a stall — on every restart and every
# chain successor, since each of them starts with a full old log.
#
# What a restart does. TERM to Nextflow (it cancels its own jobs if it can
# still shut down), KILL after --grace-seconds, then scancel whatever this
# run still has in the queue — found by working directory, so no other run's
# jobs are touched. A running task cannot be adopted by a new Nextflow process,
# so leaving it would only duplicate it. Then the same command again with
# -resume, which skips everything Nextflow had recorded as done.
#
# Signals. sbatch --signal=B:TERM reaches this script (the batch shell execs
# it). It stops Nextflow within a budget strictly inside Slurm's warning —
# --signal-lead-seconds minus TERM_RESERVE_SECONDS, two thirds of it after TERM
# and the rest after KILL — then scancels this run's leftover jobs, and exits.
# An unbounded wait here once meant a wedged Nextflow (the very state this
# script exists for) held the job until Slurm's KILL at the limit, with the
# cleanup never reached and every task left running unwatched.
#
# Which stops a --chain successor may take over from. A chain's standbys
# wait on --dependency=afternotok, so every non-zero exit starts the next one,
# and a failure that is deterministic — a config error, a slide that fails
# every time, "Unable to acquire lock" — then burns through the whole chain in
# seconds. So a stop that a successor cannot fix writes --stop-marker (a file
# in the run's out_dir, holding the reason), and every head job checks for it
# before starting Nextflow and exits 0 without running if it is there; the rest
# of the chain is then cleared by --kill-on-invalid-dep. Final, marker written:
#   - Nextflow exited non-zero on its own (a real pipeline failure).
#   - A TERM that is not the time-limit warning, i.e. a manual scancel.
#   - Nextflow survived KILL: it may still hold .nextflow/cache/<id>/db/LOCK,
#     and a successor would die on the lock and wake the next one, and so on.
# Transient, no marker: the stall restarts ran out (75), and the time-limit
# TERM. The submitter clears a stale marker on a fresh submission.
#
# Telling the time-limit TERM from a scancel. Both are a TERM. Slurm sends the
# --signal one when the walltime left drops to the lead (up to 60 s early, by
# its own documentation); a scancel comes at any time. So the walltime left
# when the TERM arrives decides it — read live from `squeue -o %L`, else from
# SLURM_JOB_END_TIME — and within --signal-lead-seconds + 90 s of the limit it
# is read as the limit. Consequently a scancel in the last three and a half
# minutes reads as the limit (the successor resumes), and a TERM whose timing
# cannot be read at all is treated as final: a chain stopped loudly costs a
# resubmission, a cancelled run resumed by its successor costs whatever it
# then does unasked.
#
# Stopping a run by hand: `scancel <the running head job>`. The marker is
# written before anything slow, because scancel's KILL follows its TERM after
# the cluster's KillWait (30 s by default); the standbys then start, find it,
# and exit 0. Cancelling the standbys too (`scancel` every id the submitter
# printed) is tidier and equivalent.
#
# Exit status: 0 if Nextflow finished — including one that finished just as a
# TERM arrived, which must not read as FAILED and wake a successor; Nextflow's
# own status if it failed; 143 after a TERM; 75 if the restarts run out or
# Nextflow cannot be killed.

set -u

log=
work_dir=
stop_marker=
stall_seconds=1800
poll_seconds=60
grace_seconds=120
max_restarts=10
signal_lead_seconds=120

# Of Slurm's --signal lead, what the TERM path keeps back from stopping
# Nextflow: the walltime lookup, squeue for the leftovers and their scancel,
# each capped at QUERY_SECONDS — 24 s — plus room for the marker and exit.
TERM_RESERVE_SECONDS=30
QUERY_SECONDS=8

usage() {
    echo "usage: nf_supervise.sh --log FILE --work-dir DIR [--stop-marker FILE]" \
         "[--stall-seconds N] [--poll-seconds N] [--grace-seconds N]" \
         "[--max-restarts N] [--signal-lead-seconds N] -- COMMAND..." >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --log)                 log=$2; shift 2 ;;
        --work-dir)            work_dir=$2; shift 2 ;;
        --stop-marker)         stop_marker=$2; shift 2 ;;
        --stall-seconds)       stall_seconds=$2; shift 2 ;;
        --poll-seconds)        poll_seconds=$2; shift 2 ;;
        --grace-seconds)       grace_seconds=$2; shift 2 ;;
        --max-restarts)        max_restarts=$2; shift 2 ;;
        --signal-lead-seconds) signal_lead_seconds=$2; shift 2 ;;
        --)                    shift; break ;;
        *)                     echo "nf_supervise: unknown option $1" >&2; usage; exit 64 ;;
    esac
done
if [ -z "$log" ] || [ -z "$work_dir" ] || [ $# -eq 0 ]; then
    usage
    exit 64
fi
if [ "$signal_lead_seconds" -le $((TERM_RESERVE_SECONDS + 2)) ]; then
    echo "nf_supervise: --signal-lead-seconds $signal_lead_seconds leaves no time to" \
         "stop Nextflow after the ${TERM_RESERVE_SECONDS}s reserved for cleanup;" \
         "ask sbatch for a longer --signal lead" >&2
    exit 64
fi
term_budget=$((signal_lead_seconds - TERM_RESERVE_SECONDS))
term_wait=$((term_budget * 2 / 3))
kill_wait=$((term_budget - term_wait))

say() { echo "[nf_supervise $(date '+%F %T')] $*" >&2; }

nf_pid=
nf_status=
nap_pid=

# A sleep that a signal can interrupt: bash runs a trap only between commands,
# and `wait` is one it will break out of, where a foreground `sleep` is not.
nap() {
    sleep "$1" &
    nap_pid=$!
    wait "$nap_pid" 2>/dev/null
    nap_pid=
}

# Run a command for at most N seconds; its stdout is ours, its stderr is left
# in $bounded_err_file for the caller to report (a file, because callers run
# this inside $(...), whose variables die with it). squeue against a
# controller that is not answering waits for its own timeout, which is longer
# than the whole of Slurm's --signal lead, so nothing on the TERM path may call
# it unbounded. (coreutils `timeout` is not on every machine this runs on.)
# Its stdout goes through a file too: a killed command's own children can hold
# a pipe open, and $(...) would then wait on them however long they live.
bounded_err_file="${TMPDIR:-/tmp}/nf_supervise.$$.err"
bounded_out_file="${TMPDIR:-/tmp}/nf_supervise.$$.out"
run_bounded() {
    local limit=$1 pid ticks=0 rc
    shift
    "$@" >"$bounded_out_file" 2>"$bounded_err_file" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null && [ "$ticks" -lt $((limit * 10)) ]; do
        sleep 0.1
        ticks=$((ticks + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        echo "no answer within ${limit}s" >>"$bounded_err_file"
        rc=124
    else
        wait "$pid"
        rc=$?
    fi
    cat "$bounded_out_file" 2>/dev/null
    return "$rc"
}
bounded_err() { tr '\n' ' ' <"$bounded_err_file" 2>/dev/null; }
trap 'rm -f "$bounded_err_file" "$bounded_out_file"' EXIT

# The stop marker, written whole (tmp then rename) so a head job reading it
# never sees half a reason.
write_stop() {
    [ -n "$stop_marker" ] || return 0
    local tmp="$stop_marker.tmp.$$"
    {
        echo "$*"
        echo "job: ${SLURM_JOB_ID:-none} on $(hostname) at $(date '+%F %T')"
        echo "No --chain successor of this run will start Nextflow while this file"
        echo "exists. Fix the cause, then resubmit — submit_anorak_nf.py clears it —"
        echo "or delete it to let a still-queued standby continue."
    } >"$tmp" && mv -f "$tmp" "$stop_marker" \
        && say "wrote stop marker $stop_marker: $*" \
        || say "could not write stop marker $stop_marker; a successor WILL start"
}

# Seconds of walltime this job has left, or nothing if that cannot be read.
# squeue first, because it is live — `scontrol update TimeLimit` does not
# change SLURM_JOB_END_TIME in a running job's environment.
walltime_left() {
    local left= days=0 total=0 part
    if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
        left=$(run_bounded "$QUERY_SECONDS" squeue -h -j "$SLURM_JOB_ID" -o %L | head -n 1)
        # D-HH:MM:SS, HH:MM:SS, MM:SS or SS; anything else (UNLIMITED,
        # INVALID, an error) is "cannot tell".
        if [[ $left =~ ^([0-9]+-)?[0-9]+(:[0-9]+)?(:[0-9]+)?$ ]]; then
            case "$left" in *-*) days=$((10#${left%%-*})); left=${left#*-} ;; esac
            IFS=: read -ra parts <<<"$left"
            for part in "${parts[@]}"; do total=$((total * 60 + 10#$part)); done
            left=$((days * 86400 + total))
        else
            left=
        fi
    fi
    if [ -z "$left" ] && [ -n "${SLURM_JOB_END_TIME:-}" ]; then
        left=$((SLURM_JOB_END_TIME - $(date +%s)))
    fi
    echo "$left"
}

# Lines the task monitor has written. grep -c prints 0 and exits 1 on no match,
# and prints nothing for a log that does not exist yet.
heartbeats() {
    local n
    n=$(grep -cF '[Task monitor]' "$log" 2>/dev/null) || true
    echo "${n:-0}"
}

cancel_run_jobs() {
    if ! command -v squeue >/dev/null 2>&1; then
        say "squeue not found; cannot look for this run's leftover jobs"
        return
    fi
    local listing ids rc
    listing=$(run_bounded "$QUERY_SECONDS" squeue -h -u "${USER:-$(id -un)}" -o '%i %Z')
    rc=$?
    # Was 2>/dev/null, so a controller that timed out looked exactly like a
    # run with nothing left in the queue — and the tasks ran on unwatched.
    if [ "$rc" -ne 0 ]; then
        say "squeue failed (exit $rc: $(bounded_err)), so this run's" \
            "leftover jobs could not be listed and none were cancelled. Find them" \
            "with: squeue -u ${USER:-\$USER} -o '%i %Z' | grep ${work_dir%/}/"
        return
    fi
    ids=$(echo "$listing" | awk -v prefix="${work_dir%/}/" 'index($2, prefix) == 1 { print $1 }')
    if [ -n "$ids" ]; then
        say "cancelling this run's jobs still in the queue: $(echo $ids)"
        # shellcheck disable=SC2086
        run_bounded "$QUERY_SECONDS" scancel $ids >/dev/null \
            || say "scancel failed ($(bounded_err)); those jobs will run on unwatched"
    fi
}

# 0 once Nextflow is gone (its status in nf_status), 1 if it survived KILL
# (uninterruptible in the kernel).
stop_nextflow() {
    local term_for=$1 kill_for=$2 waited=0
    kill -TERM "$nf_pid" 2>/dev/null
    while kill -0 "$nf_pid" 2>/dev/null && [ "$waited" -lt "$term_for" ]; do
        nap 1
        waited=$((waited + 1))
    done
    if kill -0 "$nf_pid" 2>/dev/null; then
        say "nextflow ignored TERM for ${term_for}s; sending KILL"
        kill -KILL "$nf_pid" 2>/dev/null
        waited=0
        while kill -0 "$nf_pid" 2>/dev/null && [ "$waited" -lt "$kill_for" ]; do
            nap 1
            waited=$((waited + 1))
        done
        if kill -0 "$nf_pid" 2>/dev/null; then
            return 1
        fi
    fi
    wait "$nf_pid" 2>/dev/null
    nf_status=$?
    return 0
}

on_term() {
    # Slurm TERMs everything again at the limit itself; this handler is
    # already doing what that would ask for.
    trap '' TERM INT
    if [ -n "$nap_pid" ]; then kill "$nap_pid" 2>/dev/null; fi
    local left status=143
    left=$(walltime_left)
    if [ -n "$left" ] && [ "$left" -le $((signal_lead_seconds + 90)) ]; then
        say "termination signal with ${left}s of walltime left: Slurm's warning" \
            "before the time limit, so a --chain successor may take over"
    else
        # Written first: after a scancel the KILL follows in KillWait (30 s by
        # default), and a marker not yet written is a successor that resumes.
        if [ -n "$left" ]; then left="${left}s"; else left="unknown (no squeue answer, no SLURM_JOB_END_TIME)"; fi
        say "termination signal with walltime left $left, which is not the" \
            "time-limit warning; treating it as a deliberate stop"
        write_stop "stopped by a termination signal with walltime left $left —" \
                   "read as a scancel, not the time limit"
    fi
    if [ -n "$nf_pid" ]; then
        say "stopping nextflow (pid $nf_pid): ${term_wait}s after TERM, then" \
            "${kill_wait}s after KILL, inside Slurm's ${signal_lead_seconds}s warning"
        if ! stop_nextflow "$term_wait" "$kill_wait"; then
            say "nextflow (pid $nf_pid) survived KILL"
            write_stop "nextflow (pid $nf_pid on $(hostname)) survived KILL after a" \
                       "termination signal and may still hold the session's" \
                       ".nextflow/cache/<id>/db/LOCK; a successor would fail with" \
                       "'Unable to acquire lock'. Check that host before resuming."
        elif [ "$nf_status" -eq 0 ]; then
            # It finished as the signal arrived. 143 here would read as FAILED
            # and start a successor against a completed run.
            status=0
            say "nextflow had already finished successfully; exiting 0"
        fi
    fi
    cancel_run_jobs
    exit "$status"
}
trap on_term TERM INT

if [ -n "$stop_marker" ] && [ -e "$stop_marker" ]; then
    say "stop marker $stop_marker is present; not starting nextflow. It says:"
    sed 's/^/    /' "$stop_marker" >&2
    # 0, so the rest of a chain waiting on afternotok is cleared rather than
    # started, one after another, into the same refusal.
    exit 0
fi

restarts=0
add_resume=0
# The submitter passes -resume itself (it is the default there), and Nextflow
# refuses the option twice — "Can only specify option -resume once" — so the
# first restart of run 1250456 died in a second, with the cohort half done.
# Append it only to a command that does not already carry it.
has_resume=0
for arg in "$@"; do
    [ "$arg" = "-resume" ] && has_resume=1
done
while :; do
    if [ "$add_resume" -eq 1 ] && [ "$has_resume" -eq 0 ]; then
        "$@" -resume &
    else
        "$@" &
    fi
    nf_pid=$!

    last=$(heartbeats)
    last_change=$(date +%s)
    armed=0
    stalled=0
    silent=0
    while kill -0 "$nf_pid" 2>/dev/null; do
        nap "$poll_seconds"
        now=$(heartbeats)
        # Only a rise is this attempt's monitor; a fall is Nextflow rotating
        # the previous attempt's log away (see the top of this file).
        if [ "$now" -gt "$last" ]; then
            last=$now
            last_change=$(date +%s)
            armed=1
            continue
        fi
        last=$now
        [ "$armed" -eq 1 ] || continue
        silent=$(( $(date +%s) - last_change ))
        if [ "$silent" -ge "$stall_seconds" ]; then
            stalled=1
            break
        fi
    done

    if [ "$stalled" -eq 0 ]; then
        wait "$nf_pid"
        nf_status=$?
        if [ "$nf_status" -ne 0 ]; then
            write_stop "nextflow exited with status $nf_status on its own — a" \
                       "pipeline failure, not a stall. The reason is at the end of $log."
        fi
        exit "$nf_status"
    fi

    say "nextflow's task monitor has logged nothing for ${silent}s" \
        "(limit ${stall_seconds}s) — it is stuck, most likely on a filesystem" \
        "read; stopping pid $nf_pid"
    if ! stop_nextflow "$grace_seconds" 60; then
        say "nextflow (pid $nf_pid) survived KILL; leaving it to Slurm and exiting"
        write_stop "nextflow (pid $nf_pid on $(hostname)) survived KILL after a" \
                   "stall and may still hold the session's .nextflow/cache/<id>/db/LOCK;" \
                   "a successor would fail with 'Unable to acquire lock'. Check that" \
                   "host before resuming."
        cancel_run_jobs
        exit 75
    fi
    nf_pid=
    cancel_run_jobs

    restarts=$((restarts + 1))
    if [ "$restarts" -gt "$max_restarts" ]; then
        say "stalled $restarts times, more than --max-restarts $max_restarts; giving up"
        exit 75
    fi
    # Resumed even if the first attempt was a fresh run: this is the same run
    # continuing, and without it the restart would redo the whole cohort.
    add_resume=1
    say "restarting nextflow with -resume (restart $restarts of $max_restarts)"
done
