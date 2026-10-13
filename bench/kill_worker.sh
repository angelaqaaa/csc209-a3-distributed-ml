#!/usr/bin/env bash
# Kill one worker in the middle of a training run, then check from the logs
# that the server dropped it and the other workers finished the run.
#
# Usage: bench/kill_worker.sh   (works from any directory; run one at a time,
# since every run rebuilds the binaries in the checkout and uses PORT)
#
# Settings come from the environment, defaults in brackets. Apart from
# OUT_DIR, each one is a whole number written without leading zeros.
#   WORKERS        workers to start, 2 to MAX_WORKERS [4]
#   KILL_AT_ROUND  kill once the script sees the server log
#                  "broadcasted weights for round KILL_AT_ROUND" [10]
#   KILL_SLOT      server slot of the worker to kill [2]
#   NUM_FEATURES   features per sample [64]
#   TOTAL_SAMPLES  samples across all shards; more samples make each round
#                  take longer [160000]
#   PORT           port the server and worker are built for [4545]
#   OUT_DIR        where the logs and the summary go, relative to the
#                  repository root [bench/results/<today>]
#   WAIT_SECS      give up on any single wait after this many seconds [300]
#
# The workers start one at a time, and each one has to show up as
# "worker <i> registered" in the server log before the next one starts.
# add_worker() hands out the lowest free slot, so the i-th worker started
# gets slot i, and KILL_SLOT picks the process by that order. The slot check
# compares the server's accept and registration lines with 0 to WORKERS-1,
# and check (b), together with the check that no other slot logged a
# disconnect line, confirms that the killed process was the one in KILL_SLOT.
#
# Once the build and gen_data have worked, and before it starts the server, a
# run deletes the summary and worker logs that an earlier run with the same
# WORKERS left in OUT_DIR, and empties the server log. A run that stops after
# that point leaves its own logs and no summary; a run that stops earlier
# leaves the earlier files as they were. The build step runs
# "make clean && make PORT=$PORT", so the server and worker left in the
# checkout use PORT; "make clean && make" goes back to the Makefile default.
#
# In the summary, checks (a) to (f) are the main ones and the unlettered
# checks (slot map, disconnect lines of other slots, how far the killed worker
# got, its exit status) are extra; a FAIL on any of them fails the run. (a)
# compares the kill time with the stamps of the kill line and of DONE, and (d)
# wants exactly one DONE line and exit status 0.

set -euo pipefail

CDPATH= cd -- "$(dirname -- "$0")/.."
REPO_DIR="$(pwd)"

WORKERS="${WORKERS:-4}"
KILL_AT_ROUND="${KILL_AT_ROUND:-10}"
KILL_SLOT="${KILL_SLOT:-2}"
NUM_FEATURES="${NUM_FEATURES:-64}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-160000}"
PORT="${PORT:-4545}"
OUT_DIR="${OUT_DIR:-bench/results/$(date +%Y-%m-%d)}"
WAIT_SECS="${WAIT_SECS:-300}"

SERVER_HOST="localhost"
SERVER_LOG="$OUT_DIR/kill_server_${WORKERS}w.txt"
SUMMARY="$OUT_DIR/kill_summary_${WORKERS}w.txt"

# Log lines from server.c and worker.c that the script waits for or checks.
KILL_LINE="broadcasted weights for round $KILL_AT_ROUND"
SERVER_DONE_LINE="training complete; broadcasting DONE"
WORKER_DONE_LINE="Training complete. Final loss:"
TOO_FAST="(a) the server printed DONE before the kill was sent; raise TOTAL_SAMPLES so each round takes longer, or lower KILL_AT_ROUND"

WORK_DIR=""
SERVER_PID=""
STAMP_PID=""
KILL_PID=""
KILL_TIME=""
SERVER_STATUS=""
KILLED_STATUS=""
EXPECTED_ORDER=""
BUILT=0
WORKER_PIDS=()
WORKER_STATUS=()
FAILED=0

# Kill whatever is still running and reap it at once, so that bash prints no
# "Killed: 9" line for it. Then give the timestamper 50 polls of 20 ms (about
# 1.5 s, since each 20 ms sleep runs long and is a separate process) to write
# out the server's last lines, and delete the temporary directory. HUP, INT,
# TERM and QUIT are ignored from here on, and PIPE for the whole run, so a
# second Ctrl-C, a closed terminal or a closed output pipe cannot stop
# cleanup halfway. "$!" covers a child started just before a signal, whose
# pid was never stored. set +u because bash 3.2 counts $! as unset before
# the first background job, even with a default, and empty arrays as unset
# too.
cleanup() {
  local pid i
  trap '' HUP INT TERM QUIT
  set +u
  for pid in "$SERVER_PID" "${WORKER_PIDS[@]}" "$!"; do
    if [[ -n "$pid" && "$pid" != "$STAMP_PID" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "$STAMP_PID" ]]; then
    for ((i = 0; i < 50; i++)); do
      kill -0 "$STAMP_PID" 2>/dev/null || break
      sleep 0.02
    done
    kill -9 "$STAMP_PID" 2>/dev/null || true
    wait "$STAMP_PID" 2>/dev/null || true
  fi
  if [[ -n "$WORK_DIR" ]]; then
    rm -rf "$WORK_DIR"
  fi
  if ((BUILT)); then
    echo "note: server and worker in the checkout use port $PORT now; make clean && make goes back to the Makefile default" || true
  fi
}

die() {
  echo "FAIL: $*" >&2
  exit 1
}

# Print a line and add it to the summary.
say() {
  printf '%s\n' "$*"
  printf '%s\n' "$*" >>"$WORK_DIR/summary.txt"
}

# check OK TEXT: record one check, OK is "yes" or "no".
check() {
  local ok="$1"
  shift
  if [[ "$ok" == "yes" ]]; then
    say "  PASS $*"
  else
    say "  FAIL $*"
    FAILED=1
  fi
}

now() {
  perl -MTime::HiRes=time -e 'printf "%.6f\n", time'
}

# later T1 T2: succeed if timestamp T1 comes after timestamp T2.
later() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 > b + 0) }'
}

# Value of a #define in protocol.h.
protocol_value() {
  awk -v name="$1" '$1 == "#define" && $2 == name { print $3 }' protocol.h
}

worker_log() {
  printf '%s\n' "$OUT_DIR/kill_worker_${WORKERS}w_slot$1.txt"
}

# Slot numbers from the server log, in log order, for the
# "accepted connection ... assigned idx=<i>" and "worker <i> registered" lines.
accept_order() {
  sed -n 's/^[0-9.]* accepted connection fd=[0-9]* assigned idx=\([0-9]*\)$/\1/p' \
    "$SERVER_LOG" | paste -sd' ' -
}

register_order() {
  sed -n 's/^[0-9.]* worker \([0-9]*\) registered (.*/\1/p' "$SERVER_LOG" |
    paste -sd' ' -
}

# wait_for_server_line TEXT PID: wait until the server log has a line
# containing TEXT. Exits the script with FAIL if the server or process PID
# exits first, or once WAIT_SECS have passed.
wait_for_server_line() {
  local text="$1" pid="$2"
  local deadline=$((SECONDS + WAIT_SECS))
  until grep -qF -- "$text" "$SERVER_LOG"; do
    kill -0 "$SERVER_PID" 2>/dev/null ||
      die "server exited before logging '$text' (see $SERVER_LOG)"
    kill -0 "$pid" 2>/dev/null ||
      die "process $pid exited before the server logged '$text' (see its log in $OUT_DIR)"
    ((SECONDS < deadline)) || die "no '$text' in $SERVER_LOG after ${WAIT_SECS}s"
    sleep 0.02
  done
}

# wait_for_exit PID NAME: wait for a child to exit. After WAIT_SECS the script
# kills it and says so in the summary.
wait_for_exit() {
  local pid="$1" name="$2"
  local deadline=$((SECONDS + WAIT_SECS))
  while kill -0 "$pid" 2>/dev/null; do
    if ((SECONDS >= deadline)); then
      say "timeout: $name (pid $pid) still running after ${WAIT_SECS}s; the script sent it SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.05
  done
}

# Read the logs and record the slot map, checks (a) to (f), the extra checks
# and the number of rounds in the summary. MAX_ROUNDS is read from protocol.h
# here as well, so that this also works on saved logs.
run_checks() {
  local accepted registered stats total before after disc_n disc_all disc_ts
  local done_n done_ts round_ts ok i log found text last_before max_sent
  local max_recv killed_log cap_lines

  accepted="$(accept_order)"
  registered="$(register_order)"
  ok=no
  if [[ "$accepted" == "$EXPECTED_ORDER" && "$registered" == "$EXPECTED_ORDER" ]]; then
    ok=yes
  fi
  check "$ok" "slot map: accepted idx [$accepted], registered [$registered], launch order [$EXPECTED_ORDER]"

  # One pass over the server log. Each line starts with its timestamp, and
  # "after" counts the aggregations that follow the first disconnect line
  # of the killed slot.
  stats="$(awk -v slot="$KILL_SLOT" -v kill_line="$KILL_LINE" \
      -v done_line="$SERVER_DONE_LINE" '
    { text = $0; sub(/^[^ ]* /, "", text) }
    text ~ /^aggregated gradients; global_loss=/ {
      total++
      if (disc_n) after++; else before++
    }
    text ~ /^worker [0-9]+ disconnected$/ {
      disc_all++
      if ($3 == slot && ++disc_n == 1) disc_ts = $1
    }
    text == done_line && ++done_n == 1 { done_ts = $1 }
    text == kill_line { round_ts = $1 }
    END {
      print total + 0, before + 0, after + 0, disc_n + 0, disc_all + 0,
        (disc_ts == "" ? "-" : disc_ts), done_n + 0,
        (done_ts == "" ? "-" : done_ts), (round_ts == "" ? "-" : round_ts)
    }' "$SERVER_LOG")"
  read -r total before after disc_n disc_all disc_ts done_n done_ts round_ts <<<"$stats"

  ok=no
  if [[ "$round_ts" != "-" ]] && later "$KILL_TIME" "$round_ts" &&
    { [[ "$done_ts" == "-" ]] || later "$done_ts" "$KILL_TIME"; }; then
    ok=yes
  fi
  text="(a) \"$KILL_LINE\" logged at $round_ts, kill time $KILL_TIME (read just before kill -9), DONE logged at $done_ts"
  if [[ "$done_ts" != "-" ]] && ! later "$done_ts" "$KILL_TIME"; then
    text="$text; DONE came first, raise TOTAL_SAMPLES"
  fi
  check "$ok" "$text"

  # Last round in the killed worker's log for each of its two per-round lines.
  killed_log="$(worker_log "$KILL_SLOT")"
  max_sent="$(awk '$1 == "sent" && $2 == "gradient" && $4 == "round" {
      if (!seen || $5 + 0 > m) m = $5 + 0; seen = 1 }
    END { print seen ? m : "none" }' "$killed_log")"
  max_recv="$(awk '$1 == "received" && $2 == "weights" && $4 == "round" {
      if (!seen || $5 + 0 > m) m = $5 + 0; seen = 1 }
    END { print seen ? m : "none" }' "$killed_log")"

  ok=no
  if ((disc_n == 1)) && later "$disc_ts" "$KILL_TIME"; then
    ok=yes
  fi
  if ((disc_n == 0)); then
    text="(b) no \"worker $KILL_SLOT disconnected\" line, kill time $KILL_TIME"
    # A round cannot end without a gradient from every connected worker, so
    # the server logs a disconnect for a worker killed before its last
    # gradient. After that gradient, the server can finish without reading
    # the closed socket; DONE in the server log and the last round's weights
    # in the killed worker's log point to that case.
    if [[ "$done_ts" != "-" && "$max_recv" != "none" ]] &&
      ((disc_all == 0 && max_recv == total - 1)); then
      text="$text; DONE was logged without it and slot $KILL_SLOT got the weights for the last round, so the kill came too late: raise TOTAL_SAMPLES or lower KILL_AT_ROUND"
    fi
    check "$ok" "$text"
  else
    check "$ok" "(b) \"worker $KILL_SLOT disconnected\" lines: $disc_n, first one logged at $disc_ts, kill time $KILL_TIME"
  fi

  ok=no
  if ((disc_all - disc_n == 0)); then
    ok=yes
  fi
  check "$ok" "\"worker <i> disconnected\" lines for the other slots: $((disc_all - disc_n))"

  if ((disc_n == 0)); then
    check no "(c) no \"worker $KILL_SLOT disconnected\" line to count aggregations from"
  else
    last_before="none"
    if ((before > 0)); then
      last_before="round $((before - 1))"
    fi
    ok=no
    if ((after >= 1)); then
      ok=yes
    fi
    check "$ok" "(c) aggregations after the disconnect line: $after; the last aggregation before it: $last_before (the first aggregation is round 0)"
  fi

  ok=no
  if ((done_n == 1 && SERVER_STATUS == 0)); then
    ok=yes
  fi
  check "$ok" "(d) server logged \"$SERVER_DONE_LINE\" $done_n time(s), exit status $SERVER_STATUS"

  for ((i = 0; i < WORKERS; i++)); do
    if ((i == KILL_SLOT)); then
      continue
    fi
    log="$(worker_log "$i")"
    found=missing
    if grep -qF -- "$WORKER_DONE_LINE" "$log"; then
      found=found
    fi
    ok=no
    if [[ "$found" == "found" ]] && ((WORKER_STATUS[i] == 0)); then
      ok=yes
    fi
    check "$ok" "(e) slot $i: \"$WORKER_DONE_LINE\" $found in ${log##*/}, exit status ${WORKER_STATUS[i]}"
  done

  ok=no
  if [[ "$max_sent" == "none" ]] || ((max_sent <= KILL_AT_ROUND + 1)); then
    ok=yes
  fi
  check "$ok" "(f) slot $KILL_SLOT: last \"sent gradient for round\" $max_sent (at most $((KILL_AT_ROUND + 1)))"

  # The server logs the kill line after aggregating round KILL_AT_ROUND - 1,
  # which waited for the killed worker's gradient. The worker logs "received
  # weights" for a round before it sends that gradient ("sent gradient" comes
  # after the send), so its log has to show at least that round.
  ok=no
  if [[ "$max_recv" != "none" ]] && ((max_recv >= KILL_AT_ROUND - 1)); then
    ok=yes
  fi
  check "$ok" "slot $KILL_SLOT: last \"received weights for round\" $max_recv (at least $((KILL_AT_ROUND - 1)))"

  ok=no
  if ((KILLED_STATUS == 137)); then
    ok=yes
  fi
  check "$ok" "slot $KILL_SLOT exit status $KILLED_STATUS (137 = 128 + 9, SIGKILL)"

  # One "aggregated gradients" line per round, rounds 0 to MAX_ROUNDS, so a
  # run that reaches the cap logs MAX_ROUNDS + 1 of them.
  cap_lines=$(($(protocol_value MAX_ROUNDS) + 1))
  if ((done_n != 1)); then
    say "rounds: $total \"aggregated gradients\" lines; the server logged DONE $done_n time(s)"
  elif ((total == cap_lines)); then
    say "rounds: $total \"aggregated gradients\" lines (rounds 0 to $((cap_lines - 1)), the MAX_ROUNDS cap)"
  elif ((total < cap_lines)); then
    say "rounds: $total \"aggregated gradients\" lines, fewer than $cap_lines, so the loss went below LOSS_THRESHOLD first"
  else
    say "rounds: $total \"aggregated gradients\" lines, more than $cap_lines"
  fi
}

# cleanup runs on every exit. Without the INT trap, bash would carry on after
# a SIGINT sent only to it while a foreground command such as sleep runs, and
# an untrapped QUIT would end the script without running cleanup. Each signal
# trap ignores HUP, INT, TERM and QUIT before it exits, so that a second
# Ctrl-C, or the second SIGHUP that closing a terminal can bring, does not cut
# cleanup short. PIPE is ignored for the whole run: when the output goes
# through a pipe whose reader has exited (a tee ended by the same Ctrl-C, or
# head), a write then fails and the script exits through cleanup instead of
# dying on the spot.
trap cleanup EXIT
trap 'trap "" HUP INT TERM QUIT; exit 129' HUP
trap 'trap "" HUP INT TERM QUIT; exit 130' INT
trap 'trap "" HUP INT TERM QUIT; exit 143' TERM
trap 'trap "" HUP INT TERM QUIT; exit 131' QUIT
trap '' PIPE

# A leading zero is rejected because bash arithmetic and -DPORT would read the
# value as octal, while atoi() in server.c and gen_data.c reads it as decimal.
# ${!name} is the value of the variable whose name is in $name.
for name in WORKERS KILL_AT_ROUND KILL_SLOT NUM_FEATURES TOTAL_SAMPLES PORT WAIT_SECS; do
  case "${!name}" in
    "" | *[!0-9]* | 0?*) die "$name must be a whole number without leading zeros, got '${!name}'" ;;
  esac
done
MAX_WORKERS="$(protocol_value MAX_WORKERS)"
MAX_FEATURES="$(protocol_value MAX_FEATURES)"
MAX_ROUNDS="$(protocol_value MAX_ROUNDS)"
((WORKERS >= 2 && WORKERS <= MAX_WORKERS)) || die "WORKERS must be 2 to $MAX_WORKERS"
((KILL_SLOT < WORKERS)) || die "KILL_SLOT must be below WORKERS"
((KILL_AT_ROUND >= 1 && KILL_AT_ROUND <= MAX_ROUNDS)) ||
  die "KILL_AT_ROUND must be 1 to $MAX_ROUNDS"
((NUM_FEATURES >= 1 && NUM_FEATURES <= MAX_FEATURES)) ||
  die "NUM_FEATURES must be 1 to $MAX_FEATURES"
((TOTAL_SAMPLES >= WORKERS)) || die "TOTAL_SAMPLES must be at least WORKERS"
((PORT >= 1 && PORT <= 65535)) || die "PORT must be 1 to 65535"
((WAIT_SECS >= 1)) || die "WAIT_SECS must be at least 1"

# An old server or another program listening on PORT would make the new
# server's bind fail or take the workers' connections, so stop before building.
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  die "something already accepts connections on 127.0.0.1:$PORT; stop it or set PORT"
fi

mkdir -p "$OUT_DIR"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kill_worker.XXXXXX")"
: >"$WORK_DIR/summary.txt"
EXPECTED_ORDER="$(seq 0 $((WORKERS - 1)) | paste -sd' ' -)"

say "worker-kill run (bench/kill_worker.sh)"
say "date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
if commit="$(git rev-parse --short HEAD 2>/dev/null)"; then
  if git diff --quiet HEAD -- '*.c' '*.h' Makefile; then
    say "commit: $commit (C sources and Makefile unchanged)"
  else
    say "commit: $commit (C sources or Makefile have local changes)"
  fi
else
  say "commit: unknown (not a git checkout)"
fi
say "script: bench/kill_worker.sh, git blob $(git hash-object bench/kill_worker.sh 2>/dev/null || echo unknown)"
cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null ||
  awk -F': *' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null ||
  echo unknown)"
cores="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo unknown)"
say "host: $(uname -srm), $cpu, $cores cores"
say "load average before the build: $(uptime | sed 's/.*load averages*: //')"
say "compiler: $(gcc --version | head -1)"

echo "building..."
if ! { make clean && make PORT="$PORT"; } >"$WORK_DIR/build.txt" 2>&1; then
  cat "$WORK_DIR/build.txt" >&2
  die "build failed"
fi
BUILT=1
say "build: make clean && make PORT=$PORT"
while IFS= read -r line; do
  say "  $line"
done < <(grep '^gcc ' "$WORK_DIR/build.txt" || true)
say "  build warnings: $(grep -c 'warning:' "$WORK_DIR/build.txt" || true)"
while IFS= read -r line; do
  say "    $line"
done < <(grep 'warning:' "$WORK_DIR/build.txt" || true)

# gen_data writes shard_<i>.csv into the current directory, so it runs in the
# temporary directory: shard files already in the checkout stay as they are,
# and the large CSVs go away with the directory at exit.
echo "generating data..."
if ! (cd "$WORK_DIR" && "$REPO_DIR/gen_data" "$NUM_FEATURES" "$TOTAL_SAMPLES" "$WORKERS") \
  2>"$WORK_DIR/gen_data.txt"; then
  cat "$WORK_DIR/gen_data.txt" >&2
  die "gen_data failed"
fi
say "data: ./gen_data $NUM_FEATURES $TOTAL_SAMPLES $WORKERS (run in a temporary directory, deleted at exit)"
for ((i = 0; i < WORKERS; i++)); do
  shard="$WORK_DIR/shard_$i.csv"
  say "  shard_$i.csv: $(wc -l <"$shard" | tr -d ' ') lines, $(wc -c <"$shard" | tr -d ' ') bytes"
done

# Delete what an earlier run with the same WORKERS left in OUT_DIR (see the
# header). Emptying the server log here also keeps the first poll below from
# reading an old copy before perl has opened the file.
rm -f "$SUMMARY" "$OUT_DIR"/kill_worker_"${WORKERS}"w_slot*.txt
: >"$SERVER_LOG"

# The server's stdout and stderr go into a FIFO that perl reads, and perl puts
# a wall-clock timestamp in front of every line (server.c prints no timestamps,
# and all of its output is on stderr). A FIFO rather than a pipe keeps $! on
# the server itself, so the kill -0 polls, the timeout kill and cleanup's kill
# act on the server and not on perl.
mkfifo "$WORK_DIR/server_fifo"
perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%.6f %s", time, $_' \
  <"$WORK_DIR/server_fifo" >"$SERVER_LOG" &
STAMP_PID=$!
./server "$WORKERS" >"$WORK_DIR/server_fifo" 2>&1 &
SERVER_PID=$!
wait_for_server_line "server listening on port $PORT" "$SERVER_PID"
say "server: ./server $WORKERS (pid $SERVER_PID), output timestamped into ${SERVER_LOG##*/}"

say "workers (each started after the previous one registered):"
for ((i = 0; i < WORKERS; i++)); do
  log="$(worker_log "$i")"
  ./worker "$SERVER_HOST" "$WORK_DIR/shard_$i.csv" >"$log" 2>&1 &
  WORKER_PIDS[i]=$!
  wait_for_server_line "worker $i registered (" "${WORKER_PIDS[i]}"
  say "  slot $i: pid ${WORKER_PIDS[i]}, shard_$i.csv, log ${log##*/}"
done
accepted="$(accept_order)"
registered="$(register_order)"
if [[ "$accepted" != "$EXPECTED_ORDER" || "$registered" != "$EXPECTED_ORDER" ]]; then
  die "accepted idx [$accepted] and registered [$registered] in $SERVER_LOG should both be [$EXPECTED_ORDER]"
fi

echo "waiting for \"$KILL_LINE\""
deadline=$((SECONDS + WAIT_SECS))
# Anchored, so that "round 1" does not also match "round 10" and later.
until grep -qE -- "$KILL_LINE\$" "$SERVER_LOG"; do
  if grep -qF -- "$SERVER_DONE_LINE" "$SERVER_LOG"; then
    die "$TOO_FAST"
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || die "server exited before \"$KILL_LINE\" (see $SERVER_LOG)"
  ((SECONDS < deadline)) || die "no \"$KILL_LINE\" after ${WAIT_SECS}s"
  sleep 0.02
done
if grep -qF -- "$SERVER_DONE_LINE" "$SERVER_LOG"; then
  die "$TOO_FAST"
fi
KILL_PID="${WORKER_PIDS[KILL_SLOT]}"
# After the kill, bash prints a "Killed: 9" notice at the first point where it
# reports job status, such as an external command, a pipeline or subshell, the
# next top-level command, or a wait. Inside this group the first such point is
# the wait below, whose stderr is discarded.
{
  KILL_TIME="$(now)"
  if ! kill -9 "$KILL_PID" 2>/dev/null; then
    if grep -qF -- "$SERVER_DONE_LINE" "$SERVER_LOG"; then
      die "$TOO_FAST"
    fi
    die "slot $KILL_SLOT (pid $KILL_PID) had already exited"
  fi
  KILLED_STATUS=0
  wait "$KILL_PID" 2>/dev/null || KILLED_STATUS=$?
}
say "kill: SIGKILL to slot $KILL_SLOT (pid $KILL_PID) after the script saw \"$KILL_LINE\"; kill time $KILL_TIME, read just before kill -9"

echo "waiting for the run to finish"
wait_for_exit "$SERVER_PID" server
SERVER_STATUS=0
wait "$SERVER_PID" || SERVER_STATUS=$?
wait_for_exit "$STAMP_PID" timestamper
wait "$STAMP_PID" || true
for ((i = 0; i < WORKERS; i++)); do
  if ((i == KILL_SLOT)); then
    continue
  fi
  wait_for_exit "${WORKER_PIDS[i]}" "worker in slot $i"
  WORKER_STATUS[i]=0
  wait "${WORKER_PIDS[i]}" || WORKER_STATUS[i]=$?
done

say "checks:"
run_checks
if ((FAILED)); then
  say "result: at least one check failed"
else
  say "result: all checks passed"
fi
cp "$WORK_DIR/summary.txt" "$SUMMARY"
echo "summary: $SUMMARY"
exit "$FAILED"
