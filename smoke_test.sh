#!/usr/bin/env bash
set -euo pipefail

NUM_FEATURES="${NUM_FEATURES:-10}"
SCENARIOS="${SCENARIOS:-1 2 3}"
SERVER_HOST="${SERVER_HOST:-localhost}"
SERVER_WAIT_SECS="${SERVER_WAIT_SECS:-1}"
ROUND_WAIT_SECS="${ROUND_WAIT_SECS:-20}"
TEST_PORT="${TEST_PORT:-4343}"

SERVER_PID=""
WORKER_PIDS=()

cleanup() {
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
      wait "$pid" 2>/dev/null || true
    fi
  done
  WORKER_PIDS=()

  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap cleanup EXIT

require_no_log_line() {
  local pattern="$1"
  local file="$2"
  if grep -q "$pattern" "$file"; then
    echo "FAIL: did not expect '$pattern' in $file"
    return 1
  fi
}

require_log_line() {
  local pattern="$1"
  local file="$2"
  if ! grep -q "$pattern" "$file"; then
    echo "FAIL: expected '$pattern' in $file"
    return 1
  fi
}

wait_for_training_output() {
  local file="$1"
  local pid="$2"
  local waited=0
  while (( waited < ROUND_WAIT_SECS )); do
    if grep -q "Training complete. Final loss:" "$file" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "FAIL: worker exited before training completed: $file"
      tail -n 50 "$file" || true
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "FAIL: timed out waiting for training output in $file"
  tail -n 50 "$file" || true
  return 1
}

run_scenario() {
  local num_workers="$1"
  local total_samples=$((num_workers * 12))
  local prefix="scenario_${num_workers}w"
  local server_log="${prefix}_server.log"

  echo
  echo "== Running scenario: ${num_workers} worker(s), ${total_samples} samples =="

  cleanup
  rm -f shard_*.csv "${prefix}"_worker_*.log "$server_log"

  ./gen_data "$NUM_FEATURES" "$total_samples" "$num_workers"

  ./server "$num_workers" >"$server_log" 2>&1 &
  SERVER_PID=$!
  echo "server pid=$SERVER_PID -> $server_log"
  sleep "$SERVER_WAIT_SECS"
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited during startup"
    tail -n 50 "$server_log" || true
    return 1
  fi

  WORKER_PIDS=()
  for i in $(seq 0 $((num_workers - 1))); do
    local worker_log="${prefix}_worker_${i}.log"
    ./worker "$SERVER_HOST" "shard_${i}.csv" >"$worker_log" 2>&1 &
    WORKER_PIDS+=("$!")
    echo "worker pid=${WORKER_PIDS[$i]} -> $worker_log"
  done

  for i in $(seq 0 $((num_workers - 1))); do
    local worker_log="${prefix}_worker_${i}.log"
    wait_for_training_output "$worker_log" "${WORKER_PIDS[$i]}"
  done

  wait "$SERVER_PID" || true

  require_log_line "server listening" "$server_log"
  require_log_line "broadcasted initial weights for round 0" "$server_log"
  require_log_line "aggregated gradients; global_loss=" "$server_log"
  require_log_line "training complete; broadcasting DONE" "$server_log"

  for i in $(seq 0 $((num_workers - 1))); do
    local worker_log="${prefix}_worker_${i}.log"
    require_log_line "connected to server" "$worker_log"
    require_log_line "sent register" "$worker_log"
    require_log_line "received weights for round" "$worker_log"
    require_log_line "sent gradient for round" "$worker_log"
    require_log_line "Training complete. Final loss:" "$worker_log"
    require_log_line "Final weights:" "$worker_log"
  done

  echo "-- ${server_log} (last 20 lines) --"
  tail -n 20 "$server_log"
  for i in $(seq 0 $((num_workers - 1))); do
    local worker_log="${prefix}_worker_${i}.log"
    echo "-- ${worker_log} (last 12 lines) --"
    tail -n 12 "$worker_log"
  done

  echo "PASS: scenario with ${num_workers} worker(s)"
  cleanup
}

run_insufficient_workers_test() {
  local expected_workers=2
  local actual_workers=1
  local total_samples=24
  local prefix="scenario_insufficient_workers"
  local server_log="${prefix}_server.log"
  local worker_log="${prefix}_worker_0.log"

  echo
  echo "== Running negative test: server expects ${expected_workers}, only ${actual_workers} connects =="

  cleanup
  rm -f shard_*.csv "${prefix}"_worker_*.log "$server_log"

  ./gen_data "$NUM_FEATURES" "$total_samples" "$expected_workers"

  ./server "$expected_workers" >"$server_log" 2>&1 &
  SERVER_PID=$!
  echo "server pid=$SERVER_PID -> $server_log"
  sleep "$SERVER_WAIT_SECS"
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited during startup"
    tail -n 50 "$server_log" || true
    return 1
  fi

  WORKER_PIDS=()
  ./worker "$SERVER_HOST" "shard_0.csv" >"$worker_log" 2>&1 &
  WORKER_PIDS+=("$!")
  echo "worker pid=${WORKER_PIDS[0]} -> $worker_log"

  sleep 3

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "FAIL: server exited unexpectedly during insufficient-worker test"
    tail -n 50 "$server_log" || true
    return 1
  fi

  require_log_line "server listening" "$server_log"
  require_log_line "worker 0 registered" "$server_log"
  require_no_log_line "broadcasted initial weights for round 0" "$server_log"
  require_no_log_line "aggregated gradients; global_loss=" "$server_log"
  require_no_log_line "training complete; broadcasting DONE" "$server_log"
  require_log_line "sent register" "$worker_log"
  require_no_log_line "received weights for round" "$worker_log"
  require_no_log_line "Training complete. Final loss:" "$worker_log"

  echo "-- ${server_log} (last 12 lines) --"
  tail -n 12 "$server_log"
  echo "-- ${worker_log} (last 12 lines) --"
  tail -n 12 "$worker_log"

  echo "PASS: insufficient-worker negative test"
  cleanup
}

echo "== Building project =="
make clean
make PORT="$TEST_PORT"

for workers in $SCENARIOS; do
  run_scenario "$workers"
done

run_insufficient_workers_test

echo
echo "All smoke tests passed for scenarios: $SCENARIOS plus insufficient-worker negative test on port $TEST_PORT"
