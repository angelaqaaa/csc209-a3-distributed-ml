# Distributed ML Training System

CSC209 Assignment 3, Category 2 (sockets). A parameter server trains a logistic
regression model with several worker processes over TCP. The server holds the
current weights, accepts worker connections, collects one gradient per worker
per round, averages them weighted by each worker's sample count, updates the
model, and decides when to stop. Each
worker loads a CSV shard from disk, registers its dataset shape, receives the
weights, computes a local gradient, and sends it back.

The design report is in `project.pdf` (source: `project.tex`).

## Build

```
make
```

This builds `server`, `worker` and `gen_data` with `-Wall -Wextra -g`. The
listening port comes from the Makefile (`PORT = 4242`) and is passed in as
`-DPORT`, so `make PORT=5000` builds against another port.

```
make clean
```

## Run

Generate shards first:

```
./gen_data <num_features> <total_samples> <num_shards>
```

It writes `shard_0.csv` through `shard_<num_shards-1>.csv`. Each line is
`feature1,...,featureN,label`, where the last column is a 0 or 1 label.

Start the server, optionally telling it how many workers to wait for (default
1):

```
./server [expected_workers]
```

Then start one worker per shard:

```
./worker <server-host> <shard_file>
```

The server broadcasts the weights once every expected worker has registered,
waits for all of their gradients, updates the model, and repeats. It stops
once the aggregated loss drops below `LOSS_THRESHOLD` or the round counter
reaches `MAX_ROUNDS`, then broadcasts the final weights with `MSG_DONE`.

## Protocol

Every message is a 5-byte header followed by a payload. The header is one type
byte plus a 4-byte payload size in network byte order. Headers and payloads are
serialized by hand with `memcpy` and `htonl`/`ntohl`, so there are no packed
structs on the wire.

| Type | Code | Direction | Payload |
| --- | --- | --- | --- |
| `MSG_REGISTER` | `0x01` | worker to server | feature count and shard sample count |
| `MSG_WEIGHTS` | `0x02` | server to workers | round number, feature count, weights |
| `MSG_GRADIENT` | `0x03` | worker to server | round number, feature count, local loss, gradients |
| `MSG_DONE` | `0x04` | server to workers | feature count, final loss, final weights |

Limits live in `protocol.h`: `MAX_WORKERS` 8, `MAX_FEATURES` 64, `MAX_ROUNDS`
100, `LEARNING_RATE` 0.01, `LOSS_THRESHOLD` 0.001.

## Concurrency

The server is a single process built around one `select()` loop, with no
threads and no fork per client. It never blocks on one worker: each socket has
its own accumulation buffer in `struct worker_info`, and a readable socket gets
exactly one `read()` call, so a message that arrives in pieces is held until it
is complete. Each round is a synchronous barrier, the server only aggregates
once every registered worker has sent its gradient for that round. A worker
that disconnects mid-round is dropped and the round continues with the
remaining workers.

## Layout

| File | Contents |
| --- | --- |
| `server.c` | select() loop, worker admission, gradient aggregation, termination |
| `worker.c` | registration, receive loop, local gradient, done handling |
| `protocol.h` | message codes, header size, limits, `struct worker_info` |
| `io_utils.c` | `read_all()`, `write_all()`, `accumulate_read()` |
| `net_utils.c` | listening and connecting socket helpers |
| `model.c` | `sigmoid()`, `compute_gradient()`, `update_weights()`, `init_weights()` |
| `data.c` | CSV shard loader |
| `gen_data.c` | synthetic shard generator |

## Tests

Unit tests build from their own makefile:

```
make -f test_Makefile test_all
```

That builds and runs `test_io_utils`, `test_model`, `test_server` and
`test_worker`.

`smoke_test.sh` is an end to end harness that generates shards, starts the
server and the workers, and checks the training run. It uses its own port
(`TEST_PORT`, default 4343) and takes `NUM_FEATURES`, `SCENARIOS` and
`SERVER_HOST` from the environment.

```
./smoke_test.sh
```

## Worker-kill run (post-course, 2026-10-13)

Post-course addition by Angela Su, 2026-10-13. The runs below were made on
2026-10-13 on Angela Su's machine after the course ended, not as part of the
graded submission. No C code or Makefile was changed for them.

`bench/kill_worker.sh` builds the binaries for port 4545, generates one shard
per worker with `gen_data` (640000 samples with 64 features in total), starts
`./server <workers>`, and then starts the workers one at a time, each after
the previous one has registered, so the slots follow the start order. Once
the script sees `broadcasted weights for round 10` in the server log, it
sends `SIGKILL` to the worker in slot 2, waits for the run to end, and checks
the logs. The shards come from `gen_data.c`, which Jinxi Li implemented
(`git log -- gen_data.c`).

The script's default is 160000 samples, so the two runs set `TOTAL_SAMPLES`:

```
WORKERS=4 TOTAL_SAMPLES=640000 OUT_DIR=bench/results/2026-10-13 \
  /bin/bash bench/kill_worker.sh
WORKERS=8 TOTAL_SAMPLES=640000 OUT_DIR=bench/results/2026-10-13 \
  /bin/bash bench/kill_worker.sh
```

At the default 160000 samples an 8-worker round was shorter than the time the
script needs to send the kill after it sees the round 10 line, so in both
8-worker attempts the kill landed after slot 2 had already sent its round 12
gradient, and check (f) in the summary, which allows gradients from the killed
worker up to round 11, failed; the 4-worker attempt passed at that limit. That
is why the committed runs use 640000; the logs of those three attempts are in
`bench/results/2026-10-13/default-samples/`.

| Workers | Samples per shard | Killed | Disconnect logged after | Aggregations after it | Summary |
| --- | --- | --- | --- | --- | --- |
| 4 | 160000 | slot 2, after the round 10 broadcast | round 9 aggregation | 91 | `bench/results/2026-10-13/kill_summary_4w.txt` |
| 8 | 80000 | slot 2, after the round 10 broadcast | round 10 aggregation | 90 | `bench/results/2026-10-13/kill_summary_8w.txt` |

Next to each summary are the timestamped server log (`kill_server_<n>w.txt`)
and one log per worker (`kill_worker_<n>w_slot<i>.txt`). In both runs the
server logged `worker 2 disconnected` after the kill, kept aggregating until
the `MAX_ROUNDS` cap (101 `aggregated gradients` lines, rounds 0 to 100) and
ended with `training complete; broadcasting DONE`. Every other worker printed
`Training complete. Final loss:`.

The server's disconnect handler (`handle_disconnect`, `server.c:278-285`)
closes the socket and clears the slot. Nothing hands the dropped shard to
another worker, so the remaining workers finish the run on their own shards.
The barrier (`all_gradients_received`, `server.c:167-180`) and the weighted
average (`aggregate_and_update`, `server.c:187-224`) only look at slots that
are still connected and were sent the weights (`fd != -1` and `state == 2`),
so the dropped worker's samples are excluded from every round after the
disconnect, here rounds 10 to 100 with 4 workers and rounds 11 to 100 with 8.
The final weights therefore differ from an all-worker run by construction.
