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
