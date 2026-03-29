#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/select.h>
#include "protocol.h"
#include "net_utils.h"
#include "io_utils.h"
#include "model.h"

/* ========== CONNECTION MANAGEMENT (Partner 2) ========== */
static struct worker_info workers[MAX_WORKERS];
static float global_weights[MAX_FEATURES];

/* forward declarations for functions used before their definitions */
void handle_disconnect(int idx);

/*
 * Find an empty slot in the server's worker table and initialize it for
 * a newly accepted client connection. Returns the worker index on
 * success or -1 if no free slot is available.
 */
int add_worker(int fd) {
    for (int i = 0; i < MAX_WORKERS; ++i) {
        if (workers[i].fd == -1) {
            workers[i].fd = fd;
            workers[i].state = 0;
            workers[i].num_samples = 0;
            workers[i].recv_len = 0;
            workers[i].gradient_received = 0;
            workers[i].local_loss = 0.0f;
            memset(workers[i].gradients, 0, sizeof(workers[i].gradients));
            return i;
        }
    }
    return -1;
}


/*
 * Serialize and send the current global weight vector to the worker at
 * index 'idx'. Payload: round(u32) + num_features(u32) + float[n].
 * Returns 0 on success or -1 on error.
 */
int send_weights_to_worker(int idx, int round, int num_features) {
    if (idx < 0 || idx >= MAX_WORKERS) return -1;
    struct worker_info *w = &workers[idx];
    if (w->fd == -1) return -1;

    uint32_t payload = (uint32_t)(8 + num_features * sizeof(float));
    uint8_t header[HEADER_SIZE];
    header[0] = MSG_WEIGHTS;
    uint32_t payload_net = htonl(payload);
    memcpy(header + 1, &payload_net, sizeof(uint32_t));

    /* build payload */
    char *buf = malloc((size_t)payload);
    if (!buf) return -1;
    int off = 0;
    uint32_t net_round = htonl((uint32_t)round);
    memcpy(buf + off, &net_round, 4);
    off += 4;
    uint32_t net_nf = htonl((uint32_t)num_features);
    memcpy(buf + off, &net_nf, 4);
    off += 4;
    memcpy(buf + off, global_weights, num_features * sizeof(float));
    off += num_features * sizeof(float);

    if (write_all(w->fd, header, HEADER_SIZE) < 0) {
        free(buf);
        return -1;
    }
    if (write_all(w->fd, buf, (size_t)payload) < 0) {
        free(buf);
        return -1;
    }

    free(buf);
    w->state = 2; /* sent_weights */
    return 0;
}

/*
 * Send the current weights to all registered workers.
 * Returns 0 on success, -1 if any send fails.
 */

int broadcast_weights(int round, int num_features) {
    int result = 0;
    for (int i = 0; i < MAX_WORKERS; ++i) {
        if (workers[i].fd != -1 && workers[i].state >= 1) {
            if (send_weights_to_worker(i, round, num_features) < 0) {
                fprintf(stderr, "failed to send weights to worker %d\n", i);
                /* treat write failure as disconnect */
                handle_disconnect(i);
                result = -1;
            }
        }
    }
    return result;
}



/* ========== TRAINING LOGIC (Partner 1) ========== */

/*
 * Deserialize MSG_GRADIENT from w->recv_buf.
 * Payload: round(u32) + num_features(u32) + local_loss(float) + gradients[n].
 * Validates round matches current_round; discards stale gradients.
 */
void handle_gradient(struct worker_info *w, int current_round) {
    int offset = HEADER_SIZE;
    uint32_t net_val;

    /* round */
    memcpy(&net_val, w->recv_buf + offset, 4);
    int round = (int)ntohl(net_val);
    offset += 4;

    /* num_features */
    memcpy(&net_val, w->recv_buf + offset, 4);
    int num_features = (int)ntohl(net_val);
    offset += 4;

    /* local_loss */
    float local_loss;
    memcpy(&local_loss, w->recv_buf + offset, sizeof(float));
    offset += sizeof(float);

    /* Validate round */
    if (round != current_round) {
        fprintf(stderr, "Stale gradient: got round %d, expected %d\n",
                round, current_round);
        return;
    }

    /* Store gradients and loss */
    memcpy(w->gradients, w->recv_buf + offset, num_features * sizeof(float));
    w->local_loss = local_loss;
    w->gradient_received = 1;
}

/*
 * Return 1 if every active worker (fd != -1, state == 2) has sent
 * its gradient for the current round.
 */
int all_gradients_received(struct worker_info *workers) {
    int i;
    for (i = 0; i < MAX_WORKERS; i++) {
        if (workers[i].fd != -1 && workers[i].state == 2) {
            if (workers[i].gradient_received != 1) {
                return 0;
            }
        }
    }
    return 1;
}

/*
 * Compute weighted average of gradients across all active workers,
 * update the global weight vector, and compute the global loss.
 * Each worker's contribution is weighted by its num_samples.
 */
void aggregate_and_update(float *weights, struct worker_info *workers,
                          int num_features, float learning_rate,
                          float *global_loss) {
    int i, j;
    float avg_gradient[MAX_FEATURES];
    int total_samples = 0;
    float weighted_loss = 0.0f;

    memset(avg_gradient, 0, num_features * sizeof(float));

    for (i = 0; i < MAX_WORKERS; i++) {
        if (workers[i].fd != -1 && workers[i].state == 2
                && workers[i].gradient_received == 1) {
            int ns = workers[i].num_samples;
            total_samples += ns;
            weighted_loss += workers[i].local_loss * ns;
            for (j = 0; j < num_features; j++) {
                avg_gradient[j] += workers[i].gradients[j] * ns;
            }
        }
    }

    if (total_samples > 0) {
        for (j = 0; j < num_features; j++) {
            avg_gradient[j] /= total_samples;
        }
        *global_loss = weighted_loss / total_samples;
    }

    update_weights(weights, avg_gradient, num_features, learning_rate);

    /* Reset gradient_received flags for next round */
    for (i = 0; i < MAX_WORKERS; i++) {
        if (workers[i].fd != -1 && workers[i].state == 2) {
            workers[i].gradient_received = 0;
        }
    }
}

/*
 * Return 1 if training should stop (loss below threshold or max rounds).
 */
int check_termination(float loss, int round) {
    if (loss < LOSS_THRESHOLD || round >= MAX_ROUNDS) {
        return 1;
    }
    return 0;
}

/*
 * Serialize and send MSG_DONE to all active workers.
 * Payload: num_features(u32) + final_loss(float) + weights[n].
 * Returns 0 on success, -1 if any write fails.
 */
int broadcast_done(struct worker_info *workers, float *weights,
                   int num_features, float final_loss) {
    char buf[HEADER_SIZE + 8 + MAX_FEATURES * sizeof(float)];
    int offset = 0;
    int i;

    /* Header */
    buf[offset] = MSG_DONE;
    offset += 1;

    uint32_t payload_size = htonl(8 + num_features * sizeof(float));
    memcpy(buf + offset, &payload_size, 4);
    offset += 4;

    /* Payload: num_features */
    uint32_t net_nf = htonl(num_features);
    memcpy(buf + offset, &net_nf, 4);
    offset += 4;

    /* Payload: final_loss */
    memcpy(buf + offset, &final_loss, sizeof(float));
    offset += sizeof(float);

    /* Payload: weights array */
    memcpy(buf + offset, weights, num_features * sizeof(float));
    offset += num_features * sizeof(float);

    /* Send to all active workers */
    int result = 0;
    for (i = 0; i < MAX_WORKERS; i++) {
        if (workers[i].fd != -1 && workers[i].state == 2) {
            if (write_all(workers[i].fd, buf, offset) == -1) {
                fprintf(stderr, "Failed to send MSG_DONE to worker %d\n", i);
                result = -1;
            }
        }
    }
    return result;
}

/* ========== skeleton (Partner 2) ========== */

/*
 * Cleanly close and reset the worker slot at index idx. This frees the
 * slot so a new connection may be assigned there.
 */
void handle_disconnect(int idx) {
    if (idx < 0 || idx >= MAX_WORKERS) return;
    if (workers[idx].fd != -1) close(workers[idx].fd);
    workers[idx].fd = -1;
    workers[idx].state = 0;
    workers[idx].recv_len = 0;
    workers[idx].gradient_received = 0;
}


/*
 * Parse a single complete message from the worker's receive buffer and
 * dispatch it to the appropriate handler.
 * Supports MSG_REGISTER, MSG_GRADIENT, and MSG_DONE.
 * After processing the message the function removes the processed bytes
 * from the worker's buffer so subsequent messages remain.
 */
void dispatch_message(int idx, int *num_features, int current_round) {
    struct worker_info *w = &workers[idx];
    if (w->recv_len < HEADER_SIZE) return;

    uint8_t type = (uint8_t)w->recv_buf[0];
    uint32_t payload_net;
    memcpy(&payload_net, w->recv_buf + 1, sizeof(uint32_t));
    uint32_t payload = ntohl(payload_net);

    if (w->recv_len < (int)(HEADER_SIZE + payload)) return; /* incomplete */

    char *p = w->recv_buf + HEADER_SIZE;

    if (type == MSG_REGISTER) {
        if (payload < 8) {
            fprintf(stderr, "bad register payload size %u\n", payload);
        } else {
            uint32_t nf_net;
            memcpy(&nf_net, p, 4);
            uint32_t nf = ntohl(nf_net);
            p += 4;
            uint32_t ns_net;
            memcpy(&ns_net, p, 4);
            uint32_t ns = ntohl(ns_net);
            w->num_samples = (int)ns;
            w->state = 1; /* registered; will become 2 when weights are sent */
            fprintf(stderr, "worker %d registered (samples=%d, num_features=%u)\n", idx, w->num_samples, nf);
            /* adjust server's num_features if necessary */
            if (nf > 0 && nf <= MAX_FEATURES) {
                if (*num_features == MAX_FEATURES) {
                    *num_features = (int)nf;
                } else if (*num_features != (int)nf) {
                    fprintf(stderr, "warning: worker %d num_features %u differs from server %d; using worker's value\n", idx, nf, *num_features);
                    *num_features = (int)nf;
                }
            }
            /* don't send weights here; server orchestration will broadcast when ready */
        }
    } else if (type == MSG_GRADIENT) {
        /* Delegate to existing handler that expects gradients located at HEADER_SIZE */
        handle_gradient(w, current_round);
    } else if (type == MSG_DONE) {
        fprintf(stderr, "worker %d sent DONE\n", idx);
    } else {
        fprintf(stderr, "unknown message type %u from worker %d\n", type, idx);
    }

    /* remove processed message from buffer, shift remainder left */
    int total_len = HEADER_SIZE + (int)payload;
    if (w->recv_len > total_len) {
        memmove(w->recv_buf, w->recv_buf + total_len, w->recv_len - total_len);
    }
    w->recv_len -= total_len;
}



int main(int argc, char **argv) {
    int port = PORT;
    int listen_fd;
    int num_features = MAX_FEATURES; /* default; partner1 may use fewer */
    int current_round = 0;
    int expected_workers = 1; /* default: 1 */
    int weights_broadcast = 0; /* whether we've broadcast for the current_round */

    if (argc >= 2) {
        expected_workers = atoi(argv[1]);
        if (expected_workers <= 0) expected_workers = 1;
    }

    fprintf(stderr, "server starting: PORT=%d expected_workers=%d\n", port, expected_workers);

    /* initialize workers */
    for (int i = 0; i < MAX_WORKERS; ++i) workers[i].fd = -1;

    /* init weights to zero */
    for (int i = 0; i < num_features; ++i) global_weights[i] = 0.0f;

    listen_fd = set_up_server_socket(port);
    if (listen_fd < 0) {
        fprintf(stderr, "failed to set up server socket on port %d\n", port);
        return 1;
    }
    fprintf(stderr, "server listening on port %d\n", port);

    /* ignore SIGPIPE so write failures return EPIPE instead of crashing */
    signal(SIGPIPE, SIG_IGN);

    while (1) {
        fd_set rfds;
        int maxfd = listen_fd;
        FD_ZERO(&rfds);
        FD_SET(listen_fd, &rfds);

        for (int i = 0; i < MAX_WORKERS; ++i) {
            if (workers[i].fd != -1) {
                FD_SET(workers[i].fd, &rfds);
                if (workers[i].fd > maxfd) maxfd = workers[i].fd;
            }
        }

        int rv = select(maxfd + 1, &rfds, NULL, NULL, NULL);
        if (rv < 0) {
            if (errno == EINTR) continue;
            perror("select");
            break;
        }

        if (FD_ISSET(listen_fd, &rfds)) {
            int cli = accept_connection(listen_fd);
            if (cli >= 0) {
                int idx = add_worker(cli);
                if (idx < 0) {
                    fprintf(stderr, "no free worker slots; closing connection\n");
                    close(cli);
                } else {
                    fprintf(stderr, "accepted connection fd=%d assigned idx=%d\n", cli, idx);
                }
            }
        }

        for (int i = 0; i < MAX_WORKERS; ++i) {
            if (workers[i].fd != -1 && FD_ISSET(workers[i].fd, &rfds)) {
                int r = accumulate_read(&workers[i]);
                if (r < 0) {
                    fprintf(stderr, "worker %d disconnected\n", i);
                    handle_disconnect(i);
                } else if (r == 1) {
                    /* process all complete messages in buffer */
                    while (workers[i].recv_len >= HEADER_SIZE) {
                        uint32_t payload_net;
                        memcpy(&payload_net, workers[i].recv_buf + 1, sizeof(uint32_t));
                        uint32_t payload = ntohl(payload_net);
                        if (workers[i].recv_len < (int)(HEADER_SIZE + payload)) break;
                        dispatch_message(i, &num_features, current_round);
                    }
                }
            }
        }

        /* After processing incoming messages, check orchestration state */
        /* Count registered workers (state >=1) */
        int registered_count = 0;
        for (int i = 0; i < MAX_WORKERS; ++i) {
            if (workers[i].fd != -1 && workers[i].state >= 1) registered_count++;
        }

        /* If we have reached expected workers and haven't broadcast weights yet, do it */
        if (!weights_broadcast && registered_count >= expected_workers) {
            if (broadcast_weights(current_round, num_features) < 0) {
                fprintf(stderr, "broadcast initial weights failed\n");
            } else {
                weights_broadcast = 1;
                fprintf(stderr, "broadcasted initial weights for round %d\n", current_round);
            }
        }

        /* If weights already broadcasted for current_round, check for gradients */
        if (weights_broadcast) {
            if (all_gradients_received(workers)) {
                float global_loss = 0.0f;
                aggregate_and_update(global_weights, workers, num_features, 0.1f, &global_loss);
                fprintf(stderr, "aggregated gradients; global_loss=%.6f\n", global_loss);
                if (check_termination(global_loss, current_round)) {
                    broadcast_done(workers, global_weights, num_features, global_loss);
                    fprintf(stderr, "training complete; broadcasting DONE\n");
                    break;
                }
                /* next round */
                current_round++;
                /* broadcast new weights for next round */
                if (broadcast_weights(current_round, num_features) < 0) {
                    fprintf(stderr, "broadcast weights for round %d failed\n", current_round);
                } else {
                    fprintf(stderr, "broadcasted weights for round %d\n", current_round);
                }
                /* continue; weights_broadcast remains true for next round */
            }
        }
    }

    close(listen_fd);
    return 0;
}
