#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include "protocol.h"
#include "net_utils.h"
#include "io_utils.h"
#include "model.h"

/* ========== CONNECTION MANAGEMENT (Partner 2) ========== */
/* TODO: add_worker() */
/* TODO: handle_register() */
/* TODO: broadcast_weights() */

/* Forward declarations for training logic functions used by skeleton */
int count_active_workers(struct worker_info *workers);

/* ========== TRAINING LOGIC (Partner 1) ========== */

/*
 * Return the number of workers with fd != -1.
 */
int count_active_workers(struct worker_info *workers) {
    int count = 0;
    int i;
    for (i = 0; i < MAX_WORKERS; i++) {
        if (workers[i].fd != -1) {
            count++;
        }
    }
    return count;
}

/*
 * Deserialize MSG_GRADIENT from w->recv_buf.
 * Payload: round(u32) + num_features(u32) + local_loss(float) + gradients[n].
 * Validates round matches current_round; discards stale gradients.
 */
int handle_gradient(struct worker_info *w, int current_round) {
    int offset = HEADER_SIZE;
    uint32_t net_val;

    /* round */
    memcpy(&net_val, w->recv_buf + offset, 4);
    int round = (int)ntohl(net_val);
    offset += 4;

    /* num_features */
    memcpy(&net_val, w->recv_buf + offset, 4);
    int nf = (int)ntohl(net_val);
    offset += 4;

    /* Bounds check on num_features */
    if (nf <= 0 || nf > MAX_FEATURES) {
        fprintf(stderr, "handle_gradient: invalid num_features %d from worker fd %d\n",
                nf, w->fd);
        return -1;
    }

    /* local_loss */
    float local_loss;
    memcpy(&local_loss, w->recv_buf + offset, sizeof(float));
    offset += sizeof(float);

    /* Validate round */
    if (round != current_round) {
        fprintf(stderr, "Stale gradient: got round %d, expected %d\n",
                round, current_round);
        return -1;
    }

    /* Store gradients and loss */
    memcpy(w->gradients, w->recv_buf + offset, nf * sizeof(float));
    w->local_loss = local_loss;
    w->gradient_received = 1;
    return 0;
}

/*
 * Return 1 if every active worker (fd != -1, state == 2) has sent
 * its gradient for the current round.
 */
int all_gradients_received(struct worker_info *workers) {
    int i;
    if (count_active_workers(workers) == 0) {
        return 0;
    }
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
int check_termination(float loss, int round, int max_rounds) {
    if (loss < LOSS_THRESHOLD || round >= max_rounds) {
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

/* ========== SHARED (skeleton) ========== */
/* TODO: dispatch_message() */
/* TODO: handle_disconnect() */
/* TODO: main() — select() loop following simpleselect.c pattern */
