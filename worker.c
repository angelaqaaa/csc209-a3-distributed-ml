#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include "protocol.h"
#include "net_utils.h"
#include "io_utils.h"
#include "model.h"
#include "data.h"

/* ========== CONNECTION + REGISTRATION (Partner 2) ========== */
/* TODO: send_register() */
/* TODO: receive_weights() */

/* ========== GRADIENT + DONE (Partner 1) ========== */

/*
 * Ignore SIGPIPE so writes to a closed socket return EPIPE
 * instead of killing the process. Called from main() at startup.
 */
static void setup_worker_signals(void) {
    signal(SIGPIPE, SIG_IGN);
}

/*
 * Serialize and send MSG_GRADIENT to the server.
 * Payload: round(u32) + num_features(u32) + loss(float) + gradients[n].
 * Returns 0 on success, -1 on error.
 */
int send_gradient(int fd, int round, int num_features,
                  float *gradients, float loss) {
    char payload[12 + MAX_FEATURES * sizeof(float)];
    uint32_t payload_len = 12 + num_features * sizeof(float);
    int offset = 0;

    /* Payload: round */
    uint32_t net_round = htonl(round);
    memcpy(payload + offset, &net_round, 4);
    offset += 4;

    /* Payload: num_features */
    uint32_t net_nf = htonl(num_features);
    memcpy(payload + offset, &net_nf, 4);
    offset += 4;

    /* Payload: loss */
    memcpy(payload + offset, &loss, sizeof(float));
    offset += sizeof(float);

    /* Payload: gradients array */
    memcpy(payload + offset, gradients, num_features * sizeof(float));

    /* Send header then payload */
    if (send_header(fd, MSG_GRADIENT, payload_len) == -1
            || write_all(fd, payload, payload_len) == -1) {
        fprintf(stderr, "send_gradient: write failed\n");
        return -1;
    }
    return 0;
}

/*
 * Read MSG_DONE payload via read_all(), deserialize and print results.
 * Payload: num_features(u32) + final_loss(float) + weights[n].
 * Returns 1 to signal main loop to exit.
 */
int handle_done(int fd, float *weights, int num_features) {
    char payload[8 + MAX_FEATURES * sizeof(float)];
    int payload_size = 8 + num_features * sizeof(float);
    int offset = 0;
    int j;

    if (read_all(fd, payload, payload_size) == -1) {
        fprintf(stderr, "handle_done: server disconnected\n");
        return -1;
    }

    /* num_features from payload (for verification) */
    uint32_t net_nf;
    memcpy(&net_nf, payload + offset, 4);
    offset += 4;
    (void)ntohl(net_nf);  /* could verify, but trust server */

    /* final_loss */
    float final_loss;
    memcpy(&final_loss, payload + offset, sizeof(float));
    offset += sizeof(float);

    /* weights */
    memcpy(weights, payload + offset, num_features * sizeof(float));

    printf("Training complete. Final loss: %.6f\n", final_loss);
    printf("Final weights:");
    for (j = 0; j < num_features; j++) {
        printf(" %.4f", weights[j]);
    }
    printf("\n");

    return 1;
}

/* TODO: main() — connect, register, training loop, clean exit */
