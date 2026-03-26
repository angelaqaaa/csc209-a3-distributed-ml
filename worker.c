/* Partner 2 drafts skeleton; both partners add their functions */

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
 * Serialize and send MSG_GRADIENT to the server.
 * Payload: round(u32) + num_features(u32) + loss(float) + gradients[n].
 * Returns 0 on success, -1 on error.
 */
int send_gradient(int fd, int round, int num_features,
                  float *gradients, float loss) {
    char buf[HEADER_SIZE + 12 + MAX_FEATURES * sizeof(float)];
    int offset = 0;

    /* Header */
    buf[offset] = MSG_GRADIENT;
    offset += 1;

    uint32_t payload_size = htonl(12 + num_features * sizeof(float));
    memcpy(buf + offset, &payload_size, 4);
    offset += 4;

    /* Payload: round */
    uint32_t net_round = htonl(round);
    memcpy(buf + offset, &net_round, 4);
    offset += 4;

    /* Payload: num_features */
    uint32_t net_nf = htonl(num_features);
    memcpy(buf + offset, &net_nf, 4);
    offset += 4;

    /* Payload: loss */
    memcpy(buf + offset, &loss, sizeof(float));
    offset += sizeof(float);

    /* Payload: gradients array */
    memcpy(buf + offset, gradients, num_features * sizeof(float));
    offset += num_features * sizeof(float);

    return write_all(fd, buf, offset);
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
        perror("read MSG_DONE payload");
        return 1;
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
