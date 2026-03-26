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

/*
 * Construct and send a MSG_REGISTER message to the server. The payload
 * specifies the number of local training samples this worker holds.
 * Returns 0 on success, -1 on error.
 */
int send_register(int fd, int num_samples) {
    uint8_t header[HEADER_SIZE];
    header[0] = MSG_REGISTER;
    uint32_t payload = (uint32_t)sizeof(uint32_t);
    uint32_t payload_net = htonl(payload);
    memcpy(header + 1, &payload_net, sizeof(uint32_t));

    uint32_t samples_net = htonl((uint32_t)num_samples);

    if (write_all(fd, header, HEADER_SIZE) < 0) return -1;
    if (write_all(fd, &samples_net, sizeof(samples_net)) < 0) return -1;
    return 0;
}

/*
 * Blocking helper used by the worker to read a MSG_WEIGHTS message
 * from the server. On success fills weights_out with up to
 * max_features floats and returns the number of floats received.
 * Returns -1 on error or unexpected message type.
 */
int receive_weights(int fd, float *weights_out, int max_features) {
    uint8_t header[HEADER_SIZE];
    if (read_all(fd, header, HEADER_SIZE) < 0) return -1;
    uint8_t type = header[0];
    uint32_t payload_net;
    memcpy(&payload_net, header + 1, sizeof(uint32_t));
    uint32_t payload = ntohl(payload_net);
    if (type != MSG_WEIGHTS) {
        fprintf(stderr, "expected MSG_WEIGHTS, got type=%u\n", type);
        /* drain payload if any */
        if (payload > 0) {
            char tmp[256];
            uint32_t left = payload;
            while (left > 0) {
                uint32_t toread = left > sizeof(tmp) ? sizeof(tmp) : left;
                if (read_all(fd, tmp, toread) < 0) return -1;
                left -= toread;
            }
        }
        return -1;
    }

    if (payload > (uint32_t)(max_features * sizeof(float))) {
        fprintf(stderr, "weights payload too large (%u)\n", payload);
        return -1;
    }

    if (read_all(fd, weights_out, (size_t)payload) < 0) return -1;
    return (int)(payload / sizeof(float));
}


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


/* ========== Partner 2 implementation ========== */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <server-host> [num_samples]\n", argv[0]);
        return 1;
    }

    const char *host = argv[1];
    int num_samples = 10;
    if (argc >= 3) num_samples = atoi(argv[2]);

    int fd = connect_to_server(host, PORT);
    if (fd < 0) return 1;

    fprintf(stderr, "connected to server %s:%d (fd=%d)\n", host, PORT, fd);

    if (send_register(fd, num_samples) < 0) {
        fprintf(stderr, "failed to send register\n");
        close(fd);
        return 1;
    }

    fprintf(stderr, "sent register (samples=%d), waiting for weights...\n", num_samples);

    float weights[MAX_FEATURES];
    int nfeatures = receive_weights(fd, weights, MAX_FEATURES);
    if (nfeatures < 0) {
        fprintf(stderr, "failed to receive weights\n");
        close(fd);
        return 1;
    }

    fprintf(stderr, "received %d weights from server:\n", nfeatures);
    for (int i = 0; i < nfeatures; ++i) {
        fprintf(stderr, " w[%d]=%f\n", i, weights[i]);
    }

    close(fd);
    return 0;
}
