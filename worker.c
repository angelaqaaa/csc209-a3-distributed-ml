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
int send_register(int fd, int num_features, int num_samples) {
    char buf[HEADER_SIZE + 8];
    int offset = 0;

    buf[offset] = MSG_REGISTER;
    offset += 1;

    uint32_t payload_size = htonl(8);
    memcpy(buf + offset, &payload_size, 4);
    offset += 4;

    uint32_t net_nf = htonl((uint32_t)num_features);
    memcpy(buf + offset, &net_nf, 4);
    offset += 4;

    uint32_t net_ns = htonl((uint32_t)num_samples);
    memcpy(buf + offset, &net_ns, 4);
    offset += 4;

    return write_all(fd, buf, (size_t)offset);
}

/*
 * Blocking helper used by the worker to read a MSG_WEIGHTS message
 * from the server. On success fills weights_out with up to
 * max_features floats and returns the number of floats received.
 * Returns -1 on error or unexpected message type.
 */
int receive_weights(int fd, float *weights_out, int *round_out,
                    int num_features, int max_features) {
    uint8_t header[HEADER_SIZE];
    if (read_all(fd, header, HEADER_SIZE) < 0) return -1;

    uint8_t type = header[0];
    uint32_t payload_net;
    memcpy(&payload_net, header + 1, sizeof(uint32_t));
    uint32_t payload = ntohl(payload_net);

    if (type != MSG_WEIGHTS) {
        fprintf(stderr, "expected MSG_WEIGHTS, got type=%u\n", type);

        if (payload > 0) {
            char tmp[256];
            uint32_t left = payload;
            while (left > 0) {
                uint32_t toread = left > sizeof(tmp) ? (uint32_t)sizeof(tmp) : left;
                if (read_all(fd, tmp, toread) < 0) return -1;
                left -= toread;
            }
        }
        return -1;
    }

    if (payload < 8 || payload > (uint32_t)(8 + max_features * (int)sizeof(float))) {
        fprintf(stderr, "bad weights payload size %u\n", payload);
        return -1;
    }

    char buf[8 + MAX_FEATURES * sizeof(float)];
    if (read_all(fd, buf, (size_t)payload) < 0) return -1;

    int offset = 0;
    uint32_t net_round;
    uint32_t net_nf;

    memcpy(&net_round, buf + offset, 4);
    *round_out = (int)ntohl(net_round);
    offset += 4;

    memcpy(&net_nf, buf + offset, 4);
    int nf = (int)ntohl(net_nf);
    offset += 4;

    if (nf < 0 || nf > max_features) {
        fprintf(stderr, "server sent invalid num_features=%d\n", nf);
        return -1;
    }

    if (nf > num_features) {
        fprintf(stderr, "server sent %d features but shard has %d; using shard's value\n",
                nf, num_features);
        nf = num_features;
    }

    memcpy(weights_out, buf + offset, (size_t)nf * sizeof(float));
    return nf;
}


/* ========== GRADIENT + DONE (Partner 1) ========== */

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


/* ========== Partner 2 implementation ========== */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <server-host> <shard_file>\n", argv[0]);
        return 1;
    }

    const char *host = argv[1];
    const char *shard_file = argv[2];

    float *X = NULL;
    float *y = NULL;
    int num_samples = 0;
    int num_features = 0;

    signal(SIGPIPE, SIG_IGN);

    if (load_data(shard_file, &X, &y, &num_samples, &num_features) < 0) {
        fprintf(stderr, "failed to load shard %s\n", shard_file);
        return 1;
    }

    int fd = connect_to_server(host, PORT);
    if (fd < 0) {
        free(X);
        free(y);
        return 1;
    }

    fprintf(stderr, "connected to server %s:%d (fd=%d)\n", host, PORT, fd);

    if (send_register(fd, num_features, num_samples) < 0) {
        fprintf(stderr, "failed to send register\n");
        close(fd);
        free(X);
        free(y);
        return 1;
    }

    fprintf(stderr, "sent register (samples=%d, num_features=%d), waiting...\n",
            num_samples, num_features);

    float *weights = malloc((size_t)MAX_FEATURES * sizeof(float));
    float *grad = malloc((size_t)MAX_FEATURES * sizeof(float));
    if (!weights || !grad) {
        fprintf(stderr, "allocation failure\n");
        close(fd); free(X); free(y); free(weights); free(grad);
        return 1;
    }

    while (1) {
        uint8_t header[HEADER_SIZE];
        if (read_all(fd, header, HEADER_SIZE) < 0) {
            fprintf(stderr, "connection closed while waiting for header\n");
            break;
        }
        uint8_t type = header[0];
        uint32_t payload_net;
        memcpy(&payload_net, header + 1, sizeof(uint32_t));
        uint32_t payload = ntohl(payload_net);

        if (type == MSG_WEIGHTS) {
            if (payload < 8) {
                fprintf(stderr, "bad weights payload size %u\n", payload);
                if (payload > 0) {
                    char tmp[128];
                    read_all(fd, tmp, payload);
                }
                continue;
            }
            char buf[8 + MAX_FEATURES * sizeof(float)];
            if (payload > sizeof(buf)) {
                fprintf(stderr, "weights payload too large\n");
                break;
            }
            if (read_all(fd, buf, payload) < 0) break;
            int off = 0;
            uint32_t net_round;
            memcpy(&net_round, buf + off, 4); off += 4;
            int round = (int)ntohl(net_round);
            uint32_t net_nf;
            memcpy(&net_nf, buf + off, 4); off += 4;
            int nf = (int)ntohl(net_nf);
            int copy_nf = (nf > num_features) ? num_features : nf;
            memcpy(weights, buf + off, (size_t)copy_nf * sizeof(float));

            fprintf(stderr, "received weights for round %d (n=%d)\n",
                    round, nf);

            float loss = compute_gradient(X, y, num_samples,
                                          num_features, weights, grad);
            if (send_gradient(fd, round, num_features, grad, loss) < 0) {
                fprintf(stderr, "failed to send gradient\n");
                break;
            }
            fprintf(stderr, "sent gradient for round %d (loss=%.6f)\n",
                    round, loss);

        } else if (type == MSG_DONE) {
            handle_done(fd, weights, num_features);
            break;
        } else {
            /* Unknown message: drain payload with stack buffer */
            char drain[256];
            uint32_t left = payload;
            while (left > 0) {
                uint32_t chunk = left > sizeof(drain)
                                 ? (uint32_t)sizeof(drain) : left;
                if (read_all(fd, drain, chunk) < 0) break;
                left -= chunk;
            }
        }
    }

    close(fd);
    free(X); free(y); free(weights); free(grad);
    return 0;
}
