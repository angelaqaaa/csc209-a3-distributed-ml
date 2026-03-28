#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <math.h>
#include "protocol.h"
#include "io_utils.h"

/* Declarations for Partner 1 worker functions */
int send_gradient(int fd, int round, int num_features,
                  float *gradients, float loss);
int handle_done(int fd, float *weights, int num_features);

/* Server function used in round-trip test */
int handle_gradient(struct worker_info *w, int current_round);

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  PASS: %s\n", msg); \
    } else { \
        printf("  FAIL: %s\n", msg); \
    } \
} while (0)

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < 1e-5f)

/* ========== send_gradient tests ========== */

static void test_send_gradient_basic(void) {
    int sv[2];
    float grads[2] = {0.1f, -0.2f};

    printf("test_send_gradient_basic\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    int ret = send_gradient(sv[1], 5, 2, grads, 0.42f);
    TEST_ASSERT(ret == 0, "send_gradient returns 0 on success");

    /* Read and verify header */
    uint8_t hdr[HEADER_SIZE];
    read(sv[0], hdr, HEADER_SIZE);
    TEST_ASSERT(hdr[0] == MSG_GRADIENT, "message type is MSG_GRADIENT");

    uint32_t net_plen;
    memcpy(&net_plen, hdr + 1, 4);
    uint32_t plen = ntohl(net_plen);
    uint32_t expected_plen = 12 + 2 * sizeof(float);
    TEST_ASSERT(plen == expected_plen, "payload length correct");

    /* Read payload */
    char payload[12 + MAX_FEATURES * sizeof(float)];
    read(sv[0], payload, plen);
    int offset = 0;

    /* round */
    uint32_t net_val;
    memcpy(&net_val, payload + offset, 4);
    int round = (int)ntohl(net_val);
    offset += 4;
    TEST_ASSERT(round == 5, "round in payload is 5");

    /* num_features */
    memcpy(&net_val, payload + offset, 4);
    int nf = (int)ntohl(net_val);
    offset += 4;
    TEST_ASSERT(nf == 2, "num_features in payload is 2");

    /* loss */
    float loss;
    memcpy(&loss, payload + offset, sizeof(float));
    offset += sizeof(float);
    TEST_ASSERT(FLOAT_EQ(loss, 0.42f), "loss in payload correct");

    /* gradients */
    float g0, g1;
    memcpy(&g0, payload + offset, sizeof(float));
    memcpy(&g1, payload + offset + sizeof(float), sizeof(float));
    TEST_ASSERT(FLOAT_EQ(g0, 0.1f), "gradient[0] correct");
    TEST_ASSERT(FLOAT_EQ(g1, -0.2f), "gradient[1] correct");

    close(sv[0]);
    close(sv[1]);
}

static void test_send_gradient_closed_fd(void) {
    int sv[2];
    float grads[1] = {0.0f};

    printf("test_send_gradient_closed_fd\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    close(sv[0]);

    int ret = send_gradient(sv[1], 0, 1, grads, 0.0f);
    TEST_ASSERT(ret == -1, "send_gradient returns -1 on closed peer");
    close(sv[1]);
}

static void test_send_gradient_single_feature(void) {
    int sv[2];
    float grads[1] = {3.14f};

    printf("test_send_gradient_single_feature\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    int ret = send_gradient(sv[1], 0, 1, grads, 1.0f);
    TEST_ASSERT(ret == 0, "send_gradient single feature returns 0");

    uint8_t hdr[HEADER_SIZE];
    read(sv[0], hdr, HEADER_SIZE);

    uint32_t net_plen;
    memcpy(&net_plen, hdr + 1, 4);
    uint32_t plen = ntohl(net_plen);
    TEST_ASSERT(plen == 12 + 1 * sizeof(float),
                "payload len for 1 feature is 16");

    close(sv[0]);
    close(sv[1]);
}

static void test_send_gradient_max_features(void) {
    int sv[2];
    float grads[MAX_FEATURES];
    int i;

    printf("test_send_gradient_max_features\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    for (i = 0; i < MAX_FEATURES; i++) {
        grads[i] = (float)i;
    }
    int ret = send_gradient(sv[1], 99, MAX_FEATURES, grads, 0.01f);
    TEST_ASSERT(ret == 0, "send_gradient MAX_FEATURES returns 0");

    uint8_t hdr[HEADER_SIZE];
    read(sv[0], hdr, HEADER_SIZE);
    uint32_t net_plen;
    memcpy(&net_plen, hdr + 1, 4);
    uint32_t plen = ntohl(net_plen);
    TEST_ASSERT(plen == 12 + MAX_FEATURES * sizeof(float),
                "payload len for MAX_FEATURES correct");

    close(sv[0]);
    close(sv[1]);
}

/* ========== handle_done tests ========== */

static void test_handle_done_basic(void) {
    int sv[2];
    float weights[2] = {0.0f, 0.0f};

    printf("test_handle_done_basic\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    /* Build MSG_DONE payload on the writer side */
    int nf = 2;
    float final_loss = 0.05f;
    float w_vals[2] = {1.5f, -0.5f};

    char payload[8 + MAX_FEATURES * sizeof(float)];
    int offset = 0;
    uint32_t net_nf = htonl(nf);
    memcpy(payload + offset, &net_nf, 4);
    offset += 4;
    memcpy(payload + offset, &final_loss, sizeof(float));
    offset += sizeof(float);
    memcpy(payload + offset, w_vals, nf * sizeof(float));

    int payload_size = 8 + nf * sizeof(float);
    write(sv[1], payload, payload_size);
    close(sv[1]);

    int ret = handle_done(sv[0], weights, nf);
    TEST_ASSERT(ret == 1, "handle_done returns 1 (exit signal)");
    TEST_ASSERT(FLOAT_EQ(weights[0], 1.5f), "weight[0] updated to 1.5");
    TEST_ASSERT(FLOAT_EQ(weights[1], -0.5f), "weight[1] updated to -0.5");

    close(sv[0]);
}

static void test_handle_done_disconnect(void) {
    int sv[2];
    float weights[1] = {0.0f};

    printf("test_handle_done_disconnect\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    /* Close writer immediately — handle_done gets EOF */
    close(sv[1]);

    int ret = handle_done(sv[0], weights, 1);
    TEST_ASSERT(ret == -1, "handle_done returns -1 on disconnect");

    close(sv[0]);
}

static void test_handle_done_single_feature(void) {
    int sv[2];
    float weights[1] = {0.0f};

    printf("test_handle_done_single_feature\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    char payload[12];
    int offset = 0;
    uint32_t net_nf = htonl(1);
    memcpy(payload + offset, &net_nf, 4);
    offset += 4;
    float final_loss = 0.001f;
    memcpy(payload + offset, &final_loss, sizeof(float));
    offset += sizeof(float);
    float w_val = 42.0f;
    memcpy(payload + offset, &w_val, sizeof(float));

    write(sv[1], payload, 8 + sizeof(float));
    close(sv[1]);

    int ret = handle_done(sv[0], weights, 1);
    TEST_ASSERT(ret == 1, "handle_done single feature returns 1");
    TEST_ASSERT(FLOAT_EQ(weights[0], 42.0f), "single weight updated correctly");

    close(sv[0]);
}

/* ========== send_gradient + handle_gradient round-trip ========== */

static void test_gradient_round_trip(void) {
    int sv[2];
    float grads[3] = {0.1f, -0.2f, 0.3f};

    printf("test_gradient_round_trip\n");
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);

    /* Worker sends gradient */
    send_gradient(sv[1], 7, 3, grads, 0.55f);
    close(sv[1]);

    /* Server reads into worker_info */
    struct worker_info w;
    memset(&w, 0, sizeof(w));
    w.fd = sv[0];

    /* Read the full message into recv_buf */
    int expected = HEADER_SIZE + 12 + 3 * sizeof(float);
    int n = read(sv[0], w.recv_buf, expected);
    w.recv_len = n;

    int ret = handle_gradient(&w, 7);
    TEST_ASSERT(ret == 0, "round-trip: handle_gradient succeeds");
    TEST_ASSERT(w.gradient_received == 1, "round-trip: gradient_received set");
    TEST_ASSERT(FLOAT_EQ(w.gradients[0], 0.1f), "round-trip: grad[0]");
    TEST_ASSERT(FLOAT_EQ(w.gradients[1], -0.2f), "round-trip: grad[1]");
    TEST_ASSERT(FLOAT_EQ(w.gradients[2], 0.3f), "round-trip: grad[2]");
    TEST_ASSERT(FLOAT_EQ(w.local_loss, 0.55f), "round-trip: loss");

    close(sv[0]);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    printf("=== worker (Partner 1) tests ===\n\n");

    test_send_gradient_basic();
    test_send_gradient_closed_fd();
    test_send_gradient_single_feature();
    test_send_gradient_max_features();

    test_handle_done_basic();
    test_handle_done_disconnect();
    test_handle_done_single_feature();

    test_gradient_round_trip();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
