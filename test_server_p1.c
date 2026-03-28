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
#include "model.h"

/* Declarations for Partner 1 server functions */
int count_active_workers(struct worker_info *workers);
int handle_gradient(struct worker_info *w, int current_round);
int all_gradients_received(struct worker_info *workers);
void aggregate_and_update(float *weights, struct worker_info *workers,
                          int num_features, float learning_rate,
                          float *global_loss);
int check_termination(float loss, int round, int max_rounds);
int broadcast_done(struct worker_info *workers, float *weights,
                   int num_features, float final_loss);

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

/* Helper: initialize workers array to empty */
static void init_workers(struct worker_info *workers) {
    int i;
    for (i = 0; i < MAX_WORKERS; i++) {
        workers[i].fd = -1;
        workers[i].state = 0;
        workers[i].num_samples = 0;
        workers[i].recv_len = 0;
        workers[i].local_loss = 0.0f;
        workers[i].gradient_received = 0;
        memset(workers[i].gradients, 0, sizeof(workers[i].gradients));
        memset(workers[i].recv_buf, 0, sizeof(workers[i].recv_buf));
    }
}

/* Helper: build a MSG_GRADIENT message in w->recv_buf */
static void build_gradient_msg(struct worker_info *w, int round,
                               int nf, float loss, float *grads) {
    int offset = 0;
    uint32_t net_val;

    /* Header */
    w->recv_buf[0] = MSG_GRADIENT;
    offset = 1;
    uint32_t payload_len = htonl(12 + nf * sizeof(float));
    memcpy(w->recv_buf + offset, &payload_len, 4);
    offset += 4;

    /* round */
    net_val = htonl(round);
    memcpy(w->recv_buf + offset, &net_val, 4);
    offset += 4;

    /* num_features */
    net_val = htonl(nf);
    memcpy(w->recv_buf + offset, &net_val, 4);
    offset += 4;

    /* local_loss */
    memcpy(w->recv_buf + offset, &loss, sizeof(float));
    offset += sizeof(float);

    /* gradients */
    memcpy(w->recv_buf + offset, grads, nf * sizeof(float));
}

/* ========== count_active_workers tests ========== */

static void test_count_active_none(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_count_active_none\n");
    init_workers(workers);
    TEST_ASSERT(count_active_workers(workers) == 0, "no active workers returns 0");
}

static void test_count_active_all(void) {
    struct worker_info workers[MAX_WORKERS];
    int i;

    printf("test_count_active_all\n");
    init_workers(workers);
    for (i = 0; i < MAX_WORKERS; i++) {
        workers[i].fd = i + 10;
    }
    TEST_ASSERT(count_active_workers(workers) == MAX_WORKERS,
                "all slots active returns MAX_WORKERS");
}

static void test_count_active_mixed(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_count_active_mixed\n");
    init_workers(workers);
    workers[0].fd = 5;
    workers[3].fd = 8;
    workers[7].fd = 12;
    TEST_ASSERT(count_active_workers(workers) == 3,
                "3 active workers in mixed slots");
}

/* ========== handle_gradient tests ========== */

static void test_handle_gradient_valid(void) {
    struct worker_info w;
    float grads[2] = {0.5f, -0.3f};

    printf("test_handle_gradient_valid\n");
    memset(&w, 0, sizeof(w));
    w.fd = 10;
    build_gradient_msg(&w, 1, 2, 0.42f, grads);

    int ret = handle_gradient(&w, 1);
    TEST_ASSERT(ret == 0, "handle_gradient returns 0 for valid gradient");
    TEST_ASSERT(w.gradient_received == 1, "gradient_received set to 1");
    TEST_ASSERT(FLOAT_EQ(w.local_loss, 0.42f), "local_loss stored correctly");
    TEST_ASSERT(FLOAT_EQ(w.gradients[0], 0.5f), "gradient[0] stored correctly");
    TEST_ASSERT(FLOAT_EQ(w.gradients[1], -0.3f), "gradient[1] stored correctly");
}

static void test_handle_gradient_stale_round(void) {
    struct worker_info w;
    float grads[2] = {0.1f, 0.2f};

    printf("test_handle_gradient_stale_round\n");
    memset(&w, 0, sizeof(w));
    w.fd = 10;
    build_gradient_msg(&w, 3, 2, 0.1f, grads);

    int ret = handle_gradient(&w, 5);
    TEST_ASSERT(ret == -1, "handle_gradient returns -1 for stale round");
    TEST_ASSERT(w.gradient_received == 0, "gradient_received stays 0");
}

static void test_handle_gradient_invalid_nf_zero(void) {
    struct worker_info w;
    float grads[1] = {0.0f};

    printf("test_handle_gradient_invalid_nf_zero\n");
    memset(&w, 0, sizeof(w));
    w.fd = 10;
    build_gradient_msg(&w, 1, 0, 0.0f, grads);

    int ret = handle_gradient(&w, 1);
    TEST_ASSERT(ret == -1, "handle_gradient rejects nf=0");
}

static void test_handle_gradient_invalid_nf_too_large(void) {
    struct worker_info w;
    float grads[1] = {0.0f};

    printf("test_handle_gradient_invalid_nf_too_large\n");
    memset(&w, 0, sizeof(w));
    w.fd = 10;
    build_gradient_msg(&w, 1, MAX_FEATURES + 1, 0.0f, grads);

    int ret = handle_gradient(&w, 1);
    TEST_ASSERT(ret == -1, "handle_gradient rejects nf > MAX_FEATURES");
}

static void test_handle_gradient_max_features(void) {
    struct worker_info w;
    float grads[MAX_FEATURES];
    int i;

    printf("test_handle_gradient_max_features\n");
    memset(&w, 0, sizeof(w));
    w.fd = 10;
    for (i = 0; i < MAX_FEATURES; i++) {
        grads[i] = (float)i * 0.01f;
    }
    build_gradient_msg(&w, 0, MAX_FEATURES, 1.0f, grads);

    int ret = handle_gradient(&w, 0);
    TEST_ASSERT(ret == 0, "handle_gradient accepts nf=MAX_FEATURES");
    TEST_ASSERT(FLOAT_EQ(w.gradients[0], 0.0f), "first gradient correct");
    TEST_ASSERT(FLOAT_EQ(w.gradients[MAX_FEATURES - 1],
                (MAX_FEATURES - 1) * 0.01f), "last gradient correct");
}

/* ========== all_gradients_received tests ========== */

static void test_all_grads_none_active(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_all_grads_none_active\n");
    init_workers(workers);
    TEST_ASSERT(all_gradients_received(workers) == 0,
                "no active workers returns 0 (not vacuously true)");
}

static void test_all_grads_all_received(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_all_grads_all_received\n");
    init_workers(workers);
    workers[0].fd = 5;
    workers[0].state = 2;
    workers[0].gradient_received = 1;
    workers[1].fd = 6;
    workers[1].state = 2;
    workers[1].gradient_received = 1;

    TEST_ASSERT(all_gradients_received(workers) == 1,
                "all workers received returns 1");
}

static void test_all_grads_partial(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_all_grads_partial\n");
    init_workers(workers);
    workers[0].fd = 5;
    workers[0].state = 2;
    workers[0].gradient_received = 1;
    workers[1].fd = 6;
    workers[1].state = 2;
    workers[1].gradient_received = 0;

    TEST_ASSERT(all_gradients_received(workers) == 0,
                "partial receipt returns 0");
}

static void test_all_grads_ignores_non_state2(void) {
    struct worker_info workers[MAX_WORKERS];

    printf("test_all_grads_ignores_non_state2\n");
    init_workers(workers);
    /* Worker in state 1 (registered but not sent weights) */
    workers[0].fd = 5;
    workers[0].state = 1;
    workers[0].gradient_received = 0;
    /* Worker in state 2 with gradient received */
    workers[1].fd = 6;
    workers[1].state = 2;
    workers[1].gradient_received = 1;

    TEST_ASSERT(all_gradients_received(workers) == 1,
                "ignores workers not in state 2");
}

/* ========== aggregate_and_update tests ========== */

static void test_aggregate_single_worker(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[2] = {1.0f, 2.0f};
    float global_loss = 0.0f;

    printf("test_aggregate_single_worker\n");
    init_workers(workers);
    workers[0].fd = 5;
    workers[0].state = 2;
    workers[0].gradient_received = 1;
    workers[0].num_samples = 10;
    workers[0].local_loss = 0.5f;
    workers[0].gradients[0] = 0.2f;
    workers[0].gradients[1] = -0.1f;

    aggregate_and_update(weights, workers, 2, 1.0f, &global_loss);

    /* avg_gradient = [0.2, -0.1] (only one worker) */
    /* weights: [1.0 - 0.2, 2.0 - (-0.1)] = [0.8, 2.1] */
    TEST_ASSERT(FLOAT_EQ(weights[0], 0.8f), "weight[0] updated to 0.8");
    TEST_ASSERT(FLOAT_EQ(weights[1], 2.1f), "weight[1] updated to 2.1");
    TEST_ASSERT(FLOAT_EQ(global_loss, 0.5f), "global_loss = single worker loss");
    TEST_ASSERT(workers[0].gradient_received == 0,
                "gradient_received reset to 0");
}

static void test_aggregate_weighted_avg(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[1] = {0.0f};
    float global_loss = 0.0f;

    printf("test_aggregate_weighted_avg\n");
    init_workers(workers);

    /* Worker 0: 100 samples, gradient=1.0 */
    workers[0].fd = 5;
    workers[0].state = 2;
    workers[0].gradient_received = 1;
    workers[0].num_samples = 100;
    workers[0].local_loss = 0.8f;
    workers[0].gradients[0] = 1.0f;

    /* Worker 1: 300 samples, gradient=3.0 */
    workers[1].fd = 6;
    workers[1].state = 2;
    workers[1].gradient_received = 1;
    workers[1].num_samples = 300;
    workers[1].local_loss = 0.2f;
    workers[1].gradients[0] = 3.0f;

    aggregate_and_update(weights, workers, 1, 1.0f, &global_loss);

    /* weighted avg = (1.0*100 + 3.0*300) / 400 = 1000/400 = 2.5 */
    /* weights: 0.0 - 1.0 * 2.5 = -2.5 */
    TEST_ASSERT(FLOAT_EQ(weights[0], -2.5f), "weighted average gradient applied");

    /* weighted loss = (0.8*100 + 0.2*300) / 400 = 140/400 = 0.35 */
    TEST_ASSERT(FLOAT_EQ(global_loss, 0.35f), "weighted average loss");
}

static void test_aggregate_skips_disconnected(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[1] = {0.0f};
    float global_loss = 0.0f;

    printf("test_aggregate_skips_disconnected\n");
    init_workers(workers);

    workers[0].fd = -1;  /* disconnected */
    workers[0].state = 2;
    workers[0].gradient_received = 1;
    workers[0].num_samples = 100;
    workers[0].gradients[0] = 99.0f;

    workers[1].fd = 6;
    workers[1].state = 2;
    workers[1].gradient_received = 1;
    workers[1].num_samples = 50;
    workers[1].local_loss = 0.3f;
    workers[1].gradients[0] = 0.5f;

    aggregate_and_update(weights, workers, 1, 1.0f, &global_loss);

    /* Only worker 1 counts: avg = 0.5, weights = 0 - 0.5 = -0.5 */
    TEST_ASSERT(FLOAT_EQ(weights[0], -0.5f),
                "disconnected worker skipped in aggregation");
    TEST_ASSERT(FLOAT_EQ(global_loss, 0.3f),
                "loss from active worker only");
}

static void test_aggregate_no_active_workers(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[1] = {5.0f};
    float global_loss = 99.0f;

    printf("test_aggregate_no_active_workers\n");
    init_workers(workers);

    aggregate_and_update(weights, workers, 1, 1.0f, &global_loss);

    /* No workers contributed: avg_gradient=0, weights unchanged by gradient */
    /* But update_weights still called with zero gradient, so weights stay same */
    TEST_ASSERT(FLOAT_EQ(weights[0], 5.0f),
                "weights unchanged when no active workers");
}

/* ========== check_termination tests ========== */

static void test_check_term_below_threshold(void) {
    printf("test_check_term_below_threshold\n");
    TEST_ASSERT(check_termination(0.0005f, 1, 100) == 1,
                "terminates when loss < LOSS_THRESHOLD");
}

static void test_check_term_above_threshold(void) {
    printf("test_check_term_above_threshold\n");
    TEST_ASSERT(check_termination(0.5f, 1, 100) == 0,
                "continues when loss above threshold");
}

static void test_check_term_max_rounds(void) {
    printf("test_check_term_max_rounds\n");
    TEST_ASSERT(check_termination(0.5f, 100, 100) == 1,
                "terminates when round >= max_rounds");
}

static void test_check_term_exactly_threshold(void) {
    printf("test_check_term_exactly_threshold\n");
    /* LOSS_THRESHOLD is 0.001f; 0.001f is NOT < 0.001f */
    TEST_ASSERT(check_termination(LOSS_THRESHOLD, 1, 100) == 0,
                "does not terminate when loss == LOSS_THRESHOLD exactly");
}

static void test_check_term_round_before_max(void) {
    printf("test_check_term_round_before_max\n");
    TEST_ASSERT(check_termination(0.5f, 99, 100) == 0,
                "continues when round < max_rounds");
}

static void test_check_term_zero_loss(void) {
    printf("test_check_term_zero_loss\n");
    TEST_ASSERT(check_termination(0.0f, 1, 100) == 1,
                "terminates when loss = 0");
}

/* ========== broadcast_done tests ========== */

static void test_broadcast_done_no_workers(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[2] = {1.0f, 2.0f};

    printf("test_broadcast_done_no_workers\n");
    init_workers(workers);
    int ret = broadcast_done(workers, weights, 2, 0.01f);
    TEST_ASSERT(ret == 0, "broadcast_done returns 0 with no workers");
}

static void test_broadcast_done_with_socketpair(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[2] = {1.5f, -0.5f};
    int sv[2];

    printf("test_broadcast_done_with_socketpair\n");
    init_workers(workers);

    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    workers[0].fd = sv[1];
    workers[0].state = 2;

    int ret = broadcast_done(workers, weights, 2, 0.05f);
    TEST_ASSERT(ret == 0, "broadcast_done returns 0");

    /* Read header */
    uint8_t hdr[HEADER_SIZE];
    read(sv[0], hdr, HEADER_SIZE);
    TEST_ASSERT(hdr[0] == MSG_DONE, "message type is MSG_DONE");

    uint32_t net_plen;
    memcpy(&net_plen, hdr + 1, 4);
    uint32_t plen = ntohl(net_plen);
    TEST_ASSERT(plen == 8 + 2 * sizeof(float),
                "payload length correct");

    /* Read payload */
    char payload[8 + MAX_FEATURES * sizeof(float)];
    read(sv[0], payload, plen);

    uint32_t net_nf;
    memcpy(&net_nf, payload, 4);
    int nf = (int)ntohl(net_nf);
    TEST_ASSERT(nf == 2, "num_features in payload is 2");

    float final_loss;
    memcpy(&final_loss, payload + 4, sizeof(float));
    TEST_ASSERT(FLOAT_EQ(final_loss, 0.05f), "final_loss in payload correct");

    float w0, w1;
    memcpy(&w0, payload + 8, sizeof(float));
    memcpy(&w1, payload + 8 + sizeof(float), sizeof(float));
    TEST_ASSERT(FLOAT_EQ(w0, 1.5f), "weight[0] in payload correct");
    TEST_ASSERT(FLOAT_EQ(w1, -0.5f), "weight[1] in payload correct");

    close(sv[0]);
    close(sv[1]);
}

static void test_broadcast_done_skips_non_state2(void) {
    struct worker_info workers[MAX_WORKERS];
    float weights[1] = {1.0f};
    int sv[2];

    printf("test_broadcast_done_skips_non_state2\n");
    init_workers(workers);

    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    workers[0].fd = sv[1];
    workers[0].state = 1;  /* not state 2, should be skipped */

    broadcast_done(workers, weights, 1, 0.0f);

    /* Try non-blocking read — should get nothing */
    close(sv[1]);
    char buf[1];
    int n = read(sv[0], buf, 1);
    TEST_ASSERT(n <= 0, "non-state-2 worker received no data");

    close(sv[0]);
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    printf("=== server (Partner 1) tests ===\n\n");

    test_count_active_none();
    test_count_active_all();
    test_count_active_mixed();

    test_handle_gradient_valid();
    test_handle_gradient_stale_round();
    test_handle_gradient_invalid_nf_zero();
    test_handle_gradient_invalid_nf_too_large();
    test_handle_gradient_max_features();

    test_all_grads_none_active();
    test_all_grads_all_received();
    test_all_grads_partial();
    test_all_grads_ignores_non_state2();

    test_aggregate_single_worker();
    test_aggregate_weighted_avg();
    test_aggregate_skips_disconnected();
    test_aggregate_no_active_workers();

    test_check_term_below_threshold();
    test_check_term_above_threshold();
    test_check_term_max_rounds();
    test_check_term_exactly_threshold();
    test_check_term_round_before_max();
    test_check_term_zero_loss();

    test_broadcast_done_no_workers();
    test_broadcast_done_with_socketpair();
    test_broadcast_done_skips_non_state2();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
