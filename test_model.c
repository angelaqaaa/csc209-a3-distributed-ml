#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "protocol.h"

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

/* ========== sigmoid tests ========== */

static void test_sigmoid_zero(void) {
    printf("test_sigmoid_zero\n");
    float r = sigmoid(0.0f);
    TEST_ASSERT(FLOAT_EQ(r, 0.5f), "sigmoid(0) = 0.5");
}

static void test_sigmoid_large_positive(void) {
    printf("test_sigmoid_large_positive\n");
    float r = sigmoid(100.0f);
    TEST_ASSERT(r > 0.999f, "sigmoid(100) close to 1");
}

static void test_sigmoid_large_negative(void) {
    printf("test_sigmoid_large_negative\n");
    float r = sigmoid(-100.0f);
    TEST_ASSERT(r < 0.001f, "sigmoid(-100) close to 0");
}

static void test_sigmoid_one(void) {
    printf("test_sigmoid_one\n");
    float r = sigmoid(1.0f);
    float expected = 1.0f / (1.0f + expf(-1.0f));
    TEST_ASSERT(FLOAT_EQ(r, expected), "sigmoid(1) = 1/(1+e^-1)");
}

static void test_sigmoid_negative_one(void) {
    printf("test_sigmoid_negative_one\n");
    float r = sigmoid(-1.0f);
    float expected = 1.0f / (1.0f + expf(1.0f));
    TEST_ASSERT(FLOAT_EQ(r, expected), "sigmoid(-1) = 1/(1+e^1)");
}

static void test_sigmoid_symmetry(void) {
    printf("test_sigmoid_symmetry\n");
    float a = sigmoid(2.0f);
    float b = sigmoid(-2.0f);
    TEST_ASSERT(FLOAT_EQ(a + b, 1.0f), "sigmoid(x) + sigmoid(-x) = 1");
}

/* ========== init_weights tests ========== */

static void test_init_weights_basic(void) {
    float w[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int i;

    printf("test_init_weights_basic\n");
    init_weights(w, 4);

    int all_zero = 1;
    for (i = 0; i < 4; i++) {
        if (w[i] != 0.0f) {
            all_zero = 0;
        }
    }
    TEST_ASSERT(all_zero, "init_weights sets all weights to 0.0");
}

static void test_init_weights_single(void) {
    float w[1] = {5.0f};

    printf("test_init_weights_single\n");
    init_weights(w, 1);
    TEST_ASSERT(w[0] == 0.0f, "init_weights with 1 feature sets to 0");
}

static void test_init_weights_max(void) {
    float w[MAX_FEATURES];
    int i;

    printf("test_init_weights_max\n");
    for (i = 0; i < MAX_FEATURES; i++) {
        w[i] = 99.0f;
    }
    init_weights(w, MAX_FEATURES);

    int all_zero = 1;
    for (i = 0; i < MAX_FEATURES; i++) {
        if (w[i] != 0.0f) {
            all_zero = 0;
        }
    }
    TEST_ASSERT(all_zero, "init_weights with MAX_FEATURES all zero");
}

/* ========== compute_gradient tests ========== */

static void test_compute_gradient_single_sample(void) {
    /* 1 sample, 2 features: X=[1,0], y=[1], weights=[0,0] */
    float X[2] = {1.0f, 0.0f};
    float y[1] = {1.0f};
    float weights[2] = {0.0f, 0.0f};
    float grad[2];

    printf("test_compute_gradient_single_sample\n");

    float loss = compute_gradient(X, y, 1, 2, weights, grad);

    /* With weights=0, sigmoid(0)=0.5, error = 0.5 - 1.0 = -0.5 */
    /* grad[0] = error * X[0] / 1 = -0.5 * 1 = -0.5 */
    /* grad[1] = error * X[1] / 1 = -0.5 * 0 = 0.0 */
    TEST_ASSERT(FLOAT_EQ(grad[0], -0.5f), "gradient[0] = -0.5");
    TEST_ASSERT(FLOAT_EQ(grad[1], 0.0f), "gradient[1] = 0.0");
    TEST_ASSERT(loss > 0.0f, "loss is positive");
    (void)loss;
}

static void test_compute_gradient_zero_weights(void) {
    /* 2 samples, 1 feature */
    float X[2] = {1.0f, 1.0f};
    float y[2] = {0.0f, 1.0f};
    float weights[1] = {0.0f};
    float grad[1];

    printf("test_compute_gradient_zero_weights\n");
    float loss = compute_gradient(X, y, 2, 1, weights, grad);

    /* sigmoid(0)=0.5 for both samples */
    /* error1 = 0.5-0.0=0.5, error2 = 0.5-1.0=-0.5 */
    /* grad = ((0.5*1) + (-0.5*1)) / 2 = 0.0 */
    TEST_ASSERT(FLOAT_EQ(grad[0], 0.0f), "balanced labels give zero gradient");
    TEST_ASSERT(loss > 0.0f, "loss is positive even with zero gradient");
    (void)loss;
}

static void test_compute_gradient_multi_feature(void) {
    /* 2 samples, 3 features */
    float X[6] = {1.0f, 2.0f, 3.0f,
                  4.0f, 5.0f, 6.0f};
    float y[2] = {0.0f, 1.0f};
    float weights[3] = {0.1f, -0.1f, 0.0f};
    float grad[3];

    printf("test_compute_gradient_multi_feature\n");
    float loss = compute_gradient(X, y, 2, 3, weights, grad);
    TEST_ASSERT(loss > 0.0f, "loss is positive");

    /* Just verify gradient is computed (non-trivial values) */
    int has_nonzero = (grad[0] != 0.0f || grad[1] != 0.0f || grad[2] != 0.0f);
    TEST_ASSERT(has_nonzero, "gradient has non-zero components");
    (void)loss;
}

static void test_compute_gradient_returns_avg_loss(void) {
    float X[2] = {1.0f, 1.0f};
    float y[2] = {1.0f, 0.0f};
    float weights[1] = {0.0f};
    float grad[1];

    printf("test_compute_gradient_returns_avg_loss\n");
    float loss = compute_gradient(X, y, 2, 1, weights, grad);

    /* Both samples: pred=0.5, avg loss = log(2) */
    float expected = logf(2.0f);
    TEST_ASSERT(fabsf(loss - expected) < 0.01f,
                "average loss is approximately log(2)");
}

/* ========== update_weights tests ========== */

static void test_update_weights_basic(void) {
    float w[3] = {1.0f, 2.0f, 3.0f};
    float grad[3] = {0.1f, 0.2f, 0.3f};

    printf("test_update_weights_basic\n");
    update_weights(w, grad, 3, 1.0f);

    /* w[i] -= 1.0 * grad[i] */
    TEST_ASSERT(FLOAT_EQ(w[0], 0.9f), "w[0] = 1.0 - 0.1 = 0.9");
    TEST_ASSERT(FLOAT_EQ(w[1], 1.8f), "w[1] = 2.0 - 0.2 = 1.8");
    TEST_ASSERT(FLOAT_EQ(w[2], 2.7f), "w[2] = 3.0 - 0.3 = 2.7");
}

static void test_update_weights_zero_lr(void) {
    float w[2] = {1.0f, 2.0f};
    float grad[2] = {5.0f, 5.0f};

    printf("test_update_weights_zero_lr\n");
    update_weights(w, grad, 2, 0.0f);

    TEST_ASSERT(FLOAT_EQ(w[0], 1.0f), "weights unchanged with lr=0");
    TEST_ASSERT(FLOAT_EQ(w[1], 2.0f), "weights unchanged with lr=0");
}

static void test_update_weights_zero_gradient(void) {
    float w[2] = {1.0f, 2.0f};
    float grad[2] = {0.0f, 0.0f};

    printf("test_update_weights_zero_gradient\n");
    update_weights(w, grad, 2, 0.01f);

    TEST_ASSERT(FLOAT_EQ(w[0], 1.0f), "weights unchanged with zero gradient");
    TEST_ASSERT(FLOAT_EQ(w[1], 2.0f), "weights unchanged with zero gradient");
}

static void test_update_weights_single(void) {
    float w[1] = {0.0f};
    float grad[1] = {1.0f};

    printf("test_update_weights_single\n");
    update_weights(w, grad, 1, 0.5f);
    TEST_ASSERT(FLOAT_EQ(w[0], -0.5f), "single weight: 0 - 0.5*1 = -0.5");
}

int main(void) {
    printf("=== model tests ===\n\n");

    test_sigmoid_zero();
    test_sigmoid_large_positive();
    test_sigmoid_large_negative();
    test_sigmoid_one();
    test_sigmoid_negative_one();
    test_sigmoid_symmetry();

    test_init_weights_basic();
    test_init_weights_single();
    test_init_weights_max();

    test_compute_gradient_single_sample();
    test_compute_gradient_zero_weights();
    test_compute_gradient_multi_feature();
    test_compute_gradient_returns_avg_loss();

    test_update_weights_basic();
    test_update_weights_zero_lr();
    test_update_weights_zero_gradient();
    test_update_weights_single();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
