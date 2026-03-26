#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "model.h"

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/*
 * Compute gradient of logistic regression loss on the given data.
 * X is num_samples x num_features in row-major order.
 * y contains labels (0 or 1).
 * Writes gradient into grad_out.
 * Returns the average loss.
 */
float compute_gradient(const float *X, const float *y,
                       int num_samples, int num_features,
                       const float *weights, float *grad_out) {
    int i, j;
    float total_loss = 0.0f;

    memset(grad_out, 0, num_features * sizeof(float));

    for (i = 0; i < num_samples; i++) {
        float dot = 0.0f;
        for (j = 0; j < num_features; j++) {
            dot += weights[j] * X[i * num_features + j];
        }

        float pred = sigmoid(dot);
        float error = pred - y[i];
        total_loss += -y[i] * logf(pred + 1e-7f)
                      - (1 - y[i]) * logf(1 - pred + 1e-7f);

        for (j = 0; j < num_features; j++) {
            grad_out[j] += error * X[i * num_features + j];
        }
    }

    for (j = 0; j < num_features; j++) {
        grad_out[j] /= num_samples;
    }

    return total_loss / num_samples;
}

void update_weights(float *weights, const float *avg_gradient,
                    int num_features, float learning_rate) {
    int i;
    for (i = 0; i < num_features; i++) {
        weights[i] -= learning_rate * avg_gradient[i];
    }
}

void init_weights(float *weights, int num_features) {
    int i;
    for (i = 0; i < num_features; i++) {
        weights[i] = 0.0f;
    }
}
