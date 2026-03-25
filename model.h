#ifndef MODEL_H
#define MODEL_H

/* Partner 1 implements */

float sigmoid(float x);

float compute_gradient(const float *X, const float *y,
                       int num_samples, int num_features,
                       const float *weights, float *grad_out);

void update_weights(float *weights, const float *avg_gradient,
                    int num_features, float learning_rate);

void init_weights(float *weights, int num_features);

#endif
