
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


/*
 * Simple data generator for testing. Produces `num_shards` CSV files
 * with `total_samples` samples distributed roughly equally across shards.
 * Each line is "f1,f2,...,fN,label" where label is 0 or 1 generated from a
 * random linear function plus noise.
 */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <num_features> <total_samples> <num_shards>\n", argv[0]);
        return 1;
    }

    int num_features = atoi(argv[1]);
    int total_samples = atoi(argv[2]);
    int num_shards = atoi(argv[3]);

    if (num_features <= 0 || total_samples <= 0 || num_shards <= 0) {
        fprintf(stderr, "invalid arguments\n");
        return 1;
    }

    /* Seed RNG */
    srand((unsigned int)time(NULL));

    /* random weight vector for label generation */
    float *w = malloc((size_t)num_features * sizeof(float));
    if (!w) return 1;
    for (int j = 0; j < num_features; ++j) {
        /* small weights */
        w[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    /* compute how many per shard */
    int base = total_samples / num_shards;
    int rem = total_samples % num_shards;

    for (int s = 0; s < num_shards; ++s) {
        char fname[256];
        snprintf(fname, sizeof(fname), "shard_%d.csv", s);
        FILE *f = fopen(fname, "w");
        if (!f) {
            perror("fopen");
            free(w);
            return 1;
        }

        int count = base + (s < rem ? 1 : 0);
        for (int i = 0; i < count; ++i) {
            /* generate a sample */
            float dot = 0.0f;
            for (int j = 0; j < num_features; ++j) {
                /* feature in [-1,1] */
                float x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                dot += w[j] * x;
                /* write feature with comma (except last will be label) */
                fprintf(f, "%f", x);
                if (j < num_features - 1) fprintf(f, ",");
            }
            /* add small noise */
            float noise = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.1f;
            float score = dot + noise;
            int label = score > 0.0f ? 1 : 0;
            fprintf(f, ",%d\n", label);

        }

        fclose(f);
        fprintf(stderr, "wrote %d samples to %s\n", count, fname);
    }

    free(w);
    return 0;
}
