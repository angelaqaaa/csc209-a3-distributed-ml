#include <stdio.h>
#include <stdlib.h>
#include "data.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <csv_file>\n", argv[0]);
        return 1;
    }
    float *X = NULL;
    float *y = NULL;
    int n = 0, d = 0;
    if (load_data(argv[1], &X, &y, &n, &d) != 0) {
        fprintf(stderr, "load_data failed\n");
        return 1;
    }
    printf("loaded samples=%d features=%d\n", n, d);
    if (n > 0) {
        printf("first sample: ");
        for (int j = 0; j < d; ++j) printf("%f ", X[j]);
        printf(" label=%f\n", y[0]);
    }
    free(X);
    free(y);
    return 0;
}
