#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

/*
 * Read a CSV file where each line is: feature1,feature2,...,featureN,label
 * Allocates and fills *X (row-major array of size num_samples*num_features)
 * and *y (array of size num_samples). On success returns 0 and
 * sets *num_samples and *num_features. On error returns -1 and
 * ensures no memory is leaked.
 */

int load_data(const char *filename, float **X, float **y,
    int *num_samples, int *num_features) {
    if (!filename || !X || !y || !num_samples || !num_features) return -1;

    FILE *f = fopen(filename, "r");
    if (!f) return -1;

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    int samples = 0;
    int features = -1;

    /* count lines and determine number of features */
    while ((read = getline(&line, &len, f)) != -1) {
        /* skip empty or whitespace-only lines */
        char *s = line;
        while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
        if (*s == '\0') continue;

        /* count commas to infer columns (features + label) */
        int commas = 0;
        for (char *p = line; *p; ++p) if (*p == ',') commas++;
        int cols = commas + 1;
        if (features == -1) {
            if (cols < 2) { /* need at least one feature + label */
                free(line);
                fclose(f);
                return -1;
            }
            features = cols - 1; /* last column is label */
        } else {
            if (cols - 1 != features) {
                /* inconsistent columns */
                free(line);
                fclose(f);
                return -1;
            }
        }
        samples++;
    }

    if (samples == 0) {
        free(line);
        fclose(f);
        return -1;
    }

    /* allocate arrays */
    float *Xbuf = (float *)malloc((size_t)samples * (size_t)features * sizeof(float));
    float *ybuf = (float *)malloc((size_t)samples * sizeof(float));
    if (!Xbuf || !ybuf) {
        free(line);
        fclose(f);
        free(Xbuf);
        free(ybuf);
        return -1;
    }

    /* parse values */
    rewind(f);
    int row = 0;
    while ((read = getline(&line, &len, f)) != -1) {
        char *s = line;
        while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
        if (*s == '\0') continue;

        /* parse feature values */
        char *tok;
        int col = 0;
        char *saveptr = NULL;
        tok = strtok_r(line, ",", &saveptr);
        while (tok != NULL) {
            if (col < features) {
                Xbuf[row * features + col] = strtof(tok, NULL);
            } else {
                /* label */
                ybuf[row] = strtof(tok, NULL);
            }
            col++;
            tok = strtok_r(NULL, ",", &saveptr);
        }
        row++;
    }

    free(line);
    fclose(f);

    *X = Xbuf;
    *y = ybuf;
    *num_samples = samples;
    *num_features = features;
    return 0;
}
