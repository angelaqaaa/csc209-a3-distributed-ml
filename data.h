/* Partner 2 implements */

#ifndef DATA_H
#define DATA_H

/*
 * Load CSV shard into malloc'd arrays.
 * Each line: feature1,feature2,...,featureN,label
 * Caller is responsible for freeing X and y.
 */
int load_data(const char *filename, float **X, float **y,
              int *num_samples, int *num_features);

#endif
