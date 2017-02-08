#include "balloon.h"

float_matrix *get_random_matrix(long nrows, long ncols) {
    float_matrix *mat = get_float_matrix(nrows, ncols);
    long i, j;
    for(i = 0; i < nrows; ++i) {
        for(j = 0; j < ncols; ++j) {
            mat->arr[i][j] = drand48();
        }
    }
    return mat;
}

double *get_random_labels(long n) {
    double *labels = calloc(n, sizeof(double));
    long i;
    for(i = 0; i < n; ++i) {
        labels[i] = (drand48() > 0.5) ? 1 : -1;
    }
    return labels;
}

void test_time(long nseeds, long npoints, long dim, long seed) {
    srand48(seed);
    struct timespec ts;
    double times[4];
    printf("\nUSING %d THREAD(S)\n\n", omp_get_max_threads());
    clockid_t clockid = CLOCK_MONOTONIC;
    clock_gettime(clockid, &ts);
    times[0] = ts.tv_sec + ts.tv_nsec / 1e9;
    printf("Creating %ld seeds, %ld points (dimension %ld)...\n", nseeds, npoints, dim);
    float_matrix *seeds = get_random_matrix(nseeds, dim);
    double *labels = get_random_labels(nseeds);
    float_matrix *points = get_random_matrix(npoints, dim);
    clock_gettime(clockid, &ts);
    times[1] = ts.tv_sec + ts.tv_nsec / 1e9;
    printf("%g seconds.\n\n", times[1] - times[0]);
    printf("Computing seed/point distances...\n");
    float_matrix *D = get_float_matrix(npoints, nseeds);
    compute_squared_distances(points, seeds, D);
    clock_gettime(clockid, &ts);
    times[2] = ts.tv_sec + ts.tv_nsec / 1e9;
    printf("%g seconds.\n\n", times[2] - times[1]);
    printf("Computing inflation ranking...\n");
    long *ranks = calloc(npoints, sizeof(long));
    balloon_rank_from_distances(D, labels, ranks, false);
    clock_gettime(clockid, &ts);
    times[3] = ts.tv_sec + ts.tv_nsec / 1e9;
    printf("%g seconds.\n\n", times[3] - times[2]);
    printf("%g seconds total.\n\n", times[3] - times[0]);
    free_float_matrix(seeds);
    free(labels);
    free_float_matrix(points);
    free_float_matrix(D);
}


int main(int argc, const char *argv[]) {

    long nseeds = 100, npoints = 1000, dim = 50, seed = 0;

    if (argc > 1) {
        nseeds = atol(argv[1]);
    }
    if (argc > 2) {
        npoints = atol(argv[2]);
    }
    if (argc > 3) {
        dim = atol(argv[3]);
    }
    if (argc > 4) {
        seed = atol(argv[4]);
    }

    test_time(nseeds, npoints, dim, seed);

    return 0;
}