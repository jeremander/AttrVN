#include "balloon.h"

bool test_correctness(long testnum) {
    long i;
    float_matrix *points, *seeds, *D;
    double *labels;
    long *true_ranking;
    bool did_test = true;

    switch(testnum) {
        case 3 :  // some equidistant points
            seeds = get_float_matrix(4, 2);
            seeds->arr[0][0] = -1;
            seeds->arr[0][1] = 0;
            seeds->arr[1][0] = 1;
            seeds->arr[1][1] = 0;
            seeds->arr[2][0] = 0;
            seeds->arr[2][1] = 1;
            seeds->arr[3][0] = 0;
            seeds->arr[3][1] = -1;
            points = get_float_matrix(5, 2);
            points->arr[0][0] = 0;
            points->arr[0][1] = 0;
            points->arr[1][0] = -2;
            points->arr[1][1] = 0;
            points->arr[2][0] = 2;
            points->arr[2][1] = 0;
            points->arr[3][0] = 0;
            points->arr[3][1] = 2;
            points->arr[4][0] = 0;
            points->arr[4][1] = -2;
            labels = calloc(4, sizeof(double));
            labels[0] = 1;
            labels[1] = 1;
            labels[2] = -1;
            labels[3] = -1;
            true_ranking = calloc(5, sizeof(long));
            true_ranking[0] = 1;
            true_ranking[1] = 2;
            true_ranking[2] = 0;
            true_ranking[3] = 3;
            true_ranking[4] = 4;
            break;
        case 5:
            seeds = get_float_matrix(5, 2);
            seeds->arr[0][0] = 1;
            seeds->arr[0][1] = -4;
            seeds->arr[1][0] = -1;
            seeds->arr[1][1] = -1;
            seeds->arr[2][0] = 3;
            seeds->arr[2][1] = -1;
            seeds->arr[3][0] = 1;
            seeds->arr[3][1] = -2;
            seeds->arr[4][0] = 0;
            seeds->arr[4][1] = 3;
            points = get_float_matrix(10, 2);
            points->arr[0][0] = -3;
            points->arr[0][1] = 2;
            points->arr[1][0] = 3;
            points->arr[1][1] = 3;
            points->arr[2][0] = 4;
            points->arr[2][1] = -3;
            points->arr[3][0] = 1;
            points->arr[3][1] = 4;
            points->arr[4][0] = 0;
            points->arr[4][1] = -1;
            points->arr[5][0] = -4;
            points->arr[5][1] = -1;
            points->arr[6][0] = 1;
            points->arr[6][1] = -4;
            points->arr[7][0] = -2;
            points->arr[7][1] = -1;
            points->arr[8][0] = 4;
            points->arr[8][1] = -3;
            points->arr[9][0] = -1;
            points->arr[9][1] = -1;
            labels = calloc(10, sizeof(double));
            labels[0] = -1;
            labels[1] = -1;
            labels[2] = -1;
            labels[3] = -1;
            labels[4] = -1;
            true_ranking = calloc(10, sizeof(long));
            true_ranking[0] = 0;
            true_ranking[1] = 5;
            true_ranking[2] = 1;
            true_ranking[3] = 2;
            true_ranking[4] = 8;
            true_ranking[5] = 3;
            true_ranking[6] = 7;
            true_ranking[7] = 4;
            true_ranking[8] = 9;
            true_ranking[9] = 6;
            break;
        case 6:
            seeds = get_float_matrix(3, 2);
            seeds->arr[0][0] = -3;
            seeds->arr[0][1] = 2;
            seeds->arr[1][0] = 3;
            seeds->arr[1][1] = 2;
            seeds->arr[2][0] = 3;
            seeds->arr[2][1] = -2;
            points = get_float_matrix(2, 2);
            points->arr[0][0] = -1;
            points->arr[0][1] = -3;
            points->arr[1][0] = 0;
            points->arr[1][1] = 1;
            labels = calloc(3, sizeof(double));
            labels[0] = -1;
            labels[1] = 1;
            labels[2] = 1;
            true_ranking = calloc(2, sizeof(long));
            true_ranking[0] = 0;
            true_ranking[1] = 1;
            break;
        default:
            did_test = false;
    }

    if (!did_test) {
        return true;
    }

    D = get_float_matrix(points->nrows, seeds->nrows);
    compute_squared_distances(points, seeds, D, omp_get_max_threads());
    printf("points\n");
    disp_float_matrix(points);
    printf("seeds\n");
    disp_float_matrix(seeds);
    printf("dists\n");
    disp_float_matrix(D);

    long *ranking = calloc(points->nrows, sizeof(long));
    balloon_rank_from_distances(D, labels, ranking, false, omp_get_max_threads());
    bool result = true;
    printf("computed ranking:\n");
    for(i = 0; i < points->nrows; ++i) {
        printf("%ld ", ranking[i]);
        if (ranking[i] != true_ranking[i]) {
            result = false;
        }
    }
    printf("\n");
    printf("true ranking:\n");
    for(i = 0; i < points->nrows; ++i) {
        printf("%ld ", true_ranking[i]);
    }
    printf("\n");

    free_float_matrix(points);
    free_float_matrix(seeds);
    free_float_matrix(D);
    free(labels);
    free(ranking);

    return result;
}


int main(int argc, const char *argv[]) {

    long t, testnum;
    bool result;

    for(t = 1; t < argc; ++t) {
        testnum = atol(argv[t]);
        printf("TEST #%ld\n\n", testnum);
        result = test_correctness(testnum);
        if (result) {
            printf("\nSUCCESS!\n\n");
        }
        else {
            printf("\nFAILURE.\n\n");
        }
    }

    return 0;
}