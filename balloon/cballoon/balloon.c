#include "balloon.h"
#include "mergesort.h"

void disp_int_arr(long *arr, long n) {
    long i;
    for(i = 0; i < n; ++i) {
        printf("%ld ", arr[i]);
    }
    printf("\n");
}

void disp_float_arr(double *arr, long n) {
    long i;
    for(i = 0; i < n; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

float_matrix *get_float_matrix(long nrows, long ncols) {
    float_matrix *mat = (float_matrix *) calloc(1, sizeof(float_matrix));
    mat->nrows = nrows;
    mat->ncols = ncols;
    mat->arr = (double **) calloc(nrows, sizeof(double *));
    long i;
    for(i = 0; i < nrows; ++i) {
        mat->arr[i] = (double *) calloc(ncols, sizeof(double));
    }
    return mat;
}

int_matrix *get_int_matrix(long nrows, long ncols) {
    int_matrix *mat = (int_matrix *) calloc(1, sizeof(int_matrix));
    mat->nrows = nrows;
    mat->ncols = ncols;
    mat->arr = (long **) calloc(nrows, sizeof(long *));
    long i;
    for(i = 0; i < nrows; ++i) {
        mat->arr[i] = (long *) calloc(ncols, sizeof(long));
    }
    return mat;
}

void free_float_matrix(float_matrix *mat) {
    long i;
    for(i = 0; i < mat->nrows; ++i) {
        free(mat->arr[i]);
    }
    free(mat->arr);
    free(mat);
}

void free_int_matrix(int_matrix *mat) {
    long i;
    for(i = 0; i < mat->nrows; ++i) {
        free(mat->arr[i]);
    }
    free(mat->arr);
    free(mat);
}

void disp_float_matrix(float_matrix *mat) {
    long i, j;
    for(i = 0; i < mat->nrows; ++i) {
        for(j = 0; j < mat->ncols; ++j) {
            printf("%6lf ", mat->arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void disp_int_matrix(int_matrix *mat) {
    long i, j;
    for(i = 0; i < mat->nrows; ++i) {
        for(j = 0; j < mat->ncols; ++j) {
            printf("%6ld ", mat->arr[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void compute_squared_distances(float_matrix *points, float_matrix *seeds, float_matrix *D, int num_threads) {
    long s = seeds->nrows;
    long dim = seeds->ncols;
    assert(points->ncols == dim);
    long n = points->nrows;

    double *point_mags = (double *) calloc(n, sizeof(double));
    double *seed_mags = (double *) calloc(s, sizeof(double));

    long i, j, k;
    double val;

    num_threads = (num_threads <= 0) ? omp_get_max_threads() : num_threads;
#if (VERBOSITY >= 1)
    printf("num_threads = %d\n", num_threads);
#endif
    omp_set_num_threads(num_threads);

# pragma omp parallel \
    shared (s, dim, n, D, point_mags, seed_mags) \
    private (i, j, k, val)
{
    # pragma omp for
    for(i = 0; i < n; ++i) {
        val = 0.0;
        for(k = 0; k < dim; ++k) {
            val += points->arr[i][k] * points->arr[i][k];
        }
        point_mags[i] = val;
    }

    # pragma omp for
    for(j = 0; j < s; ++j) {
        val = 0.0;
        for(k = 0; k < dim; ++k) {
            val += seeds->arr[j][k] * seeds->arr[j][k];
        }
        seed_mags[j] = val;
    }

    # pragma omp for
    for(i = 0; i < n; ++i) {
        for(j = 0; j < s; ++j) {
            val = 0.0;
            for(k = 0; k < dim; ++k) {
                val += points->arr[i][k] * seeds->arr[j][k];
            }
            D->arr[i][j] = point_mags[i] + seed_mags[j] - 2 * val;
        }
    }
}

    free(point_mags);
    free(seed_mags);
}


int compare_val_index_infl(const void *pa, const void *pb) {
    const double *a = pa;
    const double *b = pb;
    if (a[0] < b[0]) {
        return -1;
    }
    else if (a[0] > b[0]) {
        return 1;
    }
    else {
        return (a[1] < b[1]) ? -1 : 1;
    }
}

int compare_val_index_defl(const void *pa, const void *pb) {
    const double *a = pa;
    const double *b = pb;
    if (a[0] < b[0]) {
        return 1;
    }
    else if (a[0] > b[0]) {
        return -1;
    }
    else {
        return (a[1] < b[1]) ? -1 : 1;
    }
}

int compare_point(void *arg, const void *a, const void *b) {
    long i1 = *((const long *) a);
    long i2 = *((const long *) b);
    compare_point_info *info = (compare_point_info *) arg;
    long j1 = -1, j2 = -1;
    long next_j1, next_j2;
    double cumsum1 = 0.0, cumsum2 = 0.0;
    double dij1, dij2;
#if (VERBOSITY >= 2)
    printf("\ni1 = %ld, i2 = %ld\n", i1, i2);
#endif
    while (MIN(j1, j2) < info->s_minus_one) {
        next_j1 = MIN(info->s_minus_one, j1 + 1);
        next_j2 = MIN(info->s_minus_one, j2 + 1);
        dij1 = info->D_tilde->arr[i1][next_j1];
        dij2 = info->D_tilde->arr[i2][next_j2];
#if (VERBOSITY >= 2)
        printf("j1 = %ld, j2 = %ld\n", j1, j2);
        printf("next_j1 = %ld, next_j2 = %ld\n", next_j1, next_j2);
        printf("dij1 = %f, dij2 = %f\n", dij1, dij2);
#endif
        if ((info->deflate && (dij1 >= dij2)) || (!info->deflate && (dij1 <= dij2))) {
            while ((next_j1 > j1) || ((j1 < info->s_minus_one) && info->G_tilde->arr[i1][j1])) {
                ++j1;
                cumsum1 += info->labels[info->I_tilde->arr[i1][j1]];
            }
        }
        if ((info->deflate && (dij1 <= dij2)) || (!info->deflate && (dij1 >= dij2))) {
            while ((next_j2 > j2) || ((j2 < info->s_minus_one) && info->G_tilde->arr[i2][j2])) {
                ++j2;
                cumsum2 += info->labels[info->I_tilde->arr[i2][j2]];

            }
        }
#if (VERBOSITY >= 2)
        printf("cumsum1 = %f, cumsum2 = %f\n", cumsum1, cumsum2);
        printf("cumsum1 %c= cumsum2\n", (cumsum1 != cumsum2) ? '!' : '=');
#endif
        if (cumsum1 != cumsum2) {
            if (info->deflate) {
#if (VERBOSITY >= 2)
                printf("return %d\n", (cumsum1 > cumsum2) ? 1 : -1);
#endif
                return (cumsum1 > cumsum2) ? 1 : -1;
            }
            else {
#if (VERBOSITY >= 2)
                printf("return %d\n", (cumsum1 > cumsum2) ? -1 : 1);
#endif
                return (cumsum1 > cumsum2) ? -1 : 1;
            }
        }
    }
#if (VERBOSITY >= 2)
    printf("return %d\n", (i1 < i2) ? -1 : 1);
#endif
    return (i1 < i2) ? -1 : 1;  // break ties by index
}

void balloon_rank_from_distances(float_matrix *D, double *labels, long *ranks, bool deflate, int num_threads) {
    long n = D->nrows;
    long s = D->ncols;
    double val_index_pairs[s][2];
    double cur_d = 0.0, last_d = 0.0;
    long i, j;
    // first get the row-sorted distance matrix and associated indices
    int_matrix *I_tilde = get_int_matrix(n, s);
    int_matrix *G_tilde = get_int_matrix(n, s - 1);
    int (*cmp_fn)(const void *, const void *) = deflate ? compare_val_index_defl : compare_val_index_infl;

    for(i = 0; i < n; ++i) {
        for(j = 0; j < s; ++j) {
            val_index_pairs[j][0] = D->arr[i][j];
            val_index_pairs[j][1] = (double) j;
        }
        qsort(val_index_pairs, s, 2 * sizeof(double), cmp_fn);
        for(j = 0; j < s; ++j) {
            cur_d = val_index_pairs[j][0];
            if (j > 0) {
                G_tilde->arr[i][j - 1] = (j > 0) && (cur_d == last_d);
            }
            D->arr[i][j] = cur_d;
            I_tilde->arr[i][j] = (long) val_index_pairs[j][1];
            last_d = cur_d;
        }
    }

#if (VERBOSITY >= 1)
    printf("D_tilde\n");
    disp_float_matrix(D);
    printf("I_tilde\n");
    disp_int_matrix(I_tilde);
    printf("G_tilde\n");
    disp_int_matrix(G_tilde);
#endif

    // now sort the point indices by ranking
    for(i = 0; i < n; ++i) {
        ranks[i] = i;
    }
    compare_point_info info = {deflate, s - 1, D, I_tilde, G_tilde, labels};

    num_threads = (num_threads <= 0) ? omp_get_max_threads() : num_threads;
#if (VERBOSITY >= 1)
    printf("num_threads = %d\n", num_threads);
#endif
    omp_set_num_threads(num_threads);
    mt_mergesort_r(ranks, n, sizeof(long), &info, compare_point, num_threads);

    //qsort_r(ranks, n, sizeof(long), &info, compare_point);

    free_int_matrix(I_tilde);
    free_int_matrix(G_tilde);
}

void balloon_rank(float_matrix *points, float_matrix *seeds, double *labels, long *ranks, bool deflate, int num_threads) {
    float_matrix *D = get_float_matrix(points->nrows, seeds->nrows);
    compute_squared_distances(points, seeds, D, num_threads);
    balloon_rank_from_distances(D, labels, ranks, deflate, num_threads);
    free_float_matrix(D);
}

void py_compute_squared_distances(long n, long s, long dim, double **points, double **seeds, double **D, int num_threads) {
    float_matrix point_mat, seed_mat, dist_mat;
    point_mat.arr = points;
    point_mat.nrows = n;
    point_mat.ncols = dim;
    seed_mat.arr = seeds;
    seed_mat.nrows = s;
    seed_mat.ncols = dim;
    dist_mat.arr = D;
    dist_mat.nrows = n;
    dist_mat.ncols = s;
    compute_squared_distances(&point_mat, &seed_mat, &dist_mat, num_threads);
}

void py_balloon_rank_from_distances(long n, long s, double **D, double *labels, long *ranks, bool deflate, int num_threads) {
    float_matrix dist_mat;
    dist_mat.arr = D;
    dist_mat.nrows = n;
    dist_mat.ncols = s;
    balloon_rank_from_distances(&dist_mat, labels, ranks, deflate, num_threads);
}

void py_balloon_rank(long n, long s, long dim, double **points, double **seeds, double *labels, long *ranks, bool deflate, int num_threads) {
    float_matrix point_mat, seed_mat;
    point_mat.arr = points;
    point_mat.nrows = n;
    point_mat.ncols = dim;
    seed_mat.arr = seeds;
    seed_mat.nrows = s;
    seed_mat.ncols = dim;
    balloon_rank(&point_mat, &seed_mat, labels, ranks, deflate, num_threads);
}