#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

#define VERBOSITY 0

typedef struct float_matrix {
    double **arr;
    long nrows;
    long ncols;
} float_matrix;

typedef struct int_matrix {
    long **arr;
    long nrows;
    long ncols;
} int_matrix;

typedef struct compare_point_info {
    bool deflate;
    long s_minus_one;
    float_matrix *D_tilde;
    int_matrix *I_tilde;
    int_matrix *G_tilde;
    double *labels;
} compare_point_info;

#define MIN(a, b) ((a > b) ? b : a)

void disp_int_arr(long *arr, long n);
void disp_float_arr(double *arr, long n);
float_matrix *get_float_matrix(long nrows, long ncols);
int_matrix *get_int_matrix(long nrows, long ncols);
void free_float_matrix(float_matrix *mat);
void free_int_matrix(int_matrix *mat);
void disp_float_matrix(float_matrix *mat);
void disp_int_matrix(int_matrix *mat);
void compute_squared_distances(float_matrix *points, float_matrix *seeds, float_matrix *D, int num_threads);
int compare_val_index_infl(const void *pa, const void *pb);
int compare_val_index_defl(const void *pa, const void *pb);
int compare_point(void *arg, const void *a, const void *b);
void balloon_rank_from_distances(float_matrix *D, double *labels, long *ranks, bool deflate, int num_threads);
void balloon_rank(float_matrix *points, float_matrix *seeds, double *labels, long *ranks, bool deflate, int num_threads);

// Python interface to distance & rank computations
void py_compute_squared_distances(long npoints, long nseeds, long dim, double **points, double **seeds, double **D, int num_threads);
void py_balloon_rank_from_distances(long npoints, long nseeds, double **D, double *labels, long *ranks, bool deflate, int num_threads);
void py_balloon_rank(long npoints, long nseeds, long dim, double **points, double **seeds, double *labels, long *ranks, bool deflate, int num_threads);
