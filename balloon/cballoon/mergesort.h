#ifndef MERGESORT_H
#define MERGESORT_H

typedef int (*Compare_fn)(void *, const void *, const void *);

void mgsort(void *arr, size_t num_elem, size_t elem_size, void *arg, Compare_fn cmp);
void mt_mergesort_r(void *arr, size_t num_elem, size_t elem_size, void *arg, Compare_fn f, int num_threads);

#endif
