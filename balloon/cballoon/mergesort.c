/*
Christopher Antol
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <err.h>
#include <sysexits.h>
#include <omp.h>
#include "mergesort.h"

/*structure to hold arguments to pass into merge sort*/
typedef struct {
  void *array;
  size_t len;
  size_t size;
  void *arg;
  Compare_fn f;
} Arguments;

/*merges the two arrays together in sorted order*/
static void *merge(void *left_side, void *right_side, unsigned int left_len,
		   unsigned int right_len, unsigned int size, void *arg, Compare_fn f) {

  char *left = left_side, *right = right_side, *combo;
  int counter = 0;

  /*allocate memory for temporary array*/
  if (!(combo = calloc((left_len + right_len), size)))
    err(EX_OSERR, "memory allocation error");

  /* until left and right are fully distributed, compare left and
    right then place element into temp array */
  while (left_len > 0 || right_len > 0) {

    /* compare left and right if other is not empty */
    if (left_len > 0 && right_len > 0) {
      /* compare left and right given function f */
      /* if left <= right, put first element of left into temp array. otherwise put first element of right in */
      if(f(arg, left, right) <= 0) {
      	memcpy((combo + (counter++ * size)), left, size);
      	left += size;
      	left_len--;
      } 
      else {
      	memcpy((combo + (counter++ * size)), right, size);
      	right += size;
      	right_len--;
      }
      /* if one array is empty, fill put the remaining elements	of other array into temp array */
    } 
    else if(left_len > 0) {
      memcpy((combo + (counter * size)), left, (size * left_len));      
      left_len = 0;
    }
    else if(right_len > 0) {
      memcpy((combo + (counter * size)), right, (size * right_len));
      right_len = 0;
    }
  }
  return combo;
}

/*splits array in half, calls self on both halves, then merges the halves*/
void mgsort(void *arr, size_t num_elem, size_t elem_size, void *arg, Compare_fn f) {

  unsigned int len = (unsigned int) num_elem, size = (unsigned int) elem_size;
  char *array = (char *) arr;
  void *combo, *left, *right;

  if (!arr || !f)
    return;

  if (len < 2)
    return;

  /*left array starts at beggining of arr while
    right array starts at the second half of arr*/
  left = array;
  right = array + ((len/2) * size);

  /*calls self on each half, but with half the length of arr
    to only account for half the elements of arr per array passed*/

  /* len - len/2 accounts for an odd lengthed array */
  mgsort(left, len/2, size, arg, f);
  mgsort(right, (len - len/2), size, arg, f);

  /* merges left and right into temporary array combo */
  combo = merge(left, right, len/2, (len - len/2), size, arg, f);

  /* replaces contents of arr with contents of temp array (which
     are sorted) and then frees the memory for temp array */
  memcpy(array, combo, len * size);
  free(combo);
}

/*splits array up and gives each part to a thread to sort
  then merges the sorted parts together*/
void mt_mergesort_r(void *arr, size_t num_elem, size_t elem_size, void *arg, Compare_fn f,
		  int num_threads) {

  unsigned int len = (unsigned int) num_elem, size = (unsigned int) elem_size;
  unsigned int i;
  Arguments *array_part;
  char **segments;

  if (!arr || !f)
    return;

  if (len < 2)
    return;

  // get highest power of two allowable threads
  int nt = 1;
  while (nt << 1 <= num_threads) {
    nt <<= 1;
  }
  num_threads = nt;

  /* allocates memory for segments, and array_part arrays*/
  array_part = (Arguments *)calloc((size_t)num_threads, sizeof(Arguments));
  segments = (char **)calloc((size_t)num_threads, sizeof(char *));

  if(!array_part || !segments)
    err(EX_OSERR, "memory allocation error");

  /* distributes arr into num_threads number of structures
     in order to split the work up among threads */
  for (i = 0; i < num_threads; i++) {
    segments[i] = ((char *)arr) + (i * (len/num_threads) * size);
    array_part[i].array = segments[i];
    array_part[i].len = len / num_threads;
    array_part[i].size = size;
    array_part[i].arg = arg;
    array_part[i].f = f;
  }

  /* if len % num_threads != 0, add leftover elements to final thread*/
  if (len % num_threads)
    array_part[num_threads-1].len += len % num_threads;

  /* perform mergesort in parallel across all the threads */
  Arguments *args;

# pragma omp parallel for \
  shared (num_threads, array_part) \
  private (i, args)
  for(i = 0; i < num_threads; ++i) {
    args = &array_part[i];
    mgsort(args->array, args->len, args->size, args->arg, args->f);
  }

  /* merges the work done by threads if merging is necessary */
  char **temp = calloc(num_threads >> 1, sizeof(char *));
  if (num_threads > 1) {
    for(nt = num_threads >> 1; nt > 0; nt >>= 1) {

# pragma omp parallel for \
      shared (nt, segments, array_part, size, arg, f) \
      private (i)
      for(i = 0; i < nt; ++i) {
        temp[i] = (char *) merge(segments[2 * i], segments[2 * i + 1], array_part[2 * i].len, array_part[2 * i + 1].len, size, arg, f);
      }

      for(i = 0; i < nt; ++i) {
        segments[i] = temp[i];
        array_part[i].array = segments[i];
        array_part[i].len = array_part[2 * i].len + array_part[2 * i + 1].len;        
      }
    }
  } 

  /* replaces contents of arr with sorted contents of original array, then frees allocated memory */
  memcpy(arr, segments[0], len * size);

  free(array_part);
  if (num_threads > 1) {
    free(segments[0]);
  }
  free(segments);   
}
