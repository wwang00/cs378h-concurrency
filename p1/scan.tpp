#include <pthread.h>
#include <stdlib.h>

#include <iostream>

#include "scan_types.h"

/* for primitive types */

template <class T>
T *pfx_scan_sequential(const T *arr, const size_t N,
                       T (*scan_op)(const T &, const T &)) {
  T *result = (T *)malloc(N * sizeof(T));
  T acc = 0;
  result[0] = acc;
  for (int i = 1; i < N; i++) {
    acc = scan_op(acc, arr[i - 1]);
    result[i] = acc;
  }
  return result;
}

template <class T>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads,
                     T (*scan_op)(const T &, const T &)) {
  T *result = (T *)malloc(N * sizeof(T));
  return result;
}

/* special case for vectors due to unknown dimension */

template <>
fp_vector *pfx_scan_sequential<fp_vector>(
    const fp_vector *arr, const size_t N,
    fp_vector (*scan_op)(const fp_vector &, const fp_vector &)) {
  fp_vector *result = (fp_vector *)malloc(N * sizeof(fp_vector));
  fp_vector acc(arr[0].dim);
  result[0] = acc;
  for (int i = 1; i < N; i++) {
    acc = fp_vector::scan_op(acc, arr[i - 1]);
    result[i] = acc;
  }
  return result;
}