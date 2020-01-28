#include <pthread.h>
#include <stdlib.h>

#include "scan_ops.h"

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_sequential(const T *arr, const size_t N) {
  T *result = (T *)malloc(N * sizeof(T));
  T acc{};
  result[0] = acc;
  for (int i = 1; i < N; i++) {
    acc = scan_op(acc, arr[i - 1]);
    result[i] = acc;
  }
  return result;
}

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads) {
  T *result = (T *)malloc(N * sizeof(T));
  return result;
}