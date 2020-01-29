#include <pthread.h>
#include <stdlib.h>

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_sequential(const T *arr, const size_t N) {
  T *result = (T *)malloc(N * sizeof(T));
  T acc = arr[0];
  result[0] = acc;
  result[0].clear(); // reset first element without knowing constructor
  for (int i = 1; i < N; i++) {
    result[i] = acc;
    acc = scan_op(acc, arr[i]);
  }
  return result;
}

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads) {
  T *result = (T *)malloc(N * sizeof(T));
  return result;
}