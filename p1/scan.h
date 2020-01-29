#ifndef _SCAN_H_
#define _SCAN_H_

template <class T>
T *pfx_scan_sequential(const T *arr, const size_t N,
                       T (*scan_op)(const T &, const T &));

template <class T>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads,
                     T (*scan_op)(const T &, const T &));

template <class T>
void *pfx_scan_parallel_worker(void *args);

template <class T>
struct pfx_scan_parallel_args {
  const int num;
  const T *arr;
  const size_t N;
  T (*scan_op)(const T &, const T &);
  T *result;

  pfx_scan_parallel_args(const int num, const T *arr, const size_t N,
                         T (*scan_op)(const T &, const T &), T *result)
      : num(num), arr(arr), N(N), scan_op(scan_op), result(result) {}
};

#include "scan.tpp"

#endif