#ifndef _SCAN_H_
#define _SCAN_H_

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_sequential(const T *arr, const size_t N);

template <class T, T (*scan_op)(const T &, const T &)>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads);

#include "scan.tpp"

#endif