#ifndef _SCAN_H_
#define _SCAN_H_

#include <stdlib.h>

template <class T>
void pfx_scan_sequential(T *arr, const int N,
                         T (*scan_op)(const T &, const T &));

template <class T>
void pfx_scan_parallel(T *arr, const int N, const int threads,
                       T (*scan_op)(const T &, const T &));

template <class T>
void *pfx_scan_parallel_worker(void *args);

class barrier;

template <class T>
struct pfx_scan_parallel_args {
  const int num;
  barrier *bar;
  int *step;
  T *temp;
  T *arr;
  const int N;
  const int threads;
  T (*scan_op)(const T &, const T &);

  pfx_scan_parallel_args() : num(-1), N(-1), threads(-1) {}

  pfx_scan_parallel_args(const int num, barrier *bar, int *step, T *temp,
                         T *arr, const int N, const int threads,
                         T (*scan_op)(const T &, const T &))
      : num(num),
        bar(bar),
        step(step),
        temp(temp),
        arr(arr),
        N(N),
        threads(threads),
        scan_op(scan_op) {}
};

#include "scan.tpp"

#endif