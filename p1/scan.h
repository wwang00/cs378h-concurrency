#ifndef _SCAN_H_
#define _SCAN_H_

#include <pthread.h>
#include <stdlib.h>

template <class T>
void pfx_scan_sequential(T *arr, const int N,
                         T (*scan_op)(const T &, const T &));

template <class T>
void pfx_scan_parallel(T *arr, const int N, const int threads, const bool s,
                       T (*scan_op)(const T &, const T &));

template <class T>
void *pfx_scan_max_parallelism(void *args);

template <class T>
void *pfx_scan_work_efficient(void *args);

class barrier;

template <class T>
struct pfx_scan_parallel_args {
  const int num;
  pthread_barrier_t *pbar;
  barrier *bar;
  T *arr;
  const int N;
  const int threads;
  const bool s;
  T (*scan_op)(const T &, const T &);

  pfx_scan_parallel_args() : num(-1), N(-1), threads(-1), s(false) {}

  pfx_scan_parallel_args(const int num, pthread_barrier_t *pbar, barrier *bar,
                         T *arr, const int N, const int threads, const bool s,
                         T (*scan_op)(const T &, const T &))
      : num(num),
        pbar(pbar),
        bar(bar),
        arr(arr),
        N(N),
        threads(threads),
        s(s),
        scan_op(scan_op) {}
};

#include "scan.tpp"

#endif
