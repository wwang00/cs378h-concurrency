#include <pthread.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>

#include "barrier.h"
#include "scan_types.h"

#define PARALLEL_WORKER pfx_scan_max_parallelism

template <class T>
void pfx_scan_sequential(T *arr, const int N,
                         T (*scan_op)(const T &, const T &)) {
  T acc = arr[0];
  for (int t = 1; t < N; t++) {
    acc = scan_op(acc, arr[t]);
    arr[t] = acc;
  }
}

template <class T>
void pfx_scan_parallel(T *arr, const int N, const int threads,
                       T (*scan_op)(const T &, const T &)) {
  pthread_t tid[threads];
  pfx_scan_parallel_args<T> args_list[threads];
  barrier bar(threads);
  int step = 1;
  for (int t = 0; t < threads; t++) {
    auto args = new (&args_list[t])
        pfx_scan_parallel_args<T>(t, &bar, &step, arr, N, threads, scan_op);
    pthread_create(&tid[t], nullptr, PARALLEL_WORKER<T>, (void *)args);
  }
  for (int t = 0; t < threads; t++) {
    pthread_join(tid[t], nullptr);
  }
}

template <class T>
void *pfx_scan_max_parallelism(void *args_) {
  auto args = (pfx_scan_parallel_args<T> *)args_;
  int num = args->num;
  barrier *bar = args->bar;
  int *step = args->step;
  T *arr = args->arr;
  int N = args->N;
  int threads = args->threads;
  auto scan_op = args->scan_op;
  int step_local;
  while ((step_local = *step) < N) {
    int elems;
    if ((elems = (N - step_local) / threads) * threads != N - step_local) {
      elems++;
    }
    int start = step_local + num * elems;
    int max_elems = N - start;
    if (max_elems <= 0) {
      bar->wait();
      bar->wait();
      continue;
    } else if (max_elems < elems) {
      elems = max_elems;
    }
    T temp[elems];
    // calculate locally
    for (int t = 0; t < elems; t++) {
      int a = start + t;
      temp[t] = scan_op(arr[a], arr[a - step_local]);
    }
    bar->wait();
    // commit results
    for (int t = 0; t < elems; t++) {
      int a = start + t;
      arr[a] = temp[t];
    }
    // one thread increments step
    if (num == 0) {
      *step *= 2;
    }
    bar->wait();
  }
  return nullptr;
}

template <class T>
void *pfx_scan_work_efficient(void *args_) {
  return nullptr;
}