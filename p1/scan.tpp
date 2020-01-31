#include <pthread.h>
#include <stdlib.h>

#include <iostream>

#include "barrier.h"
#include "scan_types.h"

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
  auto args_list = (pfx_scan_parallel_args<T> *)malloc(
      threads * sizeof(pfx_scan_parallel_args<T>));
  bar = new barrier(threads);
  step = 1;
  for (int t = 0; t < threads; t++) {
    auto args = new (&args_list[t])
        pfx_scan_parallel_args<T>(t, arr, N, threads, scan_op);
    pthread_create(&tid[t], nullptr, pfx_scan_parallel_worker<T>, (void *)args);
  }
  for (int t = 0; t < threads; t++) {
    pthread_join(tid[t], nullptr);
  }
  free(args_list);
  delete bar;
}

template <class T>
void *pfx_scan_parallel_worker(void *args_) {
  auto args = (pfx_scan_parallel_args<T> *)args_;
  int num = args->num;
  T *arr = args->arr;
  int N = args->N;
  int threads = args->threads;
  auto scan_op = args->scan_op;
  while (step < N) {
    int start = step + num;
    if (start >= N) {
      bar->wait();
      bar->wait();
      continue;
    }
    int elems;
    if ((elems = (N - start) / threads) * threads != N - start) {
      elems++;
    }
    T temp[elems];
    // calculate locally
    for (int t = 0; t < elems; t++) {
      int a = start + t * threads;
      temp[t] = scan_op(arr[a], arr[a - step]);
    }
    bar->wait();
    // commit results
    for (int t = 0; t < elems; t++) {
      int a = start + t * threads;
      arr[a] = temp[t];
    }
    // one thread increments step
    if (num == 0) {
      step *= 2;
    }
    bar->wait();
  }
  return nullptr;
}