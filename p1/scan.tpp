#include <pthread.h>
#include <stdlib.h>

#include <iostream>

#include "barrier.h"
#include "scan_types.h"

template <class T>
void pfx_scan_sequential(T *arr, const int N,
                         T (*scan_op)(const T &, const T &)) {
  T acc = arr[0];
  for (int i = 1; i < N; i++) {
    acc = scan_op(acc, arr[i]);
    arr[i] = acc;
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
    T temp[N];
    int start = step + num;
    // calculate locally
    for (int i = start; i < N; i += threads) {
      temp[i] = scan_op(arr[i], arr[i - step]);
    }
    bar->wait();
    // commit results
    for (int i = start; i < N; i += threads) {
      arr[i] = temp[i];
    }
    if (num == 0) {
      step *= 2;
    }
    bar->wait();
  }
  return nullptr;
}