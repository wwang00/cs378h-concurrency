#include <pthread.h>
#include <stdlib.h>

#include <iostream>

#include "barrier.h"
#include "scan_types.h"

template <class T>
T *pfx_scan_sequential(const T *arr, const size_t N,
                       T (*scan_op)(const T &, const T &)) {
  T *result = (T *)calloc(N, sizeof(T));
  T acc = arr[0];
  acc = 0;
  result[0] = acc;
  for (int i = 0; i < N - 1; i++) {
    acc = scan_op(acc, arr[i]);
    result[i + 1] = acc;
  }
  return result;
}

template <class T>
T *pfx_scan_parallel(const T *arr, const size_t N, const size_t threads,
                     T (*scan_op)(const T &, const T &)) {
  T *result = (T *)calloc(N, sizeof(T));
  result[0] = 0;
  pthread_t tid[threads];
  pfx_scan_parallel_args<T> *args_list = (pfx_scan_parallel_args<T> *)malloc(
      threads * sizeof(pfx_scan_parallel_args<T>));
  bar = new barrier(threads);
  step = 1;
  for (int t = 0; t < threads; t++) {
    pfx_scan_parallel_args<T> *args = new (&args_list[t])
        pfx_scan_parallel_args<T>(t, &arr[1], N - 1, scan_op, &result[1]);
    pthread_create(&tid[t], nullptr, pfx_scan_parallel_worker<T>, (void *)args);
  }
  for (int t = 0; t < threads; t++) {
    pthread_join(tid[t], nullptr);
  }
  free(args_list);
  delete bar;
  return result;
}

template <class T>
void *pfx_scan_parallel_worker(void *args_) {
  pfx_scan_parallel_args<T> *args = (pfx_scan_parallel_args<T> *)args_;
  //   while(step < args->N) {

  //     bar->wait();
  //   }
  return nullptr;
}