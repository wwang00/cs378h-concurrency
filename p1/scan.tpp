#include <pthread.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>

#include "barrier.h"
#include "scan_types.h"

#define PARALLEL_VAR 1

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
void pfx_scan_parallel(T *arr, const int N, const int threads, const bool s,
                       T (*scan_op)(const T &, const T &)) {
  pthread_t tid[threads];
  pfx_scan_parallel_args<T> args_list[threads];
  pthread_barrier_t pbar;
  pthread_barrier_init(&pbar, nullptr, threads);
  barrier bar(threads);
  void *(*worker)(void *);
  switch (PARALLEL_VAR) {
    case 1: {
      worker = pfx_scan_work_efficient<T>;
      break;
    }
    default: {
      worker = pfx_scan_max_parallelism<T>;
      break;
    }
  }
  for (int t = 0; t < threads; t++) {
    auto args = new (&args_list[t])
        pfx_scan_parallel_args<T>(t, &pbar, &bar, arr, N, threads, s, scan_op);
    pthread_create(&tid[t], nullptr, worker, (void *)args);
  }
  for (int t = 0; t < threads; t++) {
    pthread_join(tid[t], nullptr);
  }
  pthread_barrier_destroy(&pbar);
}

// https://commons.wikimedia.org/wiki/File:Hillis-Steele_Prefix_Sum.svg
template <class T>
void *pfx_scan_max_parallelism(void *args_) {
  auto args = (pfx_scan_parallel_args<T> *)args_;
  int num = args->num;
  pthread_barrier_t *pbar = args->pbar;
  barrier *bar = args->bar;
  T *arr = args->arr;
  int N = args->N;
  int threads = args->threads;
  bool s = args->s;
  auto scan_op = args->scan_op;
  int step = 1;
  while (step <= N) {
    int elems = N - step;
    elems = elems / threads + 1;
    int start = step + num * elems;
    // compute locally
    T temp[elems];
    for (int t = 0; t < elems; t++) {
      int a = start + t;
      if (a >= N) {
        break;
      }
      temp[t] = scan_op(arr[a], arr[a - step]);
    }
    if (s) {
      bar->wait();
    } else {
      pthread_barrier_wait(pbar);
    }
    // commit results
    for (int t = 0; t < elems; t++) {
      int a = start + t;
      if (a >= N) {
        break;
      }
      arr[a] = temp[t];
    }
    step *= 2;
    if (s) {
      bar->wait();
    } else {
      pthread_barrier_wait(pbar);
    }
  }
  return nullptr;
}

// https://commons.wikimedia.org/wiki/File:Prefix_sum_16.svg
template <class T>
void *pfx_scan_work_efficient(void *args_) {
  auto args = (pfx_scan_parallel_args<T> *)args_;
  int num = args->num;
  pthread_barrier_t *pbar = args->pbar;
  barrier *bar = args->bar;
  T *arr = args->arr;
  int N = args->N;
  int threads = args->threads;
  bool s = args->s;
  auto scan_op = args->scan_op;
  int step = 2;
  // up sweep
  while (step <= N) {
    int first = step - 1;
    int elems = ((N - 1) - first) / step + 1;
    elems = elems / threads + 1;
    int start = first + num * elems * step;
    for (int t = 0; t < elems; t++) {
      int a = start + t * step;
      if (a >= N) {
        break;
      }
      arr[a] = scan_op(arr[a], arr[a - step / 2]);
    }
    step *= 2;
    if (s) {
      bar->wait();
    } else {
      pthread_barrier_wait(pbar);
    }
  }
  if (step / 2 == N) {
    step /= 4;
  } else {
    step /= 2;
  }
  // down sweep
  while (step >= 2) {
    int first = (step / 2) * 3 - 1;
    if (first >= N) {
      step /= 2;
      continue;
    }
    int elems = ((N - 1) - first) / step + 1;
    elems = elems / threads + 1;
    int start = first + num * elems * step;
    for (int t = 0; t < elems; t++) {
      int a = start + t * step;
      if (a >= N) {
        break;
      }
      arr[a] = scan_op(arr[a], arr[a - step / 2]);
    }
    step /= 2;
    if (s) {
      bar->wait();
    } else {
      pthread_barrier_wait(pbar);
    }
  }
  return nullptr;
}
