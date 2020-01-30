#ifndef _BARRIER_H_
#define _BARRIER_H_

#include <pthread.h>

class barrier {
  int limit, count;
  bool phase;
  pthread_mutex_t mutex;
  pthread_cond_t ready;

 public:
  barrier(int limit) : limit(limit), count(limit), phase(false) {
    mutex = PTHREAD_MUTEX_INITIALIZER;
    ready = PTHREAD_COND_INITIALIZER;
  }

  void wait() {
    pthread_mutex_lock(&mutex);
    if (--count == 0) {
      count = limit;
      phase = !phase;
      pthread_cond_broadcast(&ready);
      pthread_mutex_unlock(&mutex);
      return;
    }
    bool p = phase;
    do {
      pthread_cond_wait(&ready, &mutex);
    } while (p == phase);
    pthread_mutex_unlock(&mutex);
  }
};

#endif