#ifndef _BARRIER_H_
#define _BARRIER_H_

#include <pthread.h>

class barrier {
  size_t limit, count;
  pthread_mutex_t mutex;
  pthread_cond_t ready;

 public:
  barrier(size_t limit) : limit(limit), count(limit) {
    mutex = PTHREAD_MUTEX_INITIALIZER;
    ready = PTHREAD_COND_INITIALIZER;
  }

  void wait() {
    pthread_mutex_lock(&mutex);
    if (--count == 0) {
      count = limit;
      pthread_cond_broadcast(&ready);
      pthread_mutex_unlock(&mutex);
      return;
    }
    do {
      pthread_cond_wait(&ready, &mutex);
    } while (count > 0);
    pthread_mutex_unlock(&mutex);
  }
};

#endif