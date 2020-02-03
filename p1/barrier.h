#ifndef _BARRIER_H_
#define _BARRIER_H_

#include <pthread.h>

class barrier {
  int limit, count;
  bool toggle;
  pthread_spinlock_t lock;

 public:
  barrier(int limit) : limit(limit), count(limit), toggle(false) {
    pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);
  }

  void wait() {
    pthread_spin_lock(&lock);
    if (--count == 0) {
      count = limit;
      toggle = !toggle;
      pthread_spin_unlock(&lock);
      return;
    }
    bool t = toggle;
    while (t == toggle) {
      pthread_spin_unlock(&lock);
      pthread_spin_lock(&lock);
    }
    pthread_spin_unlock(&lock);
  }

  ~barrier() { pthread_spin_destroy(&lock); }
};

#endif
