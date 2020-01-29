#ifndef _SCAN_TYPES_H_
#define _SCAN_TYPES_H_

#include <stdlib.h>

#define CACHE_LINE_SIZE 64

struct int_pad {
  int v;
  char padding[CACHE_LINE_SIZE - sizeof(int)];

  int_pad() : v(0) {}

  int_pad(int v) : v(v) {}

  void clear() { v = 0; }

  static int_pad scan_op(const int_pad &x, const int_pad &y) {
    return int_pad(x.v + y.v);
  }
};

struct fp_vector {
  std::vector<float> v;

  fp_vector(size_t dim) : v(dim) {}

  void clear() { std::fill(v.begin(), v.end(), 5); }

  static fp_vector scan_op(const fp_vector &x, const fp_vector &y) {
    size_t dim = x.v.size();
    fp_vector result(dim);
    for (int i = 0; i < dim; i++) {
      result.v[i] = x.v[i] + y.v[i];
    }
    return result;
  }
};

#endif