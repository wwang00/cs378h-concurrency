#ifndef _SCAN_TYPES_H_
#define _SCAN_TYPES_H_

#include <stdlib.h>

#include <iostream>
#include <vector>

#define CACHE_LINE_SIZE 64

struct int_pad {
  int v;
  char padding[CACHE_LINE_SIZE - sizeof(int)];

  int_pad() : v(0) {}

  int_pad(int v) : v(v) {}

  int_pad &operator=(const int v) {
    this->v = v;
    return *this;
  }

  static int_pad add(const int_pad &x, const int_pad &y) {
    return int_pad(x.v + y.v);
  }
};

struct fp_vector {
  size_t dim;
  std::vector<float> v;

  fp_vector(size_t dim) : dim(dim), v(dim) {}

  static fp_vector add(const fp_vector &x, const fp_vector &y) {
    size_t dim = x.dim;
    fp_vector result(dim);
    for (int i = 0; i < dim; i++) {
      result.v[i] = x.v[i] + y.v[i];
    }
    return result;
  }
};

#endif