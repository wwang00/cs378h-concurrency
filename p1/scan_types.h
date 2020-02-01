#ifndef _SCAN_TYPES_H_
#define _SCAN_TYPES_H_

#include <stdlib.h>

#include <iostream>

struct int_pad {
  int v;

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
  int dim;
  float *v;

  fp_vector() : dim(0), v(nullptr) {
    // std::cout << "creating fp_vector default" << std::endl;
  }

  fp_vector(int dim) : dim(dim), v((float *)calloc(dim, sizeof(float))) {
    // std::cout << "creating fp_vector" << std::endl;
  }

  fp_vector(const fp_vector &copy)
      : dim(copy.dim), v((float *)malloc(copy.dim * sizeof(float))) {
    // std::cout << "copying fp_vector" << std::endl;
    for (int i = 0; i < dim; i++) {
      v[i] = copy.v[i];
    }
  }

  fp_vector &operator=(const fp_vector &copy) {
    // std::cout << "copying fp_vector assignment" << std::endl;
    dim = copy.dim;
    if (v != nullptr) {
      free(v);
    }
    v = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
      v[i] = copy.v[i];
    }
    return *this;
  }

  fp_vector &operator=(float x) {
    // std::cout << "assigning fp_vector int" << std::endl;
    if (v != nullptr) {
      for (int i = 0; i < dim; i++) {
        v[i] = x;
      }
    }
    return *this;
  }

  ~fp_vector() {
    // std::cout << "deleting fp_vector" << std::endl;
    if (v != nullptr) {
      free(v);
    }
  }

  static fp_vector add(const fp_vector &x, const fp_vector &y) {
    int dim = x.dim;
    fp_vector result(dim);
    for (int i = 0; i < dim; i++) {
      result.v[i] = x.v[i] + y.v[i];
    }
    return result;
  }
};

#endif