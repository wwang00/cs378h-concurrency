#ifndef _SCAN_OPS_
#define _SCAN_OPS_

#include <stdlib.h>

int add_int(const int &x, const int &y);

template <size_t N>
float *add_vector_float(const float *&x, const float *&y);

#include "scan_ops.tpp"

#endif