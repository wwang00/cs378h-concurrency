#include <stdlib.h>

template <size_t N>
float *add_vector_float(const float *&x, const float *&y) {
  float *result = (float *)malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    result[i] = x[i] + y[i];
  }
  return result;
}