#include "scan.h"

#include "barrier.h"

barrier *bar;
int step;

/* special case for vectors */

template <>
fp_vector *pfx_scan_sequential<fp_vector>(
    const fp_vector *arr, const size_t N,
    fp_vector (*scan_op)(const fp_vector &, const fp_vector &)) {
  fp_vector *result = (fp_vector *)malloc(N * sizeof(fp_vector));
  fp_vector acc(arr[0].dim);
  result[0] = acc;
  for (int i = 1; i < N; i++) {
    acc = scan_op(acc, arr[i - 1]);
    result[i] = acc;
  }
  return result;
}