#ifndef _FUNCTORS_H_
#define _FUNCTORS_H_

#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include "defs.h"

struct centroid_minimizer {
  Parameters P;
  d_ptr centroids;
  d_ptr features;
  int p;

  // reduce: two centroid indices -> the closer one to the point p
  __host__ __device__
  int operator()(const int &a, const int &b) {
    if(a < 0) return b; if(b < 0) return a;
    double da = 0, db = 0, diff;
    for(int d = 0; d < P.dims; d++) {
      diff = centroids[a * P.dims + d] - features[p * P.dims + d];
      da += diff * diff;
      diff = centroids[b * P.dims + d] - features[p * P.dims + d];
      db += diff * diff;
    }
    return da < db ? a : b;
  }
};

struct feature_labeler {
  Parameters P;
  d_ptr features;
  d_ptr centroids;

  // transform: point index -> closest centroid index
  __host__ __device__
  int operator()(const int p) {
    thrust::counting_iterator<int> begin(0);
    return thrust::reduce(thrust::device,
			  begin,
			  begin + P.clusters,
			  -1,
			  centroid_minimizer{P, centroids, features, p});
  }
};

#endif
