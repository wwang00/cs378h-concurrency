#ifndef _FUNCTORS_H_
#define _FUNCTORS_H_

#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "defs.h"

struct convergence_checker {
  double threshold;
  
  // transform: if elements within threshold, return true
  __host__ __device__
  bool operator()(const thrust::tuple<double, double> &e) {
    double diff = thrust::get<0>(e) - thrust::get<1>(e);
    return -threshold <= diff && diff <= threshold;
  }
};

struct feature_labeler {
  Parameters P;
  d_ptr features;
  d_ptr centroids;
  
  struct centroid_minimizer {
    Parameters P;
    d_ptr features;
    d_ptr centroids;
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

  // transform: point index -> closest centroid index
  __host__ __device__
  int operator()(const int &p) {
    thrust::counting_iterator<int> begin(0);
    return thrust::reduce(thrust::device,
			  begin,
			  begin + P.clusters,
			  -1,
			  centroid_minimizer{P, features, centroids, p});
  }
};

struct centroid_calculator {
  Parameters P;
  d_ptr_int labels;
  d_ptr features;
  d_ptr totals;
  d_ptr_int counts;
  
  struct centroid_accumulator {
    Parameters P;
    d_ptr_int labels;
    d_ptr features;
    d_ptr totals;
    d_ptr_int counts;
    int c;
    
    // for_each: point index -> if point near centroid, update centroid
    __host__ __device__
    void operator()(const int p) {
      if(labels[p] != c) return;
      for(int d = 0; d < P.dims; d++)
	totals[c * P.dims + d] = totals[c * P.dims + d] + features[p * P.dims + d];
      counts[c] = counts[c] + 1;
    }
  };
  
  // for_each: centroid index -> update all points near that centroid
  __host__ __device__
  void operator()(const int c) {
    thrust::counting_iterator<int> begin(0);
    thrust::for_each(thrust::device,
		     begin,
		     begin + P.points,
		     centroid_accumulator{P, labels, features, totals, counts, c});
  }
};

struct centroid_updater {
  Parameters P;
  d_ptr centroids;
  d_ptr totals;
  d_ptr_int counts;

  struct centroid_avg {
    int count;

    // transform: take total / count = avg and put it into centroids
    __host__ __device__
    double operator()(const double &val) {
      return val / count;
    }
  };

  //for_each: centroid index -> update the actual centroid value if needed
  __host__ __device__
  void operator()(const int c) {
    int count = counts[c];
    if(count == 0) return;
    d_ptr begin = totals + (c * P.dims);
    thrust::transform(thrust::device,
		      begin,
		      begin + P.dims,
		      centroids + (c * P.dims),
		      centroid_avg{count});
  }
};  

#endif
