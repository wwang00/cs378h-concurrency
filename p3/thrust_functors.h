#ifndef _THRUST_FUNCTORS_H_
#define _THRUST_FUNCTORS_H_

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

#include "params.h"
#include "thrust_defs.h"

struct convergence_checker {
  double threshold;
  
  // transform: if elements within threshold, return true
  __host__ __device__
  bool operator()(const thrust::tuple<double, double> &e) {
    double diff = thrust::get<0>(e) - thrust::get<1>(e);
    return -threshold <= diff && diff <= threshold;
  }
};

struct centroid_calculator {
  Parameters P;
  d_ptr      features;
  d_ptr      centroids;
  d_ptr_int  labels;
  d_ptr_int  counts;
  d_ptr      totals;
  
  struct centroid_minimizer {
    Parameters P;
    d_ptr      features;
    d_ptr      centroids;
    int        p;
    
    // reduce: two centroid indices -> the closer one to the point p
    __device__
    int operator()(const int &a, const int &b) {
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
  
  struct centroid_accumulator {
    Parameters P;
    d_ptr      features;
    d_ptr      totals;
    int        c;
    int        p;
    
    // add point coordinate to closest centroid coordinate atomically
    __device__
    void operator()(const int d) {
      atomicAdd((totals + (c * P.dims + d)).get(), (double)features[p * P.dims + d]);
    }
  };
  
  // for_each: point -> get closest centroid, update label, update count, update total
  __device__
  void operator()(const int p) {
    thrust::counting_iterator<int> begin(0);
    int closest = thrust::reduce(thrust::device,
				 begin,
				 begin + P.clusters,
				 0,
				 centroid_minimizer{P, features, centroids, p});
    labels[p] = closest;
    atomicAdd((counts + closest).get(), 1);
    begin = thrust::make_counting_iterator<int>(0);
    thrust::for_each(thrust::device,
		     begin,
		     begin + P.dims,
		     centroid_accumulator{P, features, totals, closest, p});
  }
};

struct centroid_updater {
  Parameters P;
  d_ptr      centroids;
  d_ptr_int  counts;
  d_ptr      totals;
  
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
