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
  d_ptr_bool conv;
  
  struct centroid_avg {
    Parameters P;
    d_ptr      centroids;
    d_ptr_int  counts;
    d_ptr      totals;
    d_ptr_bool conv;
    int        count;
    
    // for_each: take total / count = avg and put it into centroids; check converged
    __host__ __device__
    void operator()(const int i) {
      double new_val = totals[i] / count;
      double diff = centroids[i] - new_val;
      if(diff < -P.threshold || diff > P.threshold)
	*conv = false;
      centroids[i] = new_val;
    }
  };
  
  //for_each: centroid index -> update the actual centroid value if needed
  __host__ __device__
  void operator()(const int c) {
    int count = counts[c];
    if(count == 0) return;
    thrust::counting_iterator<int> begin(c * P.dims);
    thrust::for_each(thrust::device,
		     begin,
		     begin + P.dims,
		     centroid_avg{P,
			 centroids,
			 counts,
			 totals,
			 conv,
			 count});
  }
};  

#endif