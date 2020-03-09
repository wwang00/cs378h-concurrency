#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "params.h"

#define BLOCKS 1
#define TPB    1

__global__
void converged(Parameters P,
	       double     *curr,
	       double     *prev,
	       bool       *conv) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  
  int begin = block * block_dim + thread;
  int step = grid_dim * block_dim;
  int end = P.clusters * P.dims;
  
  for(int c = begin; c < end; c += step) {
    double diff = curr[c] - prev[c];
    if(diff < -P.threshold || diff > P.threshold) {
      *conv = false;
      return;
    }
  }
}

__global__
void centroid_calculator(Parameters P,
			 double     *features,
			 double     *centroids,
			 int        *labels,
			 int        *counts,
			 double     *totals) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  
  int begin = block * block_dim + thread;
  int step = grid_dim * block_dim;
  int end = P.points;
  
  for(int p = begin; p < end; p += step) {
    int closest;
    double min_dist_sq = 1e9;
    for(int c = 0; c < P.clusters; c++) {
      double dist_sq = 0;
      for(int d = 0; d < P.dims; d++) {
	double diff = features[p * P.dims + d] - centroids[c * P.dims + d];
	dist_sq += diff * diff;
      }
      if(dist_sq < min_dist_sq) {
	min_dist_sq = dist_sq;
	closest = c;
      }
    }
    labels[p] = closest;
    atomicAdd(counts + closest, 1);
    for(int d = 0; d < P.dims; d++) {
      atomicAdd(totals + (closest * P.dims + d), features[p * P.dims + d]);
    }
  }
}

__global__
void centroid_updater(Parameters P,
		      double     *centroids,
		      int        *counts,
		      double     *totals) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  
  int begin = block * block_dim + thread;
  int step = grid_dim * block_dim;
  int end = P.clusters;
  
  for(int c = begin; c < end; c += step) {
    int count = counts[c];
    if(count > 0) {
      for(int d = 0; d < P.dims; d++) {
	centroids[c * P.dims + d] = totals[c * P.dims + d] / count;
      }
    }
  }
}  

#endif