#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#define BLOCKS 16
#define TPB    256

#include "params.h"

__global__
void centroid_calculator(Parameters P,
			 float      *features,
			 float      *centroids,
			 int        *labels,
			 int        *counts,
			 float      *totals) {
  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  
  int begin;
  int step;
  int end;
  
  // reset global counts, totals

  begin = block * block_dim + thread;
  step = grid_dim * block_dim;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts[c] = 0;
    for(int d = 0; d < P.dims; d++) {
      totals[c * P.dims + d] = 0;
    }
  }
  __syncthreads();
  
  // work on points
  
  end = P.points;
  for(int p = begin; p < end; p += step) {
    int closest;
    float min_dist_sq = 1e9;
    for(int c = 0; c < P.clusters; c++) {
      float dist_sq = 0;
      for(int d = 0; d < P.dims; d++) {
	float diff = features[p * P.dims + d] - centroids[c * P.dims + d];
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
		      float      *centroids,
		      int        *counts,
		      float      *totals,
		      bool       *conv) {
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
	int i = c * P.dims + d;
	float new_val = totals[i] / count;
	float diff = centroids[i]  - new_val;
	if(diff < -P.threshold || diff > P.threshold)
	  *conv = false;
	centroids[i] = new_val;
      }
    }
  }
}  

#endif
