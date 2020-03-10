#ifndef _SHMEM_UTIL_H_
#define _SHMEM_UTIL_H_

#include "params.h"

//#define BLOCKS 1024
//#define TPB    64
#define SH_SZ  0xc000

__global__
void centroid_calculator_sh(Parameters P,
			    float      *features,
			    float      *centroids,
			    int        *labels,
			    int        *counts,
			    float      *totals) {
  extern __shared__ char sh[];
  int *counts_sh = (int *)sh;
  float *totals_sh = (float *)sh + P.clusters;

  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;

  int begin = block * block_dim + thread;
  int step = grid_dim * block_dim;
  int end;
  
  // reset shared counts and totals to 0
  
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts_sh[c] = 0;
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      totals_sh[idx] = 0;
    }
  }
  __syncthreads();

  // compute shared totals
  
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
    atomicAdd(counts_sh + closest, 1);
    for(int d = 0; d < P.dims; d++) {
      atomicAdd(totals_sh + (closest * P.dims + d), features[p * P.dims + d]);
    }
  }
  __syncthreads();

  // copy into global mem

  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts[c] = counts_sh[c];
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      totals[idx] = totals_sh[idx];
    }
  }
}

__global__
void centroid_updater_sh(Parameters P,
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
