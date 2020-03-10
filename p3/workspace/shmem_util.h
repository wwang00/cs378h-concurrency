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
  float *totals_sh = (float *)(counts_sh + P.clusters);
  float *centroids_sh = (float *)(totals_sh + P.clusters * P.dims);

  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  int all = grid_dim * block_dim;
  int id = block * block_dim + thread;
  
  int begin;
  int step;
  int end;
  
  // copy centroids locally

  begin = thread;
  step = block_dim;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts_sh[c] = 0;
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      totals_sh[idx] = 0;
      centroids_sh[idx] = centroids[idx];
    }
  }
  __syncthreads();

  // reset global counts, totals

  begin = id;
  step = all;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts[c] = 0;
    for(int d = 0; d < P.dims; d++) {
      totals[c * P.dims + d] = 0;
    }
  }
  __syncthreads();
  
  // work on points

  begin = id;
  step = all;
  end = P.points;
  for(int p = begin; p < end; p += step) {
    int closest;
    float min_dist_sq = 1e9;
    for(int c = 0; c < P.clusters; c++) {
      float dist_sq = 0;
      for(int d = 0; d < P.dims; d++) {
	float diff = features[p * P.dims + d] - centroids_sh[c * P.dims + d];
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

  // store to global

  begin = id;
  step = all;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    counts[c] = counts_sh[c];
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      totals[idx] = totals_sh[idx];
      centroids[idx] = centroids_sh[idx];
    }
  }
}

__global__
void centroid_updater_sh(Parameters P,
			 float      *centroids,
			 int        *counts,
			 float      *totals,
			 bool       *conv) {
  extern __shared__ char sh[];
  float *centroids_sh = (float *)sh;

  int block = blockIdx.x;
  int thread = threadIdx.x;
  int grid_dim = gridDim.x;
  int block_dim = blockDim.x;
  int all = grid_dim * block_dim;
  int id = block * block_dim + thread;
  
  int begin;
  int step;
  int end;
  
  // copy centroids locally

  begin = thread;
  step = block_dim;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      centroids_sh[idx] = centroids[idx];
    }
  }
  __syncthreads();

  // work on clusters

  begin = id;
  step = all;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    int count = counts[c];
    if(count > 0) {
      for(int d = 0; d < P.dims; d++) {
	int i = c * P.dims + d;
	float new_val = totals[i] / count;
	float diff = centroids_sh[i]  - new_val;
	if(diff < -P.threshold || diff > P.threshold)
	  *conv = false;
	centroids_sh[i] = new_val;
      }
    }
  }
  __syncthreads();

  // store to global

  begin = id;
  step = all;
  end = P.clusters;
  for(int c = begin; c < end; c += step) {
    for(int d = 0; d < P.dims; d++) {
      int idx = c * P.dims + d;
      centroids[idx] = centroids_sh[idx];
    }
  }
}  

#endif
