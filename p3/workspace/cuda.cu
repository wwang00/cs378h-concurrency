#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "argparse.h"
#include "cuda_util.h"
#include "rng.h"

using namespace std;

Parameters P;

int main(int argc, char **argv) {
  // get gpu props
  
  int device;
  cudaDeviceProp device_props;
  if(cudaGetDevice(&device)) return -1;
  if(cudaGetDeviceProperties(&device_props, device)) return -1;
  
  // get algorithm props
  
  unordered_set<string> flags{"-c", "-q"};
  unordered_set<string> opts{"-k", "-d", "-i", "-m", "-t", "-s"};
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  P.clusters = stoi(args["-k"]);
  P.dims = stoi(args["-d"]);
  string input_file = args["-i"];
  P.iterations = stoi(args["-m"]);
  P.threshold = stof(args["-t"]);
  P.output_centroids = args.count("-c");
  P.seed = stoi(args["-s"]);
  ifstream fin(input_file);
  fin >> P.points;

  // read in points
  
  vector<float> features_host(P.points * P.dims);
  for(int p = 0; p < P.points; p++) {
    for(int d = -1; d < P.dims; d++) {
      float val; fin >> val;
      if(d < 0) continue;
      features_host[p * P.dims + d] = val;
    }
  }
  float *features = 0;
  size_t features_size = P.points * P.dims * sizeof(float);
  if(cudaMalloc(&features, features_size)) return -1;
  if(cudaMemcpy(features, &features_host[0], features_size, cudaMemcpyHostToDevice)) return -1;
  
  // initialize random centroids
  
  vector<float> centroids_host(P.clusters * P.dims);
  kmeans_srand(P.seed);
  for(int c = 0; c < P.clusters; c++) {
    int index = kmeans_rand() % P.points;
    copy(features_host.begin() + index * P.dims,
	 features_host.begin() + (index + 1) * P.dims,
	 centroids_host.begin() + c * P.dims);
  }
  float *centroids;
  size_t centroids_size = P.clusters * P.dims * sizeof(float);
  if(cudaMalloc(&centroids, centroids_size)) return -1;
  if(cudaMemcpy(centroids, &centroids_host[0], centroids_size, cudaMemcpyHostToDevice)) return -1;
  
  // do centroid calculation
  
  vector<int> labels_host(P.points);
  int *labels;
  size_t labels_size = P.points * sizeof(int);
  if(cudaMalloc(&labels, labels_size)) return -1;
  
  int *counts;
  size_t counts_size = P.clusters * sizeof(int);
  if(cudaMalloc(&counts, counts_size)) return -1;
  
  float *totals;
  size_t totals_size = P.clusters * P.dims * sizeof(float);
  if(cudaMalloc(&totals, totals_size)) return -1;
  
  int iter = 0;
  bool conv_host;
  bool *conv;
  if(cudaMalloc(&conv, sizeof(bool))) return -1;
  
  auto t0 = chrono::system_clock::now();
  do {
    iter++;
    if(cudaMemset(conv, true, sizeof(bool))) return -1;
    
    // calculate centroid totals
    
    if(cudaMemset(counts, 0, counts_size)) return -1;
    //if(cudaMemset(totals, 0, totals_size)) return -1;
    centroid_calculator<<<BLOCKS, TPB>>>(P,
					 features,
					 centroids,
					 labels,
					 counts,
					 totals);
    cudaDeviceSynchronize();
    
    // update centroids and check convergence
    
    centroid_updater<<<BLOCKS, TPB>>>(P,
				      centroids,
				      counts,
				      totals,
				      conv);
    cudaDeviceSynchronize();
    if(cudaMemcpy(&conv_host, conv, sizeof(bool), cudaMemcpyDeviceToHost)) return -1;
  } while(!(iter == P.iterations || conv_host));
#ifdef DEBUG
  auto t1 = chrono::system_clock::now();
  long elapsed = (long)((t1 - t0) / chrono::microseconds(1));
  printf("%ld\n", elapsed / iter);
#else
  auto t1 = chrono::system_clock::now();
  float elapsed = (float)((t1 - t0) / chrono::milliseconds(1));
  printf("%d,%.5f\n", iter, elapsed / iter);

  if(P.output_centroids) {
    if(cudaMemcpy(&centroids_host[0], centroids, centroids_size, cudaMemcpyDeviceToHost)) return -1;
    for (int c = 0; c < P.clusters; c ++){
      printf("%d ", c);
      for (int d = 0; d < P.dims; d++)
	printf("%.5f ", centroids_host[c * P.dims + d]);
      printf("\n");
    }
  } else {
    if(cudaMemcpy(&labels_host[0], labels, labels_size, cudaMemcpyDeviceToHost)) return -1;
    printf("clusters:");
    for (int p = 0; p < P.points; p++)
      printf(" %d", labels_host[p]);
  }
#endif
  cudaFree(features);
  cudaFree(centroids);
  cudaFree(labels);
  cudaFree(counts);
  cudaFree(totals);
  cudaFree(conv);
  fin.close();
  return 0;
}
