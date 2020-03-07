#include <stdlib.h>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "argparse.h"
#include "defs.h"
#include "functors.h"
#include "rng.h"
#include "strided_range.h"

using namespace std;

Parameters P;

bool converged(h_vec &curr, h_vec &prev) {
  for(int i = 0; i < curr.size(); i++) {
    if(fabs(curr[i] - prev[i]) > P.threshold)
      return false;
  }
  return true;
}

int main(int argc, char **argv) {
  unordered_set<string> flags{"-c"};
  unordered_set<string> opts{"-k", "-d", "-i", "-m", "-t", "-s"};
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  P.clusters = stoi(args["-k"]);
  P.dims = stoi(args["-d"]);
  string input_file = args["-i"];
  P.iterations = stoi(args["-m"]);
  P.threshold = stod(args["-t"]);
  P.output_centroids = args.count("-c");
  P.seed = stoi(args["-s"]);
  ifstream fin(input_file);
  fin >> P.points;

  // read in points

  d_vec features(P.points * P.dims);
  h_vec features_host(P.points * P.dims);
  for(int p = 0; p < P.points; p++) {
    for(int d = -1; d < P.dims; d++) {
      double val; fin >> val;
      if(d < 0) continue;
      features_host[p * P.dims + d] = val;
    }
  }
  thrust::copy(features_host.begin(), features_host.end(), features.begin());
  
  // initialize random centroids

  d_vec centroids(P.clusters * P.dims);
  h_vec centroids_host(P.clusters * P.dims);
  kmeans_srand(P.seed);
  for(int c = 0; c < P.clusters; c++) {
    int index = kmeans_rand() % P.points;
    thrust::copy(features_host.begin() + index * P.dims,
		 features_host.begin() + (index + 1) * P.dims,
		 centroids_host.begin() + c * P.dims);
  }
  thrust::copy(centroids_host.begin(), centroids_host.end(), centroids.begin());

  // do centroid calculation

  d_vec_int labels(P.points);
  int iter = 0;
  d_vec old_centroids(P.clusters * P.dims);
  h_vec old_centroids_host(P.clusters * P.dims);

  auto t0 = chrono::system_clock::now();
  do {
    //thrust::copy(centroids.begin(), centroids.end(), old_centroids.begin());
    thrust::copy(centroids_host.begin(), centroids_host.end(), old_centroids_host.begin());
    iter++;

    // find nearest centroids

    thrust::counting_iterator<int> begin(0);
    thrust::transform(thrust::device,
		      begin,
		      begin + P.points,
		      labels.begin(),
		      feature_labeler{P, features.data(), centroids.data()});
    
    // calculate new centroids

    d_vec totals(P.clusters * P.dims);
    totals.clear();
    d_vec_int counts(P.clusters);
    counts.clear();
    begin = thrust::make_counting_iterator<int>(0);
    thrust::for_each(thrust::device,
		     begin,
		     begin + P.clusters,
		     centroid_calculator{
		       P, labels.data(), features.data(), totals.data(), counts.data()});
    for(int c = 0; c < P.clusters; c++) cout << counts[c] << " "; cout << endl;
    
    // update centroids

    begin = thrust::make_counting_iterator<int>(0);
    thrust::for_each(thrust::device,
		     begin,
		     begin + P.clusters,
		     centroid_updater{P, centroids.data(), totals.data(), counts.data()});

    // copy

    thrust::copy(centroids.begin(), centroids.end(), centroids_host.begin());
  } while(!(iter == P.iterations || converged(centroids_host, old_centroids_host)));

  auto t1 = chrono::system_clock::now();
  double elapsed = (double)((t1 - t0) / chrono::milliseconds(1));
  printf("%d,%lf\n", iter, elapsed / iter);
  if(P.output_centroids) {
    for (int c = 0; c < P.clusters; c ++){
      printf("%d ", c);
      for (int d = 0; d < P.dims; d++)
	printf("%.5lf ", centroids_host[c * P.dims + d]);
      printf("\n");
    }
  } else {
    printf("clusters:");
    //for (int p = 0; p < P.points; p++)
      //printf(" %d", labels[p]);
  }
  fin.close();
  return 0;
}
