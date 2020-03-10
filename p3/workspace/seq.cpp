#include <stdlib.h>
#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "argparse.h"
#include "rng.h"

using namespace std;

int    _points;
int    _clusters;
int    _dims;
string _input_file;
int    _iterations;
double _threshold;
bool   _output_centroids;
int    _seed;

vector<double> features;
vector<double> centroids;
vector<int> labels;

int main(int argc, char **argv) {
  unordered_set<string> flags{"-c", "-q"};
  unordered_set<string> opts{"-k", "-d", "-i", "-m", "-t", "-s"};
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  _clusters = stoi(args["-k"]);
  _dims = stoi(args["-d"]);
  _input_file = args["-i"];
  _iterations = stoi(args["-m"]);
  _threshold = stod(args["-t"]);
  _output_centroids = args.count("-c");
  _seed = stoi(args["-s"]);
  ifstream fin(_input_file);
  fin >> _points;
  for(int i = 0; i < _points * (_dims + 1); i++) {
    double d; fin >> d;
    if(i % (_dims + 1) == 0) continue;
    features.push_back(d);
  }
  
  // initialize random centroids
  
  centroids.resize(_clusters * _dims);
  kmeans_srand(_seed);
  for(int c = 0; c < _clusters; c++) {
    int index = kmeans_rand() % _points;
    copy(features.begin() + _dims * index,
	 features.begin() + _dims * (index + 1),
	 centroids.begin() + _dims * c);
  }
  
  // do centroid calculation
  
  labels.resize(_points);

  int iter = 0;
  bool conv;
  
  auto t0 = chrono::system_clock::now();
  do {
    iter++;
    conv = true;
    
    // calculate centroid totals
    
    vector<int> counts(_clusters, 0);
    vector<double> totals(_clusters * _dims, 0);
    for(int p = 0; p < _points; p++) {
      int closest;
      double min_dist_sq = 1e9;
      for(int c = 0; c < _clusters; c++) {
	double dist_sq = 0;
	for(int d = 0; d < _dims; d++) {
	  double diff = features[p * _dims + d] - centroids[c * _dims + d];
	  dist_sq += diff * diff;
	}
	if(dist_sq < min_dist_sq) {
	  min_dist_sq = dist_sq;
	  closest = c;
	}
      }
      labels[p] = closest;
      counts[closest]++;
      for(int d = 0; d < _dims; d++) {
	totals[closest * _dims + d] += features[p * _dims + d];
      }
    }

    // update centroids and convergence
    
    for(int c = 0; c < _clusters; c++) {
      int count = counts[c];
      if(count > 0) {
	for(int d = 0; d < _dims; d++) {
	  int i = c * _dims + d;
	  double new_val = totals[i] / count;
	  double diff = centroids[i] - new_val;
	  if(diff < -_threshold || diff > _threshold)
	    conv = false;
	  centroids[i] = new_val;
	}
      }
    }
  } while(!(iter == _iterations || conv));
  /*
  auto t1 = chrono::system_clock::now();
  long elapsed = (long)((t1 - t0) / chrono::microseconds(1));
  printf("%ld\n", elapsed / iter);
  */
  auto t1 = chrono::system_clock::now();
  double elapsed = (double)((t1 - t0) / chrono::milliseconds(1));
  printf("%d,%.5lf\n", iter, elapsed / iter);

  if(_output_centroids) {
    for (int c = 0; c < _clusters; c ++){
      printf("%d ", c);
      for (int d = 0; d < _dims; d++)
	printf("%.5lf ", centroids[c * _dims + d]);
      printf("\n");
    }
  } else {
    printf("clusters:");
    for (int p = 0; p < _points; p++)
      printf(" %d", labels[p]);
  }

  fin.close();
  return 0;
}
