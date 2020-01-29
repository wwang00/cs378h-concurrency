#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "argparse.h"
#include "scan.h"
#include "scan_types.h"

using namespace std;

int main(int argc, char **argv) {
  size_t N;        // number of input data
  size_t dim;      // FP vector dimension
  size_t threads;  // parallelism
  // parse args
  unordered_set<string> flags{"-s"};
  unordered_set<string> opts{"-n", "-i", "-o"};
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  if (!args.count("-n") || !args.count("-i") || !args.count("-o")) {
    cout << "missing arguments" << endl;
    return -1;
  }
  threads = stoi(args["-n"]);
  ifstream fin(args["-i"]);
  ofstream fout(args["-o"]);
  // read input and do work
  fin >> dim >> N;
  if (dim > 0) {  // FP vector data
    fp_vector *input = (fp_vector *)malloc(N * sizeof(fp_vector));
    for (int i = 0; i < N; i++) {
      fp_vector vec = *new (&input[i]) fp_vector(dim);
      for (int j = 0; j < dim; j++) {
        fin >> vec.v[j];
      }
    }
    // timed work
    fp_vector *result;
    auto t0 = chrono::system_clock::now();
    if (threads == 0) {  // sequential
      result = pfx_scan_sequential<fp_vector, fp_vector::scan_op>(input, N);
    } else {  // parallel
      result =
          pfx_scan_parallel<fp_vector, fp_vector::scan_op>(input, N, threads);
    }
    auto t1 = chrono::system_clock::now();
    cout << (t1 - t0) / chrono::milliseconds(1) << endl;
    // write to output file
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < dim; j++) {
        fout << result[i].v[j] << " ";
      }
      fout << endl;
    }
  } else {  // integer data
    int_pad *input = (int_pad *)malloc(N * sizeof(int_pad));
    for (int i = 0; i < N; i++) {
      fin >> input[i].v;
    }
    // timed work
    int_pad *result;
    auto t0 = chrono::system_clock::now();
    if (threads == 0) {  // sequential
      result = pfx_scan_sequential<int_pad, int_pad::scan_op>(input, N);
    } else {  // parallel
      result = pfx_scan_parallel<int_pad, int_pad::scan_op>(input, N, threads);
    }
    auto t1 = chrono::system_clock::now();
    cout << (t1 - t0) / chrono::milliseconds(1) << endl;
    // write to output file
    for (int i = 0; i < N; i++) {
      fout << result[i].v << endl;
    }
  }
  fin.close();
  fout.close();
  return 0;
}
