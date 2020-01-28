#include <stdlib.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "argparse.h"
#include "scan.h"
#include "scan_ops.h"

using namespace std;

size_t N;        // number of input data
size_t dim;      // FP vector dimension
size_t threads;  // parallelism

unordered_set<string> flags{"-s"};
unordered_set<string> opts{"-n", "-i", "-o"};

int main(int argc, char **argv) {
  // parse args
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
  } else {        // integer data
    int *arr = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
      fin >> arr[i];
    }
    int *result;
    // timed work
    auto t0 = chrono::system_clock::now();
    if (threads == 0) {  // sequential
      result = pfx_scan_sequential<int, add_int>(arr, N);
    } else {  // parallel
      result = pfx_scan_parallel<int, add_int>(arr, N, threads);
    }
    auto t1 = chrono::system_clock::now();
    cout << (t1 - t0) / chrono::milliseconds(1) << endl;
    // write to output file
    for (int i = 0; i < N; i++) {
      fout << result[i] << endl;
    }
  }
  fin.close();
  fout.close();
  return 0;
}
