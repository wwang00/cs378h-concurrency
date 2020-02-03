#include <stdlib.h>

#include <chrono>
#include <fstream>

#include "argparse.h"
#include "scan.h"
#include "scan_types.h"

using namespace std;

int main(int argc, char **argv) {
  int N;        // number of input data
  int dim;      // FP vector dimension
  int threads;  // parallelism
  bool s;
  // parse args
  unordered_set<string> flags{"-s"};
  unordered_set<string> opts{"-n", "-i", "-o"};
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  if (!args.count("-n") || !args.count("-i") || !args.count("-o")) {
    cout << "missing arguments" << endl;
    return -1;
  }
  threads = stoi(args["-n"]);
  s = args.count("-s");
  ifstream fin(args["-i"]);
  ofstream fout(args["-o"]);
  // read input and do work
  fin >> dim >> N;
  if (dim > 0) {  // FP vector data
    fp_vector *arr = (fp_vector *)calloc(N, sizeof(fp_vector));
    string elem;
    for (int i = 0; i < N; i++) {
      fp_vector vec(dim);
      for (int j = 0; j < dim; j++) {
        getline(fin, elem, j == dim - 1 ? '\n' : ',');
        vec.v[j] = stof(elem);
      }
      arr[i] = vec;
    }
    // timed work
    auto t0 = chrono::system_clock::now();
    if (threads == 0) {  // sequential
      pfx_scan_sequential<fp_vector>(arr, N, fp_vector::add);
    } else {  // parallel
      pfx_scan_parallel<fp_vector>(arr, N, threads, s, fp_vector::add);
    }
    auto t1 = chrono::system_clock::now();
    cout << (t1 - t0) / chrono::microseconds(1) << endl;
    // write to output file
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < dim; j++) {
        fout << arr[i].v[j] << (j == dim - 1 ? "" : ",");
      }
      fout << endl;
    }
  } else {  // integer data
    int_pad *arr = (int_pad *)malloc(N * sizeof(int_pad));
    for (int i = 0; i < N; i++) {
      fin >> arr[i].v;
    }
    // timed work
    auto t0 = chrono::system_clock::now();
    if (threads == 0) {  // sequential
      pfx_scan_sequential<int_pad>(arr, N, int_pad::add);
    } else {  // parallel
      pfx_scan_parallel<int_pad>(arr, N, threads, s, int_pad::add);
    }
    auto t1 = chrono::system_clock::now();
    cout << (t1 - t0) / chrono::microseconds(1) << endl;
    // write to output file
    for (int i = 0; i < N; i++) {
      fout << arr[i].v << endl;
    }
  }
  fin.close();
  fout.close();
  return 0;
}
