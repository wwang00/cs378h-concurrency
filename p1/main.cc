#include <pthread.h>
#include <stdio.h>

#include <chrono>
#include <vector>

#include "argparse.h"
#include "scan.h"

using namespace std;

int N;    // number of input data
int dim;  // FP vector dimension

unordered_set<string> flags{"-s"};
unordered_set<string> opts{"-n", "-i", "-o"};

int main(int argc, char **argv) {
  // parse args
  unordered_map<string, string> args = parse_args(argc, argv, flags, opts);
  if (!args.count("-n") || !args.count("-i") || !args.count("-o")) {
    printf("missing arguments\n");
    return -1;
  }
  // open input file
  auto fin = fopen(args["-i"].c_str(), "r");
  if (fin == nullptr) {
    printf("unable to open input file %s\n", args["-i"].c_str());
    return -1;
  }
  // open output file
  auto fout = fopen(args["-o"].c_str(), "w");
  if (fin == nullptr) {
    printf("unable to open input file %s\n", args["-o"].c_str());
    return -1;
  }
  // read input and do work
  fscanf(fin, "%d\n%d\n", &dim, &N);
  if (dim == 0) {  // integer data
    vector<int> arr(N);
    for (int i = 0; i < N; i++) {
      fscanf(fin, "%d\n", &arr[i]);
    }
    // timed work
    auto t0 = chrono::system_clock::now();
    auto result = pfx_scan_seq<int, add_int>(arr);
    auto t1 = chrono::system_clock::now();
    printf("%lld\n", (t1 - t0) / chrono::milliseconds(1));
    // write to output file
    for (int r : result) {
      fprintf(fout, "%d\n", r);
    }
  } else if (dim > 0) {  // FP vector data
    printf("not implemented\n");
    return -1;
  } else {
    printf("invalid FP vector dim %d\n", dim);
    return -1;
  }
  return 0;
}
