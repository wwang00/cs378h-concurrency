#include "scan.h"

using namespace std;

int add_int(const int &x, const int &y) { return x + y; }

vector<float> add_vector_float(const vector<float> &x, const vector<float> &y) {
  vector<float> result;
  for (int i = 0; i < min(x.size(), y.size()); i++) {
    result.push_back(x[i] + y[i]);
  }
  return result;
}

template <class T, T (*scan_op)(const T &, const T &)>
vector<T> pfx_scan_seq(const vector<T> &arr) {
  T acc{};
  vector<T> result{acc};
  for (int i = 1; i < arr.size(); i++) {
    acc = scan_op(acc, arr[i]);
    result.push_back(acc);
  }
  return result;
}

template vector<int> pfx_scan_seq<int, add_int>(const vector<int> &arr);
template vector<vector<float>> pfx_scan_seq<vector<float>, add_vector_float>(
    const vector<vector<float>> &arr);