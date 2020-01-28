#ifndef _SCAN_H_
#define _SCAN_H_

#include <vector>

int add_int(const int &x, const int &y);
std::vector<float> add_vector_float(const std::vector<float> &x,
                                    const std::vector<float> &y);

template <class T, T (*scan_op)(const T &, const T &)>
std::vector<T> pfx_scan_seq(const std::vector<T> &arr);

#endif