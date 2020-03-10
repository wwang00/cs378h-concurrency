#ifndef _THRUST_DEFS_H_
#define _THRUST_DEFS_H_

typedef typename thrust::host_vector<float>   h_vec;
typedef typename thrust::device_vector<float> d_vec;
typedef typename thrust::device_ptr<float>    d_ptr;
typedef typename thrust::host_vector<int>     h_vec_int;
typedef typename thrust::device_vector<int>   d_vec_int;
typedef typename thrust::device_ptr<int>      d_ptr_int;
typedef typename thrust::device_vector<bool>  d_vec_bool;
typedef typename thrust::device_ptr<bool>     d_ptr_bool;

#endif
