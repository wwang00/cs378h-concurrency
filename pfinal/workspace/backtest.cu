#include <algorithm>
#include <fstream>
#include <iostream>
#define dbg 1
using namespace std;

const int threadsPerBlock = 32;
int data_size, num_tests;
double *h_price_data, *h_shuffled_dataset, *h_output;
double *d_price_data, *d_shuffled_dataset, *d_output;

__device__ void kalman(double *d_price_data, double *d_output, int sz) {
	for(int i = 0; i < sz; i++) {
		if(d_price_data[i] > 4)
			d_output[i] = 1;
		else
			d_output[i] = 0;
	}
	return;
}

__global__ void ExecuteStrategy(double *d_output, double *d_shuffled_dataset,
                                int num_tests, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= num_tests)
		return;
	int base = i * n;
	kalman(&d_shuffled_dataset[base], &d_output[base], n);
	return;
}

void load_data(string filename) {
	// load data
	ifstream fin(filename);
	fin >> data_size;
	h_price_data = (double *)malloc(sizeof(double) * data_size);
	if(dbg)
		cout << "og dataset" << endl;
	for(int i = 0; i < data_size; i++) {
		fin >> h_price_data[i];
		if(dbg)
			cout << h_price_data[i] << " ";
	}
	if(dbg)
		cout << endl;
}

void gen_data() {
	// clone and shuffle
	h_shuffled_dataset =
	    (double *)malloc(data_size * sizeof(double) * num_tests);
	for(int i = 0; i < num_tests; i++) {
		int off = i * data_size;
		memcpy(&h_shuffled_dataset[off], h_price_data,
		       data_size * sizeof(double));
		random_shuffle(&h_shuffled_dataset[off],
		               &h_shuffled_dataset[off] + data_size);
	}

	if(dbg)
		cout << "shuffled datasets" << endl;
	for(int i = 0; i < num_tests; i++) {
		for(int j = 0; j < data_size; j++) {
			int idx = i * data_size + j;
			if(dbg)
				cout << h_shuffled_dataset[idx] << " ";
		}
		if(dbg)
			cout << endl;
	}
}

void backtest() {
	// copy to device
	cudaMalloc(&d_shuffled_dataset, data_size * sizeof(double) * num_tests);
	cudaMemcpy(d_shuffled_dataset, h_shuffled_dataset,
	           data_size * sizeof(double) * num_tests, cudaMemcpyHostToDevice);

	// output array
	h_output = (double *)malloc(data_size * sizeof(double) * num_tests);
	cudaMalloc(&d_output, data_size * sizeof(double) * num_tests);

	// execute strategy
	int blocksPerGrid = (num_tests + threadsPerBlock - 1) / threadsPerBlock;
	ExecuteStrategy<<<blocksPerGrid, threadsPerBlock>>>(
	    d_output, d_shuffled_dataset, num_tests, data_size);
	cudaDeviceSynchronize();

	// device -> host
	cudaMemcpy(h_output, d_output, data_size * sizeof(double) * num_tests,
	           cudaMemcpyDeviceToHost);

	// print trades
	cout << "trades\n";
	for(int i = 0; i < num_tests; i++) {
		for(int j = 0; j < data_size; j++) {
			int off = i * data_size + j;
			cout << h_output[off] << " ";
		}
		cout << endl;
	}
	return;
}

int main() {
	// TODO: param parser
	string filename = "data.txt";
	num_tests = 10;
	load_data(filename);
	gen_data();
	backtest();
	return 0;
}
