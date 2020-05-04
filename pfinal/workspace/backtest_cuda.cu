#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "argparse.h"

#define DEBUG

using namespace std;

#define BLOCK_DIM 32

int stonks, days, tests, data_bytes;

double *h_price_data, *h_shuffled_dataset, *h_output;
double *d_price_data, *d_shuffled_dataset, *d_output;
vector<vector<double>> raw_price_data;

__device__ void kalman(double *d_price_data, double *d_output, int stonks,
                       int days) {
	for(int i = 0; i < days; i++) {
		for(int j = 0; j < stonks; j++) {
			int off = i * stonks + j;
			d_output[off] = d_price_data[off] < 3;
		}
	}
	return;
}

__global__ void ExecuteStrategy(double *d_output, double *d_shuffled_dataset,
                                int tests, int stonks, int days) {
	// check bounds
	int test_id = blockDim.x * blockIdx.x + threadIdx.x;
	if(test_id >= tests)
		return;

	// flat array
	int base = test_id * stonks * days;
	kalman(&d_shuffled_dataset[base], &d_output[base], stonks, days);
	return;
}

void load_data(string filename) {
	// load data
	ifstream fin(filename);
	fin >> stonks >> days;

	// size of all data
	data_bytes = stonks * days * tests * sizeof(double);

	raw_price_data.resize(days, vector<double>(stonks));
	for(int i = 0; i < days; i++) {
		for(int j = 0; j < stonks; j++) {
			double price;
			fin >> price;
			if(i == 0) {
				raw_price_data[i][j] = price;
			} else {
				raw_price_data[i][j] = price - raw_price_data[i - 1][j];
			}
		}
	}

#ifdef DEBUG
	for(int i = 0; i < days; i++) {
		for(int j = 0; j < stonks; j++) {
			cout << raw_price_data[i][j] << " ";
		}
		cout << endl;
	}
#endif
}

void gen_data() {
	h_shuffled_dataset = (double *)malloc(data_bytes);
	for(int t = 0; t < tests; t++) {
		// shuffle whole day arrays
		random_shuffle(++raw_price_data.begin(), raw_price_data.end());

		// copy shuffled arrays into 1D array
		for(int i = 0; i < days; i++) {
			int off = stonks * (t * days + i);
			memcpy(&h_shuffled_dataset[off], &raw_price_data[i][0],
			       stonks * sizeof(double));

#ifdef DEBUG
			for(int j = 0; j < stonks; j++) {
				cout << off << " " << h_shuffled_dataset[off + j] << endl;
			}
#endif
#ifdef DEBUG
			cout << endl;
#endif
		}
#ifdef DEBUG
		cout << endl;
#endif
	}
}

void backtest() {
	// copy to device
	cudaMalloc(&d_shuffled_dataset, data_bytes);
	cudaMemcpy(d_shuffled_dataset, h_shuffled_dataset, data_bytes,
	           cudaMemcpyHostToDevice);

	// output array
	h_output = (double *)malloc(data_bytes);
	cudaMalloc(&d_output, data_bytes);

	// execute strategy
	int grid_dim = (tests + BLOCK_DIM - 1) / BLOCK_DIM;
	ExecuteStrategy<<<grid_dim, BLOCK_DIM>>>(d_output, d_shuffled_dataset,
	                                         tests, stonks, days);
	cudaDeviceSynchronize();

	// device -> host
	cudaMemcpy(h_output, d_output, data_bytes, cudaMemcpyDeviceToHost);

	// print trades
	cout << "trades" << endl;
	for(int t = 0; t < tests; t++) {
		for(int i = 0; i < days; i++) {
			for(int j = 0; j < stonks; j++) {
				int off = stonks * (t * days + i) + j;
				// h_shuffled_dataset[off] = raw_price_data[i][j];
				cout << off << " " << h_output[off] << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	return;
}

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-t"};

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);
	auto filename = args["-i"];
	tests = stoi(args["-t"]);
	load_data(filename);
	gen_data();
	backtest();
	return 0;
}
