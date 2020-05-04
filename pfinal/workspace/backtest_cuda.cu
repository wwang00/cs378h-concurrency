#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "argparse.h"
#include "kalman.h"

#define DEBUG

using namespace std;

#define BLOCK_DIM 32

int stonks, days, tests, data_bytes;

double *h_price_data, *h_shuffled_dataset, *h_output;
double *d_price_data, *d_shuffled_dataset, *d_output;
vector<vector<double>> raw_price_data;

__device__ void kalman(double *d_price_data, double *d_output, int stonks,
                       int days) {
	int N = stonks;
	int N2 = N * N;

	double x[N];
	double P[N2];
	for(int j = 0; j < N; j++) {
		x[j] = 0;
		for(int i = 0; i < N; i++) {
			P[i + j * N] = i == j ? P0 : 0;
		}
	}
	auto Q = DELTA / (1 - DELTA);

	kalman_update(stonks, days, x, P, d_price_data[0], d_price_data, Q);
    d_output[0] = 777;
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
#ifdef DEBUG
	printf("load_data called......\n");
#endif
	// load data
	ifstream fin(filename);
	fin >> stonks >> days;

	// size of all data
	data_bytes = stonks * days * tests * sizeof(double);

	raw_price_data.resize(days, vector<double>(stonks));
	vector<double> last_prices(stonks);
	for(int i = 0; i < days; i++) {
		for(int j = 0; j < stonks; j++) {
			double price;
			fin >> price;
			if(i == 0) {
				raw_price_data[i][j] = price;
			} else {
				raw_price_data[i][j] = price - last_prices[j];
			}
			last_prices[j] = price;
		}
	}
	fin.close();

#ifdef DEBUG
	for(int i = 0; i < days; i++) {
		for(int j = 0; j < stonks; j++) {
			printf("%.4lf\t", raw_price_data[i][j]);
		}
		printf("\n");
	}
	printf("load_data exited......\n");
#endif
}

void gen_data() {
#ifdef DEBUG
	printf("gen_data called......\n");
#endif
	h_shuffled_dataset = (double *)malloc(data_bytes);
	for(int t = 0; t < tests; t++) {
		// shuffle whole day arrays
		random_shuffle(++raw_price_data.begin(), raw_price_data.end());

		// copy shuffled arrays into 1D array
		for(int i = 0; i < days; i++) {
			int off = stonks * (t * days + i);
			memcpy(&h_shuffled_dataset[off], &raw_price_data[i][0],
			       stonks * sizeof(double));
		}
	}
#ifdef DEBUG
	for(int t = 0; t < tests; t++) {
		printf("test %d\n", t);
		for(int i = 0; i < days; i++) {
			printf("\tday %d\n", i);
			int off = stonks * (t * days + i);
			printf("\t\t");
			for(int j = 0; j < stonks; j++) {
				printf("%.4lf\t", h_shuffled_dataset[off + j]);
			}
			printf("\n");
		}
	}

	printf("gen_data exited......\n");
#endif
}

void backtest() {
	printf("backtest called......\n");

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
	printf("trades\n");
	for(int t = 0; t < tests; t++) {
		printf("test %d\n", t);
		for(int i = 0; i < days; i++) {
			printf("\tday %d\n", i);
			int off = stonks * (t * days + i);
			printf("\t\t");
			for(int j = 0; j < stonks; j++) {
				printf("%.4lf\t", h_output[off + j]);
			}
			printf("\n");
		}
	}

	printf("backtest exited......\n");

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
