#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "argparse.h"
#include "kalman.h"

#define DEBUG

using namespace std;

#define GRID_DIM 1
#define BLOCK_DIM 1

int stonks, days, tests, data_bytes;

double *h_price_data, *h_shuffled_dataset, *h_output;
double *d_price_data, *d_shuffled_dataset, *d_output;
// first entry: initial prices
// following entries: changes in prices
vector<vector<double>> raw_price_data;

__device__ KalmanResult kalman_cuda_update(const int N, const int obs, double *x, double *P,
                                           const double *prices) {
	auto N2 = N * N;
	auto Q = DELTA / (1 - DELTA);
	auto z = prices[obs];

	/////////////
	// predict //
	/////////////

	// state prediction

	auto x_apriori = x;

	// state covariance prediction

	double *P_apriori;
	cudaMalloc(&P_apriori, N2 * sizeof(double));
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			auto idx = i + j * N;
			P_apriori[idx] = P[idx];
			if(i == j)
				P_apriori[idx] += Q;
		}
	}

	///////////////
	// calculate //
	///////////////

	// innovation residual

	double y = z;
	int i_x = 0;
    for(int i_p = 0; i_p < N; i_p++) {
	    if(i_p == obs) {
		    y -= x_apriori[N - 1];
	    } else {
		    y -= prices[i_p] * x_apriori[i_x];
		    i_x++;
	    }
    }

	// innovation covariance

	double s = R;
	double *P_H;
    cudaMalloc(&P_H, N * sizeof(double));
    for(int j = 0; j < N; j++) {
	    for(int i = 0; i < N; i++) {
		    P_H[i] += H[i] * P_apriori[i + j * N];
	    }
    }
    for(int i = 0; i < N; i++) {
	    s += H[i] * P_H[i];
    }

    // kalman gain

    double *K;
    cudaMalloc(&K, N * sizeof(double));
    for(int i = 0; i < N; i++) {
	    K[i] = P_H[i] / s;
    }

	////////////
	// update //
	////////////

	// state update

    for(int i = 0; i < N; i++) {
	    x[i] += y * K[i];
    }

    // covariance update

    double *diff;
    cudaMalloc(&diff, N2 * sizeof(double));
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			diff[i + j * N] = (i == j ? 1 : 0) - K[i] * H[j];
		}
	}
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			double dot = 0;
			for(int k = 0; k < N; k++) {
				dot += diff[i + k * N] * P_apriori[k + j * N];
			}
			P[i + j * N] = dot;
		}
	}

	return KalmanResult{y, s};
}

__device__ void kalman_cuda(int N, int days, const double *prices, double *pnl) {
	auto N2 = N * N;

	double *x;
	cudaMalloc(&x, N * sizeof(double));
	double *P;
	cudaMalloc(&x, N2 * sizeof(double));
	for(int j = 0; j < N; j++) {
		x[j] = 0;
		for(int i = 0; i < N; i++) {
			P[i + j * N] = i == j ? P0 : 0;
		}
	}

	int position = 0;
	double exit_zscore;
	double last_beta;
	double last_port;
	double total_pnl = 0;
	int total_trades = 0;
	for(int d = 0; d < days; d++) {
		auto px = prices[d * N];
		auto py = prices[d * N + 1];
		auto beta = x[0];
		auto intc = x[1];
		auto result = kalman_cuda_update(N, 1, x, P, prices);
		if(d < TRAINING_DAYS) {
			pnl[d] = 0;
			continue;
		}
		int sgn = result.y < 0 ? -1 : 1;
		auto zscore = sgn * result.y * result.y / (result.s - R);

		if(zscore < -ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore + ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
			total_trades++;
			position = 1;
		}
		if(zscore > exit_zscore && position == 1) {
			auto port = py - px * last_beta;
			total_pnl += port - last_port;
			total_trades++;
			position = 0;
		}
		if(zscore > ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore - ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
			total_trades++;
			position = -1;
		}
		if(zscore < exit_zscore && position == -1) {
			auto port = py - px * last_beta;
			total_pnl += last_port - port;
			total_trades++;
			position = 0;
		}

		pnl[d] = total_pnl;
	}
}

__global__ void ExecuteStrategy(double *d_output, double *d_shuffled_dataset,
                                int tests, int stonks, int days) {
	// check bounds
	int test_id = blockDim.x * blockIdx.x + threadIdx.x;
	if(test_id >= tests)
		return;

	// flat array
	int base = test_id * stonks * days;
	kalman_cuda(stonks, days, &d_shuffled_dataset[base], &d_output[base]);
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

		// copy and accumulate shuffled arrays
		for(int i = 0; i < days; i++) {
			int off = stonks * (t * days + i);
			for(int j = 0; j < stonks; j++) {
				if(i == 0) {
					h_shuffled_dataset[off + j] = raw_price_data[i][j];
				} else {
					h_shuffled_dataset[off + j] =
					    raw_price_data[i][j] + h_shuffled_dataset[off - stonks + j];
				}
			}
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
	ExecuteStrategy<<<GRID_DIM, BLOCK_DIM>>>(d_output, d_shuffled_dataset,
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
