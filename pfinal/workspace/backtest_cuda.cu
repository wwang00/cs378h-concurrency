#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "argparse.h"
#include "kalman.h"

//#define DEBUG

using namespace std;

#define GRID_DIM tests
#define BLOCK_DIM 32

int stonks, days, tests;
int data_bytes, out_bytes, scratch_bytes;

double *h_prices, *h_pnl;
double *d_prices, *d_pnl;
double *scratch; // workspace for cuda routine

// first entry: initial prices
// following entries: changes in prices
vector<vector<double>> raw_price_data;

__device__ KalmanResult kalman_cuda_update(const int N, const int obs, double *x, double *P,
                                           const double *prices, double *scratch) {
	auto tid = threadIdx.x;

	auto N2 = N * N;
	auto Q = DELTA / (1 - DELTA);
	auto z = prices[obs];
	auto H = scratch + 0;
	int i_H = 0;
	for(int i = 0; i < N; i++) {
		if(i == obs)
			continue;
		H[i_H] = prices[i];
		i_H++;
	}
	H[i_H] = 1;


	/////////////
	// predict //
	/////////////

	// state prediction

	auto x_apriori = x;

	// state covariance prediction

	auto P_apriori = H + N;
	for(int j = tid; j < N; j += BLOCK_DIM) {
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
    for(int i = 0; i < N; i++) {
	    y -= H[i] * x_apriori[i];
    }

	// innovation covariance

	double s = R;
	auto P_H = P_apriori + N2;
    for(int i = tid; i < N; i += BLOCK_DIM) {
	    double dot = 0;
	    for(int j = 0; j < N; j++) {
		    dot += P_apriori[i + j * N] * H[j];
	    }
	    P_H[i] = dot;
    }
    for(int i = 0; i < N; i++) {
	    s += H[i] * P_H[i];
    }

    // kalman gain

    auto K = P_H + N;
    for(int i = tid; i < N; i += BLOCK_DIM) {
	    K[i] = P_H[i] / s;
    }

	////////////
	// update //
	////////////

	// state update

    for(int i = tid; i < N; i += BLOCK_DIM) {
	    x[i] += y * K[i];
    }

    // covariance update

    auto diff = K + N;
	for(int j = tid; j < N; j += BLOCK_DIM) {
		for(int i = 0; i < N; i++) {
			diff[i + j * N] = (i == j ? 1 : 0) - K[i] * H[j];
		}
	}
	for(int j = tid; j < N; j += BLOCK_DIM) {
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

__device__ void kalman_cuda(int N, int days, const double *prices, double *pnl, double *scratch) {
#ifdef DEBUG
	printf("kalman_cuda called......\n");
#endif

	auto N2 = N * N;

	auto *x = scratch + 0;
	auto *P = x + N;
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
		//auto intc = x[1];
		auto result = kalman_cuda_update(N, 1, x, P, &prices[d * N], P + N2);
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
#ifdef DEBUG
	printf("kalman_cuda exited......\n");
#endif
}

__global__ void ExecuteStrategy(int tests, int stonks, int days, double *prices, double *pnl, double *scratch) {
#ifdef DEBUG
	printf("ExecuteStrategy called......\n");
#endif

    // check bounds
	int test_id = blockIdx.x;
    if(test_id >= tests) return;

#ifdef DEBUG
    printf("Hello from block %d, thread %d, test_id %d of %d\n", blockIdx.x, threadIdx.x, test_id, tests);
#endif

    // flat array
    int data_base = test_id * stonks * days;
    int out_base = test_id * days;
    int scratch_base = test_id * (3 * stonks * stonks + 4 * stonks);
	kalman_cuda(stonks, days, &prices[data_base], &pnl[out_base], &scratch[scratch_base]);

#ifdef DEBUG
	printf("ExecuteStrategy exited......\n");
#endif
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
	h_prices = (double *)malloc(data_bytes);
	for(int t = 0; t < tests; t++) {
		// shuffle whole day arrays
		srand(time(0));
		//random_shuffle(++raw_price_data.begin(), raw_price_data.end());

		// copy and accumulate shuffled arrays
		for(int i = 0; i < days; i++) {
			int off = stonks * (t * days + i);
			for(int j = 0; j < stonks; j++) {
				if(i == 0) {
					h_prices[off + j] = raw_price_data[i][j];
				} else {
					h_prices[off + j] =
					    raw_price_data[i][j] + h_prices[off - stonks + j];
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
				printf("%.4lf\t", h_prices[off + j]);
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
	cudaMalloc(&d_prices, data_bytes);
	cudaMemcpy(d_prices, h_prices, data_bytes,
	           cudaMemcpyHostToDevice);

	out_bytes = tests * days * sizeof(double);
	// output array
	h_pnl = (double *)malloc(out_bytes);
	cudaMalloc(&d_pnl, out_bytes);

	scratch_bytes = tests * (3 * stonks * stonks + 4 * stonks) * sizeof(double);
	cudaMalloc(&scratch, scratch_bytes);

	// execute strategy
	ExecuteStrategy<<<GRID_DIM, BLOCK_DIM>>>(tests, stonks, days, d_prices, d_pnl, scratch);
	cudaDeviceSynchronize();

	// device -> host
	cudaMemcpy(h_pnl, d_pnl, out_bytes, cudaMemcpyDeviceToHost);

	for(int t = 0; t < tests; t++) {
		int off = t * days;
		printf("Test %d - total P&L: %.4lf\n", t, h_pnl[off + days - 1]);
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
