#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argparse.h"
#include "kalman.h"

//#define DEBUG

using namespace std;

#define SCRATCH_SIZE (3 * N * N + 4 * N)

int N, days, grid_dim, block_dim;

__device__ KalmanResult kalman_cuda_update(int N, int obs,
                                           double *x, double *P,
                                           const double *prices,
                                           double *scratch) {
	auto tid = threadIdx.x;
	auto block_dim = blockDim.x;

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

#ifdef DEBUG
	if(obs == 1 && tid == 0) {
		printf("H\n");
		for(int i = 0; i < N; i++) {
			printf("%.2lf\t", H[i]);
		}
		printf("\n");
	}
#endif

	/////////////
	// predict //
	/////////////

	// state prediction

	auto x_apriori = x;

#ifdef DEBUG
	if(obs == 1 && tid == 0) {
		printf("x_apriori\n");
		for(int i = 0; i < N; i++) {
			printf("%.2lf\t", x_apriori[i]);
		}
		printf("\n");
	}
#endif

	// state covariance prediction

	auto P_apriori = H + N;
	for(int j = tid; j < N; j += block_dim) {
		for(int i = 0; i < N; i++) {
			auto idx = i + j * N;
			P_apriori[idx] = P[idx];
			if(i == j)
				P_apriori[idx] += Q;
		}
	}

#ifdef DEBUG
    if(obs == 1 && tid == 0) {
        printf("P_apriori\n");
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                printf("%.2lf\t", P_apriori[i + j * N]);
            }
            printf("\n");
        }
    }
#endif

	///////////////
	// calculate //
	///////////////

	// innovation residual

	double y = z;
	for(int i = 0; i < N; i++) {
		y -= H[i] * x_apriori[i];
	}

#ifdef DEBUG
	if(obs == 1 && tid == 0) {
		printf("y %.2lf\n", y);
	}
#endif

	// innovation covariance

	double s = R;
	auto P_H = P_apriori + N2;
	for(int i = tid; i < N; i += block_dim) {
		double dot = 0;
		for(int j = 0; j < N; j++) {
			dot += P_apriori[i + j * N] * H[j];
		}
		P_H[i] = dot;
	}
	for(int i = 0; i < N; i++) {
		s += H[i] * P_H[i];
	}

#ifdef DEBUG
	if(obs == 1 && tid == 0) {
		printf("s %.2lf\n", s);
	}
#endif

	// kalman gain

	auto K = P_H + N;
	for(int i = tid; i < N; i += block_dim) {
		K[i] = P_H[i] / s;
	}

	////////////
	// update //
	////////////

	// state update

	for(int i = tid; i < N; i += block_dim) {
		x[i] += y * K[i];
	}

	// covariance update

	auto diff = K + N;
	for(int j = tid; j < N; j += block_dim) {
		for(int i = 0; i < N; i++) {
			diff[i + j * N] = (i == j ? 1 : 0) - K[i] * H[j];
		}
	}
	for(int j = tid; j < N; j += block_dim) {
		for(int i = 0; i < N; i++) {
			double dot = 0;
			for(int k = 0; k < N; k++) {
				dot += diff[i + k * N] * P_apriori[k + j * N];
			}
			P[i + j * N] = dot;
		}
	}

#ifdef DEBUG
    if(obs == 1 && tid == 0) {
        printf("\n\n");
    }
#endif

	return KalmanResult{y, s};
}

__global__ void kalman_cuda(int N, double *d_prices, char *d_positions,
                            double *d_scratch) {
#ifdef DEBUG
	printf("kalman_cuda called......\n");
#endif

	auto N2 = N * N;
	auto grid_dim = gridDim.x;

	for(int obs = blockIdx.x; obs < N; obs += grid_dim) {
		auto scratch_off = obs * SCRATCH_SIZE;
		auto x = d_scratch + scratch_off;
		auto P = x + N;
		auto prices = d_prices;
		auto scratch = P + N2;
		auto result = kalman_cuda_update(N, obs, x, P, prices, scratch);
	}

#ifdef DEBUG
	printf("kalman_cuda exited......\n");
#endif
}

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-o", "-g", "-b"};

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	auto ifile = fopen(args["-i"].c_str(), "r");
	auto ofile = fopen(args["-o"].c_str(), "w");
	fscanf(ifile, "%d %d", &N, &days);

	// read price series

	auto h_prices = (double *)calloc(days * N, sizeof(double));
	for(int d = 0; d < days; d++) {
		double price;
		for(int i = 0; i < N; i++) {
			fscanf(ifile, "%lf", &price);
			h_prices[d * N + i] = price;
		}
	}

	// init memory regions
	
	double *d_prices;
	cudaMalloc(&d_prices, N * sizeof(double));
	char *d_positions;
	cudaMalloc(&d_positions, N);
	double *d_scratch;
	cudaMalloc(&d_scratch, N * SCRATCH_SIZE * sizeof(double));

	auto h_scratch = (double *)malloc(N * SCRATCH_SIZE * sizeof(double));
	for(int obs = 0; obs < N; obs++) {
		auto scratch_off = obs * SCRATCH_SIZE;
		auto x = h_scratch + scratch_off;
		auto P = x + N;
		for(int i = 0; i < N; i++) {
			x[i] = 0;
			P[i + i * N] = P0;
		}
	}
	cudaMemcpy(d_scratch, h_scratch, N * SCRATCH_SIZE * sizeof(double),
	           cudaMemcpyHostToDevice);

	// do kalman

	grid_dim = stoi(args["-g"]);
	block_dim = stoi(args["-b"]);

	for(int d = 0; d < days; d++) {
		printf("day %d\n", d);

		auto t0 = chrono::system_clock::now();
		long elapsed;

		cudaMemcpy(d_prices, h_prices + d * N, N * sizeof(double),
		           cudaMemcpyHostToDevice);

		auto t1 = chrono::system_clock::now();
		elapsed = (long)((t1 - t0) / chrono::microseconds(1));
		printf("memcpy %ld\n", elapsed);

		kalman_cuda<<< grid_dim, block_dim >>>(N, d_prices, d_positions,
		                                       d_scratch);
		cudaDeviceSynchronize();

		auto t2 = chrono::system_clock::now();
		elapsed = (long)((t2 - t1) / chrono::microseconds(1));
		printf("kalman %ld\n", elapsed);
	}

	// print

	// printf("total P&L: %.4lf\n", total_pnl);
	// printf("%.4lf\n", total_pnl);

	fclose(ifile);
	fclose(ofile);

	return 0;
}
