#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argparse.h"
#include "cblas.h"

using namespace std;

static constexpr int N = 2;
static constexpr int N2 = N * N;

static constexpr double P0 = 10;
static constexpr double DELTA = 0.000001;
static constexpr double R = 10;

static constexpr int TRAINING = 250;
static constexpr double ZSCORE_ENTRY = 1.0;
static constexpr double ZSCORE_EXIT = 0.0;
static constexpr double ZSCORE_DELTA = 1.0;

static constexpr double ZEROS[N] = {0};
static constexpr double ONES[N2] = {1, 0, 0, 1};

/**
 * result of kalman update
 *
 * y: innovation residual
 * s: innovation covariance
 */
struct KalmanResult {
	double y;
	double s;
};

/**
 * perform one kalman update iteration
 *
 * <params>
 * n: dimension of observation model
 * x: previous hidden state estimate
 * P: previous covariance matrix estimate
 * z: current observable variable observation
 * H: current observation matrix
 * Q: process noise covariance matrix
 * R: observation noise covariance matrix
 *
 * <effects>
 * x: current hidden state estimate
 * P: current covariance matrix estimate
 */
KalmanResult kalman_update(double *x, double *P, const double z,
                           const double *H, const double *Q, const double R) {
	// printf("kalman_update called......\n");
	// printf("x: %.4lf, %.4lf\n", x[0], x[1]);
	// printf("z: %.4lf\n", z);
	// printf("H: %.4lf, %.4lf\n", H[0], H[1]);

	/////////////
	// predict //
	/////////////

	// state prediction

	double x_apriori[N];
	cblas_dcopy(N, x, 1, x_apriori, 1);

	// state covariance prediction

	double P_apriori[N2];
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			auto idx = i + j * N;
			P_apriori[idx] = P[idx] + Q[idx];
		}
	}

	///////////////
	// calculate //
	///////////////

	// innovation residual

	double y = z - cblas_ddot(N, H, 1, x_apriori, 1);
	// printf("y: %.4lf\n", y);

	// innovation covariance

	double s;
	double P_H[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1, P_apriori, N, H, 1, 0,
	            P_H, 1);
	s = cblas_ddot(N, H, 1, P_H, 1) + R;
	// printf("s: %.4lf\n", s);

	// kalman gain

	double K[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1 / s, P_apriori, N, H, 1, 0,
	            K, 1);
	// printf("K: %.4lf, %.4lf\n", K[0], K[1]);

	////////////
	// update //
	////////////

	// state update

	cblas_dcopy(N, x_apriori, 1, x, 1);
	cblas_daxpy(N, y, K, 1, x, 1);

	// covariance update

	double diff[N2];
	cblas_dcopy(N2, ONES, 1, diff, 1);
	cblas_dger(CblasColMajor, N, N, -1, K, 1, H, 1, diff, N);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, diff, N,
	            P_apriori, N, 0, P, N);

	// printf("kalman_update exited......\n");
	return KalmanResult{y, s};
}

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i1", "-i2", "-o", "-p"};

int N_PTS;

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	auto ifile1 = fopen(args["-i1"].c_str(), "r");
	auto ifile2 = fopen(args["-i2"].c_str(), "r");
	auto ofile = fopen(args["-o"].c_str(), "w");
	N_PTS = stoi(args["-p"]);

	// read price series

	auto z = (double *)calloc(N_PTS, sizeof(double));
	auto H = (double *)calloc(N_PTS * N, sizeof(double));
	for(int i = 0; i < N_PTS; i++) {
		double x, y;
		fscanf(ifile1, "%lf", &x);
		fscanf(ifile2, "%lf", &y);
		H[i * N] = x;
		H[i * N + 1] = 1;
		z[i] = y;
	}

	// do kalman

	auto x = (double *)calloc(N, sizeof(double));
	auto P = (double *)calloc(N2, sizeof(double));
	auto Q = (double *)calloc(N2, sizeof(double));
	auto q = DELTA / (1 - DELTA);
	for(int i = 0; i < N; i++) {
		P[i + i * N] = P0;
		Q[i + i * N] = q;
	}

	int position = 0;
	double exit_zscore;
	double last_beta;
	double last_port;
	double total_pnl = 0;
	int total_trades = 0;
	for(int p = 0; p < N_PTS; p++) {
		auto px = H[p * N];
		auto py = z[p];
		auto beta = x[0];
		auto intc = x[1];
		auto result = kalman_update(x, P, z[p], &H[p * N], Q, R);
		if(p < TRAINING)
			continue;
        int sgn = result.y < 0 ? -1 : 1;
		auto zscore = sgn * result.y * result.y / (result.s - R);
		// printf("zscore %.4lf\n", zscore);

		if(zscore < -ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore + ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
			printf("BUY-I\t%d\tcurr: %.2lf\n", p, last_port);
			total_trades++;
			position = 1;
		}
		if(zscore > exit_zscore && position == 1) {
			auto port = py - px * last_beta;
			printf("SEL-O\t%d\tcurr: %.2lf\tprev: %.2lf\n", p, port, last_port);
			total_pnl += port - last_port;
			total_trades++;
			position = 0;
		}
		if(zscore > ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore - ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
			printf("SEL-I\t%d\tcurr: %.2lf\n", p, last_port);
			total_trades++;
			position = -1;
		}
		if(zscore < exit_zscore && position == -1) {
			auto port = py - px * last_beta;
			printf("BUY-O\t%d\tcurr: %.2lf\tprev: %.2lf\n", p, port, last_port);
			total_pnl += last_port - port;
			total_trades++;
			position = 0;
		}

		fprintf(ofile, "%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", beta, intc,
		        result.y, result.s, total_pnl);
	}

	// print

	printf("trades: %d\n", total_trades);
	printf("coefficient: %.4lf, intercept: %.4lf\n", x[0], x[1]);
	printf("total P&L: %.4lf\n", total_pnl);

	fclose(ifile1);
	fclose(ifile2);
	fclose(ofile);

	return 0;
}
