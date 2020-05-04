#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argparse.h"
#include "cblas.h"

// #define DEBUG

using namespace std;

int N, N2;
int days;

static constexpr double P0 = 10;
static constexpr double DELTA = 0.000001;
static constexpr double R = 10;

static constexpr int TRAINING_DAYS = 250;

static constexpr double ZSCORE_ENTRY = 1.0;
static constexpr double ZSCORE_EXIT = 0.0;
static constexpr double ZSCORE_DELTA = 1.0;

// TODO generalize
static constexpr double ZEROS[2] = {0};
static constexpr double ONES[4] = {1, 0, 0, 1};

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

	// innovation covariance

	double s;
	double P_H[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1, P_apriori, N, H, 1, 0,
	            P_H, 1);
	s = cblas_ddot(N, H, 1, P_H, 1) + R;

	// kalman gain

	double K[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1 / s, P_apriori, N, H, 1, 0,
	            K, 1);

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

	return KalmanResult{y, s};
}

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-o"};

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	auto ifile = fopen(args["-i"].c_str(), "r");
	auto ofile = fopen(args["-o"].c_str(), "w");
	fscanf(ifile, "%d %d", &N, &days);
	N2 = N * N;

	// read price series

	auto z = (double *)calloc(days, sizeof(double));
	auto H = (double *)calloc(days * N, sizeof(double));
	for(int d = 0; d < days; d++) {
		double x, y;
		fscanf(ifile, "%lf\t%lf", &x, &y);
		H[d * N] = x;
		H[d * N + 1] = 1;
		z[d] = y;
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
	for(int d = 0; d < days; d++) {
		auto px = H[d * N];
		auto py = z[d];
		auto beta = x[0];
		auto intc = x[1];
		auto result = kalman_update(x, P, z[d], &H[d * N], Q, R);
		if(d < TRAINING_DAYS)
			continue;
		int sgn = result.y < 0 ? -1 : 1;
		auto zscore = sgn * result.y * result.y / (result.s - R);

		if(zscore < -ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore + ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
#ifdef DEBUG
			printf("BUY-I\t%d\tcurr: %.2lf\n", d, last_port);
#endif
			total_trades++;
			position = 1;
		}
		if(zscore > exit_zscore && position == 1) {
			auto port = py - px * last_beta;
#ifdef DEBUG
			printf("SEL-O\t%d\tcurr: %.2lf\tprev: %.2lf\n", d, port, last_port);
#endif
			total_pnl += port - last_port;
			total_trades++;
			position = 0;
		}
		if(zscore > ZSCORE_ENTRY && position == 0) {
			exit_zscore = zscore - ZSCORE_DELTA;
			last_beta = beta;
			last_port = py - px * beta;
#ifdef DEBUG
			printf("SEL-I\t%d\tcurr: %.2lf\n", d, last_port);
#endif
			total_trades++;
			position = -1;
		}
		if(zscore < exit_zscore && position == -1) {
			auto port = py - px * last_beta;
#ifdef DEBUG
			printf("BUY-O\t%d\tcurr: %.2lf\tprev: %.2lf\n", d, port, last_port);
#endif
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
	// printf("%.4lf\n", total_pnl);

	fclose(ifile);
	fclose(ofile);

	return 0;
}
