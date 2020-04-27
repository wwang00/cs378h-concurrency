#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argparse.h"
#include "cblas.h"

using namespace std;

static constexpr int N = 2;
static constexpr int N2 = N * N;

static constexpr double DELTA = 0.0001;
static constexpr double MU_INV = DELTA / (1 - DELTA);
static constexpr double R = 0.001;

static constexpr double ZEROS[N] = {0};
static constexpr double ONES[N2] = {1, 0, 0, 1};

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
void kalman_update(double *x, double *P, const double z, const double *H,
                   const double *Q, const double R) {
	printf("kalman_update called......\n");
	printf("x: %.4lf, %.4lf\n", x[0], x[1]);
	printf("z: %.4lf\n", z);
	printf("H: %.4lf, %.4lf\n", H[0], H[1]);

	/////////////
	// predict //
	/////////////

	// state prediction

	double x_apriori[N];
	cblas_dcopy(N, x, 1, x_apriori, 1);

	// state covariance prediction

	double P_apriori[N2];
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			auto idx = i + j * N;
			P_apriori[idx] = P[idx] + Q[idx];
		}
	}

	///////////////
	// calculate //
	///////////////

	// innovation residual

	double y = z - cblas_ddot(N, H, 1, x, 1);
	printf("y: %.4lf\n", y);

	// innovation covariance

	double s;
	double P_H[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1, P_apriori, N, H, 1, 0,
	            P_H, 1);
	s = cblas_ddot(N, H, 1, P_H, 1) + R;
	printf("s: %.4lf\n", s);

	// kalman gain

	double K[N];
	cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1 / s, P_apriori, N, H, 1, 0,
	            K, 1);
	printf("K: %.4lf, %.4lf\n", K[0], K[1]);

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

	printf("kalman_update exited......\n");
}

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i"};

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	auto ifile = fopen(args["-i"].c_str(), "r");

	int n_pts;
	fscanf(ifile, "%d", &n_pts);

	// read price series

	auto z = (double *)calloc(n_pts, sizeof(double));
	auto H = (double *)calloc(n_pts * N, sizeof(double));
	for(int i = 0; i < n_pts; i++) {
		double x, y;
		fscanf(ifile, "%lf,%lf", &x, &y);
		H[i * N] = x;
		H[i * N + 1] = 1;
		z[i] = y;
	}

	// do kalman

	auto x = (double *)calloc(N, sizeof(double));
	auto P = (double *)calloc(N2, sizeof(double));
	auto Q = (double *)calloc(N2, sizeof(double));
	for(int i = 0; i < N; i++) {
		Q[i + i * N] = MU_INV;
	}

	for(int p = 0; p < n_pts; p++)
		kalman_update(x, P, z[p], &H[p * N], Q, R);

	// print

	printf("coefficient: %.4lf, intercept: %.4lf\n", x[0], x[1]);

	return 0;
}
