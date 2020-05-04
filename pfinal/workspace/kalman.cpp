#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas.h"
#include "kalman.h"

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
KalmanResult kalman_update(int stonks, int days, double *x, double *P,
                           const double z, const double *H, const double Q) {
	int N = stonks;
	int N2 = N * N;

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
			P_apriori[idx] = P[idx];
			if(i == j)
				P_apriori[idx] += Q;
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
	for(int j = 0; j < N; j++) {
		for(int i = 0; i < N; i++) {
			diff[i + j * N] = i == j ? 1 : 0;
		}
	}
	cblas_dger(CblasColMajor, N, N, -1, K, 1, H, 1, diff, N);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, diff, N,
	            P_apriori, N, 0, P, N);

	return KalmanResult{y, s};
}