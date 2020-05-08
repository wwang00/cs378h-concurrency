#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cblas.h"
#include "kalman.h"

KalmanResult kalman_update(int N, int obs, double *x, double *P,
                           const double *prices) {
	auto N2 = N * N;
	auto Q = DELTA / (1 - DELTA);
	auto z = prices[obs];
    double H[N];
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

	double y = z;
	for(int i = 0; i < N; i++) {
		y -= H[i] * x_apriori[i];
	}

	// innovation covariance

	double s = R;
	double P_H[N];
	for(int i = 0; i < N; i++) {
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

	double K[N];
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

	double diff[N2];
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