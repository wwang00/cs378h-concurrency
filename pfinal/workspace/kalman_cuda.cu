#include "kalman.h"

__device__ KalmanResult kalman_cuda_update(const int N, double *x, double *P,
                                           const double *prices) {
	int N2 = N * N;

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
    for(int i = 0; i < N; i++) {
        y -= H[i] * x_apriori[i];
    }

	// innovation covariance

	double s;
	double *P_H;
    cudaMalloc(&P_H, N * sizeof(double));

	// cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1, P_apriori, N, H, 1, 0,
	//             P_H, 1);
	// s = cblas_ddot(N, H, 1, P_H, 1) + R;

	// // kalman gain

	// double K[N];
	// cblas_dgemv(CblasColMajor, CblasNoTrans, N, N, 1 / s, P_apriori, N, H, 1, 0,
	//             K, 1);

	// ////////////
	// // update //
	// ////////////

	// // state update

	// cblas_daxpy(N, y, K, 1, x, 1);

	// // covariance update

	// double diff[N2];
	// for(int j = 0; j < N; j++) {
	// 	for(int i = 0; i < N; i++) {
	// 		diff[i + j * N] = i == j ? 1 : 0;
	// 	}
	// }
	// cblas_dger(CblasColMajor, N, N, -1, K, 1, H, 1, diff, N);
	// cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, diff, N,
	//             P_apriori, N, 0, P, N);

	return KalmanResult{y, s};
}