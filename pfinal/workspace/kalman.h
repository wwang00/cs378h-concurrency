#ifndef _KALMAN_H_
#define _KALMAN_H_

#define P0 10.0
#define DELTA 0.000001
#define R 10.0

#define TRAINING_DAYS 250

#define ZSCORE_ENTRY 1.0
#define ZSCORE_EXIT 0.0
#define ZSCORE_DELTA 1.0

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
KalmanResult kalman_update(int stonks, int days, double *x, double *P,
                           const double z, const double *H, const double Q);

KalmanResult kalman_cuda_update(const int N, double *x, double *P,
                                const double *prices);
#endif