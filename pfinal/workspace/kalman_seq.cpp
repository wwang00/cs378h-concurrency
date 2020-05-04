#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argparse.h"
#include "cblas.h"
#include "kalman.h"

// #define DEBUG

using namespace std;

int N, days;

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-o"};

int main(int argc, char **argv) {
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	auto ifile = fopen(args["-i"].c_str(), "r");
	auto ofile = fopen(args["-o"].c_str(), "w");
	fscanf(ifile, "%d %d", &N, &days);
	int N2 = N * N;

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
	for(int i = 0; i < N; i++) {
		P[i + i * N] = P0;
	}
	auto Q = DELTA / (1 - DELTA);

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
		auto result = kalman_update(N, days, x, P, z[d], &H[d * N], Q);
		if(d < TRAINING_DAYS)
			continue;
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
