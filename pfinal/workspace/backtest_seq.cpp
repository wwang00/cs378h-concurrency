#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <vector>

#include "argparse.h"
#include "kalman.h"

//#define DEBUG

using namespace std;

int stonks, days, tests;
int data_bytes, out_bytes, scratch_bytes;

double *h_prices, *h_pnl;
double *scratch; // workspace for cuda routine

// first entry: initial prices
// following entries: changes in prices
vector<vector<double>> raw_price_data;


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
		random_shuffle(++raw_price_data.begin(), raw_price_data.end());

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

	out_bytes = tests * days * sizeof(double);
	// output array
	h_pnl = (double *)malloc(out_bytes);

	for(int t = 0; t < tests; t++) {
		// flat array
		int data_base = t * stonks * days;
		int out_base = t * days;
		// kalman_cuda(stonks, days, &prices[data_base], &pnl[out_base], &scratch[scratch_base]);
	}

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
