#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>

using namespace std;

int main() {
	srand(5208);
	for(int n = 200; n <= 500; n += 100) {
		char buf[100];
		sprintf(buf, "nb-%d.txt", n);
		string fname(buf);
		ofstream ofile(fname);
		ofile << std::scientific;
		ofile << n << endl;
		for(int p = 0; p < n; p++) {
			ofile << p << "\t" << ((double)rand() / (double)RAND_MAX) * 4.0
			      << "\t" << ((double)rand() / (double)RAND_MAX) * 4.0 << "\t"
			      << ((double)rand() / (double)RAND_MAX) * 4.0 << "\t" << 0.0
			      << "\t" << 0.0 << endl;
		}
        ofile.close();
	}
}