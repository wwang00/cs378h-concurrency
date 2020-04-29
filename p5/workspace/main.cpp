#include <algorithm>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>

#include "argparse.h"
#include "datatypes.h"
#include "tree.h"

using namespace std;

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-o", "-s", "-t", "-d"};

double random(double m) { return (double)rand() / (double)RAND_MAX * m; }

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &M);
	MPI_Comm_rank(MPI_COMM_WORLD, &R);
	// printf("enter processor %d of %d\n", R, M);

	init_MPI_structs();

	auto args = parse_args(argc, argv, FLAGS, OPTS);
	THETA = stof(args["-t"]);
	DT = stof(args["-d"]);

	ifstream ifile(args["-i"]);

	// init local tree
	ifile >> N_PTS;
	auto tree = Tree();

	// read particle start configurations
	for(int p = 0; p < N_PTS; p++) {
		int id;
		double x, y, m, vx, vy;
		ifile >> id >> x >> y >> m >> vx >> vy;
		tree.particles[p] =
		    Particle{PointMass{Point{x, y}, m}, Point{vx, vy}, Point()};
	}

	auto iters = stoi(args["-s"]);

	if(M == 1) { // sequential
		// do work
		auto t0 = MPI_Wtime();
		for(int s = 0; s < iters; s++) {
			// printf("%d\n", s);
			tree.build();
			// printf("%s\n", tree.to_string().c_str());
			tree.update_seq();
			// printf("%s\n", tree.to_string().c_str());
		}
		auto t1 = MPI_Wtime();
		cout << (t1 - t0) << endl;
	} else { // parallel
		if(R == 0) {
			// do work
			auto t0 = MPI_Wtime();
			for(int s = 0; s < iters; s++) {
				// printf("%d\n", s);
				tree.build();
				// printf("%s\n", tree.to_string().c_str());
				tree.update();
				// printf("%s\n", tree.to_string().c_str());
			}
			auto t1 = MPI_Wtime();
			cout << (t1 - t0) << endl;
		} else {
			// do work
			for(int s = 0; s < iters; s++) {
				tree.build();
				tree.update();
			}
		}
	}

	// write results
	if(R == 0) {
		// cout << tree.to_string() << endl;

		ofstream ofile(args["-o"]);
		ofile << std::scientific;
		ofile << N_PTS << endl;

		for(int p = 0; p < N_PTS; p++) {
			auto particle = tree.particles[p];
			ofile << p << "\t" << particle.pm.p.x << "\t" << particle.pm.p.y
			      << "\t" << particle.pm.m << "\t" << particle.v.x << "\t"
			      << particle.v.y << endl;
		}

		ofile.close();
	}

	ifile.close();

	// Finalize the MPI environment.
	MPI_Finalize();
}