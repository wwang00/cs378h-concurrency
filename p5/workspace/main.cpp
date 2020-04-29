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
	printf("enter processor %d of %d\n", R, M);

	init_MPI_structs();

	auto args = parse_args(argc, argv, FLAGS, OPTS);
	THETA = stof(args["-t"]);
	DT = stof(args["-d"]);

	ifstream ifile(args["-i"]);

	// init local tree
	ifile >> N_PTS;
	auto tree = Tree();

	if(R == 0) {
		// read particle start configurations
		for(int p = 0; p < N_PTS; p++) {
			auto particle = Particle();
			int id;
			ifile >> id >> particle.pm.p.x >> particle.pm.p.y >>
			    particle.pm.m >> particle.v.x >> particle.v.y;
			tree.particles[p] = particle;
		}
	}

	if(M == 1) { // sequential
		// do work
		auto t0 = MPI_Wtime();
		for(int s = 0; s < stoi(args["-s"]); s++) {
			printf("%d\n", s);
			tree.build_seq();
			// printf("%s\n", tree.to_string().c_str());
			tree.compute_coms_seq();
			// printf("%s\n", tree.to_string().c_str());
			tree.compute_forces_seq();
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
			for(int s = 0; s < stoi(args["-s"]); s++) {
				printf("%d\n", s);
				tree.build_master();
				tree.compute_coms_master();
				tree.compute_forces_master();
				tree.update_master();
			}
			auto t1 = MPI_Wtime();
			cout << (t1 - t0) << endl;
		} else {
			// do work
			for(int s = 0; s < stoi(args["-s"]); s++) {
				tree.build();
				tree.compute_coms();
				tree.compute_forces();
				tree.update();
			}
		}
	}

	// write results
	if(R == 0) {
		cout << tree.to_string() << endl;

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