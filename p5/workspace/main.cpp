#include <algorithm>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>

#include "argparse.h"
#include "tree.h"

using namespace std;

unordered_set<string> FLAGS{"-n"};
unordered_set<string> OPTS{"-i", "-o", "-s", "-t", "-d"};

int M;
int R;
float theta;
float dt;
int n_particles;

float random(float m) { return (float)rand() / (float)RAND_MAX * m; }

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &M);
	MPI_Comm_rank(MPI_COMM_WORLD, &R);

	auto args = parse_args(argc, argv, FLAGS, OPTS);
	theta = stof(args["-t"]);
	dt = stof(args["-d"]);

	ifstream ifile(args["-i"]);

	// init tree
	ifile >> n_particles;
	auto tree = Tree();

	// read particle start configurations
	for(int p = 0; p < n_particles; p++) {
		auto particle = Particle();
		int id;
		ifile >> id >> particle.pm.p.x >> particle.pm.p.y >> particle.pm.m >>
		    particle.vel.x >> particle.vel.y;
		tree.particles[p] = particle;
	}

	// do work
	auto t0 = MPI_Wtime();
	if(args.count("-n")) {
		for(int s = 0; s < stoi(args["-s"]); s++) {
			tree.build();
			tree.compute_coms_seq();
			tree.compute_forces_seq();
			tree.update_seq();
		}
	} else {
		for(int s = 0; s < stoi(args["-s"]); s++) {
			tree.build();
			tree.compute_coms();
			tree.compute_forces();
			tree.update();
		}
	}

	auto t1 = MPI_Wtime();
	cout << (t1 - t0) << endl;

	cout << tree.to_string() << endl;

	ofstream ofile(args["-o"]);
	ofile << std::scientific;
	ofile << n_particles << endl;

	for(int p = 0; p < n_particles; p++) {
		auto particle = tree.particles[p];
		ofile << p << "\t" << particle.pm.p.x << "\t" << particle.pm.p.y << "\t"
		      << particle.pm.m << "\t" << particle.vel.x << "\t"
		      << particle.vel.y << endl;
	}

	ifile.close();
	ofile.close();

	// Finalize the MPI environment.
	MPI_Finalize();
}