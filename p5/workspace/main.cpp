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

unordered_set<string> FLAGS{};
unordered_set<string> OPTS{"-i", "-o", "-s", "-t", "-d"};

float random(float m) { return (float)rand() / (float)RAND_MAX * m; }

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	auto args = parse_args(argc, argv, FLAGS, OPTS);

	// Get the number of processes
	int M;
	MPI_Comm_size(MPI_COMM_WORLD, &M);

	// Get the rank of the process
	int R;
	MPI_Comm_rank(MPI_COMM_WORLD, &R);

	// Print off a hello world message
	printf("ENTER processor %d out of %d processors\n", R, M);

	ifstream ifile(args["-i"]);

	// init tree
	int n_particles;
	ifile >> n_particles;
	Tree tree(stof(args["-t"]), stof(args["-d"]), n_particles);

	// read particle start configurations
	for(int p = 0; p < n_particles; p++) {
		Particle particle;
		int id;
		ifile >> id >> particle.pm.p.x >> particle.pm.p.y >> particle.pm.m >>
		    particle.vel.x >> particle.vel.y;
		particle.force = Point();
        printf("inserting particle %s\n", particle.to_string().c_str());
		tree.particles[p] = particle;
	}

	for(int s = 0; s < stoi(args["-s"]); s++) {
        printf("begin iteration %d......\n", s);
		tree.build();
		tree.compute_coms();
		tree.compute_forces();
        tree.update();
	}

    cout << tree.to_string() << endl;

	cout << "particles after: {";
	for(int p = 0; p < n_particles; p++) {
		printf("\n%d\t%s", p, tree.particles[p].to_string().c_str());
	}
	cout << "\n}" << endl;

	printf("EXIT processor %d out of %d processors\n", R, M);

	// Finalize the MPI environment.
	MPI_Finalize();
}