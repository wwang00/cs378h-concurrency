#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>

#include "argparse.h"
#include "tree.h"

#define N 8

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

	Tree tree(stof(args["-t"]), stof(args["-d"]));
	vector<Particle> particles;
	PointMass com{};
	for(int i = 0; i < N; i++) {
		PointMass pm{Point{random(4), random(4)}, random(10) + 10};
		Particle particle = Particle{pm, Point(), Point()};
		particles.push_back(particle);
		com.p.x += pm.p.x * pm.m;
		com.p.y += pm.p.y * pm.m;
		com.m += pm.m;
		tree.insert(particle);
	}
	com.p.x /= com.m;
	com.p.y /= com.m;

	tree.compute_coms();
	tree.compute_forces();

	cout << "particles: {";
	for(int p = 0; p < N; p++) {
		printf("\n%d\t%s", p, particles[p].to_string().c_str());
	}
	cout << "\n}" << endl;

	cout << "tree after" << endl << tree.to_string() << endl;

	cout << "true com " << com.to_string() << endl;

	std::copy(tree.particles.begin(), tree.particles.end(), particles.begin());

	cout << "particles after: {";
	for(int p = 0; p < N; p++) {
		printf("\n%d\t%s", p, particles[p].to_string().c_str());
	}
	cout << "\n}" << endl;

	printf("EXIT processor %d out of %d processors\n", R, M);

	// Finalize the MPI environment.
	MPI_Finalize();
}