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

float random(int m) { return (float)rand() / RAND_MAX * m; }

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
	printf("Hello world from processor %d out of %d processors\n", R, M);

	Tree tree;
	for(int i = 0; i < 8; i++)
		tree.insert(PointMass{Point{random(4), random(4)}, random(10)});

	cout << "tree" << endl << tree.to_string() << endl;

	cout << "EXIT" << endl;

	// Finalize the MPI environment.
	MPI_Finalize();
}