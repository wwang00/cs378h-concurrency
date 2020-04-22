#include "datatypes.h"
#include "tree.h"

MPI_Datatype PointMPI;
MPI_Datatype PointMassMPI;
MPI_Datatype ParticleMPI, ParticleVectorMPI;
MPI_Datatype CellMPI, CellVectorMPI;

void init_structsMPI() {
    init_PointMPI();
    init_PointMassMPI();
    init_ParticleMPI();
    init_CellMPI();
}

void init_PointMPI() {
	const int count = 1;
	int blocklengths[count] = {2};
	MPI_Aint displacements[count] = {0};
	MPI_Datatype types[count] = {MPI_FLOAT};
	MPI_Type_create_struct(count, blocklengths, displacements, types, &PointMPI);
}

void init_PointMassMPI() {
	const int count = 2;
	int blocklengths[count] = {1, 1};
	MPI_Aint displacements[count] = {0, sizeof(Point)};
	MPI_Datatype types[count] = {PointMPI, MPI_FLOAT};
	MPI_Type_create_struct(count, blocklengths, displacements, types, &PointMassMPI);
}

void init_ParticleMPI() {
    const int count = 2;
	int blocklengths[count] = {1, 2};
	MPI_Aint displacements[count] = {0, sizeof(PointMass)};
	MPI_Datatype types[count] = {PointMassMPI, PointMPI};
	MPI_Type_create_struct(count, blocklengths, displacements, types, &ParticleMPI);
}

void init_CellMPI() {
    const int count = 4;
	int blocklengths[count] = {4, 1, 1, 1};
	MPI_Aint displacements[count] = {0, 16, 20, 20 + sizeof(Point)};
	MPI_Datatype types[count] = {MPI_INT, MPI_FLOAT, PointMPI, PointMassMPI};
	MPI_Type_create_struct(count, blocklengths, displacements, types, &CellMPI);
}