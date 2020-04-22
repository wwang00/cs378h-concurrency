#ifndef _DATATYPES_H_
#define _DATATYPES_H_

#include <mpi.h>

extern MPI_Datatype PointMPI;
extern MPI_Datatype PointMassMPI;
extern MPI_Datatype ParticleMPI;
extern MPI_Datatype CellMPI;

void init_MPI_structs();

void init_PointMPI();
void init_PointMassMPI();
void init_ParticleMPI();
void init_CellMPI();

#endif