#ifndef _DATATYPES_H_
#define _DATATYPES_H_

#include <mpi.h>

extern MPI_Datatype PointMPI;
extern MPI_Datatype PointMassMPI;
extern MPI_Datatype ParticleMPI, ParticleVectorMPI;
extern MPI_Datatype CellMPI, CellVectorMPI;

void init_structsMPI();

void init_PointMPI();
void init_PointMassMPI();
void init_ParticleMPI();
void init_CellMPI();

#endif