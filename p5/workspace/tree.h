#ifndef _TREE_H_
#define _TREE_H_

#include <string>
#include <vector>

#include "datatypes.h"
#include "globals.h"

struct Point {
	double x;
	double y;

	void add(Point p);
	double mag();
	Point diff(Point p);

	std::string to_string();
} __attribute__((packed));

struct PointMass {
	Point p;
	double m;

	void add(PointMass pm);
	void normalize();
	Point force(PointMass pm);

	std::string to_string();
} __attribute__((packed));

struct Particle {
	PointMass pm;
	Point vel, force;

	std::string to_string();
} __attribute__((packed));

enum CellState { Empty, Full, Split };

struct Cell {
	CellState state;
	int parent;
	int child_base; // Split
	int pid; // Full
	double dim;
	Point loc;
	PointMass com;

    Cell();
	Cell(Point loc, double dim, int parent);

	std::string to_string();
} __attribute__((packed));

class Tree {
	std::vector<Cell> cells;

	bool mac(Particle p, Cell c);

public:
	std::vector<Particle> particles;

	Tree();

	void build();
	void compute_coms();
	void compute_forces();
	void update();

	void build_master();
	void compute_coms_master();
	void compute_forces_master();
	void update_master();

	void build_seq();
	void compute_coms_seq();
	void compute_forces_seq();
	void update_seq();

	std::string to_string();
};

#endif