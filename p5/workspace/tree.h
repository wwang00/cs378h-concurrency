#ifndef _TREE_H_
#define _TREE_H_

#include <string>
#include <vector>

#include "globals.h"

struct Point {
	float x;
	float y;

	void add(Point p);
	float mag();
	Point diff(Point p);

	std::string to_string();
};

struct PointMass {
	Point p;
	float m;

	void add(PointMass pm);
	void normalize();
	Point force(PointMass pm);

	std::string to_string();
};

struct Particle {
	PointMass pm;
	Point vel, force;

	std::string to_string();
};

enum CellState { Empty, Full, Split };

struct Cell {
	CellState state;
	Point loc;
	float dim;
	int parent;
	int child_base; // Split
	PointMass com;
	int pid; // Full

	Cell(Point loc, float dim, int parent);

	std::string to_string();
};

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

	void compute_coms_seq();
	void compute_forces_seq();
	void update_seq();

	std::string to_string();
};

#endif