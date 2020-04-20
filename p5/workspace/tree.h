#ifndef _TREE_H_
#define _TREE_H_

#include <string>
#include <vector>

#define MAX_DIM 4.0
#define G 0.0001
#define MIN_R 0.03

struct Point {
	float x;
	float y;

	std::string to_string();
};

struct PointMass {
	Point p;
	float m;

	void join(const PointMass &pm);

	std::string to_string();
};

enum CellState { Empty, Full, Split };

struct Cell {
	CellState state;
	Point loc, vel, acc;
	float dim;
	int parent;
	int child_base;
	PointMass com;

	Cell(Point loc, float dim, int parent);

	std::string to_string();
};

class Tree {
	std::vector<Cell> cells;
	const float theta;
	const float dt;

public:
	Tree(float theta, float dt);

	void insert(PointMass particle);
	void compute_coms();
	void compute_forces();

	std::string to_string();
};

#endif