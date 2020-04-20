#ifndef _TREE_H_
#define _TREE_H_

#include <string>
#include <vector>

#define MAX_DIM 4.0

struct Point {
    float x;
    float y;

    std::string to_string();
};

struct PointMass {
    Point p;
	float m;

	std::string to_string();
};

enum CellState { Empty, Full, Split };

struct Cell {
	CellState state;
	Point loc;
	float dim;
	int parent;
	int child_base;
	PointMass com;

	Cell(Point loc, float dim, int parent);

	std::string to_string();
};

class Tree {
	std::vector<Cell> cells;

  public:
	Tree();
	void insert(PointMass particle);

	std::string to_string();
};

#endif