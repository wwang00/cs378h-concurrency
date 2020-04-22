#ifndef _TREE_H_
#define _TREE_H_

#include <string>
#include <vector>

#define MAX_DIM 4.0
#define G 0.0001f
#define MIN_R 0.03f

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

	Particle(PointMass pm);
	Particle(PointMass pm, Point vel, Point force);

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
	const float theta;
	const float dt;

public:
	std::vector<Cell> cells;
	std::vector<Particle> particles;

	Tree(float theta, float dt);

	void insert(Particle particle);
	void compute_coms();
	void compute_forces();
	bool mac(Particle p, Cell c);
	void reset();

	std::string to_string();
};

#endif