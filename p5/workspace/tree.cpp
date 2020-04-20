#include "tree.h"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string>

using namespace std;

/**
 * 1 | 3
 * --+--
 * 0 | 2
 */
int quadrant(Point p, Point m) {
	if(p.x < m.x) {
		if(p.y < m.y) {
			return 0;
		}
		return 1;
	}
	if(p.y < m.y) {
		return 2;
	}
	return 3;
}

Cell::Cell(Point loc, float dim, int parent)
    : state(CellState::Empty), loc(loc), dim(dim), parent(parent),
      child_base(0), com(PointMass{Point{0, 0}, 0}) {}

Tree::Tree() { cells.push_back(Cell(Point{0, 0}, MAX_DIM, -1)); }

void Tree::insert(PointMass particle) {
	cout << "Tree::insert called......" << endl;
	cout << "particle " << particle.to_string() << endl;

	int cid = 0; // current cell id in tree traversal

	while(true) {
		cout << "Tree::insert cid " << cid << endl;
		auto curr = cells[cid];
		auto dim_half = curr.dim / 2;
		auto xc = curr.loc.x;
		auto yc = curr.loc.y;
		auto xm = xc + dim_half;
		auto ym = yc + dim_half;
		Point mid{xm, ym};

		switch(curr.state) {
		case CellState::Empty: {
			cout << "Tree::insert switch state Empty" << endl;

			// found a place to insert
			curr.state = CellState::Full;
			curr.com = particle;
			cells[cid] = curr;

			cout << "Tree::insert exited......" << endl;
			return;
		}
		case CellState::Full: {
			cout << "Tree::insert switch state Full" << endl;

			// split and make empty subcells
			curr.child_base = cells.size();
			cells.push_back(Cell(Point{xc, yc}, dim_half, cid));
			cells.push_back(Cell(Point{xc, ym}, dim_half, cid));
			cells.push_back(Cell(Point{xm, yc}, dim_half, cid));
			cells.push_back(Cell(Point{xm, ym}, dim_half, cid));

			// move current cell's particle into its subcell
			auto qc = quadrant(curr.com.p, mid);
			auto curr_moved = cells[curr.child_base + qc];
			curr_moved.state = CellState::Full;
			curr_moved.com = curr.com;
			cells[curr.child_base + qc] = curr_moved;

			// go straight to case Split
			curr.state = CellState::Split;
			cells[cid] = curr;
		}
		case CellState::Split: {
			cout << "Tree::insert switch state Split" << endl;

			// find the right quadrant to recurse into
			auto qp = quadrant(particle.p, mid);
			cid = curr.child_base + qp;
			break;
		}
		default:
			cout << "Tree::insert BAD STATE" << endl;
			return;
		}
	}
}

///////////////
// to_string //
///////////////

string Point::to_string() {
	char buf[1024];
	sprintf(buf, "(%.5f, %.5f)", x, y);
	return string(buf);
}

string PointMass::to_string() {
	char buf[1024];
	sprintf(buf, "{ loc: %s, mass: %.5f }", p.to_string().c_str(), m);
	return string(buf);
}

string Cell::to_string() {
	char buf[1024];
	string state_str;
	switch(state) {
	case CellState::Empty:
		state_str = "Empty";
		break;
	case CellState::Full:
		state_str = "Full";
		break;
	case CellState::Split:
		state_str = "Split";
		break;
	default:
		cout << "Cell::to_string BAD STATE" << endl;
		state_str = "";
		break;
	}
	auto state_c = state_str.c_str();
	auto loc_c = loc.to_string().c_str();
	auto com_c = com.to_string().c_str();
	sprintf(buf,
	        "{ state: %s, parent: %d, child_base: %d, loc: %s, dim: %.5f, com: "
	        "%s }",
	        state_c, parent, child_base, loc_c, dim, com_c);
	return string(buf);
}

string Tree::to_string() {
	stringstream ss;
	ss << "cells: {";
	for(int c = 0; c < cells.size(); c++) {
		ss << "\n" << c << "\t" << cells[c].to_string();
	}
	ss << "\n}";
	return ss.str();
}