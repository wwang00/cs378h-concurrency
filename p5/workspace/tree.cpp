#include "tree.h"

#include <iostream>
#include <queue>
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

void PointMass::join(const PointMass &pm) {
	auto mj = m + pm.m;
	auto xj = (p.x * m + pm.p.x * pm.m) / mj;
	auto yj = (p.y * m + pm.p.y * pm.m) / mj;
	p = Point{xj, yj};
	m = mj;
}

Particle::Particle(PointMass pm, Point vel, Point force)
    : pm(pm), vel(vel), force(force) {}

Cell::Cell(Point loc, float dim, int parent)
    : state(CellState::Empty), loc(loc), dim(dim), parent(parent),
      child_base(-1), com(PointMass()), pid(-1) {}

Tree::Tree(float theta, float dt) : theta(theta), dt(dt) {
	cells.push_back(Cell(Point(), MAX_DIM, -1));
}

void Tree::insert(Particle particle) {
	cout << "Tree::insert called......" << endl;
	cout << "particle " << particle.to_string() << endl;

	auto pp = particle.pm.p;

	// insert particle into particles
	if(pp.x < 0 || pp.x > MAX_DIM || pp.y < 0 || pp.y > MAX_DIM)
		return;
	int pid = particles.size();
	particles.push_back(particle);

	// update cells
	int cid = 0;

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
			curr.pid = pid;
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
			auto qc = quadrant(particles[curr.pid].pm.p, mid);
			auto moved_idx = curr.child_base + qc;
			auto curr_moved = cells[moved_idx];
			curr_moved.state = CellState::Full;
			curr_moved.pid = curr.pid;
			cells[moved_idx] = curr_moved;
			curr.pid = -1;

			// go straight to case Split
			curr.state = CellState::Split;
			cells[cid] = curr;
		}
		case CellState::Split: {
			cout << "Tree::insert switch state Split" << endl;

			// recurse into the right quadrant
			auto qp = quadrant(pp, mid);
			cid = curr.child_base + qp;
			break;
		}
		default:
			cout << "Tree::insert BAD STATE" << endl;
			return;
		}
	}
}

void Tree::compute_coms() {
	// find level-order traversal
	vector<int> order;
	queue<int> q;
	q.push(0);
	while(!q.empty()) {
		auto c = q.front();
		q.pop();
		auto cell = cells[c];
		if(cell.state == CellState::Empty)
			continue;

		// add Full or Split cell to level order
		order.push_back(c);
		auto base = cell.child_base;
		if(base < 0)
			continue;
		for(int ch = base; ch < base + 4; ch++) {
			q.push(ch);
		}
	}

	// calculate coms in reverse level-order
	for(auto cit = order.rbegin(); cit != order.rend(); cit++) {
		auto c = *cit;
		auto cell = cells[c];
		auto com = PointMass();
		if(cell.state == CellState::Full) {
			com = particles[cell.pid].pm;
		} else { // Split
			auto base = cell.child_base;
			for(int ch = base; ch < base + 4; ch++) {
				auto child = cells[ch];
				auto comc = child.com;
				com.p.x += comc.p.x * comc.m;
				com.p.y += comc.p.y * comc.m;
				com.m += comc.m;
			}
			com.p.x /= com.m;
			com.p.y /= com.m;
		}
		cell.com = com;
		cells[c] = cell;
	}
}

///////////////
// to_string //
///////////////

string Point::to_string() {
	char buf[1024];
	sprintf(buf, "(%.3f, %.3f)", x, y);
	return string(buf);
}

string PointMass::to_string() {
	char buf[1024];
	sprintf(buf, "{ loc: %s, mass: %.3f }", p.to_string().c_str(), m);
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
	        "{ state: %s,\tloc: %s,\tdim: %.3f,\tparent: %d,\tchild_base: "
	        "%d,\tcom: %s }",
	        state_c, loc_c, dim, parent, child_base, com_c);
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

string Particle::to_string() {
	char buf[1024];
	sprintf(buf, "{ pm: %s, vel: %s, force: %s }", pm.to_string().c_str(),
	        vel.to_string().c_str(), force.to_string().c_str());
	return string(buf);
}