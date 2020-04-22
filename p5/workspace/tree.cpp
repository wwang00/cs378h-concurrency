#include "tree.h"

#include <iostream>
#include <math.h>
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

bool Tree::mac(Particle p, Cell c) {
	return c.dim / p.pm.p.diff(c.com.p).mag() < theta;
}

inline Point Point::diff(Point p) { return Point{p.x - x, p.y - y}; }

inline float Point::mag() { return sqrtf(x * x + y * y); }

inline void Point::add(Point p) {
	x += p.x;
	y += p.y;
}

inline void PointMass::add(PointMass pm) {
	p.x += pm.p.x * pm.m;
	p.y += pm.p.y * pm.m;
	m += pm.m;
}

inline void PointMass::normalize() {
	p.x /= m;
	p.y /= m;
}

Point PointMass::force(PointMass pm) {
	auto dp = p.diff(pm.p);
	auto d = fmax(dp.mag(), MIN_R);
	auto c = G * m * pm.m / (d * d * d);
	return Point{c * dp.x, c * dp.y};
}

Cell::Cell(Point loc, float dim, int parent)
    : state(CellState::Empty), loc(loc), dim(dim), parent(parent),
      child_base(-1), com(PointMass()), pid(-1) {}

Tree::Tree(float theta, float dt, int n_particles)
    : theta(theta), dt(dt), n_particles(n_particles) {
	particles.resize(n_particles);
}

void Tree::build() {
	cout << "Tree::build called......" << endl;

	cells.clear();
	cells.push_back(Cell(Point(), MAX_DIM, -1));
	for(int p = 0; p < n_particles; p++) {
		cout << "particle " << p << endl;
		auto particle = particles[p];
		if(particle.pm.m < 0)
			continue;

		// update cells
		int cid = 0;
		bool working = true;
		while(working) {
			cout << "\tat cid " << cid << endl;
			auto curr = cells[cid];
			auto dim_half = curr.dim / 2;
			auto xc = curr.loc.x;
			auto yc = curr.loc.y;
			auto xm = xc + dim_half;
			auto ym = yc + dim_half;
			Point mid{xm, ym};

			switch(curr.state) {
			case CellState::Empty: {
				cout << "\tswitch state Empty" << endl;

				// found a place to insert
				curr.state = CellState::Full;
				curr.pid = p;
				cells[cid] = curr;
				working = false;
				break;
			}
			case CellState::Full: {
				cout << "\tswitch state Full" << endl;

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
				cout << "\tswitch state Split" << endl;

				// recurse into the right quadrant
				auto qp = quadrant(particle.pm.p, mid);
				cid = curr.child_base + qp;
				break;
			}
			default:
				cout << "Tree::build BAD STATE" << endl;
				return;
			}
		}
	}

	cout << "Tree::build exited......" << endl;
}

void Tree::compute_coms() {
	cout << "Tree::compute_coms called......" << endl;

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
		if(cell.state == CellState::Full)
			continue;

		// add Split children to queue
		auto base = cell.child_base;
		for(int ch = base; ch < base + 4; ch++) {
			q.push(ch);
		}
	}

	// calculate coms in reverse level-order
	for(auto cit = order.rbegin(); cit != order.rend(); cit++) {
		auto c = *cit;
		auto cell = cells[c];
		PointMass com;
		if(cell.state == CellState::Full) {
			com = particles[cell.pid].pm;
		} else { // Split
			com = PointMass();
			auto base = cell.child_base;
			for(int ch = base; ch < base + 4; ch++) {
				auto child = cells[ch];
				com.add(child.com);
			}
			com.normalize();
		}
		cell.com = com;
		cells[c] = cell;
	}

	cout << "Tree::compute_coms exited......" << endl;
}

void Tree::compute_forces() {
	cout << "Tree::compute_forces called......" << endl;

	queue<int> q; // queue of cell ids
	for(int p = 0; p < particles.size(); p++) {
		auto particle = particles[p];
		auto force = Point();
		q.push(0);
		while(!q.empty()) {
			auto c = q.front();
			q.pop();
			auto cell = cells[c];
			if(cell.state == CellState::Empty)
				continue;
			if(cell.state == CellState::Full) {
				if(cell.pid != p)
					force.add(particle.pm.force(particles[cell.pid].pm));
				continue;
			}
			// Split
			if(mac(particle, cell)) {
				cout << "mac satisfied for particle " << p << " cell " << c << endl;
				force.add(particle.pm.force(cell.com));
				continue;
			}
			auto base = cell.child_base;
			for(int ch = base; ch < base + 4; ch++) {
				q.push(ch);
			}
		}
		particle.force = force;
		particles[p] = particle;
	}

	cout << "Tree::compute_forces exited......" << endl;
}

///////////////
// to_string //
///////////////

string Point::to_string() {
	char buf[1024];
	sprintf(buf, "(%.6f, %.6f)", x, y);
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
	        "%d,  \tpid: %d, \tcom: %s }",
	        state_c, loc_c, dim, parent, child_base, pid, com_c);
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