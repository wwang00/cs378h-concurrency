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

bool mac(Particle p, Cell c) {
	return c.dim / p.pm.p.diff(c.com.p).norm() < THETA;
}

Point Point::diff(Point p) { return Point{p.x - x, p.y - y}; }

double Point::norm() { return sqrt(x * x + y * y); }

void Point::add(Point p) {
	x += p.x;
	y += p.y;
}

void PointMass::join(PointMass pm) {
	auto xt = p.x * m + pm.p.x * pm.m;
	auto yt = p.y * m + pm.p.y * pm.m;
	auto mt = m + pm.m;
	p.x = xt / mt;
	p.y = yt / mt;
	m = mt;
}

Point PointMass::force(PointMass pm) {
	auto dp = p.diff(pm.p);
	auto d = dp.norm();
	double d3;
	if(d < MIN_R) {
		// printf("rlimit reached\n");
		d3 = MIN_R * MIN_R * d;
	} else {
		d3 = d * d * d;
	}
	auto c = G * m * pm.m / d3;
	return Point{c * dp.x, c * dp.y};
}

Cell::Cell() {}

Cell::Cell(Point loc, double dim, int parent)
    : state(Empty), parent(parent), child_base(-1), pid(-1), dim(dim), loc(loc),
      com(PointMass()) {}

Tree::Tree() {
	particles.resize(N_PTS);
	n_changed = N_PTS / M;
	if(n_changed * M != N_PTS)
		n_changed++;
}

void Tree::build() {
	// printf("[%d] Tree::build called......\n", R);

	build_seq();

	// printf("[%d] Tree::build exited......\n", R);
}

void Tree::update() {
	// printf("[%d] Tree::update called......\n", R);

	auto start = R * n_changed;
	if(start >= N_PTS)
		return;
	auto end = start + n_changed;
	if(end > N_PTS)
		end = N_PTS;
	update_particles(start, end);
	for(int r = 0; r < M; r++) {
		auto start = r * n_changed;
		if(start >= N_PTS)
			continue;
		auto end = start + n_changed;
		if(end > N_PTS)
			end = N_PTS;
		auto count = end - start;
		MPI_Bcast(&particles[start], count * sizeof(Particle), MPI_BYTE, r,
		          MPI_COMM_WORLD);
	}

	// printf("[%d] Tree::update exited......\n", R);
}

// void Tree::build_master() {
// 	// printf("Tree::build_master called......\n");

// 	MPI_Bcast(&particles[0], N_PTS * sizeof(Particle), MPI_BYTE, 0,
// 	          MPI_COMM_WORLD);
// 	build_seq();

// 	// printf("Tree::build_master exited......\n");
// }

// void Tree::update_master() {
// 	// printf("[%d] Tree::update_master called......\n", R);

// 	auto stride = M - 1;
// 	MPI_Status status;
// 	for(int r = 1; r < M; r++) {
// 		MPI_Recv(&particles_changed[0], n_changed * sizeof(Particle), MPI_BYTE,
// 		         MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
// 		int base = status.MPI_SOURCE - 1;
// 		int changed_idx = 0;
// 		for(int p = base; p < N_PTS; p += stride) {
// 			particles[p] = particles_changed[changed_idx++];
// 		}
// 	}

// 	// printf("[%d] Tree::update_master exited......\n", R);
// }

void Tree::build_seq() {
	// printf("Tree::build_seq called......\n");

	cells.clear();
	cells.push_back(Cell(Point(), MAX_DIM, -1));
	for(int p = 0; p < N_PTS; p++) {
		// printf("particle %d\n", p);
		auto particle = particles[p];
		if(particle.pm.m < 0)
			continue;

		// update cells
		int cid = 0;
		bool working = true;
		while(working) {
			// printf("\tat cid %d\n", cid);
			auto curr = cells[cid];
			curr.com.join(particle.pm);
			// printf("\tnew com %s\n", curr.com.to_string().c_str());

			auto dim_half = curr.dim / 2;
			auto xc = curr.loc.x;
			auto yc = curr.loc.y;
			auto xm = xc + dim_half;
			auto ym = yc + dim_half;
			Point mid{xm, ym};

			switch(curr.state) {
			case Empty: {
				// printf("\tswitch state Empty\n");

				// found a place to insert
				curr.state = Full;
				curr.pid = p;
				cells[cid] = curr;

				working = false;
				break;
			}
			case Full: {
				// printf("\tswitch state Full\n");

				// split and make empty subcells
				curr.state = Split;
				curr.child_base = cells.size();
				cells.push_back(Cell(Point{xc, yc}, dim_half, cid));
				cells.push_back(Cell(Point{xc, ym}, dim_half, cid));
				cells.push_back(Cell(Point{xm, yc}, dim_half, cid));
				cells.push_back(Cell(Point{xm, ym}, dim_half, cid));

				// move current cell's particle into its subcell
				auto qc = quadrant(particles[curr.pid].pm.p, mid);
				auto moved_idx = curr.child_base + qc;
				auto curr_moved = cells[moved_idx];
				curr_moved.state = Full;
				curr_moved.pid = curr.pid;
				curr_moved.com = particles[curr.pid].pm;
				cells[moved_idx] = curr_moved;
				curr.pid = -1;
				cells[cid] = curr;

				// recurse into the right quadrant
				auto qp = quadrant(particle.pm.p, mid);
				cid = curr.child_base + qp;
				break;
			}
			case Split: {
				// printf("\tswitch state Split\n");

				cells[cid] = curr;

				// recurse into the right quadrant
				auto qp = quadrant(particle.pm.p, mid);
				cid = curr.child_base + qp;
				break;
			}
			default:
				printf("Tree::build_seq BAD STATE\n");
				return;
			}
		}
	}

	// printf("Tree::build_seq exited......\n");
}

void Tree::update_seq() {
	// printf("Tree::update_seq called......\n");

	update_particles(0, N_PTS);

	// printf("Tree::update_seq exited......\n");
}

void Tree::update_particles(int start, int end) {
	// compute forces
	queue<int> q; // queue of cell ids
	for(int p = start; p < end; p++) {
		// printf("particle %d\n", p);
		auto particle = particles[p];
		if(particle.pm.m < 0)
			continue;

		// compute force
		auto force = Point();
		q.push(0);
		while(!q.empty()) {
			auto c = q.front();
			q.pop();
			auto cell = cells[c];
			if(cell.state == Empty)
				continue;
			if(cell.state == Full) {
				if(cell.pid != p)
					force.add(particle.pm.force(cell.com));
				continue;
			}
			// Split
			if(mac(particle, cell)) {
				// printf("\tparticle %d mac satisfied cell %d\n", p, c);
				force.add(particle.pm.force(cell.com));
				continue;
			}
			auto base = cell.child_base;
			for(int ch = base; ch < base + 4; ch++) {
				q.push(ch);
			}
		}
		particle.f = force;

		// update particle
		auto ax_dt = (particle.f.x / particle.pm.m) * DT;
		auto ay_dt = (particle.f.y / particle.pm.m) * DT;
		Point loc_new{particle.pm.p.x + (particle.v.x + 0.5 * ax_dt) * DT,
		              particle.pm.p.y + (particle.v.y + 0.5 * ay_dt) * DT};
		Point vel_new{particle.v.x + ax_dt, particle.v.y + ay_dt};
		particle.pm.p = loc_new;
		particle.v = vel_new;

		// handle lost particle
		if(loc_new.x < 0 || loc_new.x > MAX_DIM || loc_new.y < 0 ||
		   loc_new.y > MAX_DIM) {
			// printf("particle %d was lost\n", p);
			particle.pm.m = -1;
		}
		particles[p] = particle;
	}
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
	case Empty:
		state_str = "Empty";
		break;
	case Full:
		state_str = "Full";
		break;
	case Split:
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
	sprintf(buf, "{ pm: %s, v: %s, f: %s }", pm.to_string().c_str(),
	        v.to_string().c_str(), f.to_string().c_str());
	return string(buf);
}