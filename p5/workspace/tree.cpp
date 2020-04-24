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

Point Point::diff(Point p) { return Point{p.x - x, p.y - y}; }

double Point::mag() { return sqrt(x * x + y * y); }

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

Cell::Cell() {}

Cell::Cell(Point loc, double dim, int parent)
    : state(CellState::Empty), parent(parent), child_base(-1), pid(-1),
      dim(dim), loc(loc), com(PointMass()) {}

Tree::Tree() { particles.resize(n_particles); }

void Tree::build() {
	printf("[%d] Tree::build called......\n", R);

	int n_cells;
	MPI_Bcast(&n_cells, 1, MPI_INT, 0, MPI_COMM_WORLD);
	cells.resize(n_cells);
	MPI_Bcast(&cells[0], cells.size(), CellMPI, 0, MPI_COMM_WORLD);
	MPI_Bcast(&particles[0], n_particles, ParticleMPI, 0, MPI_COMM_WORLD);

	printf("[%d] Tree::build exited......\n", R);
}

void Tree::compute_coms() {
	printf("[%d] Tree::compute_coms called......\n", R);

	int start = cells.size() - 1;
	if(R - 1 > start) {
		printf("[%d] Tree::compute_coms exited......\n", R);
		return;
	}
	while(start % (M - 1) != R - 1) {
		start--;
	}
	for(int c = start; c >= 0; c -= M - 1) {
		PointMass com;
		auto cell = cells[c];
		switch(cell.state) {
		case CellState::Empty:
			com = PointMass();
			break;
		case CellState::Full:
			com = particles[cell.pid].pm;
			break;
		case CellState::Split: {
			com = PointMass();
			auto base = cell.child_base;
			for(int ch = base; ch < base + 4; ch++) {
				auto child = cells[ch];
				if(child.state == CellState::Empty)
					continue;
				if(child.state == CellState::Full) {
					com.add(particles[child.pid].pm);
					continue;
				}
				// recv from Split child
				PointMass child_com;
				auto source = ch % (M - 1) + 1;
				if(source == R) {
					child_com = child.com;
				} else {
					MPI_Recv(&child_com, 1, PointMassMPI, source, ch,
					         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				com.add(child_com);
			}
			com.normalize();
			// send to parent
			if(c > 0) {
				auto dest = cell.parent % (M - 1) + 1;
				if(dest != R)
					MPI_Send(&com, 1, PointMassMPI, dest, c, MPI_COMM_WORLD);
			}
			break;
		}
		default:
			printf("[%d] Tree::compute_coms BAD STATE %d\n", R, cell.state);
			break;
		}
		// store locally
		cell.com = com;
		cells[c] = cell;
		// send to master
		MPI_Send(&com, 1, PointMassMPI, 0, c, MPI_COMM_WORLD);
	}

	printf("[%d] Tree::compute_coms exited......\n", R);
}

void Tree::compute_forces() {
	printf("[%d] Tree::compute_forces called......\n", R);

	MPI_Bcast(&cells[0], cells.size(), CellMPI, 0, MPI_COMM_WORLD);
	compute_forces_work(R - 1, M - 1);

	printf("[%d] Tree::compute_forces exited......\n", R);
}

void Tree::update() {
	printf("[%d] Tree::update called......\n", R);

	for(int p = R - 1; p < n_particles; p += M - 1) {
		auto particle = update_get(p);
		MPI_Send(&particle, 1, ParticleMPI, 0, p, MPI_COMM_WORLD);
	}

	printf("[%d] Tree::update exited......\n", R);
}

void Tree::build_master() {
	printf("Tree::build_master called......\n");

	build_seq();

	int n_cells = cells.size();
	MPI_Bcast(&n_cells, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cells[0], cells.size(), CellMPI, 0, MPI_COMM_WORLD);
	MPI_Bcast(&particles[0], n_particles, ParticleMPI, 0, MPI_COMM_WORLD);

	printf("Tree::build_master exited......\n");
}

void Tree::compute_coms_master() {
	printf("Tree::compute_coms_master called......\n");

	MPI_Status status;
	for(int i = 0; i < cells.size(); i++) {
		PointMass com;
		MPI_Recv(&com, 1, PointMassMPI, MPI_ANY_SOURCE, MPI_ANY_TAG,
		         MPI_COMM_WORLD, &status);
		auto c = status.MPI_TAG;
		cells[c].com = com;
	}

	printf("Tree::compute_coms_master exited......\n");
}

void Tree::compute_forces_master() {
	printf("Tree::compute_forces_master called......\n");

	MPI_Bcast(&cells[0], cells.size(), CellMPI, 0, MPI_COMM_WORLD);

	printf("Tree::compute_forces_master exited......\n");
}

void Tree::update_master() {
	printf("[%d] Tree::update_master called......\n", R);

	MPI_Status status;
	for(int i = 0; i < n_particles; i++) {
		Particle particle;
		MPI_Recv(&particle, 1, ParticleMPI, MPI_ANY_SOURCE, MPI_ANY_TAG,
		         MPI_COMM_WORLD, &status);
		auto p = status.MPI_TAG;
		particles[p] = particle;
	}

	printf("[%d] Tree::update_master exited......\n", R);
}

void Tree::build_seq() {
	printf("Tree::build_seq called......\n");

	cells.clear();
	cells.push_back(Cell(Point(), MAX_DIM, -1));
	for(int p = 0; p < n_particles; p++) {
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
			auto dim_half = curr.dim / 2;
			auto xc = curr.loc.x;
			auto yc = curr.loc.y;
			auto xm = xc + dim_half;
			auto ym = yc + dim_half;
			Point mid{xm, ym};

			switch(curr.state) {
			case CellState::Empty: {
				// printf("\tswitch state Empty\n");

				// found a place to insert
				curr.state = CellState::Full;
				curr.pid = p;
				cells[cid] = curr;
				working = false;
				break;
			}
			case CellState::Full: {
				// printf("\tswitch state Full\n");

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
				// printf("\tswitch state Split\n");

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

	printf("Tree::build_seq exited......\n");
}

void Tree::compute_coms_seq() {
	printf("Tree::compute_coms_seq called......\n");

	// calculate coms in reverse order
	for(int c = cells.size() - 1; c >= 0; c--) {
		auto cell = cells[c];
		if(cell.state == CellState::Empty)
			continue;
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

	printf("Tree::compute_coms_seq exited......\n");
}

void Tree::compute_forces_seq() {
	printf("Tree::compute_forces_seq called......\n");

	compute_forces_work(0, 1);

	printf("Tree::compute_forces_seq exited......\n");
}

void Tree::update_seq() {
	printf("Tree::update_seq called......\n");

	for(int p = 0; p < n_particles; p++) {
		particles[p] = update_get(p);
	}

	printf("Tree::update_seq exited......\n");
}

void Tree::compute_forces_work(int base, int stride) {
	queue<int> q; // queue of cell ids
	for(int p = base; p < particles.size(); p += stride) {
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
			if(cell.state == CellState::Empty)
				continue;
			if(cell.state == CellState::Full) {
				if(cell.pid != p)
					force.add(particle.pm.force(particles[cell.pid].pm));
				continue;
			}
			// Split
			if(mac(particle, cell)) {
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
}

Particle Tree::update_get(int p) {
	auto particle = particles[p];
	if(particle.pm.m < 0)
		return particle;
	auto ax_dt_2 = 0.5 * (particle.force.x / particle.pm.m) * dt;
	auto ay_dt_2 = 0.5 * (particle.force.y / particle.pm.m) * dt;
	Point loc_new{particle.pm.p.x + (particle.vel.x + ax_dt_2) * dt,
	              particle.pm.p.y + (particle.vel.y + ay_dt_2) * dt};
	Point vel_new{particle.vel.x + ax_dt_2, particle.vel.y + ay_dt_2};
	particle.pm.p = loc_new;
	particle.vel = vel_new;

	// handle lost particles
	if(loc_new.x < 0 || loc_new.x > MAX_DIM || loc_new.y < 0 ||
	   loc_new.y > MAX_DIM) {
		printf("particle %d was lost\n", p);
		particle.pm.m = -1;
	}

	return particle;
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