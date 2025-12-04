#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#include "IterSolver.h"

class CGSolver : public IterSolver{
public:
	CGSolver(DomainDecomposition& domain, LinearOperator& A, int max_iter, double tolerance) : IterSolver(domain, A, max_iter, tolerance) {}

	void solve(const double* f, double* x) override;
};

#endif
