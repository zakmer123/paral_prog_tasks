#ifndef ITER_SOLVER_H
#define ITER_SOLVER_H

#include "LinearOperator.h"
#include "LinearAlgebra.h"

class IterSolver {
protected:
    DomainDecomposition& domain_;
    LinearOperator& A_;
    int max_iter_;
    double tolerance_;
    int elapsed_iter_ = -1;

    IterSolver(DomainDecomposition& domain, LinearOperator& A, int max_iter, double tolerance) : domain_(domain), A_(A), max_iter_(max_iter), tolerance_(tolerance) {}
public:
    virtual ~IterSolver() = default;
    virtual void solve(const double* f, double* x) = 0;

    int max_iter() const { return max_iter_; }
    double tolerance() const { return tolerance_; }
    int elapsed_iter() const { return elapsed_iter_; }

};

#endif

