#ifndef LAPLACIAN_OPERATOR_H
#define LAPLACIAN_OPERATOR_H

#include "LinearOperator.h"
#include "ConductivityField.h"

class LaplacianOperator : public LinearOperator {
private:
    DomainDecomposition& domain_;
    ConductivityField& cond_;
    double hx_, hy_;
    double hx2_, hy2_;

public:
    LaplacianOperator(DomainDecomposition& domain, ConductivityField& conductivity);

    void apply(const double* x, double* y) override;
};

#endif