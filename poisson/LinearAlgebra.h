#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include "DomainDecomposition.h"

namespace LinearAlgebra {
	double ddot(DomainDecomposition& domain, const double* x, const double* y);
	void spectral_decomp(DomainDecomposition& domain, double* spectrum, double* supdiag, double* Q);
	void tridiag_factor(DomainDecomposition& domain, double* spectrum, double* D, double* L);
}

#endif
