#include "CGSolver.h"
#include <iostream>

void CGSolver::solve(const double* f, double* x) {
    int N = domain_.localNxNy();
    std::vector<double> Ap(N);
    std::vector<double> p(N);
    std::vector<double> r(N);

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rr = 0.0;
    double ff = LinearAlgebra::ddot(domain_, f, f);
    double eps = pow(10, -8) * sqrt(ff);

    A_.apply(x, p.data());
    //cblas_dcopy(N, f, 1, r.data(), 1);
    //cblas_daxpy(N, -1.0, p.data(), 1, r.data(), 1);
    //cblas_dcopy(N, r.data(), 1, p.data(), 1);
    for (int i = 0; i < N; ++i) r[i] = f[i];
    for (int i = 0; i < N; ++i) r[i] -= p[i];
    for (int i = 0; i < N; ++i) p[i] = r[i];

    rr = LinearAlgebra::ddot(domain_, r.data(), r.data());

    while (sqrt(rr) > eps && iter < max_iter_) {
        A_.apply(p.data(), Ap.data());
        alpha = rr / LinearAlgebra::ddot(domain_, Ap.data(), p.data());
        beta = rr;
        //cblas_daxpy(N, alpha, p.data(), 1, x, 1);
        //cblas_daxpy(N, -alpha, Ap.data(), 1, r.data(), 1);
        for (int i = 0; i < N; ++i) x[i] += alpha * p[i];
        for (int i = 0; i < N; ++i) r[i] -= alpha * Ap[i];
        rr = LinearAlgebra::ddot(domain_, r.data(), r.data());
        beta = rr / beta;
        //cblas_dscal(N, beta, p.data(), 1);
        //cblas_daxpy(N, 1.0, r.data(), 1, p.data(), 1);
        for (int i = 0; i < N; ++i) p[i] *= beta;
        for (int i = 0; i < N; ++i) p[i] += r[i];
        iter++;
    }
    elapsed_iter_ = iter;
	return;
}