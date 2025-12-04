#include "LinearAlgebra.h"

namespace LinearAlgebra {
	double ddot(DomainDecomposition& domain, const double* x, const double* y) {
        double local_ddot = 0.0;
        for (int i = domain.offsetLeftX(); i < domain.localNx() - domain.offsetRightX(); ++i) {
            /*local_ddot += cblas_ddot(domain.localNy() - domain.offsetBottomY() - domain. offsetTopY(), 
                x + i * domain.localNy() + domain.offsetBottomY(), 1, y + i * domain.localNy() + domain.offsetBottomY(), 1);*/
            double tmp = 0.0;
            int ny = domain.localNy() - domain.offsetBottomY() - domain.offsetTopY();
            const double* xi = x + i * domain.localNy() + domain.offsetBottomY();
            const double* yi = y + i * domain.localNy() + domain.offsetBottomY();
            for (int j = 0; j < ny; ++j) {
                tmp += xi[j] * yi[j];
            }
            local_ddot += tmp;
        }
        double global_ddot;
        MPI_Allreduce(&local_ddot, &global_ddot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return global_ddot;
	}
}