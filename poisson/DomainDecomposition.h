#ifndef DOMAIN_DECOMPOSITION_H
#define DOMAIN_DECOMPOSITION_H

#include <mpi.h>
#include <vector>
#include <string>
#include <math.h>
//#include <mkl.h>
#include <algorithm>

#define ind(N, i, j) ((i) * (N) + (j))
#define sqr(x) ((x) * (x))
#define rev(x) (1.0 / (x))
#define reav(x, y) (2.0 / ((x) + (y)))

class DomainDecomposition {
private:
    MPI_Comm comm_;
    int rank_, size_;
    int prx_, pry_;
    int globalNx_, globalNy_;
    int leftNeighbor_, rightNeighbor_, bottomNeighbor_, topNeighbor_;
    int localNx_, localNy_;
    int offsetLeftX_, offsetRightX_;
    int offsetBottomY_, offsetTopY_;

public:
    DomainDecomposition(MPI_Comm comm, int prx, int pry, int globalNx, int globalNy);
    
    int rank() const { return rank_; }
    int size() const { return size_; }
    int prx() const { return prx_; }
    int pry() const { return pry_; }

    int globalNx() const { return globalNx_; }
    int globalNy() const { return globalNy_; }
    int leftNeighbor() const { return leftNeighbor_; }
    int rightNeighbor() const { return rightNeighbor_; }
    int bottomNeighbor() const { return bottomNeighbor_; }
    int topNeighbor() const { return topNeighbor_; }
    int localNx() const { return localNx_; }
    int localNy() const { return localNy_; }
    int offsetLeftX() const { return offsetLeftX_; }
    int offsetRightX() const { return offsetRightX_; }
    int offsetBottomY() const { return offsetBottomY_; }
    int offsetTopY() const { return offsetTopY_; }
    int realocalNx() const { return localNx_ - offsetLeftX_ - offsetRightX_; }
    int realocalNy() const { return localNy_ - offsetBottomY_ - offsetTopY_; }

    int globalNxNy() const { return globalNx_ * globalNy_; }
    int localNxNy() const { return localNx_ * localNy_; }

    void resizeDomain(int scale);
};

#endif