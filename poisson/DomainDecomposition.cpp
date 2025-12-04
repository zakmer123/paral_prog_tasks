#include "DomainDecomposition.h"

DomainDecomposition::DomainDecomposition(MPI_Comm comm, int prx, int pry, int globalNx, int globalNy) : 
	comm_(comm), prx_(prx), pry_(pry), globalNx_(globalNx), globalNy_(globalNy) {
	MPI_Comm_rank(comm_, &rank_);
	MPI_Comm_size(comm_, &size_);

	leftNeighbor_ = (rank_ / pry_ == 0) ? MPI_PROC_NULL : rank_ - pry_;
	rightNeighbor_ = (rank_ / pry_ == prx_ - 1) ? MPI_PROC_NULL : rank_ + pry_;
	bottomNeighbor_ = (rank_ % pry_ == 0) ? MPI_PROC_NULL : rank_ - 1;
	topNeighbor_ = (rank_ % pry_ == pry_ - 1) ? MPI_PROC_NULL : rank_ + 1;

	offsetLeftX_ = (leftNeighbor_ == MPI_PROC_NULL) ? 0 : 1;
	offsetRightX_ = (rightNeighbor_ == MPI_PROC_NULL) ? 0 : 1;
	offsetBottomY_ = (bottomNeighbor_ == MPI_PROC_NULL) ? 0 : 1;
	offsetTopY_ = (topNeighbor_ == MPI_PROC_NULL) ? 0 : 1;

	localNx_ = (globalNx_ / prx_) + offsetLeftX_ + offsetRightX_;
	localNy_ = (globalNy_ / pry_) + offsetBottomY_ + offsetTopY_;
}
void DomainDecomposition::resizeDomain(int scale) {
	globalNx_ *= scale;
	globalNy_ *= scale;
	localNx_ = (globalNx_ / prx_) + offsetLeftX_ + offsetRightX_;
	localNy_ = (globalNy_ / pry_) + offsetBottomY_ + offsetTopY_;
}

