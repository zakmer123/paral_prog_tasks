#include "LaplacianOperator.h"

LaplacianOperator::LaplacianOperator(DomainDecomposition& domain, ConductivityField& conductivity) : domain_(domain), cond_(conductivity){
	hx_ = rev(domain_.globalNx() + 1);
	hy_ = rev(domain_.globalNy() + 1);
	hx2_ = rev(sqr(hx_));
	hy2_ = rev(sqr(hy_));
}
void LaplacianOperator::apply(const double* x, double* y) {
	MPI_Request requests[8];
	const int TAG_TOP_HALO = 0;  
	const int TAG_BOTTOM_HALO = 1;
	const int TAG_RIGHT_HALO = 2;
	const int TAG_LEFT_HALO = 3;

	int s;
	std::vector<double> send_top_halo(1), recv_top_halo(1);
	std::vector<double> send_bottom_halo(1), recv_bottom_halo(1);
	std::vector<double> send_right_halo(1), recv_right_halo(1);
	std::vector<double> send_left_halo(1), recv_left_halo(1);

	if (domain_.topNeighbor() != MPI_PROC_NULL) {
		send_top_halo.resize(domain_.localNx());
		recv_top_halo.resize(domain_.localNx());

		s = ind(domain_.localNy(), 0, domain_.localNy() - 2);
		send_top_halo[0] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
			+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]));

		for (int i = 1; i < domain_.localNx() - 1; ++i) {
			s = ind(domain_.localNy(), i, domain_.localNy() - 2);
			send_top_halo[i] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
				+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
				+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()])
				+ reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
		}

		s = ind(domain_.localNy(), domain_.localNx() - 1, domain_.localNy() - 2);
		send_top_halo[domain_.localNx() - 1] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
			+ hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));

		//cblas_dcopy(domain_.localNx(), send_top_halo.data(), 1, y + ind(domain_.localNy(), 0, domain_.localNy() - 2), domain_.localNy());
		int nx = domain_.localNx();
		double* src = send_top_halo.data();
		double* dst = y + ind(domain_.localNy(), 0, domain_.localNy() - 2);

		for (int j = 0; j < nx; ++j) {
			dst[j * domain_.localNy()] = src[j];
		}
	}
	if (domain_.bottomNeighbor() != MPI_PROC_NULL) {
		send_bottom_halo.resize(domain_.localNx());
		recv_bottom_halo.resize(domain_.localNx());

		s = ind(domain_.localNy(), 0, 1);
		send_bottom_halo[0] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
			+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]));

		for (int i = 1; i < domain_.localNx() - 1; ++i) {
			s = ind(domain_.localNy(), i, 1);
			send_bottom_halo[i] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
				+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
				+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()])
				+ reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
		}

		s = ind(domain_.localNy(), domain_.localNx() - 1, 1);
		send_bottom_halo[domain_.localNx() - 1] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]))
			+ hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));

		//cblas_dcopy(domain_.localNx(), send_bottom_halo.data(), 1, y + ind(domain_.localNy(), 0, 1), domain_.localNy());
		int nx = domain_.localNx();
		double* src = send_bottom_halo.data();
		double* dst = y + ind(domain_.localNy(), 0, 1);

		for (int j = 0; j < nx; ++j) {
			dst[j * domain_.localNy()] = src[j];
		}
	}
	if (domain_.rightNeighbor() != MPI_PROC_NULL) {
		send_right_halo.resize(domain_.localNy());
		recv_right_halo.resize(domain_.localNy());

		s = ind(domain_.localNy(), domain_.localNx() - 2, 0);
		send_right_halo[0] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
			+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
			+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]) + rev(cond_[s]) * x[s]);

		for (int j = 1; j < domain_.localNy() - 1; ++j) {
			s = ind(domain_.localNy(), domain_.localNx() - 2, j);
			send_right_halo[j] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
				+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
				+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]));
		}

		s = ind(domain_.localNy(), domain_.localNx() - 2, domain_.localNy() - 1);
		send_right_halo[domain_.localNy() - 1] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
			+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
			+ hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]) + rev(cond_[s]) * x[s]);

		//cblas_dcopy(domain_.localNy(), send_right_halo.data(), 1, y + ind(domain_.localNy(), domain_.localNx() - 2, 0), 1);
		memcpy(
			y + ind(domain_.localNy(), domain_.localNx() - 2, 0),
			send_right_halo.data(),
			domain_.localNy() * sizeof(double)
		);
	}
	if (domain_.leftNeighbor() != MPI_PROC_NULL) {
		send_left_halo.resize(domain_.localNy());
		recv_left_halo.resize(domain_.localNy());

		s = ind(domain_.localNy(), 1, 0);
		send_left_halo[0] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
			+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
			+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1]) + rev(cond_[s]) * x[s]);

		for (int j = 1; j < domain_.localNy() - 1; ++j) {
			s = ind(domain_.localNy(), 1, j);
			send_left_halo[j] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
				+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
				+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]));
		}

		s = ind(domain_.localNy(), 1, domain_.localNy() - 1);
		send_left_halo[domain_.localNy() - 1] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
			+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
			+ hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]) + rev(cond_[s]) * x[s]);

		//cblas_dcopy(domain_.localNy(), send_left_halo.data(), 1, y + ind(domain_.localNy(), 1, 0), 1);
		memcpy(
			y + ind(domain_.localNy(), 1, 0),
			send_left_halo.data(),
			domain_.localNy() * sizeof(double)
		);
	}

	MPI_Isend(send_top_halo.data(), domain_.localNx(), MPI_DOUBLE, domain_.topNeighbor(),
		TAG_BOTTOM_HALO, MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(recv_top_halo.data(), domain_.localNx(), MPI_DOUBLE, domain_.topNeighbor(),
		TAG_TOP_HALO, MPI_COMM_WORLD, &requests[1]);
	MPI_Isend(send_bottom_halo.data(), domain_.localNx(), MPI_DOUBLE, domain_.bottomNeighbor(),
		TAG_TOP_HALO, MPI_COMM_WORLD, &requests[2]);
	MPI_Irecv(recv_bottom_halo.data(), domain_.localNx(), MPI_DOUBLE, domain_.bottomNeighbor(),
		TAG_BOTTOM_HALO, MPI_COMM_WORLD, &requests[3]);
	MPI_Isend(send_right_halo.data(), domain_.localNy(), MPI_DOUBLE, domain_.rightNeighbor(),
		TAG_LEFT_HALO, MPI_COMM_WORLD, &requests[4]);
	MPI_Irecv(recv_right_halo.data(), domain_.localNy(), MPI_DOUBLE, domain_.rightNeighbor(),
		TAG_RIGHT_HALO, MPI_COMM_WORLD, &requests[5]);
	MPI_Isend(send_left_halo.data(), domain_.localNy(), MPI_DOUBLE, domain_.leftNeighbor(),
		TAG_RIGHT_HALO, MPI_COMM_WORLD, &requests[6]);
	MPI_Irecv(recv_left_halo.data(), domain_.localNy(), MPI_DOUBLE, domain_.leftNeighbor(),
		TAG_LEFT_HALO, MPI_COMM_WORLD, &requests[7]);

	for (int i = 1 + domain_.offsetLeftX(); i < domain_.localNx() - 1 - domain_.offsetRightX(); ++i) {
		for (int j = 1 + domain_.offsetBottomY(); j < domain_.localNy() - 1 - domain_.offsetTopY(); ++j) {
			s = ind(domain_.localNy(), i, j);
			y[s] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()])
				+ reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
				+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]));
		}
	}

	if (domain_.topNeighbor() == MPI_PROC_NULL) {
		for (int i = 1 + domain_.offsetLeftX(); i < domain_.localNx() - 1 - domain_.offsetRightX(); ++i) {
			s = ind(domain_.localNy(), i, domain_.localNy() - 1);
			y[s] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
				+ rev(cond_[s]) * x[s])
				+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()])
				+ reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
		}
	}
	if (domain_.bottomNeighbor() == MPI_PROC_NULL) {
		for (int i = 1 + domain_.offsetLeftX(); i < domain_.localNx() - 1 - domain_.offsetRightX(); ++i) {
			s = ind(domain_.localNy(), i, 0);
			y[s] = hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ rev(cond_[s]) * x[s])
				+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()])
					+ reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
		}
	}
	if (domain_.rightNeighbor() == MPI_PROC_NULL) {
		for (int j = 1 + domain_.offsetBottomY(); j < domain_.localNy() - 1 - domain_.offsetTopY(); ++j) {
			s = ind(domain_.localNy(), domain_.localNx() - 1, j);
			y[s] = hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]))
				+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]));
		}
	}
	if (domain_.leftNeighbor() == MPI_PROC_NULL) {
		for (int j = 1 + domain_.offsetBottomY(); j < domain_.localNy() - 1 - domain_.offsetTopY(); ++j) {
			s = ind(domain_.localNy(), 0, j);
			y[s] = hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]))
				+ hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
				+ reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1]));
		}
	}

	if (domain_.topNeighbor() == MPI_PROC_NULL && domain_.rightNeighbor() == MPI_PROC_NULL) {
		s = ind(domain_.localNy(), domain_.localNx() - 1, domain_.localNy() - 1);
		y[s] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ rev(cond_[s]) * x[s])
			+ hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
	}
	if (domain_.bottomNeighbor() == MPI_PROC_NULL && domain_.rightNeighbor() == MPI_PROC_NULL) {
		s = ind(domain_.localNy(), domain_.localNx() - 1, 0);
		y[s] = hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
			+ rev(cond_[s]) * x[s])
			+ hx2_ * (reav(cond_[s], cond_[s - domain_.localNy()]) * (x[s] - x[s - domain_.localNy()]));
	}
	if (domain_.topNeighbor() == MPI_PROC_NULL && domain_.leftNeighbor() == MPI_PROC_NULL) {
		s = ind(domain_.localNy(), 0, domain_.localNy() - 1);
		y[s] = hy2_ * (reav(cond_[s], cond_[s - 1]) * (x[s] - x[s - 1])
			+ rev(cond_[s]) * x[s])
			+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]));
	}
	if (domain_.bottomNeighbor() == MPI_PROC_NULL && domain_.leftNeighbor() == MPI_PROC_NULL) {
		s = ind(domain_.localNy(), 0, 0);
		y[s] = hy2_ * (reav(cond_[s], cond_[s + 1]) * (x[s] - x[s + 1])
			+ rev(cond_[s]) * x[s])
			+ hx2_ * (reav(cond_[s], cond_[s + domain_.localNy()]) * (x[s] - x[s + domain_.localNy()]));
	}

	MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);

	/*if (domain_.topNeighbor() != MPI_PROC_NULL) cblas_dcopy(domain_.localNx(), recv_top_halo.data(), 1, y + ind(domain_.localNy(), 0, domain_.localNy() - 1), domain_.localNy());
	if (domain_.bottomNeighbor() != MPI_PROC_NULL) cblas_dcopy(domain_.localNx(), recv_bottom_halo.data(), 1, y + ind(domain_.localNy(), 0, 0), domain_.localNy());
	if (domain_.rightNeighbor() != MPI_PROC_NULL) cblas_dcopy(domain_.localNy(), recv_right_halo.data(), 1, y + ind(domain_.localNy(), domain_.localNx() - 1, 0), 1);
	if (domain_.leftNeighbor() != MPI_PROC_NULL) cblas_dcopy(domain_.localNy(), recv_left_halo.data(), 1, y + ind(domain_.localNy(), 0, 0), 1);*/

	if (domain_.topNeighbor() != MPI_PROC_NULL) {
		int nx = domain_.localNx();
		int ny = domain_.localNy();

		const double* src = recv_top_halo.data();
		for (int i = 0; i < nx; ++i) {
			y[ind(ny, i, ny - 1)] = src[i];
		}
	}

	if (domain_.bottomNeighbor() != MPI_PROC_NULL) {
		int nx = domain_.localNx();
		int ny = domain_.localNy();

		const double* src = recv_bottom_halo.data();
		for (int i = 0; i < nx; ++i) {
			y[ind(ny, i, 0)] = src[i];
		}
	}

	if (domain_.rightNeighbor() != MPI_PROC_NULL) {
		memcpy(
			y + ind(domain_.localNy(), domain_.localNx() - 1, 0),
			recv_right_halo.data(),
			domain_.localNy() * sizeof(double)
		);
	}

	if (domain_.leftNeighbor() != MPI_PROC_NULL) {
		memcpy(
			y + ind(domain_.localNy(), 0, 0),
			recv_left_halo.data(),
			domain_.localNy() * sizeof(double)
		);
	}

	return;
}