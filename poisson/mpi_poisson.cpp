#include <iostream>

#include "DomainDecomposition.h"
#include "ConductivityField.h"
#include "SolutionField.h"
#include "RightHandSideField.h"
#include "LaplacianOperator.h"
#include "CGSolver.h"

void procedure(int argc, char** argv, int rank, int size) {
    int prx = std::stoi(argv[1]);
    int pry = std::stoi(argv[2]);
    int Nx = std::stoi(argv[3]);
    int Ny = std::stoi(argv[4]);
    int scale = std::stoi(argv[5]);
    int por = std::stoi(argv[6]);
    int cx = std::stoi(argv[7]);;
    int cy = std::stoi(argv[8]);;
    int num = std::stoi(argv[9]);
    double sigma = std::stod(argv[10]);

    DomainDecomposition domain(MPI_COMM_WORLD, prx, pry, Nx, Ny);

    std::string filename = "twophase/N=" + std::to_string(Nx) + "; por=" + std::to_string((int)por) + "; cx=" + std::to_string((int)cx) + "; cy=" + std::to_string((int)cy) + "; num=" + std::to_string(num) + ".bin";
    ConductivityField conductivity(domain, filename);
    conductivity.considerContrast(sigma);

    conductivity.resizeConductivityField(scale);
    domain.resizeDomain(scale);

    SolutionField solution(domain);
    solution.initialize();
    RightHandSideField rhs(domain, conductivity);
    LaplacianOperator laplace(domain, conductivity);

    CGSolver cg(domain, laplace, domain.globalNxNy(), pow(10, -8));
    double start_time0 = MPI_Wtime();
    cg.solve(rhs.data(), solution.data());
    double end_time0 = MPI_Wtime();
    if (rank == 0) std::cout << prx << ' ' << pry << ' ' << scale * Nx << ' ' << scale * Ny << ' ' << sigma << ' ' << cg.elapsed_iter() << ' ' << end_time0 - start_time0 << std::endl;

    return;
}

// mpiexec -n 4 mpi_poisson.exe 4 1 100 100 1 30 5 5 0 -1
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    procedure(argc, argv, rank, size);

    MPI_Finalize();

    return 0;
}