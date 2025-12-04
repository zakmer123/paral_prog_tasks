#define _USE_MATH_DEFINES  
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <vector>

using namespace std;

#define indA(y, x, n2) ((y) * (n2) + (x))  
#define indB(y, x, n2) ((x) * (n2) + (y))  
#define indC(y, x, n3) ((y) * (n3) + (x))   

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n1 = argc < 2 ? 5 : atoi(argv[1]);   
    int n2 = argc < 2 ? 7 : atoi(argv[2]);   
    int n3 = argc < 2 ? 5 : atoi(argv[3]);   
    int p1 = argc < 2 ? 1 : atoi(argv[4]);   
    int p2 = argc < 2 ? 1 : atoi(argv[5]);   

    //задаем тополгию
    int dims[2] = { p1, p2 };
    int periods[2] = { 0, 0 };
    MPI_Comm grid_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);

    //коммуникаторы
    MPI_Comm row_comm, col_comm;
    int remain_dims_row[2] = { false, true };  
    int remain_dims_col[2] = { true, false };  
    MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
    MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);

    //размеры подматриц
    int local_n1 = n1 / p1;   
    int local_n3 = n3 / p2;   

    //подматрицы
    vector<double> A_local(local_n1 * n2);
    vector<double> B_local(n2 * local_n3);
    vector<double> C_local(local_n1 * local_n3, 0.0);

    //инициализация
    vector<double> A, B, C;
    if (rank == 0) {
        A.resize(n1 * n2, 0.0);
        B.resize(n2 * n3, 0.0);
        C.resize(n1 * n3, 0.0);

        ////A единичная
        //for (int i = 0; i < n1 && i < n2; i++) {
        //    A[indA(i, i, n2)] = 1.0;
        //}

        ////B единичная
        //for (int i = 0; i < n2 && i < n3; i++) {
        //    B[indB(i, i, n2)] = 1.0;
        //}

        //A ортогональная
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                double norm = sqrt(2.0 / (n2 + 1));
                A[indA(i, j, n2)] = norm * sin(M_PI * (i + 1) * (j + 1) / (n2 + 1));
            }
        }

        //B ортогональная
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n3; j++) {
                double norm = sqrt(2.0 / (n2 + 1));
                B[indB(i, j, n2)] = norm * sin(M_PI * (j + 1) * (i + 1) / (n2 + 1));
            }
        }

        /*cout << "Matrix A (" << n1 << "x" << n2 << "):" << endl;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                cout << A[indA(i, j, n2)] << " ";
            }
            cout << endl;
        }

        cout << "\nMatrix B (" << n2 << "x" << n3 << "):" << endl;
        for (int i = 0; i < n2; i++) {
            for (int j = 0; j < n3; j++) {
                cout << B[indB(i, j, n2)] << " ";
            }
            cout << endl;
        }*/
    }

    double start = MPI_Wtime();

    //A scatter
    if (coords[1] == 0) {
        MPI_Scatter(A.data(), local_n1 * n2, MPI_DOUBLE,
            A_local.data(), local_n1 * n2, MPI_DOUBLE,
            0, col_comm);
    }

    //A bcast
    MPI_Bcast(A_local.data(), local_n1 * n2, MPI_DOUBLE, 0, row_comm);

    //B scatter
    if (coords[0] == 0) {
        MPI_Scatter(B.data(), n2 * local_n3, MPI_DOUBLE,
            B_local.data(), n2 * local_n3, MPI_DOUBLE,
            0, row_comm);
    }

    //B bcast
    MPI_Bcast(B_local.data(), n2 * local_n3, MPI_DOUBLE, 0, col_comm);

    //умножение подматриц
    for (int i = 0; i < local_n1; i++) {
        for (int j = 0; j < local_n3; j++) {
            double sum = 0.0;
            for (int k = 0; k < n2; k++) {
                sum += A_local[indA(i, k, n2)] * B_local[indB(k, j, n2)];
            }
            C_local[indC(i, j, local_n3)] = sum;
        }
    }

    //
    MPI_Datatype block_type;
    MPI_Type_vector(local_n1, local_n3, n3, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);

    //
    MPI_Datatype resized_block_type;
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);

    //
    vector<int> recvcounts(size, 1);  
    vector<int> displs(size);

    //смещения
    for (int i = 0; i < size; i++) {
        int coords[2];
        MPI_Cart_coords(grid_comm, i, 2, coords);
        displs[i] = coords[0] * local_n1 * n3 + coords[1] * local_n3;
    }

    //сбор
    MPI_Gatherv(C_local.data(), local_n1 * local_n3, MPI_DOUBLE,
        C.data(), recvcounts.data(), displs.data(), resized_block_type,
        0, grid_comm);

    double end = MPI_Wtime();

    MPI_Type_free(&block_type);
    MPI_Type_free(&resized_block_type);

    //консоль
    if (rank == 0) {
        /*cout << "\nResult matrix C (" << n1 << "x" << n3 << "):" << endl;
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n3; j++) {
                if (abs(C[indC(i, j, n3)]) < 1e-12) C[indC(i, j, n3)] = 0.0;
                cout << C[indC(i, j, n3)] << " ";
            }
            cout << endl;
        }*/
        //вычитание 1 из диагонали
        for (int i = 0; i < min(n1, n3); i++) {
            C[indC(i, i, n3)] -= 1.0;
        }
        //норма
        double norm = 0.0;
        for (int i = 0; i < n1 * n3; i++) {
            norm += C[i] * C[i];
        }
        norm = sqrt(norm);
        cout << norm << ' ' << n1 << ' ' << n2 << ' ' << n3 << ' ' << p1 << ' ' << p2 << ' ' << end - start << endl;
    }

    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    MPI_Finalize();
    return 0;
}