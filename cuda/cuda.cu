#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// 
__global__ void matrixVectorMultiply(const double* matrix, const double* vec, double* result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col] * vec[col];
        }
        result[row] = sum;
    }
}

// 
void cudaMatrixVectorMultiply(const std::vector<double>& matrix,
    const std::vector<double>& vec, std::vector<double>& result,
    int rows, int cols, int blockSize) {
    double* d_matrix, * d_vector, * d_result;
    cudaMalloc(&d_matrix, rows * cols * sizeof(double));
    cudaMalloc(&d_vector, cols * sizeof(double));
    cudaMalloc(&d_result, rows * sizeof(double));
    cudaMemcpy(d_matrix, matrix.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vec.data(), cols * sizeof(double), cudaMemcpyHostToDevice);

    int numBlocks = (rows + blockSize - 1) / blockSize;
    matrixVectorMultiply <<<numBlocks, blockSize >>> (d_matrix, d_vector, d_result, rows, cols);

    cudaMemcpy(result.data(), d_result, rows * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix); cudaFree(d_vector); cudaFree(d_result);
}

// 
std::vector<double> cpuMatrixVectorMultiply(const std::vector<double>& matrix, const std::vector<double>& vec, int rows, int cols) {
    std::vector<double> result(rows, 0.0);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result[i] += matrix[i * cols + j] * vec[j];
    return result;
}

// умножение матрицы на вектор
void Ax(const std::vector<double>& matrix, const std::vector<double>& x, std::vector<double>& y, int rows, int cols,
    bool use_gpu, int blockSize) {
    if (use_gpu) {
        cudaMatrixVectorMultiply(matrix, x, y, rows, cols, blockSize);
    }
    else {
        y = cpuMatrixVectorMultiply(matrix, x, rows, cols);
    }
}

// скалярное произведение
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

// решение СЛАУ
void CG(const std::vector<double>& matrix, const std::vector<double>& f, std::vector<double>& x,
    int rows, int cols, double eps, bool use_gpu, int blockSize) {
    int max_iter = 20000;
    int iter = 0;
    std::vector<double> r = f;
    std::vector<double> p = r;
    std::vector<double> Ap(cols);
    double rr = dot(r, r);
    eps *= sqrt(dot(f, f));
    while (sqrt(rr) > eps && iter < max_iter) {
        Ax(matrix, p, Ap, rows, cols, use_gpu, blockSize);
        double alpha = rr / dot(p, Ap);
        for (int i = 0; i < cols; ++i) x[i] += alpha * p[i];
        for (int i = 0; i < cols; ++i) r[i] -= alpha * Ap[i];
        double rr_new = dot(r, r);
        double beta = rr_new / rr;
        for (int i = 0; i < cols; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
        iter++;
    }
    cout << iter << ' ';
}

// корретировка диагонали матрицы
void correct_diag(int rows, int cols, std::vector<double>& matrix) {
    for (int i = 0; i < cols; ++i) {
        matrix[i * cols + i] += (double)(i + 1) / (10 * rows);
    }
}

// пргрев CUDA
void warmupCuda() {
    const int size = 1000;
    double* d_temp;
    cudaMalloc(&d_temp, size * sizeof(double));
    cudaMemset(d_temp, 0, size * sizeof(double));
    cudaFree(d_temp);
    cudaDeviceSynchronize();
}

//
int main(int argc, char** argv) {
    int rows = argc >= 2 ? stoi(argv[1]) : 1024;
    int cols = rows; 
    int blockSize = argc >= 2 ? stoi(argv[2]) : 256;
    double eps = 1e-8;

    vector<double> matrix(rows * cols, 1.0);
    correct_diag(rows, cols, matrix);

    vector<double> f(cols, 1.0);
    vector<double> x_cpu(cols, 0.0);
    vector<double> x_gpu(cols, 0.0);

    warmupCuda();

    cout << rows << ' ';
    auto start_cpu = chrono::high_resolution_clock::now();
    CG(matrix, f, x_cpu, rows, cols, eps, false, blockSize);
    auto time_cpu = chrono::high_resolution_clock::now() - start_cpu;
    cout << chrono::duration_cast<chrono::milliseconds>(time_cpu).count()  << ' ';

    auto start_gpu = chrono::high_resolution_clock::now();
    CG(matrix, f, x_gpu, rows, cols, eps, true, blockSize);
    auto time_gpu = chrono::high_resolution_clock::now() - start_gpu;
    cout << chrono::duration_cast<chrono::milliseconds>(time_gpu).count()  << ' ';

    return 0;
}

