#include <iostream>
#include <cuda_runtime.h>

// Kernel for matrix addition: C = A + B
__global__ void matrixAdd(const float *A, const float *B, float *C, int N, int M) {
    // Compute row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        // Compute linear index and perform addition
        C[row * M + col] = A[row * M + col] + B[row * M + col];
    }
}

// Kernel for matrix multiplication: C = A * B
// A is of size (A_rows x A_cols) and B is of size (A_cols x B_cols)
__global__ void matrixMul(const float *A, const float *B, float *C,
                          int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        // Multiply row of A with column of B
        for (int k = 0; k < A_cols; k++) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

// Kernel for matrix transpose: B = A^T
// A is of size (N x M) and B will be (M x N)
__global__ void matrixTranspose(const float *A, float *B, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        // Transpose: switch row and column indices
        B[col * N + row] = A[row * M + col];
    }
}

// Kernel for matrix inversion using Gauss-Jordan elimination.
// This kernel is launched with one thread and works only for small matrices.
// It computes the inverse of matrix A (of size N x N) and writes the result to inv.
__global__ void matrixInverse(const float *A, float *inv, int N) {
    const int MAX_N = 32;  // Maximum supported matrix size (adjust as needed)
    // Create an augmented matrix [A | I] stored in a local array.
    float aug[MAX_N][2 * MAX_N];
    
    // Load A into the left half of aug and initialize the right half as the identity matrix.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            aug[i][j] = A[i * N + j];
        }
        for (int j = N; j < 2 * N; j++) {
            aug[i][j] = (i == (j - N)) ? 1.0f : 0.0f;
        }
    }
    
    // Perform Gauss-Jordan elimination to convert [A | I] into [I | A^-1]
    for (int i = 0; i < N; i++) {
        // Get the pivot element.
        float pivot = aug[i][i];
        // Normalize the pivot row.
        for (int j = 0; j < 2 * N; j++) {
            aug[i][j] /= pivot;
        }
        // Eliminate the current column elements in all other rows.
        for (int k = 0; k < N; k++) {
            if (k != i) {
                float factor = aug[k][i];
                for (int j = 0; j < 2 * N; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    
    // Copy the inverse matrix (right half of aug) into the output array.
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            inv[i * N + j] = aug[i][j + N];
        }
    }
}

// Helper function to print a matrix stored in a 1D array
void printMatrix(const float *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Define matrix dimensions (using 3x3 matrices for this example)
    int N = 3, M = 3;
    int size = N * M * sizeof(float);
    
    // Define host matrices
    // Matrix A
    float h_A[9] = {
         1, 2, 3,
         0, 1, 4,
         5, 6, 0
    };
    // Matrix B
    float h_B[9] = {
         7, 8, 9,
         1, 3, 5,
         2, 4, 6
    };
    // Array to hold results
    float h_C[9];
    
    // Allocate device memory for matrices A, B, and C.
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // Copy host matrices A and B to device memory.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions for 2D kernels.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // --- Matrix Addition ---
    // Compute C = A + B
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "Matrix Addition (A + B):" << std::endl;
    printMatrix(h_C, N, M);
    
    // --- Matrix Multiplication ---
    // Compute C = A * B
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, M);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "\nMatrix Multiplication (A * B):" << std::endl;
    printMatrix(h_C, N, M);
    
    // --- Matrix Transpose ---
    // Compute C = A^T
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N, M);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "\nMatrix Transpose (A^T):" << std::endl;
    // Note: Transposed dimensions are (M x N)
    printMatrix(h_C, M, N);
    
    // --- Matrix Inversion ---
    // Compute C = A^-1 using Gauss-Jordan elimination.
    // Launch this kernel with a single thread.
    matrixInverse<<<1, 1>>>(d_A, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    std::cout << "\nMatrix Inverse (A^-1):" << std::endl;
    printMatrix(h_C, N, M);
    
    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
