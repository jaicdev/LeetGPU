#include <iostream>         // For standard input/output
#include <cuda_runtime.h>   // For CUDA runtime functions

// CUDA kernel that performs basic mathematical operations on two numbers.
// The results are stored in the 'results' array as follows:
// results[0] = addition, results[1] = subtraction,
// results[2] = multiplication, results[3] = division.
__global__ void basicMathOps(float a, float b, float* results) {
    // Perform addition
    results[0] = a + b;
    
    // Perform subtraction
    results[1] = a - b;
    
    // Perform multiplication
    results[2] = a * b;
    
    // Perform division with a check to avoid division by zero
    if(b != 0)
        results[3] = a / b;
    else
        results[3] = 0.0f;  // Define division by zero result as 0.0 (or handle as needed)
}

int main() {
    // Define two numbers for the operations.
    float a = 10.0f;
    float b = 2.0f;

    // Host array to hold the results from the device.
    float h_results[4];

    // Pointer for the results array on the device.
    float* d_results;

    // Allocate device memory for 4 float elements.
    cudaMalloc((void**)&d_results, 4 * sizeof(float));

    // Launch the kernel with a single block containing one thread.
    // This thread computes all four operations.
    basicMathOps<<<1, 1>>>(a, b, d_results);

    // Copy the results from device memory to host memory.
    cudaMemcpy(h_results, d_results, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the results.
    std::cout << "Addition: " << h_results[0] << "\n";
    std::cout << "Subtraction: " << h_results[1] << "\n";
    std::cout << "Multiplication: " << h_results[2] << "\n";
    std::cout << "Division: " << h_results[3] << "\n";

    // Free the allocated device memory.
    cudaFree(d_results);

    return 0;
}
