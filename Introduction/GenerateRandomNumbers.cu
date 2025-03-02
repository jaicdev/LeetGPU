#include <iostream>         // For standard input/output operations
#include <cuda_runtime.h>   // For CUDA runtime API functions
#include <curand_kernel.h>  // For CURAND functions and types

// CUDA kernel that generates random numbers using CURAND.
// Each thread generates one random float and stores it in the 'results' array.
__global__ void generateRandomNumbers(float* results, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out-of-bounds.
    if (idx < n) {
        // Declare a CURAND state
        curandState state;
        // Initialize the CURAND state using a seed, a sequence number (idx), and an offset.
        curand_init(seed, idx, 0, &state);
        
        // Generate a random float uniformly distributed in the interval (0, 1]
        results[idx] = curand_uniform(&state);
    }
}

int main() {
    // Number of random numbers to generate.
    int n = 5;
    
    // Allocate host memory for the random numbers.
    float* h_results = new float[n];
    
    // Allocate device memory for the random numbers.
    float* d_results;
    cudaMalloc(&d_results, n * sizeof(float));
    
    // Define kernel launch parameters.
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;  // Ensures we cover all 'n' numbers.
    
    // Launch the kernel. Use current time as seed.
    generateRandomNumbers<<<gridSize, blockSize>>>(d_results, n, time(0));
    
    // Copy the generated random numbers from device to host.
    cudaMemcpy(h_results, d_results, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the generated random numbers.
    std::cout << "Random numbers generated on the GPU:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << h_results[i] << std::endl;
    }
    
    // Free device and host memory.
    cudaFree(d_results);
    delete[] h_results;
    
    return 0;
}

