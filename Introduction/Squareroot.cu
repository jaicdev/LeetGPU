#include <iostream>         // For standard input/output
#include <cuda_runtime.h>   // For CUDA runtime API

// CUDA kernel to compute the square root of a number.
// The computed square root is stored in the memory location pointed by 'result'.
__global__ void sqrtKernel(float number, float* result) {
    // Compute the square root using the CUDA device function sqrtf.
    *result = sqrtf(number);
}

int main() {
    // Define the number for which the square root will be computed.
    float number = 15.0f;
    
    // Variable to hold the result on the host.
    float hostResult;
    
    // Device pointer to store the result on the GPU.
    float* deviceResult;
    
    // Allocate device memory to store one float.
    cudaMalloc(&deviceResult, sizeof(float));
    
    // Launch the kernel with one block and one thread.
    // This is sufficient since we are computing the square root of one number.
    sqrtKernel<<<1, 1>>>(number, deviceResult);
    
    // Copy the result from the device memory to host memory.
    cudaMemcpy(&hostResult, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the computed square root.
    std::cout << "Square root of " << number << " is " << hostResult << std::endl;
    
    // Free the allocated device memory.
    cudaFree(deviceResult);
    
    return 0;  // Return 0 to indicate successful execution.
}
