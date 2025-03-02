#include <iostream>         // For standard input/output functions
#include <cuda_runtime.h>   // For CUDA runtime API functions

// CUDA kernel to compute the area of a triangle.
// The formula used is: Area = 0.5 * base * height
__global__ void triangleAreaKernel(float base, float height, float* result) {
    // Compute the area and store it in the device memory pointed to by 'result'.
    *result = 0.5f * base * height;
}

int main() {
    // Define the base and height of the triangle.
    float base = 10.0f;
    float height = 5.0f;

    // Host variable to store the computed area.
    float hostArea;

    // Device pointer to store the area computed on the GPU.
    float* deviceArea;

    // Allocate device memory to hold one float value (the area).
    cudaMalloc(&deviceArea, sizeof(float));

    // Launch the kernel with one block and one thread.
    // A single thread is sufficient since we are performing a single calculation.
    triangleAreaKernel<<<1, 1>>>(base, height, deviceArea);

    // Copy the computed area from device memory back to the host.
    cudaMemcpy(&hostArea, deviceArea, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the computed area.
    std::cout << "Area of the triangle with base " << base 
              << " and height " << height 
              << " is: " << hostArea << std::endl;

    // Free the allocated device memory.
    cudaFree(deviceArea);

    return 0;  // Return 0 to indicate successful execution.
}
