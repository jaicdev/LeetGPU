#include <iostream>         // For standard I/O operations
#include <cuda_runtime.h>   // For CUDA runtime API functions

// CUDA kernel to swap two integers.
// This kernel swaps the values pointed to by 'a' and 'b'.
__global__ void swapKernel(int* a, int* b) {
    // Use a temporary variable to hold the value of *a.
    int temp = *a;
    // Assign the value of *b to *a.
    *a = *b;
    // Assign the original value of *a (stored in temp) to *b.
    *b = temp;
}

int main() {
    // Host variables to be swapped.
    int h_a = 10;
    int h_b = 20;

    // Device pointers for the two integers.
    int *d_a, *d_b;

    // Allocate memory on the device for each integer.
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));

    // Copy the values from the host to the device memory.
    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with a single block and one thread since only one swap is needed.
    swapKernel<<<1, 1>>>(d_a, d_b);

    // Copy the swapped values back from the device to the host.
    cudaMemcpy(&h_a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_b, d_b, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the swapped values.
    std::cout << "After swap: a = " << h_a << ", b = " << h_b << std::endl;

    // Free the allocated device memory.
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;  // Indicate successful execution.
}

