#include <iostream>         // For standard input/output
#include <cuda_runtime.h>   // For CUDA runtime functions

// CUDA kernel to check if a number is even or odd.
// It takes an integer 'number' and writes the result (1 for even, 0 for odd)
// to the memory location pointed to by 'result'.
__global__ void checkOddEven(int number, int* result) {
    // Use modulus operator to check for even number.
    // If number is divisible by 2, it is even.
    if (number % 2 == 0) {
        *result = 1;  // Even
    } else {
        *result = 0;  // Odd
    }
}

int main() {
    // Define the number to be checked.
    int number = 12;  // You can change this value to test with different numbers
    

    // Host variable to store the result after copying from device.
    int hostResult;

    // Device pointer to store the result computed on the GPU.
    int* deviceResult;

    // Allocate memory on the device for one integer.
    cudaMalloc(&deviceResult, sizeof(int));

    // Launch the kernel with a single block and one thread,
    // since only one number is being processed.
    checkOddEven<<<1, 1>>>(number, deviceResult);

    // Copy the result from device memory to host memory.
    cudaMemcpy(&hostResult, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);

    // Interpret and print the result.
    if (hostResult == 1) {
        std::cout << number << " is Even." << std::endl;
    } else {
        std::cout << number << " is Odd." << std::endl;
    }

    // Free the allocated device memory.
    cudaFree(deviceResult);

    return 0;  // Return 0 to indicate successful execution.
}
