#include <iostream>         // For standard I/O operations
#include <cuda_runtime.h>   // For CUDA runtime API functions

// CUDA kernel to perform block-level multiplication reduction.
// Each block loads elements into shared memory and then reduces them (by multiplying)
// to a single product. The result from each block is written to d_out.
__global__ void product_reduce(unsigned long long* d_in, unsigned long long* d_out, int n) {
    extern __shared__ unsigned long long sdata[];

    // Thread index within the block.
    int tid = threadIdx.x;
    // Global index: each thread processes two elements to improve efficiency.
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Initialize local product to 1 (multiplicative identity).
    unsigned long long myProduct = 1ULL;

    // Load the first element if within bounds.
    if (i < n)
        myProduct = d_in[i];
    // If a second element exists, multiply it.
    if (i + blockDim.x < n)
        myProduct *= d_in[i + blockDim.x];

    // Store the computed product in shared memory.
    sdata[tid] = myProduct;
    __syncthreads();

    // Perform reduction in shared memory.
    // Each step halves the number of active threads.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] *= sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's product to global memory.
    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

int main() {
    // Define the number for which factorial is to be computed.
    // For example, n = 5 computes 5! (5 factorial).
    const int n = 5;
    // Calculate the total number of elements (numbers 1 through n).
    size_t size = n * sizeof(unsigned long long);

    // Allocate host memory and initialize the array with numbers from 1 to n.
    unsigned long long* h_data = new unsigned long long[n];
    for (int i = 0; i < n; i++) {
        h_data[i] = static_cast<unsigned long long>(i + 1);
    }

    // Allocate device memory for the input array.
    unsigned long long *d_in, *d_out;
    cudaMalloc(&d_in, size);
    // Allocate maximum required memory for the output array.
    cudaMalloc(&d_out, size);

    // Copy the input data from host to device.
    cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice);

    // Set up reduction parameters.
    int current_size = n;
    int threads = 256;  // Threads per block.
    int blocks;         // Number of blocks per grid.

    // Iteratively perform multiplication reduction until one value remains.
    while (current_size > 1) {
        // Each thread handles 2 elements, so compute the number of blocks needed.
        blocks = (current_size + threads * 2 - 1) / (threads * 2);
        // Launch the kernel with dynamic shared memory size (threads * sizeof(unsigned long long)).
        product_reduce<<<blocks, threads, threads * sizeof(unsigned long long)>>>(d_in, d_out, current_size);
        
        // The output of this reduction becomes the input for the next iteration.
        current_size = blocks;
        
        // Swap pointers: d_in now points to the reduced results.
        unsigned long long* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    // Copy the final result (factorial) from device to host.
    unsigned long long factorial;
    cudaMemcpy(&factorial, d_in, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Print the computed factorial.
    std::cout << "Factorial of " << n << " is: " << factorial << std::endl;

    // Clean up allocated memory.
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;

    return 0;  // Indicate successful execution.
}
