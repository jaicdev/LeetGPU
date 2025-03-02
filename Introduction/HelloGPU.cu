#include <iostream>         // Include I/O stream library for printing output
#include <cuda_runtime.h>   // Include CUDA runtime API for CUDA functions
#include <cstring>          // Include C-string library for strlen()

// Declare constant memory on the device to hold a message of up to 20 characters.
// Constant memory is read-only and cached, making it efficient when all threads access the same data.
__constant__ char d_message[20];

// Define the CUDA kernel function 'welcome' that copies data from constant memory to global memory.
__global__ void welcome(char* msg) {
    // Calculate the unique index for each thread.
    // blockIdx.x is the block index, blockDim.x is the number of threads per block,
    // and threadIdx.x is the thread index within the block.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy one character from the constant memory (d_message) to the global memory array (msg)
    msg[idx] = d_message[idx];
}

int main() {
    // Declare pointers for device (GPU) memory and host (CPU) memory.
    char* d_msg;
    char* h_msg;
    
    // Define the message string to be copied.
    const char message[] = "Welcome to LeetGPU!";
    // Compute the length of the message including the null terminator.
    const int length = strlen(message) + 1;
    
    // Allocate memory on the host (CPU) to store the message.
    h_msg = (char*)malloc(length * sizeof(char));
    
    // Allocate memory on the device (GPU) for the message.
    cudaMalloc(&d_msg, length * sizeof(char));
    
    // Copy the message from the host to the GPU's constant memory (d_message).
    cudaMemcpyToSymbol(d_message, message, length);
    
    // Launch the 'welcome' kernel on the GPU.
    // The kernel is launched with one block and 'length' threads, where each thread copies one character.
    welcome<<<1, length>>>(d_msg);
    
    // Copy the message from the GPU's global memory (d_msg) back to the host memory (h_msg).
    cudaMemcpy(h_msg, d_msg, length * sizeof(char), cudaMemcpyDeviceToHost);
    
    // Ensure the string is properly null-terminated.
    h_msg[length-1] = '\0';
    
    // Print the copied message to the console.
    std::cout << h_msg << "\n";
    
    // Free the allocated host memory.
    free(h_msg);
    
    // Free the allocated device memory.
    cudaFree(d_msg);
    
    // Return 0 to indicate successful execution.
    return 0;
}
