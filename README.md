# LeetGPU

LeetGPU is a collection of CUDA C++ examples designed to help you build a solid foundation in GPU programming. This repository contains several small programs that demonstrate fundamental CUDA concepts and basic operations using NVIDIA GPUs.

## Contents

- **AreaOfTriangle.cu**  
  Computes the area of a triangle using the formula: Area = 0.5 x base x height

- **BasicMathsOps.cu**  
  Performs basic arithmetic operations (addition, subtraction, multiplication, and division) on two numbers using a CUDA kernel.

- **CheckOddEven.cu**  
  Checks if a given integer is odd or even on the GPU.

- **Factorial.cu**  
  Computes the factorial of a given number using a parallel reduction approach.

- **GenerateRandomNumbers.cu**  
  Uses the CURAND library to generate random numbers on the GPU.

- **HelloGPU.cu**  
  A simple "Hello, GPU!" program to get started with CUDA programming.

- **MatrixOps.cu**  
  Demonstrates several matrix operations including addition, multiplication, transpose, and inversion using CUDA kernels.

- **Squareroot.cu**  
  Computes the square root of a given number using CUDA device functions.

- **SwapTwoVariables.cu**  
  Swaps two variables using a CUDA kernel.

## Requirements

- NVIDIA GPU with CUDA capability.
- CUDA Toolkit installed (includes the `nvcc` compiler).
- For **GenerateRandomNumbers.cu**, the CURAND library is required.

## How to Compile and Run

Each program can be compiled using NVIDIA's CUDA compiler (`nvcc`). For example, to compile and run `HelloGPU.cu`:

```bash
nvcc -o HelloGPU HelloGPU.cu
./HelloGPU
```

Replace `HelloGPU` with the corresponding file name (without the `.cu` extension) for other programs.

## Description

LeetGPU is designed as a practical guide for beginners to learn CUDA programming. It covers a variety of operations from simple arithmetic and memory management to more complex tasks like matrix operations and parallel reductions. By studying these examples, you'll gain hands-on experience with:

- Kernel launches and thread organization (grids, blocks, threads).
- Memory management (global, shared, constant memory).
- Synchronization and error checking in CUDA.
- Using CUDA libraries (e.g., CURAND) for specialized operations.

Whether you're new to GPU programming or looking to solidify your understanding of CUDA fundamentals, these examples provide a great starting point.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
