## Project Overview  
The goal of this project is to explore and implement different versions of a neural network for the classic digit-classification problem (e.g., using the MNIST dataset) in pure C, both for CPU and GPU. The implementations vary from hand-crafted matrix multiplications to leveraging highly optimized libraries (ie cBlas and cuBlas).  
This provides insight into:  
- how a basic neural network works under the hood  
- performance trade-offs between pure code vs. library calls  
- CPU vs. GPU execution for numerical computing  
---

## Table of Contents  
- [Project Overview](#project-overview)  
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
  - [Dependencies](#dependencies)  
  - [Build Instructions](#build-instructions)  
- [Usage](#usage)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Repository Structure  
Here’s a high-level overview of the directories:  

- /MNIST/ – Data loader and helper files for the MNIST dataset
- /blas/ – Implementations using BLAS / OpenMP, etc.
- /gpu/ – GPU implementation (custom matrix multiplication)
- /gpuBlas/ – GPU implementation using cuBLAS
- /include/ – Header files and common definitions
- /milestone1/ – Code and materials for Milestone 1
- /milestone2/ – Code and materials for Milestone 2
- /finalBatches/ – Final batch-run results
- /final/ – Final implementation and scripts
- /plotting/ – Scripts to generate performance/accuracy plots
- Project_3_ML_Milestone_1_2025.pdf – Project Milestone 1 write-up

---

## Getting Started  

### Dependencies  
Make sure you have the following installed:  
- C compiler such as `gcc` or `clang`  
- (For CPU BLAS version) BLAS and/or OpenMP libraries  
- (For GPU versions) NVIDIA CUDA toolkit, cuBLAS library, and compatible GPU hardware  
- `make` utility (used via Makefiles)  
- (Optional) Python + Matplotlib/gnuplot for plotting results  

### Build Instructions  
```bash
# Build pure CPU version
cd blas
make clean && make cpu

# Build GPU custom matrix multiply
cd ../gpu
make clean && make all

# Build GPU cuBLAS version
cd ../gpuBlas
make clean && make all
```

### Acknowledgements
This built during Profesor Andrew Siegel's High Performance Computing course.
