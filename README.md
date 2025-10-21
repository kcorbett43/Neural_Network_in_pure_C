## Project Overview  
The goal of this project is to explore and implement different versions of a neural network for the classic digit-classification problem (e.g., using the MNIST dataset) in pure C, both for CPU and GPU. The implementations vary from hand-crafted matrix multiplications to leveraging highly optimized libraries (ie cBlas and cuBlas).  
This provides insight into:  
- how a basic neural network works under the hood  
- performance trade-offs between pure code vs. library calls  
- CPU vs. GPU execution for numerical computing   

---

## Repository Structure  
Here’s a high-level overview of the directories:  

- /MNIST/ – Data loader and helper files for the MNIST dataset
- /blas/ – Implementations using BLAS / OpenMP, etc.
- /gpu/ – GPU implementation (custom matrix multiplication)
- /gpuBlas/ – GPU implementation using cuBLAS
- /include/ – Header files and common definitions
- /finalBatches/ – Final batch-run results
- /plotting/ – Scripts to generate performance/accuracy plots

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
## Results and Discussion

### Implementation 1

Initial implementation achieved **90.3 % accuracy** with the following metrics:

| Metric | Value |
|:--|--:|
| Success Rate | 0.903 |
| Grind Rate | 112 samples / second |
| Total Training Time | 535 s |
| Total Inference Time | 0.99 s |
| Learning Rate | 0.05 |
| Batch Size | 100 |

This baseline demonstrated functional correctness but revealed opportunities for major performance improvement through vectorization, BLAS usage, and GPU acceleration.

---

### Implementation 2

After extensive refactoring—fixing matrix transposition, dimensional mismatches, and delta accumulation—accuracy again reached **> 90 %**, with far better runtime across optimized versions.

| Version | Tile Size | Core Count | 256 Time (s) | 512 Time (s) | 256 Grind Rate | 512 Grind Rate | Accuracy 256 | Accuracy 512 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| CPU | 64 | 1 | 104.35 | 105.42 | 2 885 | 2 857 | 0.91 | 0.88 |
| OpenMP (no BLAS) | 64 | 16 | 28.02 | 17.08 | 10 714 | 17 647 | 0.91 | 0.88 |
| BLAS | 64 | 1 | 26.91 | 26.00 | 11 111 | 11 538 | 0.91 | 0.88 |
| GPU | N/A | N/A | 5.15 | 4.06 | 58 252 | 73 892 | 0.91 | 0.88 |

**Observations**

- GPU implementations achieved the highest throughput (≈ 20–25× speedup over CPU BLAS).  
- OpenMP scaling produced strong gains on CPU but still trailed GPU by a wide margin.  
- Accuracy stabilized near 0.9 across all versions after sufficient epochs.  

**Loss Curves**

<img width="730" height="496" alt="image" src="https://github.com/user-attachments/assets/da43fd47-cd45-4240-82a9-0219267c07d3" />

---

### Final Results

Final tests on an **A100 GPU** demonstrated near-identical accuracy (~ 0.94) across all architectures, with massive differences in throughput:

| Version | Processor | Accuracy | Grind Rate (samples/s) | Training Time (s) | TPB / Cores |
|:--|:--|--:|--:|--:|--:|
| GPU Native | A100 | 0.94 | 85 959 | 34.9 | 256 |
| GPU cuBLAS | A100 | 0.94 | 121 000 | 24.7 | N/A |
| CPU Native | — | 0.94 | 37 641 | 79.7 | 16 |
| CPU BLAS | — | 0.94 | 9 933 | 302.3 | 1 |

**Key Takeaways**

- **cuBLAS GPU** implementation offered > 3× speedup vs CPU BLAS and ~ 5× vs CPU Native.  
- Most remaining bottlenecks stem from **CPU–GPU transfer overhead** and **non-optimized I/O**.  
- Further improvements could include keeping all data resident on-GPU, tuning batch-tile sizes, and integrating additional cuBLAS routines.

---

### Acknowledgements
This was built during Profesor Andrew Siegel's High Performance Computing course.
