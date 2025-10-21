# Project 3


Milestone 2:
In the blas folder are the make files for openmp and blas without openmp.
In the gpu folder is the make file for running on the gpu.


Milestone 1:
mnist.h is from: https://github.com/projectgalateia/mnist/blob/master/mnist.h
compile with: gcc cpuFF.c -o cpuFF -O3 -ffast-math -lm -lgsl -lgslcblas
or run make to create the ./cpuFF binary