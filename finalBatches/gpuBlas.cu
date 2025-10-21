
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "../include/mnist.h"
#include <stdio.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include <math.h>
#include<time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>

#ifdef HAVE_OPENBLAS
    #include <cblas.h>
#endif
#ifdef _OPENMP
    #include<omp.h>
#endif
int NUM_THREADS = 16;
//int num_batches = 10; // Number of mini batches
int img_len_2d = 28; // 28 * 28 images
int img_len = 784;  // 784 = 28*28 images
int num_layer1_neurons = 128;
int num_layer2_neurons = 256;
//int examples_per_batch;
int num_epochs = 50;
unsigned int num_examples; // number of training examples
int TILE_SIZE = 64;


// Compute the mean of an array of floats
float compute_mean(float *data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

// Compute the variance of an array given its mean
float compute_variance(float *data, int size, float mean) {
    float sum_sq_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / size;
}

// Compute the minimum value in an array
float compute_min(float *data, int size) {
    float min_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val)
            min_val = data[i];
    }
    return min_val;
}

// Compute the maximum value in an array
float compute_max(float *data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val)
            max_val = data[i];
    }
    return max_val;
}

// Print gradient statistics: mean, variance, min, and max
void print_gradient_stats(const char *grad_name, float *data, int size) {
    float mean = compute_mean(data, size);
    float variance = compute_variance(data, size, mean);
    float min_val = compute_min(data, size);
    float max_val = compute_max(data, size);
    printf("%s: mean = %f, variance = %f, min = %f, max = %f\n",
           grad_name, mean, variance, min_val, max_val);
}


gsl_rng **rngs;
void init_rng() {
  gsl_rng_env_setup();  
  rngs = (gsl_rng **)malloc(NUM_THREADS * sizeof(gsl_rng *)); 
  for (int i=0; i<NUM_THREADS;i++){
    rngs[i] = gsl_rng_alloc(gsl_rng_mt19937);
  }
  #ifdef _OPENMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    gsl_rng_set(rngs[tid], 4238811 * tid);
  } 
  #else
    gsl_rng_set(rngs[0], 4238811);
  #endif
}
float randoMM(float min, float max) {
    #ifdef _OPENMP
      int tid = omp_get_thread_num();
      return (float)(min + gsl_rng_uniform(rngs[tid]) * (max - min));
    #else
      return (float)(min + gsl_rng_uniform(rngs[0]) * (max - min));
    #endif
}
// [min, max)
int randoMM_int(int min, int max) {
  #ifdef _OPENMP
    int tid = omp_get_thread_num();
    return min + (int)(gsl_rng_uniform(rngs[tid]) * (max - min));
  #else
    return min + (int)(gsl_rng_uniform(rngs[0]) * (max - min)); 
  #endif
}
float random_normal(float std) {
  #ifdef _OPENMP
    int tid = omp_get_thread_num();
    return (float)(gsl_ran_gaussian(rngs[tid], std));
  #else
    return (float)(gsl_ran_gaussian(rngs[0], std));
  #endif
}

// Randomly shuffle data.
void shuffleIndices(int *indices, int len) {
  int id;
  for (int i = len-1; i > 0; i--) {
    id = randoMM_int(0, i+1); // random index 0 to i
    int temp = indices[i];
    indices[i] = indices[id];
    indices[id] = temp;
  }
}


void naiveMatMul(float *output, float *data1, float *data2, int m, int n, int p) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
        for (int k = 0; k < n; k++) {
          output[i * p + j] += data1[i * n + k] * data2[k * p + j]; 
        }
    }
  }
}


void tileMatMul(float *output, float *data1, float *data2, int m, int n, int p) {
  int endi;
  int endj;
  int endk;
  for (int is = 0; is < m; is+=TILE_SIZE) {
    endi = (is+TILE_SIZE < m) ? is + TILE_SIZE : m;
    for (int js = 0; js < p; js+=TILE_SIZE) {
      endj = (js+TILE_SIZE < p) ? js + TILE_SIZE : p;
      for (int ks = 0; ks<n; ks+= TILE_SIZE) {
        endk = (ks+TILE_SIZE < n) ? ks + TILE_SIZE : n;
        for (int i = is; i<endi; i++) {
          for (int j = js; j<endj; j++) {
            for (int k = ks; k<endk; k++) {
              output[i * p + j] += data1[i * n + k] * data2[k * p + j]; 
            }
          }
        }
      }
    }
  }
}


__global__ void naiveGpuMatMul(float *output, float *data1, float *data2, int m, int n, int p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m*p) {
    return;
  }
  int row = idx / p;
  int col = idx % p;

  for (int k = 0; k < n; k++) {
    output[row * p + col] += data1[row * n + k] * data2[k * p + col]; 
  }
}

void gpuMatMul(float *output, float *data1, float *data2, int m, int n, int p, cublasHandle_t handle) {
    // Allocate memory on device
    float *dev_output;
    cudaMalloc((void **)&dev_output, m*p*sizeof(float));
    float *dev_data1;
    cudaMalloc((void **)&dev_data1, m*n*sizeof(float));
    float *dev_data2;
    cudaMalloc((void **)&dev_data2, n*p*sizeof(float));
  
    // Copy data to device
    cudaMemset(dev_output, 0, m*p*sizeof(float));
    cudaMemcpy(dev_data1, data1, m*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data2, data2, n*p*sizeof(float), cudaMemcpyHostToDevice);
  
    // Determine threads
    int threads_per_block = 256;
    int num_blocks = (m*p + threads_per_block - 1) / threads_per_block;
  
    // run kernel
    //naiveGpuMatMul<<<num_blocks, threads_per_block>>>(dev_output, dev_data1, dev_data2, m, n, p);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, dev_data2, p, dev_data1, n, &beta, dev_output, p);
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    // copy kernel output to host
    cudaMemcpy(output, dev_output, m*p*sizeof(float), cudaMemcpyDeviceToHost);
  
    // free cuda data
    cudaFree(dev_data1);
    cudaFree(dev_data2);
    cudaFree(dev_output);
}

void matMul(float *output, float *data1, float *data2, int m, int n, int p, cublasHandle_t handle) {
  //naiveMatMul(output, data1, data2, m, n, p);
  //tileMatMul(output, data1, data2, m, n, p);
  gpuMatMul(output, data1, data2, m, n, p, handle);
}



void matAdd(float *output, float *bias, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] += bias[j];
    }
  }
}
void ReLU(float *output, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] = fmaxf(0, output[i*cols + j]);
    }
  }
}

void softmax(float *output, int len, int batch_size) {
  float total_exp;
  for (int j = 0; j < batch_size; j++){
    total_exp = 0.0f;
    for (int i = 0; i < len; i++) {
      output[j*len + i] = expf(output[j*len + i]);
      total_exp += output[j*len + i];
    }
    for (int i = 0; i < len; i++) {
      output[j*len + i] = output[j*len + i] / total_exp;
    }
  }
}

float crossEntropyLoss(float *output, int *one_hots, int batch_size) {
  softmax(output, 10, batch_size);
  float epsilon = 1e-10f;
  int correct_prob_label;
  float loss = 0.0f;
  for (int j = 0; j < batch_size; j++){
    for (int i = 0; i < 10; i++) {
      if (one_hots[j*10 + i] == 1) {
        correct_prob_label = i;
        break;
      }
    }
    loss += -logf(output[j*10 + correct_prob_label] + epsilon);
  }
  return loss;
} 


void linearLayer(float *output, float *input, float *weights, float *bias, int m, int n, int p, cublasHandle_t handle) {
  #ifdef HAVE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, p, n, 1.0, input, n, weights, p, 0.0, output, p); 
    for(int i=0; i<m; i++) { 
      cblas_saxpy(p, 1.0, bias, 1, output+i*p, 1);
    }
  #else
    matMul(output, input, weights, m, n, p, handle); 
    matAdd(output, bias, m, p);
  #endif
}


// Take 28 * 28 image data and convert to 784.
float **flattenData(mnist_data *data, int len) {
  float **flattened_data = (float **)malloc(len * sizeof(float *));
  for (int i = 0; i < len; i++) {
    float *flattened = (float *)malloc(img_len * sizeof(float));
    for (int j = 0; j < img_len_2d; j++) {
      for (int k = 0; k < img_len_2d; k++) {
        flattened[j * img_len_2d + k] = data[i].data[j][k];
      }
    }
    flattened_data[i] = flattened;
  }
  return flattened_data;
}
void oneHotLabels(int **labels, mnist_data *data, int len) {
  for (int i = 0; i < len; i++) {
    int *encoding = (int *)calloc(10, sizeof(int)); // 10 = number of digits.
    int digit = data[i].label;
    encoding[digit] = 1;
    labels[i] = encoding;
  }
}

void kaimingInit(float *weights, int num_inputs, int num_outputs) {
  float std = sqrtf(2.0f / num_inputs);  
  for (int i = 0; i < num_inputs * num_outputs; i++) {
      weights[i] = random_normal(std);
  }
}
// computes x - y and stores in output
void matrixSub(float *output, float *x, float *y, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] = x[i*cols + j] - y[i*cols + j];
    }
  }
}
// If x is float and y is int, computes x - y and stores in output.
void matrixFloatIntSub(float *output, float *x, int *y, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] = x[i*cols + j] - y[i*cols + j];
    }
  }
}

// calcs A^T * B
void naiveMatMulTranspose(float *output, float *transpose, float *mat, int m, int n, int p){
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        output[i * p + j] += transpose[k * m + i] * mat[k * p + j];
      }
    }
  }
}
// calcs A^T * B using tiling
void tileMatMulTranspose(float *output, float *transpose, float *mat, int m, int n, int p) {
  int endi;
  int endj;
  int endk;
  for (int is = 0; is < m; is+=TILE_SIZE) {
    endi = (is+TILE_SIZE < m) ? is + TILE_SIZE : m;
    for (int js = 0; js < p; js+=TILE_SIZE) {
      endj = (js+TILE_SIZE < p) ? js + TILE_SIZE : p;
      for (int ks = 0; ks<n; ks+= TILE_SIZE) {
        endk = (ks+TILE_SIZE < n) ? ks + TILE_SIZE : n;
        for (int i = is; i<endi; i++) {
          for (int j = js; j<endj; j++) {
            for (int k = ks; k<endk; k++) {
              output[i * p + j] += transpose[k * m + i] * mat[k * p + j];
            }
          }
        }
      }
    }
  }
}


__global__ void naiveGPUMatMulTranspose(float *output, float *transpose, float *mat, int m, int n, int p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m*p) {
    return;
  }
  int row = idx / p;
  int col = idx % p;

  for (int k = 0; k < n; k++) {
    output[row * p + col] += transpose[k * m + row] * mat[k * p + col];
  }
}

void gpuMatMulTranspose(float *output, float *transpose, float *mat, int m, int n, int p, cublasHandle_t handle){
  // Allocate memory on device
  float *dev_output;
  cudaMalloc((void **)&dev_output, m*p*sizeof(float));
  float *dev_transpose;
  cudaMalloc((void **)&dev_transpose, m*n*sizeof(float));
  float *dev_mat;
  cudaMalloc((void **)&dev_mat, n*p*sizeof(float));

  // Copy data to device
  cudaMemset(dev_output, 0, m*p*sizeof(float));
  cudaMemcpy(dev_transpose, transpose, m*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mat, mat, n*p*sizeof(float), cudaMemcpyHostToDevice);

  // Determine threads
  int threads_per_block = 256;
  int num_blocks = (m*p + threads_per_block - 1) / threads_per_block;

  // run kernel
  //naiveGPUMatMulTranspose<<<num_blocks, threads_per_block>>>(dev_output, dev_transpose, dev_mat, m, n, p);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, p, m, n, &alpha, dev_mat, p, dev_transpose, m, &beta, dev_output, p);
  cudaGetLastError();
  cudaDeviceSynchronize();
  
  // copy kernel output to host
  cudaMemcpy(output, dev_output, m*p*sizeof(float), cudaMemcpyDeviceToHost);

  // free cuda data
  cudaFree(dev_transpose);
  cudaFree(dev_mat);
  cudaFree(dev_output);
}


void matMulTranspose(float *output, float *transpose, float *mat, int m, int n, int p, cublasHandle_t handle){
  //tileMatMulTranspose(output, transpose, mat, m, n, p);
  gpuMatMulTranspose(output, transpose, mat, m, n, p, handle);
}


void naiveMatMulTransposeOpp(float *output, float *mat, float *transpose, int m, int n, int p) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        output[i * p + j] += mat[i * n + k] * transpose[j * n + k];
      }
    }
  }
}


void tileMatMulTransposeOpp(float *output, float *mat, float *transpose, int m, int n, int p) {
  int endi;
  int endj;
  int endk;
  for (int is = 0; is < m; is+=TILE_SIZE) {
    endi = (is+TILE_SIZE < m) ? is + TILE_SIZE : m;
    for (int js = 0; js < p; js+=TILE_SIZE) {
      endj = (js+TILE_SIZE < p) ? js + TILE_SIZE : p;
      for (int ks = 0; ks<n; ks+= TILE_SIZE) {
        endk = (ks+TILE_SIZE < n) ? ks + TILE_SIZE : n;
        for (int i = is; i<endi; i++) {
          for (int j = js; j<endj; j++) {
            for (int k = ks; k<endk; k++) {
              output[i * p + j] += mat[i * n + k] * transpose[j * n + k];
            }
          }
        }
      }
    }
  }
}

__global__ void naiveGPUMatMulTransposeOpp(float *output, float *mat, float *transpose, int m, int n, int p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m*p) {
    return;
  }
  int row = idx / p;
  int col = idx % p;

  for (int k = 0; k < n; k++) {
    output[row * p + col] += mat[row * n + k] * transpose[col * n + k];
  }
}

void gpuMatMulTransposeOpp(float *output, float *transpose, float *mat, int m, int n, int p, cublasHandle_t handle){
  // Allocate memory on device
  float *dev_output;
  cudaMalloc((void **)&dev_output, m*p*sizeof(float));
  float *dev_transpose;
  cudaMalloc((void **)&dev_transpose, n*p*sizeof(float));
  float *dev_mat;
  cudaMalloc((void **)&dev_mat, m*n*sizeof(float));

  // Copy data to device
  cudaMemset(dev_output, 0, m*p*sizeof(float));
  cudaMemcpy(dev_transpose, transpose, n*p*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mat, mat, m*n*sizeof(float), cudaMemcpyHostToDevice);

  // Determine threads
  int threads_per_block = 256;
  int num_blocks = (m*p + threads_per_block - 1) / threads_per_block;

  // run kernel
  //naiveGPUMatMulTransposeOpp<<<num_blocks, threads_per_block>>>(dev_output, dev_mat, dev_transpose, m, n, p);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, dev_transpose, n, dev_mat, n, &beta, dev_output, p);
  //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, dev_transpose, m, dev_mat, p, &beta, dev_output, p);

  cudaGetLastError();
  cudaDeviceSynchronize();
  
  // copy kernel output to host
  cudaMemcpy(output, dev_output, m*p*sizeof(float), cudaMemcpyDeviceToHost);

  // free cuda data
  cudaFree(dev_transpose);
  cudaFree(dev_mat);
  cudaFree(dev_output);
}


// calcs A * B^T
void matMulTransposeOpp(float *output, float *mat, float *transpose, int m, int n, int p, cublasHandle_t handle){
  //naiveMatMulTransposeOpp(output, mat, transpose, m, n, p);
  //tileMatMulTransposeOpp(output, mat, transpose, m, n, p);
  gpuMatMulTransposeOpp(output, transpose, mat, m, n, p, handle);
}

void matrixScalar(float *output, float scalar, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] *= scalar; 
    }
  }
}
void hadamard(float *output, float *a, float *b, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = a[i] * b[i];
  }
}
void reluDerivative(float *output, float *z, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = (z[i] > 0) ? 1.0f : 0.0f;
  }
}
void accumulate(float *output, float *additions, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[i*cols + j] += additions[i*cols + j]; 
    }
  }
}

void accumulateB(float *output, float *additions, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      output[j] += additions[i*cols + j];
    }
  }
}

void applyAccumulation(float *W, float *b, float *acc_dW, float *acc_db, int cols, int w_rows, float lr, int batch_size) {
  #ifdef HAVE_OPENBLAS
    cblas_sscal(w_rows*cols, (lr/batch_size), acc_dW, 1);
    cblas_sscal(cols, (lr/batch_size), acc_db, 1);
    cblas_saxpy(w_rows * cols, -1.0f, acc_dW, 1, W, 1);
    cblas_saxpy(cols, -1.0f, acc_db, 1, b, 1);
  #else
    matrixScalar(acc_dW, (lr/batch_size), w_rows, cols);
    matrixScalar(acc_db, (lr/batch_size), 1, cols);
    matrixSub(W, W, acc_dW, w_rows, cols); // Update weights
    matrixSub(b, b, acc_db, 1, cols);  
  #endif
}

void hiddenBackward(float *acc_dW, float *acc_db, float *next_W, float *next_del, float *a, int batch_size, int cols, int w_rows, float *prev_a, float *del_WL, int prev_w_len, cublasHandle_t handle) {
  float *relu_deriv = (float *)malloc(cols * batch_size * sizeof(float));
  float *grad_W = (float *)calloc(cols * prev_w_len, sizeof(float));
  #ifdef HAVE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_size, cols, w_rows, 1.0f, next_del, w_rows, next_W, w_rows, 0.0f, del_WL, cols);
  #else
    matMulTransposeOpp(del_WL, next_del, next_W, batch_size, w_rows, cols, handle);
  #endif
  reluDerivative(relu_deriv, a, cols * batch_size);
  hadamard(del_WL, del_WL, relu_deriv, batch_size * cols);
  #ifdef HAVE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, prev_w_len, cols, batch_size, 1.0f, prev_a, prev_w_len, del_WL, cols, 0.0f, grad_W, cols);
    cblas_saxpy(prev_w_len * cols, 1.0f, grad_W, 1, acc_dW, 1);
  #else
    matMulTranspose(grad_W, prev_a, del_WL, prev_w_len, batch_size, cols, handle);
    accumulate(acc_dW, grad_W, prev_w_len, cols);
  #endif
  accumulateB(acc_db, del_WL, batch_size, cols);
  free(grad_W);
  free(relu_deriv);
}
void softmaxBackward(float *acc_dW, float *acc_db, float *del_L, float *a_l, float *a_l_1, int *batch_labels, int batch_size, int cols, int w_rows, cublasHandle_t handle) { 
  matrixFloatIntSub(del_L, a_l, batch_labels, batch_size, cols); // 4  
  float *del_WL = (float *)calloc(w_rows * cols, sizeof(float));
  #ifdef HAVE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, w_rows, cols, batch_size, 1.0f, a_l_1, w_rows, del_L, cols, 0.0f, del_WL, cols);
    cblas_saxpy(w_rows * cols, 1.0f, del_WL, 1, acc_dW, 1);
  #else
    matMulTranspose(del_WL, a_l_1, del_L, w_rows, batch_size, cols, handle); // 7
    accumulate(acc_dW, del_WL, w_rows, cols);
  #endif
  accumulateB(acc_db, del_L, batch_size, cols);
  free(del_WL);
}

int main(int argc, char **argv) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  // Default Values
  img_len = 784;  // 784 = 28*28 images
  num_layer1_neurons = 128;
  num_layer2_neurons = 256;
  int batch_size = 500;
  //examples_per_batch = num_examples / num_batches;
  int num_outputs = 10; // digits 0-9
  float loss = 0.0f;
  float lr = 0.01f;
  int num_correct = 0;


  #ifdef HAVE_OPENBLAS
    printf("Compiled with blas.\n");
  #else
    printf("GPU version, no blas.\n");
    printf("Tile Size: %d\n", TILE_SIZE);
  #endif
  if (argc > 1) {
    batch_size = (int) (atoi(argv[1]));
  }
  printf("Batch Size: %d, Learning Rate: %f\n", batch_size, lr);
  


  init_rng();
  // MNIST loading code is from https://github.com/projectgalateia/mnist/blob/master/mnist.h
  mnist_data *mnist_data;
  int ret;
  if (ret = mnist_load("../MNIST/train-images-idx3-ubyte", "../MNIST/train-labels-idx1-ubyte", &mnist_data, &num_examples)) {
    printf("An error occured: %d\n", ret);
  } else {
    printf("image count: %d\n", num_examples);
  }
  int num_training = 50000;
  int num_validation = 10000;
  int num_batches = (num_training / batch_size);

  // Flattened fully connected layer weights and biases
  // Weights
  float *W_1 = (float *)malloc(img_len * num_layer1_neurons * sizeof(float)); 
  float *W_2 = (float *)malloc(num_layer1_neurons * num_layer2_neurons * sizeof(float)); 
  float *W_3 = (float *)malloc(num_layer2_neurons * num_outputs * sizeof(float));
  // Bias
  float *b_1 = (float *)calloc(num_layer1_neurons, sizeof(float)); 
  float *b_2 = (float *)calloc(num_layer2_neurons, sizeof(float)); 
  float *b_3 = (float *)calloc(num_outputs, sizeof(float));
  // Accumulation for weights
  float *acc_dW_1 = (float *)calloc(img_len * num_layer1_neurons, sizeof(float)); 
  float *acc_dW_2 = (float *)calloc(num_layer1_neurons * num_layer2_neurons, sizeof(float)); 
  float *acc_dW_3 = (float *)calloc(num_layer2_neurons * num_outputs, sizeof(float));
  // Accumulation for bias
  float *acc_db_1 = (float *)calloc(num_layer1_neurons, sizeof(float)); 
  float *acc_db_2 = (float *)calloc(num_layer2_neurons, sizeof(float)); 
  float *acc_db_3 = (float *)calloc(num_outputs, sizeof(float));
  // Deltas
  float *del_L = (float *)malloc(batch_size * num_outputs * sizeof(float));
  float *del_l1 = (float *)malloc(batch_size * num_layer1_neurons * sizeof(float));
  float *del_l2 = (float *)malloc(batch_size * num_layer2_neurons * sizeof(float));

  // flatted the 28*28 image array to 1d 784 size
  float **data;
  data = flattenData(mnist_data, num_examples);
  int **labels = (int **)malloc(num_examples * sizeof(int *));
  oneHotLabels(labels, mnist_data, num_examples);

  // Initialize weights and biases with random values between 0 and 1 with kaiming
  kaimingInit(W_1, img_len, num_layer1_neurons);
  kaimingInit(W_2, num_layer1_neurons, num_layer2_neurons);
  kaimingInit(W_3, num_layer2_neurons, num_outputs);

  int *indices = (int *)malloc(num_examples * sizeof(int));
  for (int i = 0; i < num_examples; i++) {
    indices[i] = i;
  }
  float *a1 = (float *)malloc(batch_size * num_layer1_neurons * sizeof(float));
  float *a2 = (float *)malloc(batch_size * num_layer2_neurons * sizeof(float));
  float *a3 = (float *)malloc(batch_size * num_outputs * sizeof(float));
  float *a1v = (float *)malloc(num_validation * num_layer1_neurons * sizeof(float));
  float *a2v = (float *)malloc(num_validation * num_layer2_neurons * sizeof(float));
  float *a3v = (float *)malloc(num_validation * num_outputs * sizeof(float));
  float *batch_inputs = (float *)malloc(batch_size * img_len * sizeof(float));
  int *batch_labels = (int *)malloc(batch_size * num_outputs * sizeof(int));
  float *val_inputs = (float *)malloc(num_validation * img_len * sizeof(float));
  int *val_labels = (int *)malloc(num_validation * num_outputs * sizeof(int));
  clock_t training_start = clock();
  for (int k = 0; k < num_epochs; k++) {
    num_correct = 0;
    shuffleIndices(indices, num_training);
    loss = 0.0f;
    for (int i = 0; i < num_batches-1; i++) { 
      for (int r = 0; r < batch_size; r++) {
        int index = indices[i * batch_size + r];  
        memcpy(&batch_inputs[r * img_len], data[index], img_len * sizeof(float));  
        memcpy(&batch_labels[r * num_outputs], labels[index], num_outputs * sizeof(int));  
      }

      linearLayer(a1, batch_inputs, W_1, b_1, batch_size, img_len, num_layer1_neurons, handle);
      ReLU(a1, batch_size, num_layer1_neurons);
      linearLayer(a2, a1, W_2, b_2, batch_size, num_layer1_neurons, num_layer2_neurons, handle);
      ReLU(a2, batch_size, num_layer2_neurons);
      linearLayer(a3, a2, W_3, b_3, batch_size, num_layer2_neurons, num_outputs, handle);
      loss += crossEntropyLoss(a3, batch_labels, batch_size);
      softmaxBackward(acc_dW_3, acc_db_3, del_L, a3, a2, batch_labels, batch_size, num_outputs, num_layer2_neurons, handle);
      hiddenBackward(acc_dW_2, acc_db_2, W_3, del_L, a2, batch_size, num_layer2_neurons, num_outputs, a1, del_l2, num_layer1_neurons, handle);
      hiddenBackward(acc_dW_1, acc_db_1, W_2, del_l2, a1, batch_size, num_layer1_neurons, num_layer2_neurons, batch_inputs, del_l1, img_len, handle);
      

      applyAccumulation(W_3, b_3, acc_dW_3, acc_db_3, num_outputs, num_layer2_neurons, lr, batch_size);
      applyAccumulation(W_2, b_2, acc_dW_2, acc_db_2, num_layer2_neurons, num_layer1_neurons, lr, batch_size);
      applyAccumulation(W_1, b_1, acc_dW_1, acc_db_1, num_layer1_neurons, img_len, lr, batch_size);


      memset(acc_dW_3, 0, num_layer2_neurons * num_outputs * sizeof(float));
      memset(acc_db_3, 0, num_outputs * sizeof(float));
      memset(acc_dW_2, 0, num_layer1_neurons * num_layer2_neurons * sizeof(float));
      memset(acc_db_2, 0, num_layer2_neurons * sizeof(float));
      memset(acc_dW_1, 0, img_len * num_layer1_neurons * sizeof(float));
      memset(acc_db_1, 0, num_layer1_neurons * sizeof(float));    
      memset(a1, 0, batch_size * num_layer1_neurons * sizeof(float));
      memset(a2, 0, batch_size * num_layer2_neurons * sizeof(float));
      memset(a3, 0, batch_size * num_outputs * sizeof(float));

      memset(del_L, 0, batch_size * num_outputs * sizeof(float));
      memset(del_l1, 0, batch_size * num_layer1_neurons * sizeof(float));
      memset(del_l2, 0, batch_size * num_layer2_neurons * sizeof(float));    

    }
    for (int bi = num_training; bi < num_training+num_validation; bi++) {
      int index = indices[bi];  
      memcpy(&val_inputs[(bi-num_training) * img_len], data[index], img_len * sizeof(float));  
      memcpy(&val_labels[(bi-num_training) * num_outputs], labels[index], num_outputs * sizeof(int));  
    }
    linearLayer(a1v, val_inputs, W_1, b_1, num_validation, img_len, num_layer1_neurons, handle);
    ReLU(a1v, num_validation, num_layer1_neurons);
    linearLayer(a2v, a1v, W_2, b_2, num_validation, num_layer1_neurons, num_layer2_neurons, handle);
    ReLU(a2v, num_validation, num_layer2_neurons);
    linearLayer(a3v, a2v, W_3, b_3, num_validation, num_layer2_neurons, num_outputs, handle);
    loss = crossEntropyLoss(a3v, val_labels, num_validation);
    printf("%f\n", loss);
    memset(a1v, 0, num_validation * num_layer1_neurons * sizeof(float));
    memset(a2v, 0, num_validation * num_layer2_neurons * sizeof(float));
    memset(a3v, 0, num_validation * num_outputs * sizeof(float));
  }
  clock_t training_end = clock();
  float training_time = (float)(training_end - training_start) / CLOCKS_PER_SEC;



  

  // INFERENCE:
  // MNIST loading code is from https://github.com/projectgalateia/mnist/blob/master/mnist.h
  mnist_data = NULL;
  unsigned int num_test_examples;
  int test_ret;
  if (test_ret = mnist_load("../MNIST/t10k-images-idx3-ubyte", "../MNIST/t10k-labels-idx1-ubyte", &mnist_data, &num_test_examples)) {
    printf("An error occured: %d\n", test_ret);
  } else {
    printf("image count: %d\n", num_test_examples);
  }
  float **test_data = flattenData(mnist_data, num_test_examples);

  num_correct = 0;
  clock_t inference_start = clock();
  for (int i = 0; i<num_test_examples; i++) {
    memset(a1, 0, batch_size * num_layer1_neurons * sizeof(float));
    memset(a2, 0, batch_size * num_layer2_neurons * sizeof(float));
    memset(a3, 0, batch_size * num_outputs * sizeof(float));


    int label = mnist_data[i].label;
    linearLayer(a1, test_data[i], W_1, b_1, 1, img_len, num_layer1_neurons, handle);
    ReLU(a1, 1, num_layer1_neurons);
    linearLayer(a2, a1, W_2, b_2, 1, num_layer1_neurons, num_layer2_neurons, handle);
    ReLU(a2, 1, num_layer2_neurons);
    linearLayer(a3, a2, W_3, b_3, 1, num_layer2_neurons, num_outputs, handle);
    int predicted_label = 0;
    float max_prob = a3[0];
    for (int c = 1; c < num_outputs; c++) {
      if (a3[c] > max_prob) {
        max_prob = a3[c];
        predicted_label = c;
      }
    }
    if (label == predicted_label) {
      num_correct++;
    }

  }
  clock_t inference_end = clock();
  float inference_time = (float)(inference_end - inference_start) / CLOCKS_PER_SEC;

  printf("total test percent correct: %f\n", ((float)num_correct/num_test_examples));
  printf("Total Training Time: %f seconds\n", training_time);
  printf("Total Inference Time: %f seconds\n", inference_time);


  for (int i = 0; i < num_examples; i++) {
    free(data[i]);
    free(labels[i]);
  }
  free(data);
  free(labels);
  free(W_1);
  free(W_2);
  free(W_3);
  free(b_1);
  free(b_2);
  free(b_3);
  free(acc_dW_1);
  free(acc_dW_2);
  free(acc_dW_3);
  free(acc_db_1);
  free(acc_db_2);
  free(acc_db_3);

  return 0;
}
