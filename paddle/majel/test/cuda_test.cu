#include <stdio.h>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

#define CHECK_ERR(x)                                    \
  if (x != cudaSuccess) {                               \
    fprintf(stderr,"%s in %s at line %d\n",             \
        cudaGetErrorString(err),__FILE__,__LINE__);	    \
    exit(-1);											\
  }                                                     \

__global__ void vecAdd (float* d_A, float* d_B, float* d_C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    d_C[i] = d_A[i] + d_B[i];
  }
}

TEST(Cuda, Equality) {
  int n = 10;
  // Memory allocation for h_A, h_B and h_C (in the host)
  float h_A[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0 };
  float h_B[10] = { 0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
  float h_C[10];
  float *d_A, *d_B, *d_C;

  // Memory allocation for d_A, d_B and d_C (in the device)
  err = cudaMalloc((void **) &d_A, sizeof(float)*n);
  CHECK_ERR(err);

  err =cudaMalloc((void **) &d_B, sizeof(float)*n);
  CHECK_ERR(err);

  err =cudaMalloc((void **) &d_C, sizeof(float)*n);
  CHECK_ERR(err);
  
  // Copying memory to device
  err = cudaMemcpy(d_A, h_A, sizeof(float)*n, cudaMemcpyHostToDevice);
  CHECK_ERR(err);

  err = cudaMemcpy(d_B, h_B, sizeof(float)*n, cudaMemcpyHostToDevice);
  CHECK_ERR(err);

  // Calling the kernel
  vecAdd<<<ceil(n/256.0), 256>>>(d_A,d_B,d_C,n);

  // Copying results back to host
  err = cudaMemcpy(h_C, d_C, sizeof(float)*n, cudaMemcpyDeviceToHost);
  CHECK_ERR(err);
  
  EXPECT_EQ(h_C[1], 1.0);
  for (size_t i = 1; i < n - 1; ++i) {
    EXPECT_EQ(h_C[i], 11.0);
  }
  EXPECT_EQ(h_C[0], 1.0);
}
