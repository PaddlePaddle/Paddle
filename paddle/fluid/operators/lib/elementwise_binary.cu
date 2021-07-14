/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <iostream>
#include "GlobalFunctor.h"
#define DTYPE float
// #define DTYPE  __half
template <typename T>
struct ADD_t {
  __device__ __forceinline__ void operator()(const T* in1, const T* in2,
                                             T* out) const {
    out[0] = in1[0] + in2[0];
  }
};

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignVec {
  T val[VecSize];
};

template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void Binary(const T* __restrict__ in1, const T* __restrict__ in2,
                       T* __restrict__ out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      compute(in1[idx].val[idy], in2[idx].val[idy], out[idx].val[idy]);
    }
  }
}

template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void CycleBinary(const T* in1, const T* in2, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      compute(in1[idx], in2[idx].val[idy], out[idx].val[idy]);
    }
  }
}

template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__device__ void Unary(const T* in, T* out) {
  OpFunc compute;
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      compute(in[idx].val[idy], out[idx].val[idy]);
    }
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void Load(const T* src, T* dst) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
    dst[idx] = src[idx * BlockSize];
  }
}

template <typename T, int NX, int NY, int BlockSize>
__device__ void Store(const T src[], T* dst) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
    dst[idx * BlockSize] = src[idx];
  }
}
/*
step1:
   NX = (num / NY) / block_size;
   NY = VecType;
step2:
   NX = 1;
   NY = VecType;
step3:
   NX = 1;
   NY = 1;
 -------------------|--------------|----------|
   NX, NY             % block_size    %VecType

*/
template <typename T, int NX, int NY, int BlockSize, class OpFunc>
__global__ void ADDKERNEL(const T* in1, const T* in2, T* out, int num) {
  using VecType = AlignVec<T, NY>;
  const VecType* src1 = reinterpret_cast<const VecType*>(in1);
  const VecType* src2 = reinterpret_cast<const VecType*>(in2);
  VecType* dst = reinterpret_cast<VecType*>(out);
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  VecType arg1[NX];
  VecType arg2[NX];
  Load<VecType, NX, NY, BlockSize>(src1 + offset, arg1);
  Load<VecType, NX, NY, BlockSize>(src2 + offset, arg2);
  Binary<VecType, NX, NY, BlockSize, OpFunc>(arg1, arg2, arg1);
  Store<VecType, NX, NY, BlockSize>(arg1, dst + offset);
}

void init(DTYPE* src1, DTYPE* src2, DTYPE* dst_cpu, int num) {
  for (int i = 0; i < num; i++) {
    src1[i] = i % 3 * ((-1) ^ i);
    src2[i] = i % 2 * ((-1) ^ i);
    dst_cpu[i] = src1[i] + src2[i];
  }
}
/*
int main() {
  int N = 16 * 128 * 257 * 257;
  float dev = 0;
  cudaSetDevice(dev);
  DTYPE* in1_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* in2_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* out_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* out_d_cpu = (DTYPE*)malloc(N * sizeof(DTYPE));
  init(in1_h, in2_h, out_h, N);

  DTYPE *in1_d, *in2_d, *out_d;
  cudaMalloc((DTYPE**)&in1_d, N * sizeof(DTYPE));
  cudaMalloc((DTYPE**)&in2_d, N * sizeof(DTYPE));
  cudaMalloc((DTYPE**)&out_d, N * sizeof(DTYPE));

  cudaMemcpy(in1_d, in1_h, N * sizeof(DTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(in2_d, in2_h, N * sizeof(DTYPE), cudaMemcpyHostToDevice);

  const int block = 256;
  const int VecType = 4;
  int grid = (N / VecType + block - 1) / block;

  for (int i = 0; i < 1000; i++) {
    ADDKERNEL<DTYPE, 1, VecType, block, ADD_t<DTYPE>><<<grid, block>>>(
        in1_d, in2_d, out_d, N);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(out_d_cpu, out_d, N * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < N; i++) {
    if (out_d_cpu[i] != out_h[i]) {
      printf("\nERROR gpu %d  cpu %d idx %d\n", out_d_cpu[i], out_h[i], i);
      break;
    }
  }
  cudaFree(in1_d);
  cudaFree(in2_d);
  cudaFree(out_d);
  free(in1_h);
  free(in2_h);
  free(out_h);
  free(out_d_cpu);
  return 0;
}
*/
