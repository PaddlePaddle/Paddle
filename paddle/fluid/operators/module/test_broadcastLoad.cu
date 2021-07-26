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
#include "paddle/fluid/framework/array.h"
#define MAX_DIM 5
//#define DTYPE float
#define DTYPE __half
namespace paddle {

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) CudaAlignedVector {
  T val[VecSize];
};
template <typename T, int NX, int NY, int BlockSize>
__device__ void load(const T* in, T* out, int strid_in) {
#pragma unroll
  for (int idx = 0; idx < NX; idx++) {
#pragma unroll
    for (int idy = 0; idy < NY; idy++) {
      out[idx + idy * NX] = in[idy + strid_in * idx];
    }
  }
}

template <typename T, typename VecType, int VecSize>
__device__ void VecToT(const VecType in, T* out) {
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    out[i] = in.val[i];
  }
}
template <typename T, int Shape_Size, int VecSize>
__device__ __forceinline__ void Load(const T* __restrict__ in, T* out,
                                     uint32_t offset,
                                     framework::Array<uint32_t, 5> shape_in,
                                     framework::Array<uint32_t, 5> shape_out) {
  uint32_t stride_in[Shape_Size];
  uint32_t stride_out[Shape_Size];

  // shape[Shape_Size - 1] = shape_in[Shape_Size - 1];
  stride_in[Shape_Size - 1] = 1;
  stride_out[Shape_Size - 1] = 1;

#pragma unroll Shape_Size - 1
  for (int i = Shape_Size - 2; i >= 0; i--) {
    stride_in[i] = stride_in[i + 1] * shape_in[i + 1];
    stride_out[i] = stride_out[i + 1] * shape_out[i + 1];
    //	shape[i] = shape_in[i];
  }

#pragma unroll
  for (uint32_t i = 0; i < VecSize; i++) {
    uint32_t index = 0;
    uint32_t idx = offset + i;
#pragma unroll
    for (uint32_t j = 0; j < Shape_Size; j++) {
      uint32_t tmp = idx / stride_out[j];
      idx = idx - tmp * stride_out[j];
      index += (tmp % shape_in[j]) * stride_in[j];
    }
    out[i] = in[index];
  }
}
template <typename T, typename OutT, int Shape_Size, int VecSize, int ET>
__global__ void broadcastLoad(
    framework::Array<const T* __restrict__, ET> in_data, OutT* out,
    framework::Array<bool, ET> use_broadcast, uint32_t out_num,
    framework::Array<framework::Array<uint32_t, MAX_DIM>, ET> shape_in,
    framework::Array<uint32_t, 5> shape_out) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  using InVecType = CudaAlignedVector<T, VecSize>;
  using OutVecType = CudaAlignedVector<OutT, VecSize>;
  OutVecType* dst = reinterpret_cast<OutVecType*>(out);
  InVecType data[ET];
  OutVecType result;
  T arg[ET][VecSize];
  T args[ET];
  const InVecType* in[ET];
#pragma unroll
  for (int i = 0; i < ET; i++) {
    in[i] = reinterpret_cast<const InVecType*>(in_data[i]);
  }

#pragma unroll
  for (uint32_t fix = idx; fix * VecSize < out_num;
       fix += blockDim.x * gridDim.x) {
// load
#pragma unroll
    for (int i = 0; i < ET; i++) {
      // broadcast load
      if (use_broadcast[i]) {
        Load<T, Shape_Size, VecSize>(in_data[i], &arg[i][0], fix * VecSize,
                                     shape_in[i], shape_out);
      } else {
        load<InVecType, 1, 1, 1>(in[i] + fix, &data[i], 0);
        VecToT<T, InVecType, VecSize>(data[i], &arg[i][0]);
      }
    }

// compute
#pragma unroll
    for (int i = 0; i < VecSize; i++) {
#pragma unroll
      for (int j = 0; j < ET; ++j) {
        args[j] = arg[j][i];
      }
      result.val[i] = __hadd(args[0], args[1]);
    }

    // store
    dst[fix] = result;
  }
}

}  // namespace paddle
void init(DTYPE* src1, DTYPE* src2, DTYPE* dst_cpu, int num) {
  for (int i = 0; i < num; i++) {
    // src1[i] = i; //i % 3 * ((-1) ^ i);
    // src2[i] = i;// % 2 * ((-1) ^ i);
    // dst_cpu[i] = src1[i] + src2[i];
  }
}

int main() {
  int N = 32 * 12 * 128 * 128;
  float dev = 0;
  cudaSetDevice(dev);
  DTYPE* in1_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* b_in1_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* in2_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* out_h = (DTYPE*)malloc(N * sizeof(DTYPE));
  DTYPE* out_d_cpu = (DTYPE*)malloc(N * sizeof(DTYPE));
  init(in1_h, in2_h, out_h, 32 * 128);
  init(in2_h, in2_h, out_h, 32 * 128 * 12 * 128);
  /*  cpu
    for(int i = 0; i < 32; i++) {
      for(int k = 0; k < 12 * 128; k++) {
            for(int j = 0; j < 128; j++) {
                b_in1_h[i * 12 * 128 * 128 + k * 128 + j] = in1_h[i * 128 + j];
            }
      }
    }

    for(int i = 0; i < 32 * 12 * 128 * 128; i++) {
      out_h[i] = b_in1_h[i] + in2_h[i];
    }
  */
  DTYPE *in1_d, *in2_d, *out_d;
  cudaMalloc((DTYPE**)&in1_d, N * sizeof(DTYPE));
  cudaMalloc((DTYPE**)&in2_d, N * sizeof(DTYPE));
  cudaMalloc((DTYPE**)&out_d, N * sizeof(DTYPE));

  cudaMemcpy(in1_d, in1_h, N * sizeof(DTYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(in2_d, in2_h, N * sizeof(DTYPE), cudaMemcpyHostToDevice);

  const int block = 256;
  const int VecType = 4;
  int grid = (N / VecType + block - 1) / block;
  paddle::framework::Array<uint32_t, MAX_DIM> shape_out;
  paddle::framework::Array<paddle::framework::Array<uint32_t, MAX_DIM>, 2>
      in_shape;
  paddle::framework::Array<const DTYPE* __restrict__, 2> in_data;
  paddle::framework::Array<bool, 2> use_broadcast;
  use_broadcast[0] = 0;
  use_broadcast[1] = 1;
  in_data[0] = in2_d;
  in_data[1] = in1_d;
  in_shape[0][0] = 32;
  in_shape[0][1] = 1;
  in_shape[0][2] = 128;
  in_shape[1][0] = 32;
  in_shape[1][1] = 12 * 128;
  in_shape[1][2] = 128;
  shape_out[0] = 32;
  shape_out[1] = 12 * 128;
  shape_out[2] = 128;

  for (int i = 0; i < 1000; i++) {
    paddle::broadcastLoad<DTYPE, DTYPE, 3, VecType, 2><<<grid, block>>>(
        in_data, out_d, use_broadcast, N, in_shape, shape_out);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(out_d_cpu, out_d, N * sizeof(DTYPE), cudaMemcpyDeviceToHost);
  /*
  for (int i = 0; i < N; i++) {
    if (out_h[i] != out_d_cpu[i]) {
      printf("\nERROR cpu %f  gpu %f idx %d %d\n", out_h[i], out_d_cpu[i], N,
  i);
      break;
    }
  }
  */
  cudaFree(in1_d);
  cudaFree(in2_d);
  cudaFree(out_d);
  free(in1_h);
  free(b_in1_h);
  free(in2_h);
  free(out_h);
  free(out_d_cpu);
  return 0;
}
