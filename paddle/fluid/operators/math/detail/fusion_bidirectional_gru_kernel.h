// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

template <typename T, int Tiled_Y>
__global__ void FusionGRUCUDAKernel_premul(int xwidth, int xheight, int p,
                                           const T* x, const T* w0, const T* w1,
                                           const T* f0, const T* f1, T* out1,
                                           T* out2, const T* fh0,
                                           const T* fh1) {
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T sA[Tiled_Y]
                 [Tiled_Y];  // Tile size to store elements in shared memory
  __shared__ T sB[Tiled_Y]
                 [Tiled_Y];  // Tile size to store elements in shared memory
  __shared__ T rsB[Tiled_Y]
                  [Tiled_Y];  // Tile size to store elements in shared memory

  T Cvalue = 0.0f;
  T RCvalue = 0.0f;

  // tile matrix mul with sm
  for (int k = 0; k < (((p - 1) / Tiled_Y) + 1); k++) {
    if ((ROW < xheight) && (threadIdx.x + (k * Tiled_Y)) < p) {
      sA[threadIdx.y][threadIdx.x] = x[ROW * p + threadIdx.x + (k * Tiled_Y)];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (COL < xwidth && (threadIdx.y + k * Tiled_Y) < p) {
      rsB[threadIdx.y][threadIdx.x] =
          w0[(threadIdx.y + k * Tiled_Y) * xwidth + COL];
      sB[threadIdx.y][threadIdx.x] =
          w1[(threadIdx.y + k * Tiled_Y) * xwidth + COL];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0f;
      rsB[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int j = 0; j < Tiled_Y; ++j) {
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
      RCvalue += sA[threadIdx.y][j] * rsB[j][threadIdx.x];
    }
    __syncthreads();
  }

  if (ROW < xheight && COL < xwidth) {
    if (f0 != nullptr && f1 != nullptr) {
      out2[ROW * xwidth + COL] = Cvalue + f1[COL] + fh1[COL];
      out1[(xheight - 1 - ROW) * xwidth + COL] = RCvalue + f0[COL] + fh0[COL];
    } else {
      out2[ROW * xwidth + COL] = Cvalue + fh1[COL];
      out1[(xheight - 1 - ROW) * xwidth + COL] = RCvalue + fh0[COL];
    }
  }
}

template <typename T, int Tiled_size>
__global__ void FusionGRUCUDAKernelGru_out(
    int p, T* x0, T* x1, T* hidden0, T* hidden1, const T* w0h, const T* w1h,
    const T* h0p, const T* h1p, T* gate0, T* gate1,
    math::detail::ActivationType act_node) {
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  T a0 = 0.0f, a1 = 0.0f;
  T b0[Tiled_size], b1[Tiled_size];
  T c0 = 0.0f, c1 = 0.0f;

  int Tiled_mask = ((1 << Tiled_size) - 1);
  // Tiled  matrix multiply with register shift
  for (int k = 0; k < (((p - 1) / Tiled_size) + 1); ++k) {
    a0 = 0, a1 = 0;
    if ((threadIdx.x + k * Tiled_size) < p) {
      a0 = gate0[threadIdx.x + (k * Tiled_size) + p];
      a1 = gate1[threadIdx.x + (k * Tiled_size) + p];
    }
    for (int i = 0; i < Tiled_size; ++i) {
      if (COL < p && (i + k * Tiled_size) < p) {
        b0[i] = w0h[(i + k * Tiled_size) * p + COL];
        b1[i] = w1h[(i + k * Tiled_size) * p + COL];
      }
    }

    __syncthreads();

    for (int i = 0; i < Tiled_size; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      c0 = c0 + __shfl_sync(Tiled_mask, a0, i, Tiled_size) * b0[i];
      c1 = c1 + __shfl_sync(Tiled_mask, a1, i, Tiled_size) * b1[i];
#else
      c0 = c0 + __shfl(a0, i, Tiled_size) * b0[i];
      c1 = c1 + __shfl(a1, i, Tiled_size) * b1[i];
#endif
    }
    __syncthreads();
  }

  if (COL < p) {
    T xt_0 = x0[COL + 2 * p];
    T xt_1 = x1[COL + 2 * p];
    T gta_0 = gate0[COL];
    T gta_1 = gate1[COL];
    T htp_0 = h0p[COL];
    T htp_1 = h1p[COL];
    c0 += xt_0;
    c1 += xt_1;
    c0 = forward::activation(c0, act_node);
    c1 = forward::activation(c1, act_node);
    hidden0[COL] = c0 * gta_0 + (1 - gta_0) * htp_0;
    hidden1[COL] = c1 * gta_1 + (1 - gta_1) * htp_1;
  }
}

template <typename T, int Tiled_size>
__global__ void FusionGRUCUDAKernelGru_gate(
    int p, T* x0, T* x1, const T* w0h, const T* w1h, const T* h0p, const T* h1p,
    T* gate0, T* gate1, math::detail::ActivationType act_gate, int i) {
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  T xt_0 = 0.0f, xt_1 = 0.0f;
  T a0 = 0.0f, a1 = 0.0f;
  T c0 = 0.0f, c1 = 0.0f;
  T b0[Tiled_size], b1[Tiled_size];

  int Tiled_mask = ((1 << Tiled_size) - 1);
  // Tiled  matrix multiply using register shift, faster than sm.
  if (i != 0) {
    for (int k = 0; k < (((p - 1) / Tiled_size) + 1); ++k) {
      a0 = 0, a1 = 0;
      if ((threadIdx.x + k * Tiled_size) < p) {
        a0 = h0p[threadIdx.x + (k * Tiled_size)];
        a1 = h1p[threadIdx.x + (k * Tiled_size)];
      }
      for (int i = 0; i < Tiled_size; ++i) {
        if (COL < p * 2 && (i + k * Tiled_size) < p) {
          b0[i] = w0h[(i + k * Tiled_size) * p * 2 + COL];
          b1[i] = w1h[(i + k * Tiled_size) * p * 2 + COL];
        }
      }

      __syncthreads();

      for (int i = 0; i < Tiled_size; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        c0 = c0 + __shfl_sync(Tiled_mask, a0, i, Tiled_size) * b0[i];
        c1 = c1 + __shfl_sync(Tiled_mask, a1, i, Tiled_size) * b1[i];
#else
        c0 = c0 + __shfl(a0, i, Tiled_size) * b0[i];
        c1 = c1 + __shfl(a1, i, Tiled_size) * b1[i];
#endif
      }

      __syncthreads();
    }
  }
  // compute update & reset: +g[0-2D]
  if (COL < p * 2) {
    T xt_0 = x0[COL];
    T xt_1 = x1[COL];
    c0 += xt_0;
    c1 += xt_1;
    c0 = forward::activation(c0, act_gate);
    c1 = forward::activation(c1, act_gate);

    if (p <= COL && COL < p * 2) {
      T htp_0 = 0.0;
      T htp_1 = 0.0;
      if (i != 0) {
        htp_0 = h0p[COL - p];
        htp_1 = h1p[COL - p];
        gate0[COL] = c0 * htp_0;
        gate1[COL] = c1 * htp_1;
      }
    } else if (COL < p) {
      gate0[COL] = c0;
      gate1[COL] = c1;
    }
  }
}

template <typename T, int Tiled_X>
__global__ void FusionGRUCUDAKernel_sufmul(int xwidth, int xheight, int p,
                                           const T* x0, const T* x1,
                                           const T* w0, const T* w1, T* out) {
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  T Cvalue = 0.0f;
  T RCvalue = 0.0f;

  // tile matrix mul with sm, faster than register, the reason i think is
  // matrix is bigger than gru, and too many ALU computes in one thread.
  // Another reason is matrix mul costs little time compare with gru
  // so keep using sm untile find a better algo.
  __shared__ T sA[Tiled_X]
                 [Tiled_X];  // Tile size to store elements in shared memory
  __shared__ T rsA[Tiled_X]
                  [Tiled_X];  // Tile size to store elements in shared memory
  __shared__ T sB[Tiled_X]
                 [Tiled_X];  // Tile size to store elements in shared memory
  __shared__ T rsB[Tiled_X]
                  [Tiled_X];  // Tile size to store elements in shared memory

  for (int k = 0; k < (((p - 1) / Tiled_X) + 1); k++) {
    if ((ROW < xheight) && (threadIdx.x + (k * Tiled_X)) < p) {
      sA[threadIdx.y][threadIdx.x] = x0[ROW * p + threadIdx.x + (k * Tiled_X)];
      rsA[threadIdx.y][threadIdx.x] = x1[ROW * p + threadIdx.x + (k * Tiled_X)];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0f;
      rsA[threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (COL < xwidth && (threadIdx.y + k * Tiled_X) < p) {
      sB[threadIdx.y][threadIdx.x] =
          w0[(threadIdx.y + k * Tiled_X) * xwidth + COL];
      rsB[threadIdx.y][threadIdx.x] =
          w1[(threadIdx.y + k * Tiled_X) * xwidth + COL];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0f;
      rsB[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int j = 0; j < Tiled_X; ++j) {
      Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
      RCvalue += rsA[threadIdx.y][j] * rsB[j][threadIdx.x];
    }
    __syncthreads();
  }

  if (ROW < xheight && COL < xwidth) {
    out[ROW * xwidth + COL] = RCvalue + Cvalue;
  }
}

template <typename T>
__global__ void FusionGRUCUDAKernel_sum(int m, int n, const T* x0, const T* x1,
                                        T* out) {
  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  int ROW = blockIdx.y * blockDim.y + threadIdx.y;

  if (ROW < m && COL < n)
    out[ROW * n + COL] = x0[ROW * n + COL] + x1[ROW * n + COL];
}
}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
