/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <iostream>
#include <utility>

namespace paddle {
namespace platform {

// element vector N {1, 2, 4, 8}
template <typename T, int N>
struct alignas(sizeof(T) * N) Element {
  T data[N];
  // constructor
  __host__ __device__ Element() {}

  __host__ __device__ Element(T val) { operator=(val); }

  template <typename... Args>
  __host__ __device__ Element(Args... args) {
    Set(0, args...);
  }

  // +
  __host__ __device__ inline Element<T, N> operator+(Element<T, N> &other) {
    Element<T, N> res;
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      res.data[idx] = data[idx] + other.data[idx];
    }
    return res;
  }

  // +=
  __device__ inline void operator+=(Element<T, N> other) {
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      data[idx] += other.data[idx];
    }
  }

  // []
  __host__ __device__ inline T &operator[](const uint32_t index) {
    if (index < N)
      return data[index];
    else
      return data[0];
  }

  // Set
  __host__ __device__ inline void Set(size_t index, T val) {
    if (index < N) data[index] = val;
  }

  template <typename... Args>
  __host__ __device__ void Set(size_t index, T first, Args... args) {
    data[index] = first;
    if (sizeof...(args) == 0 || index >= N) {
      return;
    }

    Set(index + 1, args...);
  }

  // =
  __host__ __device__ inline void operator=(T val) {
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      data[idx] = val;
    }
  }

  template <typename... Args>
  __host__ __device__ void operator=(Args... args) {
    Set(0, args...);
  }

  __host__ __device__ inline void operator=(const Element<T, N> &other) {
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      data[idx] = other.data[idx];
    }
  }

  // sum
  __host__ __device__ __inline__ T Sum() {
    T val = data[0];
#pragma unroll
    for (int idx = 1; idx < N; ++idx) {
      val += data[idx];
    }
    return val;
  }
};

/*
 * @annotation blocksize x < y
 * @param shared_addr shared for reshape
 * @param value value in thread
 * @param tx,ty thread idx,idy
 * @param (x = col, y = row):(8, 16),(8, 32),(8,64),(8,128),(16,32),(16,64)
 */

template <typename T, int N, int X, int Y, int num_warps, int warp_size>
struct Transpose {
  __device__ void operator()(Element<T, N> *value, int tx, int ty,
                             Element<T, N>) {
    const int R = Y / X;
    __shared__ Element<T, N> shared_addr_0[X][Y];
    // transpose data
    shared_addr_0[threadIdx.x]
                 [((threadIdx.y % R) * X + threadIdx.y / R + threadIdx.x) % Y] =
                     *value;
    __syncthreads();
    // reduce
    if (tx < warp_size) {
      // load data from shared
      *value = shared_addr_0[ty][tx];
// sum the rest data
#pragma unroll
      for (int idx = 1; idx < num_warps; ++idx) {
        *value += shared_addr_0[ty][idx * warp_size + tx];
      }

#pragma unroll
      for (int offset = warp_size >> 1; offset > 0; offset = offset >> 1) {
#pragma unroll
        for (int idx = 0; idx < N; ++idx) {
          value->data[idx] +=
              __shfl_down_sync(0xffffffff, value->data[idx], offset, warp_size);
        }
      }
    }
    __syncthreads();
  }

  __device__ void operator()(Element<T, N> *value, int tx, int ty, T) {
    const int R = Y / X;
    __shared__ T shared_addr_0[X][Y];
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      // transpose
      shared_addr_0[threadIdx.x]
                   [((threadIdx.y % R) * X + threadIdx.y / R + threadIdx.x) %
                    Y] = value->data[idx];
      __syncthreads();
      // reduction
      if (tx < warp_size) {
        // load data from shared
        value->data[idx] = shared_addr_0[ty][tx];
// sum the rest data
#pragma unroll
        for (int idy = 1; idy < num_warps; ++idy) {
          value->data[idx] += shared_addr_0[ty][idy * warp_size + tx];
        }

#pragma unroll
        for (int offset = warp_size >> 1; offset > 0; offset = offset >> 1) {
          value->data[idx] +=
              __shfl_down_sync(0xffffffff, value->data[idx], offset, warp_size);
        }
      }
      __syncthreads();
    }
  }

  __device__ void operator()(Element<T, N> *value, int tx, int ty, T *) {
    const int R = Y / X;
    __shared__ T shared_addr_0[N][X][Y];
// transpose
#pragma unroll
    for (int idx = 0; idx < N; ++idx) {
      shared_addr_0[idx][threadIdx.x]
                   [((threadIdx.y % R) * X + threadIdx.y / R + threadIdx.x) %
                    Y] = value->data[idx];
    }
    __syncthreads();
    // reduction
    if (tx < warp_size) {
#pragma unroll
      for (int idx = 0; idx < N; ++idx) {
        // load data from shared
        value->data[idx] = shared_addr_0[idx][ty][tx];
// sum the rest data
#pragma unroll
        for (int idy = 1; idy < num_warps; ++idy) {
          value->data[idx] += shared_addr_0[idx][ty][idy * warp_size + tx];
        }
#pragma unroll
        for (int offset = warp_size >> 1; offset > 0; offset = offset >> 1) {
          value->data[idx] +=
              __shfl_down_sync(0xffffffff, value->data[idx], offset, warp_size);
        }
      }
    }
    __syncthreads();
  }
};

template <typename T, int N, int x, int y, typename ST = Element<T, N>,
          int ROW = 0, int COL = 0, int num_warps = (y + 31) / 32,
          int warp_size = y <= 32 ? y : 32>
__global__ void BlockReduce(const void *input, const int row, const int col,
                             void *result) {
  const int num_col = col / N;
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // compute new thread idx in block [x,y]
  const int _tx = (blockDim.x * threadIdx.y + threadIdx.x) % blockDim.y;
  const int _ty = (blockDim.x * threadIdx.y + threadIdx.x) / blockDim.y;

  // read data from global memery
  const Element<T, N> *_input = static_cast<const Element<T, N> *>(input) +
                                threadIdx.y * num_col + col_idx;
  Element<T, N> value = T(0);
  if (col_idx < num_col) {
    for (int idx = 0; idx < row; idx += y) {
      value += *_input;
      _input += y * num_col;
    }
  }
  // tranpose and reduce
  Transpose<T, N, x, y, num_warps, warp_size> transpose;
  transpose(&value, _tx, _ty, ST());

  // write value into tmp_shared
  __shared__ Element<T, N> tmp_shared[x];
  if (_tx == 0) {
    tmp_shared[_ty] = value;
  }
  // do thread synchronization
  __syncthreads();
  // write tmp_shared into result
  if (col_idx < num_col && threadIdx.y == 0) {
    static_cast<Element<T, N> *>(result)[col_idx] = tmp_shared[threadIdx.x];
  }
}

}  // namespace platform
}  // namespace paddle
