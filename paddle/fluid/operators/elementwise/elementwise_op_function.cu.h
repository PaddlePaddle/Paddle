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

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename Functor, typename T, typename OutType = T>
static __global__ void RowWiseKernel(Functor func, const T *x, const T *y,
                                     OutType *z, int prev, int n) {
  int n_step = blockDim.x * gridDim.x;
  for (int n_i = blockIdx.x * blockDim.x + threadIdx.x; n_i < n;
       n_i += n_step) {
    T value_y = y[n_i];
    for (int prev_i = threadIdx.y; prev_i < prev; prev_i += blockDim.y) {
      T value_x = x[prev_i * n + n_i];
      z[prev_i * n + n_i] = static_cast<OutType>(func(value_x, value_y));
    }
  }
}

template <typename Functor, typename T, typename OutType = T>
static __global__ void MidWiseKernel(Functor func, const T *x, const T *y,
                                     OutType *z, int prev, int n, int post) {
  if (blockDim.y > 1) {
    for (int n_i = blockIdx.x; n_i < n; n_i += gridDim.x) {
      T value_y = y[n_i];
      for (int prev_i = threadIdx.y; prev_i < prev; prev_i += blockDim.y) {
        int offset = (prev_i * n + n_i) * post;
        for (int post_i = threadIdx.x; post_i < post; post_i += blockDim.x) {
          T value_x = x[offset + post_i];
          z[offset + post_i] = static_cast<OutType>(func(value_x, value_y));
        }
      }
    }
  } else {
    for (int prev_i = blockIdx.y; prev_i < prev; prev_i += gridDim.y) {
      for (int n_i = blockIdx.x; n_i < n; n_i += gridDim.x) {
        T value_y = y[n_i];
        int offset = (prev_i * n + n_i) * post;
        for (int post_i = threadIdx.x; post_i < post; post_i += blockDim.x) {
          T value_x = x[offset + post_i];
          z[offset + post_i] = static_cast<OutType>(func(value_x, value_y));
        }
      }
    }
  }
}

template <typename Functor, typename T, typename OutType>
class TransformFunctor<Functor, T, platform::CUDADeviceContext, OutType> {
 public:
  TransformFunctor(const framework::Tensor *x, const framework::Tensor *y,
                   framework::Tensor *z, const platform::CUDADeviceContext &ctx,
                   Functor func)
      : x_(x->data<T>()),
        y_(y->data<T>()),
        z_(z->mutable_data<OutType>(ctx.GetPlace())),
        nx_(x->numel()),
        ctx_(ctx),
        func_(func) {}

  inline void Run() const {
    platform::Transform<platform::CUDADeviceContext> trans;
    trans(ctx_, x_, x_ + nx_, y_, z_, func_);
  }

  inline void RunRowWise(int n, int prev) const {
    const int kThreadsPerBlock = ELEMWISE_MAX_BLOCK_DIM;
    const int kMaximumBlocks = 65535;

    // post is 1
    int block_dim_x = 32;
    int block_dim_y = prev * block_dim_x < kThreadsPerBlock
                          ? prev
                          : kThreadsPerBlock / block_dim_x;
    dim3 block(block_dim_x, block_dim_y);

    int grid_dim_x = ((n + 31) >> 5) << 5;
    dim3 grid(grid_dim_x);

    RowWiseKernel<Functor, T, OutType><<<grid, block, 0, ctx_.stream()>>>(
        func_, x_, y_, z_, prev, n);
  }

  inline void RunMidWise(int n, int prev, int post) const {
    const int kThreadsPerBlock = ELEMWISE_MAX_BLOCK_DIM;
    const int kMaximumBlocks = 65535;

    int block_dim_x = (post > kThreadsPerBlock) ? kThreadsPerBlock
                                                : (((post + 31) >> 5) << 5);
    int block_dim_y = block_dim_x * prev < kThreadsPerBlock ? prev : 1;
    dim3 block(block_dim_x, block_dim_y);

    int grid_dim_x = n;
    // int grid_dim_y = (n * prev > kMaximumBlocks) ? kMaximumBlocks / n : prev;
    int grid_dim_y = block_dim_y == 1 ? prev : 1;
    dim3 grid(grid_dim_x, grid_dim_y);

    MidWiseKernel<Functor, T, OutType><<<grid, block, 0, ctx_.stream()>>>(
        func_, x_, y_, z_, prev, n, post);
  }

 private:
  const T *x_;
  const T *y_;
  OutType *z_;
  int64_t nx_;
  const platform::CUDADeviceContext &ctx_;
  Functor func_;
};

}  // namespace operators
}  // namespace paddle
