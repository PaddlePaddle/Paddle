// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <thrust/device_ptr.h>
#include <thrust/iterator/reverse_iterator.h>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace paddle {
namespace operators {
namespace math {

template <typename InputIterator, typename OutputIterator, typename BinaryOp,
          typename Context>
static void CubInclusiveScan(InputIterator x_iter, OutputIterator y_iter,
                             size_t n, BinaryOp op, const Context &dev_ctx) {
  memory::AllocationPtr allocation;
  void *temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  for (size_t i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(cub::DeviceScan::InclusiveScan(
        temp_storage, temp_storage_bytes, x_iter, y_iter, op,
        static_cast<int>(n),  // Maybe overflow?
        dev_ctx.stream()));
    if (i == 0 && temp_storage_bytes > 0) {
      allocation = memory::Alloc(dev_ctx.GetPlace(), temp_storage_bytes);
      temp_storage = allocation->ptr();
    }
  }
}

template <typename T>
static auto MakeThrustReverseIterator(T *x) {
  return thrust::reverse_iterator<thrust::device_ptr<T>>(
      thrust::device_pointer_cast(x));
}

template <typename T, typename BinaryOp, bool kReverse>
struct InclusiveScanOuterOrMidDimFunctor {
  HOSTDEVICE InclusiveScanOuterOrMidDimFunctor(const T *x, T *y, size_t mid_dim,
                                               size_t inner_dim, T init,
                                               BinaryOp op)
      : x_(x),
        y_(y),
        mid_dim_(mid_dim),
        inner_dim_(inner_dim),
        init_(init),
        op_(op) {}

  HOSTDEVICE void operator()(size_t idx) const {
    auto outer_idx = idx / inner_dim_;
    auto inner_idx = idx % inner_dim_;
    if (kReverse) {
      idx = outer_idx * mid_dim_ * inner_dim_ + (mid_dim_ - 1) * inner_dim_ +
            inner_idx;
    } else {
      idx = outer_idx * mid_dim_ * inner_dim_ + inner_idx;
    }

    auto x_ptr = x_ + idx;
    auto y_ptr = y_ + idx;
    T acc_value = init_;
    for (size_t i = 0; i < mid_dim_; ++i) {
      acc_value = op_(acc_value, *x_ptr);
      *y_ptr = acc_value;
      if (kReverse) {
        x_ptr -= inner_dim_;
        y_ptr -= inner_dim_;
      } else {
        x_ptr += inner_dim_;
        y_ptr += inner_dim_;
      }
    }
  }

 private:
  const T *x_;
  T *y_;
  size_t mid_dim_;
  size_t inner_dim_;
  T init_;
  BinaryOp op_;
};

// Reference to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp

template <typename T, typename BinaryOp, size_t kThreadNumX, size_t kThreadNumY,
          bool kReverse>
static __global__ void InclusiveScanInnerDimCUDAKernel(const T *x, T *y,
                                                       size_t num_rows,
                                                       size_t row_size, T init,
                                                       BinaryOp op) {
  using RealT = phi::dtype::Real<T>;
  constexpr auto kSharedBufferSize =
      framework::IsComplex<T>::value ? 4 * kThreadNumX : 2 * kThreadNumX;
  __shared__ RealT sbuf[kThreadNumY][kSharedBufferSize];
  T *row_buf = reinterpret_cast<T *>(sbuf[threadIdx.y]);

  size_t block_row = static_cast<size_t>(blockIdx.x * kThreadNumY);
  size_t block_row_stride = static_cast<size_t>(gridDim.x * kThreadNumY);
  for (; block_row < num_rows; block_row += block_row_stride) {
    size_t row = block_row + threadIdx.y;
    T block_total = init;

    const T *row_x = x + row * row_size;
    T *row_y = y + row * row_size;
    for (size_t block_col = 0; block_col < row_size;
         block_col += 2 * kThreadNumX) {
      size_t col1, col2;
      if (kReverse) {
        col1 = row_size - 1 - block_col - threadIdx.x;
        col2 = col1 - kThreadNumX;
      } else {
        col1 = block_col + threadIdx.x;
        col2 = col1 + kThreadNumX;
      }

      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_x[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[kThreadNumX + threadIdx.x] = row_x[col2];
        } else {
          row_buf[kThreadNumX + threadIdx.x] = init;
        }

        if (threadIdx.x == 0) {
          row_buf[0] = op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      for (size_t s = kThreadNumX, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          size_t offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      for (size_t s = 2, d = kThreadNumX / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          size_t offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      if (row < num_rows) {
        if (col1 < row_size) row_y[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_y[col2] = row_buf[kThreadNumX + threadIdx.x];
      }
      block_total = row_buf[2 * kThreadNumX - 1];
      __syncthreads();
    }
  }
}

template <typename T, typename BinaryOp, typename Context>
static void InclusiveScanInnerDim(const T *x, T *y, size_t outer_dim,
                                  size_t inner_dim, T init, BinaryOp op,
                                  bool reverse, const Context &dev_ctx) {
  constexpr size_t kThreadNumX = 16;
  constexpr size_t kThreadNumY = 32;

  size_t grid_dim = (outer_dim + kThreadNumY - 1) / kThreadNumY;
  grid_dim = std::min<size_t>(grid_dim, dev_ctx.GetCUDAMaxGridDimSize()[0]);
  dim3 thread_dims(kThreadNumX, kThreadNumY);
  if (reverse) {
    InclusiveScanInnerDimCUDAKernel<
        T, BinaryOp, kThreadNumX, kThreadNumY,
        /*kReverse=*/true><<<grid_dim, thread_dims, 0, dev_ctx.stream()>>>(
        x, y, outer_dim, inner_dim, init, op);
  } else {
    InclusiveScanInnerDimCUDAKernel<
        T, BinaryOp, kThreadNumX, kThreadNumY,
        /*kReverse=*/false><<<grid_dim, thread_dims, 0, dev_ctx.stream()>>>(
        x, y, outer_dim, inner_dim, init, op);
  }
}

template <typename T, typename BinaryOp, typename Context>
void InclusiveScan(const T *x, T *y, size_t outer_dim, size_t mid_dim,
                   size_t inner_dim, T init, BinaryOp op, bool reverse,
                   const Context &dev_ctx) {
  if (outer_dim == 0 || mid_dim == 0 || inner_dim == 0) return;

  if (outer_dim == 1 && inner_dim == 1) {
    if (reverse) {
      auto x_reverse_iter = MakeThrustReverseIterator(x + mid_dim);
      auto y_reverse_iter = MakeThrustReverseIterator(y + mid_dim);
      CubInclusiveScan(x_reverse_iter, y_reverse_iter, mid_dim, op, dev_ctx);
    } else {
      CubInclusiveScan(x, y, mid_dim, op, dev_ctx);
    }
  } else if (inner_dim != 1) {
    platform::ForRange<Context> for_range(dev_ctx, outer_dim * inner_dim);
    if (reverse) {
      for_range(
          InclusiveScanOuterOrMidDimFunctor<T, BinaryOp, /*kReverse=*/true>(
              x, y, mid_dim, inner_dim, init, op));
    } else {
      for_range(
          InclusiveScanOuterOrMidDimFunctor<T, BinaryOp, /*kReverse=*/false>(
              x, y, mid_dim, inner_dim, init, op));
    }
  } else {
    InclusiveScanInnerDim<T, BinaryOp>(x, y, outer_dim, mid_dim, init, op,
                                       reverse, dev_ctx);
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
