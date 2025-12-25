// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_ptr.h>
#include <thrust/iterator/reverse_iterator.h>
#include <algorithm>
#include <climits>
#include "paddle/phi/kernels/funcs/cub.h"

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {
namespace funcs {

template <typename T>
struct IsComplex : public std::false_type {};

template <>
struct IsComplex<phi::complex64> : public std::true_type {};

template <>
struct IsComplex<phi::complex128> : public std::true_type {};

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
static void CubInclusiveScan(InputIterator x_iter,
                             OutputIterator y_iter,
                             size_t n,
                             BinaryOp op,
                             const phi::GPUContext &dev_ctx) {
  phi::Allocator::AllocationPtr allocation;
  void *temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  for (size_t i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cub::DeviceScan::InclusiveScan(temp_storage,
                                       temp_storage_bytes,
                                       x_iter,
                                       y_iter,
                                       op,
                                       static_cast<int>(n),
                                       dev_ctx.stream()));
    if (i == 0 && temp_storage_bytes > 0) {
      allocation =
          phi::memory_utils::Alloc(dev_ctx.GetPlace(), temp_storage_bytes);
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
  HOSTDEVICE InclusiveScanOuterOrMidDimFunctor(
      const T *x, T *y, size_t mid_dim, size_t inner_dim, T init, BinaryOp op)
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

template <typename T,
          typename BinaryOp,
          size_t kThreadNumX,
          size_t kThreadNumY,
          bool kReverse>
static __global__ void InclusiveScanInnerDimCUDAKernel(
    const T *x, T *y, size_t num_rows, size_t row_size, T init, BinaryOp op) {
  using RealT = phi::dtype::Real<T>;
  constexpr auto kSharedBufferSize =
      IsComplex<T>::value ? 4 * kThreadNumX : 2 * kThreadNumX;
  __shared__ RealT sbuf[kThreadNumY][kSharedBufferSize];
  T *row_buf = reinterpret_cast<T *>(sbuf[threadIdx.y]);

  size_t block_row = static_cast<size_t>(blockIdx.x * kThreadNumY);
  size_t block_row_stride = static_cast<size_t>(gridDim.x * kThreadNumY);
  for (; block_row < num_rows; block_row += block_row_stride) {
    size_t row = block_row + static_cast<size_t>(threadIdx.y);
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
          size_t offset = (2 * static_cast<size_t>(threadIdx.x) + 1) * d - 1;
          row_buf[offset + d] = op(row_buf[offset], row_buf[offset + d]);
        }
        __syncthreads();
      }

      for (size_t s = 2, d = kThreadNumX / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          size_t offset = 2 * (static_cast<size_t>(threadIdx.x) + 1) * d - 1;
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

template <typename T, typename BinaryOp>
static void InclusiveScanInnerDim(const T *x,
                                  T *y,
                                  size_t outer_dim,
                                  size_t inner_dim,
                                  T init,
                                  BinaryOp op,
                                  bool reverse,
                                  const phi::GPUContext &dev_ctx) {
  constexpr size_t kThreadNumX = 16;
  constexpr size_t kThreadNumY = 32;

  size_t grid_dim = (outer_dim + kThreadNumY - 1) / kThreadNumY;
  grid_dim = std::min<size_t>(grid_dim, dev_ctx.GetCUDAMaxGridDimSize()[0]);
  dim3 thread_dims(kThreadNumX, kThreadNumY);
  if (reverse) {
    InclusiveScanInnerDimCUDAKernel<T,
                                    BinaryOp,
                                    kThreadNumX,
                                    kThreadNumY,
                                    /*kReverse=*/true>
        <<<grid_dim, thread_dims, 0, dev_ctx.stream()>>>(
            x, y, outer_dim, inner_dim, init, op);
  } else {
    InclusiveScanInnerDimCUDAKernel<T,
                                    BinaryOp,
                                    kThreadNumX,
                                    kThreadNumY,
                                    /*kReverse=*/false>
        <<<grid_dim, thread_dims, 0, dev_ctx.stream()>>>(
            x, y, outer_dim, inner_dim, init, op);
  }
}

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename Integer>
constexpr inline Integer GetLogNumThreadsX(Integer num_rows, Integer row_size) {
  Integer log_num_threads_x = 0;
  Integer log_num_threads_y = 0;

  while (((Integer)1 << log_num_threads_x) < row_size) {
    ++log_num_threads_x;
  }

  while (((Integer)1 << log_num_threads_y) < num_rows) {
    ++log_num_threads_y;
  }

  Integer diff = log_num_threads_x - log_num_threads_y;

  log_num_threads_x = ((Integer)9 + diff) / (Integer)2;

  log_num_threads_x =
      std::min(std::max((Integer)4, log_num_threads_x), (Integer)9);

  return log_num_threads_x;
}

template <typename T, typename index_t, class BinaryFunction>
__device__ void InclusiveScanInnerDimSklanskyImpl(
    T *row_buf,
    T *tgt_,
    const T *src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    const uint32_t log_num_threads_x,
    T init,
    BinaryFunction binary_op) {
  const index_t num_threads_x = 1 << log_num_threads_x;

  for (index_t block_row = blockIdx.x * (index_t)blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    index_t row = block_row + (index_t)threadIdx.y;
    T block_total = init;

    const T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;
    const bool row_exists = row < num_rows;

    for (index_t block_col = 0; block_col < row_size;
         block_col += 2 * num_threads_x) {
      index_t col1 = block_col + (index_t)threadIdx.x;
      index_t col2 = block_col + num_threads_x + (index_t)threadIdx.x;

      if (row_exists) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        if (threadIdx.x == 0) {
          row_buf[0] = binary_op(row_buf[0], block_total);
        }
      }
      __syncthreads();

      for (int m = 0; m <= log_num_threads_x; ++m) {
        if (row_exists) {
          index_t s = 1 << m;
          auto a = static_cast<index_t>((threadIdx.x >> m) << (m + 1)) | s;
          index_t ti = a + (threadIdx.x % s);
          index_t si = a - 1;

          row_buf[ti] = binary_op(row_buf[ti], row_buf[si]);
        }
        __syncthreads();
      }

      if (row_exists) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size)
          row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }

      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <typename T, class BinaryFunction>
__global__ void InclusiveScanInnerDimSklanskyKernel(
    T *tgt_,
    const T *src_,
    const uint32_t num_rows,
    const uint32_t row_size,
    const uint32_t log_num_threads_x,
    T init,
    BinaryFunction binary_op) {
  extern __shared__ char sbuf[];
  T *sbuf2 = reinterpret_cast<T *>(sbuf);

  const uint32_t num_threads_x = 1 << log_num_threads_x;
  T *row_buf = reinterpret_cast<T *>(sbuf2 + num_threads_x * 2 * threadIdx.y);

  if (static_cast<size_t>(num_rows) * static_cast<size_t>(row_size) <=
      UINT_MAX) {
    InclusiveScanInnerDimSklanskyImpl<T, uint32_t>(row_buf,
                                                   tgt_,
                                                   src_,
                                                   num_rows,
                                                   row_size,
                                                   log_num_threads_x,
                                                   init,
                                                   binary_op);
  } else {
    InclusiveScanInnerDimSklanskyImpl<T, size_t>(row_buf,
                                                 tgt_,
                                                 src_,
                                                 num_rows,
                                                 row_size,
                                                 log_num_threads_x,
                                                 init,
                                                 binary_op);
  }
}

template <typename T, typename BinaryOp>
void InclusiveScanInnerDimSklansky(const T *src,
                                   T *tgt,
                                   size_t outer_dim,
                                   size_t inner_dim,
                                   T init,
                                   BinaryOp op,
                                   const phi::GPUContext &dev_ctx) {
  int64_t num_rows = outer_dim;
  int64_t row_size = inner_dim;

  const uint32_t num_threads = 512;
  const uint32_t log_num_threads_x = GetLogNumThreadsX(num_rows, row_size);
  const uint32_t num_threads_x = (1 << log_num_threads_x);
  const uint32_t num_threads_y = num_threads / num_threads_x;

  dim3 threads(num_threads_x, num_threads_y);

  int64_t max_grid_dim = dev_ctx.GetCUDAMaxGridDimSize()[0];
  int64_t grid_y = CeilDiv(num_rows, int64_t{threads.y});
  dim3 grid(std::min(max_grid_dim, grid_y));

  size_t shared_mem_bytes = num_threads_y * (num_threads_x * 2) * sizeof(T);

  InclusiveScanInnerDimSklanskyKernel<T, BinaryOp>
      <<<grid, threads, shared_mem_bytes, dev_ctx.stream()>>>(
          tgt,
          src,
          static_cast<uint32_t>(num_rows),
          static_cast<uint32_t>(row_size),
          log_num_threads_x,
          init,
          op);
}

template <typename T, typename BinaryOp>
void InclusiveScan(const T *x,
                   T *y,
                   size_t outer_dim,
                   size_t mid_dim,
                   size_t inner_dim,
                   T init,
                   BinaryOp op,
                   bool reverse,
                   const phi::GPUContext &dev_ctx) {
  if (outer_dim == 0 || mid_dim == 0 || inner_dim == 0) return;

  if (outer_dim == 1 && inner_dim == 1) {
    if (reverse) {
      auto x_reverse_iter = thrust::make_reverse_iterator(x + mid_dim);
      auto y_reverse_iter = thrust::make_reverse_iterator(y + mid_dim);
      CubInclusiveScan(x_reverse_iter, y_reverse_iter, mid_dim, op, dev_ctx);
    } else {
      CubInclusiveScan(x, y, mid_dim, op, dev_ctx);
    }
  } else if (inner_dim != 1) {
    phi::funcs::ForRange<phi::GPUContext> for_range(dev_ctx,
                                                    outer_dim * inner_dim);
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
    if (FLAGS_use_accuracy_compatible_kernel && !reverse) {
      InclusiveScanInnerDimSklansky<T, BinaryOp>(
          x, y, outer_dim, mid_dim, init, op, dev_ctx);
    } else {
      InclusiveScanInnerDim<T, BinaryOp>(
          x, y, outer_dim, mid_dim, init, op, reverse, dev_ctx);
    }
  }
}

}  // namespace funcs
}  // namespace phi
