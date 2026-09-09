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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/for_range.h"

#include "paddle/common/flags.h"

COMMON_DECLARE_bool(cudnn_deterministic);
COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

namespace phi {
namespace funcs {

template <typename T>
struct IsComplex : public std::false_type {};

template <>
struct IsComplex<phi::complex64> : public std::true_type {};

template <>
struct IsComplex<phi::complex128> : public std::true_type {};

template <typename T>
struct AddFunctor;

// The deterministic scan hard-codes summation (BlockScan::InclusiveSum), so it
// only applies to the plus-like ops that cumsum passes in.
template <typename BinaryOp>
struct IsPlusOp : public std::false_type {};

template <typename T>
struct IsPlusOp<std::plus<T>> : public std::true_type {};

template <typename T>
struct IsPlusOp<AddFunctor<T>> : public std::true_type {};

template <>
struct IsPlusOp<cub::Sum> : public std::true_type {};

// Integer addition is exactly associative, so only inexact types need the
// deterministic path.
template <typename T>
struct IsInexact : public std::is_floating_point<T> {};

template <>
struct IsInexact<phi::dtype::float16> : public std::true_type {};

template <>
struct IsInexact<phi::dtype::bfloat16> : public std::true_type {};

template <typename T>
struct IsInexact<phi::dtype::complex<T>> : public std::true_type {};

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
static void CubInclusiveScan(InputIterator x_iter,
                             OutputIterator y_iter,
                             size_t n,
                             BinaryOp op,
                             const GPUContext &dev_ctx) {
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
                                  const GPUContext &dev_ctx) {
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

template <typename T, typename index_t, bool Reverse, class BinaryFunction>
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
      index_t col1, col2;
      if (Reverse) {
        col1 = row_size - 1 - block_col - (index_t)threadIdx.x;
        col2 = row_size - 1 - block_col - (index_t)threadIdx.x - num_threads_x;
      } else {
        col1 = block_col + (index_t)threadIdx.x;
        col2 = block_col + num_threads_x + (index_t)threadIdx.x;
      }

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

template <typename T, bool Reverse, class BinaryFunction>
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
    InclusiveScanInnerDimSklanskyImpl<T, uint32_t, Reverse>(row_buf,
                                                            tgt_,
                                                            src_,
                                                            num_rows,
                                                            row_size,
                                                            log_num_threads_x,
                                                            init,
                                                            binary_op);
  } else {
    InclusiveScanInnerDimSklanskyImpl<T, size_t, Reverse>(row_buf,
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
                                   bool reverse,
                                   const GPUContext &dev_ctx) {
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

  if (reverse) {
    InclusiveScanInnerDimSklanskyKernel<T, true, BinaryOp>
        <<<grid, threads, shared_mem_bytes, dev_ctx.stream()>>>(
            tgt,
            src,
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(row_size),
            log_num_threads_x,
            init,
            op);
  } else {
    InclusiveScanInnerDimSklanskyKernel<T, false, BinaryOp>
        <<<grid, threads, shared_mem_bytes, dev_ctx.stream()>>>(
            tgt,
            src,
            static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(row_size),
            log_num_threads_x,
            init,
            op);
  }
}

// Callback operator entered by the first warp of threads in the block.
// Thread-0 is responsible for returning a value for seeding the block-wide
// scan.
template <typename T>
struct InclusiveScanBlockPrefixCallbackOp {
  T running_total;

  HOSTDEVICE explicit InclusiveScanBlockPrefixCallbackOp(T running_total)
      : running_total(running_total) {}

  HOSTDEVICE T operator()(T block_aggregate) {
    T old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <size_t kSize>
constexpr size_t GetDeterministicScanBlockThreads() {
  if (kSize >= 16) {
    return 128;
  } else if (kSize >= 8) {
    return 256;
  } else {
    return 512;
  }
}

// Per-thread reduction mirroring CCCL 3.2's ThreadReduce dispatch (sequential
// for sizeof(T) >= 8, binary tree otherwise); Paddle's pinned CUB 2.2 is always
// sequential, which breaks bitwise alignment with torch for smaller types.
template <size_t kItemsPerThread, typename T>
__device__ __forceinline__ T
InclusiveScanThreadSum(const T (&data)[kItemsPerThread]) {
  if constexpr (sizeof(T) >= 8) {
    T acc = data[0];
    for (size_t j = 1; j < kItemsPerThread; ++j) acc = acc + data[j];
    return acc;
  } else {
    T tmp[kItemsPerThread];
    for (size_t j = 0; j < kItemsPerThread; ++j) tmp[j] = data[j];
    for (size_t len = kItemsPerThread; len > 1; len /= 2) {
      for (size_t j = 0; j < len / 2; ++j) tmp[j] = tmp[2 * j] + tmp[2 * j + 1];
    }
    return tmp[0];
  }
}

// Each CTA reduces the tiles it owns into a single aggregate, so that the final
// scan pass can seed its prefix from a fixed number of aggregates. This keeps
// the summation order independent of the launch configuration.
template <size_t kBlockThreads,
          size_t kItemsPerThread,
          typename T,
          typename InputIter>
static __global__ void InclusiveScanCalcBlockSumsCUDAKernel(InputIter x,
                                                            T *agg,
                                                            int64_t numel,
                                                            int iters_per_cta) {
  int64_t offset = kBlockThreads * kItemsPerThread * iters_per_cta *
                   static_cast<int64_t>(blockIdx.x);
  int64_t remaining = numel - offset;
  if (remaining <= 0) return;
  x += offset;

  using BlockLoadT = cub::
      BlockLoad<T, kBlockThreads, kItemsPerThread, cub::BLOCK_LOAD_STRIPED>;
  using BlockReduceT = cub::BlockReduce<T, kBlockThreads>;
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockReduceT::TempStorage reduce;
  } temp_storage;

  T data[kItemsPerThread];
  T agg_val = static_cast<T>(0);
  for (int i = 0; i < iters_per_cta; ++i) {
    if (remaining >= static_cast<int64_t>(kBlockThreads * kItemsPerThread)) {
      BlockLoadT(temp_storage.load).Load(x, data);
    } else {
      BlockLoadT(temp_storage.load).Load(x, data, remaining, static_cast<T>(0));
    }
    __syncthreads();
    agg_val += BlockReduceT(temp_storage.reduce)
                   .Sum(InclusiveScanThreadSum<kItemsPerThread>(data));

    x += kBlockThreads * kItemsPerThread;
    remaining -= kBlockThreads * kItemsPerThread;
    if (remaining <= 0) break;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    agg[blockIdx.x] = agg_val;
  }
}

template <size_t kBlockThreads,
          size_t kItemsPerThread,
          typename T,
          typename InputIter,
          typename OutputIter>
static __global__ void InclusiveScanFinalScanCUDAKernel(
    InputIter x, OutputIter y, T *agg, int64_t numel, int iters_per_cta) {
  int64_t offset = kBlockThreads * kItemsPerThread * iters_per_cta *
                   static_cast<int64_t>(blockIdx.x);
  int64_t remaining = numel - offset;
  if (remaining <= 0) return;
  x += offset;
  y += offset;

  using BlockLoadT = cub::BlockLoad<T,
                                    kBlockThreads,
                                    kItemsPerThread,
                                    cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<T,
                                      kBlockThreads,
                                      kItemsPerThread,
                                      cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScanT =
      cub::BlockScan<T, kBlockThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  using BlockReduceT = cub::BlockReduce<T, kBlockThreads>;
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
    typename BlockReduceT::TempStorage reduce;
  } temp_storage;

  // Reduce the aggregates of all preceding CTAs into this CTA's initial prefix.
  T agg_data = threadIdx.x >= blockIdx.x ? static_cast<T>(0) : agg[threadIdx.x];
  // There may be fewer threads than preceding aggregates, so a thread may need
  // to accumulate more than one of them.
  for (unsigned int i = threadIdx.x + blockDim.x; i < blockIdx.x;
       i += blockDim.x) {
    agg_data += agg[i];
  }
  T aggregate = BlockReduceT(temp_storage.reduce).Sum(agg_data);
  __syncthreads();
  InclusiveScanBlockPrefixCallbackOp<T> prefix_op(aggregate);

  T data[kItemsPerThread];
  for (int i = 0; i < iters_per_cta; ++i) {
    if (remaining >= static_cast<int64_t>(kBlockThreads * kItemsPerThread)) {
      BlockLoadT(temp_storage.load).Load(x, data);
    } else {
#pragma unroll
      for (size_t j = 0; j < kItemsPerThread; ++j) {
        data[j] = static_cast<T>(0);
      }
      BlockLoadT(temp_storage.load).Load(x, data, remaining);
    }
    __syncthreads();

    // Inlined InclusiveSum whose per-thread reduction mirrors CCCL 3.2 (torch's
    // version); revert to the one-liner once Paddle's CUB reaches 300200.
    T partial = InclusiveScanThreadSum<kItemsPerThread>(data);
    T thread_prefix;
    BlockScanT(temp_storage.scan)
        .ExclusiveSum(partial, thread_prefix, prefix_op);
    T acc = thread_prefix;
    for (int j = 0; j < kItemsPerThread; ++j) {
      acc = acc + data[j];
      data[j] = acc;
    }

    __syncthreads();

    if (remaining >= static_cast<int64_t>(kBlockThreads * kItemsPerThread)) {
      BlockStoreT(temp_storage.store).Store(y, data);
    } else {
      BlockStoreT(temp_storage.store).Store(y, data, remaining);
    }
    x += kBlockThreads * kItemsPerThread;
    y += kBlockThreads * kItemsPerThread;
    remaining -= kBlockThreads * kItemsPerThread;
    if (remaining <= 0) return;
    __syncthreads();
  }
}

// Deterministic inclusive sum: the number of CTAs is capped by the SM count and
// each CTA processes a fixed range, so the summation order does not depend on
// the number of items per launch.
template <typename T, typename InputIter, typename OutputIter>
static void LaunchInclusiveDeterministicScan(InputIter x_iter,
                                             OutputIter y_iter,
                                             int64_t numel,
                                             const GPUContext &dev_ctx) {
  constexpr size_t kBlockThreads =
      GetDeterministicScanBlockThreads<sizeof(T)>();
  constexpr size_t kItemsPerThread = 16;
  constexpr int64_t kItemsPerCTAIter = kBlockThreads * kItemsPerThread;

  int64_t grid_size = CeilDiv(numel, kItemsPerCTAIter);
  int64_t num_sms = dev_ctx.GetSMCount();
  int iters_per_cta = static_cast<int>(CeilDiv(grid_size, num_sms));
  grid_size = std::min(num_sms, grid_size);

  auto agg =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), grid_size * sizeof(T));
  auto *agg_ptr = reinterpret_cast<T *>(agg->ptr());

  InclusiveScanCalcBlockSumsCUDAKernel<kBlockThreads, kItemsPerThread, T>
      <<<grid_size, kBlockThreads, 0, dev_ctx.stream()>>>(
          x_iter, agg_ptr, numel, iters_per_cta);
  InclusiveScanFinalScanCUDAKernel<kBlockThreads, kItemsPerThread, T>
      <<<grid_size, kBlockThreads, 0, dev_ctx.stream()>>>(
          x_iter, y_iter, agg_ptr, numel, iters_per_cta);
}

// A reverse scan is a forward scan over reversed iterators, so it keeps the
// same summation order as the forward case.
template <typename T>
static void InclusiveDeterministicScan(
    const T *x, T *y, int64_t numel, bool reverse, const GPUContext &dev_ctx) {
  if (reverse) {
    LaunchInclusiveDeterministicScan<T>(
        thrust::make_reverse_iterator(x + numel),
        thrust::make_reverse_iterator(y + numel),
        numel,
        dev_ctx);
  } else {
    LaunchInclusiveDeterministicScan<T>(x, y, numel, dev_ctx);
  }
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
                   const GPUContext &dev_ctx) {
  if (outer_dim == 0 || mid_dim == 0 || inner_dim == 0) return;

  if (outer_dim == 1 && inner_dim == 1) {
    // Same condition as torch: the deterministic scan is only used when the
    // scan dimension covers the whole tensor.
    if constexpr (IsPlusOp<BinaryOp>::value && IsInexact<T>::value) {
      if (FLAGS_use_accuracy_compatible_kernel && FLAGS_cudnn_deterministic) {
        InclusiveDeterministicScan<T>(
            x, y, static_cast<int64_t>(mid_dim), reverse, dev_ctx);
        return;
      }
    }
    if (reverse) {
      auto x_reverse_iter = thrust::make_reverse_iterator(x + mid_dim);
      auto y_reverse_iter = thrust::make_reverse_iterator(y + mid_dim);
      CubInclusiveScan(x_reverse_iter, y_reverse_iter, mid_dim, op, dev_ctx);
    } else {
      CubInclusiveScan(x, y, mid_dim, op, dev_ctx);
    }
  } else if (inner_dim != 1) {
    funcs::ForRange<GPUContext> for_range(dev_ctx, outer_dim * inner_dim);
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
    if (FLAGS_use_accuracy_compatible_kernel) {
      InclusiveScanInnerDimSklansky<T, BinaryOp>(
          x, y, outer_dim, mid_dim, init, op, reverse, dev_ctx);
    } else {
      InclusiveScanInnerDim<T, BinaryOp>(
          x, y, outer_dim, mid_dim, init, op, reverse, dev_ctx);
    }
  }
}

}  // namespace funcs
}  // namespace phi
