// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <assert.h>
#include <cuda.h>

#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#define CUDART_INF_F __int_as_float(0x7f800000)

namespace phi {
namespace funcs {

constexpr int kWarpSize = 32;

template <typename T>
struct AddFunctor {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

template <typename T>
struct MaxFunctor {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max(a, b);
  }
};
template <typename T>
__inline__ __device__ T Inf();

template <>
__inline__ __device__ float Inf<float>() {
  return CUDART_INF_F;
}

template <>
__inline__ __device__ double Inf<double>() {
  return CUDART_INF;
}

template <typename T,
          template <typename>
          class Functor,
          int thread_group_width = kWarpSize>
__inline__ __device__ T WarpAllReduce(T val) {
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    val = Functor<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

inline cudaError_t GetNumBlocks(int64_t block_size,
                                int64_t max_blocks,
                                int64_t waves,
                                int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int sm_count;
  {
    cudaError_t err =
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  int tpm;
  {
    cudaError_t err = cudaDeviceGetAttribute(
        &tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) {
      return err;
    }
  }
  *num_blocks = std::max<int>(
      1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}
template <typename T, int N>
struct GetPackType {
  using type =
      typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};

template <typename SourceType, typename TargetType>
struct Load {
  const SourceType* src;
  int64_t row_size;

  Load(const SourceType* src, int64_t row_size)
      : src(src), row_size(row_size) {}

  template <int N>
  __device__ void load(TargetType* tar, int64_t row, int64_t col) const {
    Pack<SourceType, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage =
        *(reinterpret_cast<const PackType<SourceType, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      tar[i] = static_cast<TargetType>(pack.elem[i]);
    }
  }
};

template <typename T,
          typename SourceType,
          typename Context,
          typename Load,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth,
          bool isPadded>
__global__ void LogsumexpWarpImpl(const Context& dev_ctx,
                                  const int64_t num_row,
                                  const int64_t num_col,
                                  Load load,
                                  SourceType* out) {
  static_assert(ColsPerThread % VecSize == 0, "");
  static_assert(ThreadGroupWidth <= kWarpSize, "");
  static_assert(kWarpSize % ThreadGroupWidth == 0, "");
  constexpr int num_read = ColsPerThread / VecSize;
  assert(num_col <= ColsPerThread * ThreadGroupWidth);
  T buffer[RowsPerThread][ColsPerThread];
  const int group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_thread_group = gridDim.x * blockDim.y;
  const int thread_id = threadIdx.x;
  const int step = num_thread_group * RowsPerThread;

  for (int64_t cur_row = group_id * RowsPerThread; cur_row < num_row;
       cur_row += step) {
    T thread_max[RowsPerThread];
// Read data
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      thread_max[row_id] = -Inf<T>();
      T* row_buffer = buffer[row_id];
#pragma unroll
      for (int read_id = 0; read_id < num_read; read_id++) {
        const int offset = read_id * VecSize;
        const int cur_col = (read_id * ThreadGroupWidth + thread_id) * VecSize;
        if (!isPadded || cur_col < num_col) {
          load.template load<VecSize>(
              row_buffer + offset, cur_row + row_id, cur_col);
#pragma unroll
          for (int i = 0; i < VecSize; i++) {
            thread_max[row_id] =
                max(thread_max[row_id], row_buffer[offset + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < VecSize; ++i) {
            row_buffer[offset + i] = -Inf<T>();
          }
        }
      }
    }
    T warp_max[RowsPerThread];
// Get warp max
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      warp_max[row_id] =
          WarpAllReduce<T, MaxFunctor, ThreadGroupWidth>(thread_max[row_id]);
    }
    T thread_sum[RowsPerThread];
// Calculate
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      thread_sum[row_id] = 0;
      T* row_buffer = buffer[row_id];
#pragma unroll
      for (int i = 0; i < ColsPerThread; i++) {
        thread_sum[row_id] += exp(row_buffer[i] - warp_max[row_id]);
      }
    }

    T warp_sum[RowsPerThread];
// Get warp sum and write
#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      warp_sum[row_id] =
          WarpAllReduce<T, AddFunctor, ThreadGroupWidth>(thread_sum[row_id]);
    }

#pragma unroll
    for (int row_id = 0; row_id < RowsPerThread; row_id++) {
      out[cur_row + row_id] =
          static_cast<SourceType>(log(warp_sum[row_id]) + warp_max[row_id]);
    }
  }
}

template <typename T,
          typename SourceType,
          typename Context,
          typename Load,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth,
          bool isPadded>
inline cudaError_t LaunchLogsumexpWarp(const Context& dev_ctx,
                                       const int64_t num_row,
                                       const int64_t num_col,
                                       Load load,
                                       SourceType* out) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % ThreadGroupWidth == 0, "");
  constexpr int thread_groups_per_block = block_size / ThreadGroupWidth;
  dim3 block_dim(ThreadGroupWidth, thread_groups_per_block);
  const int64_t num_blocks =
      (num_row / RowsPerThread + thread_groups_per_block - 1) /
      thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) {
      return err;
    }
  }
  LogsumexpWarpImpl<T,
                    SourceType,
                    Context,
                    Load,
                    VecSize,
                    ColsPerThread,
                    RowsPerThread,
                    ThreadGroupWidth,
                    isPadded><<<grid_dim_x, block_dim, 0, dev_ctx.stream()>>>(
      dev_ctx, num_row, num_col, load, out);
  return cudaPeekAtLastError();
}

template <typename T,
          typename SourceType,
          typename Context,
          typename Load,
          int VecSize,
          int ColsPerThread,
          int RowsPerThread,
          int ThreadGroupWidth>
inline cudaError_t DispatchLogsumexpWarpWithPadding(const Context& dev_ctx,
                                                    const int64_t num_row,
                                                    const int64_t num_col,
                                                    Load load,
                                                    SourceType* out) {
  if (num_col == ColsPerThread * ThreadGroupWidth) {
    return LaunchLogsumexpWarp<T,
                               SourceType,
                               Context,
                               Load,
                               VecSize,
                               ColsPerThread,
                               RowsPerThread,
                               ThreadGroupWidth,
                               false>(dev_ctx, num_row, num_col, load, out);
  } else {
    return LaunchLogsumexpWarp<T,
                               SourceType,
                               Context,
                               Load,
                               VecSize,
                               ColsPerThread,
                               RowsPerThread,
                               ThreadGroupWidth,
                               true>(dev_ctx, num_row, num_col, load, out);
  }
}

template <typename T,
          typename SourceType,
          typename Context,
          typename Load,
          int VecSize>
typename std::enable_if<VecSize == 1, cudaError_t>::type
DispatchLogsumexpWarpCols(const Context& dev_ctx,
                          const int64_t num_row,
                          const int64_t num_col,
                          Load load,
                          SourceType* out) {
  if (num_col <= 0) {
    return cudaErrorInvalidValue;
  }
#define HANDLE_ROWS(thread_group_width)                            \
  else if (num_col <= (thread_group_width)*VecSize) {              \
    if (num_row % 2 == 0) {                                        \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              Load,                \
                                              VecSize,             \
                                              VecSize,             \
                                              2,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, load, out);                   \
    } else {                                                       \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              Load,                \
                                              VecSize,             \
                                              VecSize,             \
                                              1,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, load, out);                   \
    }                                                              \
  }
  HANDLE_ROWS(1)
  HANDLE_ROWS(2)
  HANDLE_ROWS(4)
  HANDLE_ROWS(8)
  HANDLE_ROWS(16)
  HANDLE_ROWS(32)
#undef HANDLE_ROWS
#define HANDLE_THREAD_GROUP(col)                        \
  else if (num_col <= (col)*kWarpSize) {                \
    return DispatchLogsumexpWarpWithPadding<T,          \
                                            SourceType, \
                                            Context,    \
                                            Load,       \
                                            VecSize,    \
                                            col,        \
                                            kWarpSize,  \
                                            1>(         \
        dev_ctx, num_row, num_col, load, out);          \
  }

  HANDLE_THREAD_GROUP(2)
  HANDLE_THREAD_GROUP(3)
  HANDLE_THREAD_GROUP(4)
  HANDLE_THREAD_GROUP(5)
  HANDLE_THREAD_GROUP(6)
  HANDLE_THREAD_GROUP(7)
  HANDLE_THREAD_GROUP(8)
  HANDLE_THREAD_GROUP(9)
  HANDLE_THREAD_GROUP(10)
  HANDLE_THREAD_GROUP(11)
  HANDLE_THREAD_GROUP(12)
  HANDLE_THREAD_GROUP(13)
  HANDLE_THREAD_GROUP(14)
  HANDLE_THREAD_GROUP(15)
  HANDLE_THREAD_GROUP(16)
  HANDLE_THREAD_GROUP(17)
  HANDLE_THREAD_GROUP(18)
  HANDLE_THREAD_GROUP(19)
  HANDLE_THREAD_GROUP(20)
  HANDLE_THREAD_GROUP(21)
  HANDLE_THREAD_GROUP(22)
  HANDLE_THREAD_GROUP(23)
  HANDLE_THREAD_GROUP(24)
  HANDLE_THREAD_GROUP(25)
  HANDLE_THREAD_GROUP(26)
  HANDLE_THREAD_GROUP(27)
  HANDLE_THREAD_GROUP(28)
  HANDLE_THREAD_GROUP(29)
  HANDLE_THREAD_GROUP(30)
  HANDLE_THREAD_GROUP(31)
  HANDLE_THREAD_GROUP(32)
#undef HANDLE_THREAD_GROUP
  else {
    return cudaErrorInvalidValue;
  }
}

template <typename T,
          typename SourceType,
          typename Context,
          typename Load,
          int VecSize>
typename std::enable_if<VecSize == 2, cudaError_t>::type
DispatchLogsumexpWarpCols(const Context& dev_ctx,
                          const int64_t num_row,
                          const int64_t num_col,
                          Load load,
                          SourceType* out) {
  if (num_col <= 0) {
    return cudaErrorInvalidValue;
  }
#define HANDLE_ROWS(thread_group_width)                            \
  else if (num_col <= (thread_group_width)*VecSize) {              \
    if (num_row % 2 == 0) {                                        \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              Load,                \
                                              VecSize,             \
                                              VecSize,             \
                                              2,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, load, out);                   \
    } else {                                                       \
      return DispatchLogsumexpWarpWithPadding<T,                   \
                                              SourceType,          \
                                              Context,             \
                                              Load,                \
                                              VecSize,             \
                                              VecSize,             \
                                              1,                   \
                                              thread_group_width>( \
          dev_ctx, num_row, num_col, load, out);                   \
    }                                                              \
  }
  HANDLE_ROWS(1)
  HANDLE_ROWS(2)
  HANDLE_ROWS(4)
  HANDLE_ROWS(8)
  HANDLE_ROWS(16)
  HANDLE_ROWS(32)
#undef HANDLE_ROWS
#define HANDLE_THREAD_GROUP(col)                        \
  else if (num_col <= (col)*kWarpSize) {                \
    return DispatchLogsumexpWarpWithPadding<T,          \
                                            SourceType, \
                                            Context,    \
                                            Load,       \
                                            VecSize,    \
                                            col,        \
                                            1,          \
                                            kWarpSize>( \
        dev_ctx, num_row, num_col, load, out);          \
  }

  HANDLE_THREAD_GROUP(4)
  HANDLE_THREAD_GROUP(6)
  HANDLE_THREAD_GROUP(8)
  HANDLE_THREAD_GROUP(10)
  HANDLE_THREAD_GROUP(12)
  HANDLE_THREAD_GROUP(14)
  HANDLE_THREAD_GROUP(16)
  HANDLE_THREAD_GROUP(18)
  HANDLE_THREAD_GROUP(20)
  HANDLE_THREAD_GROUP(22)
  HANDLE_THREAD_GROUP(24)
  HANDLE_THREAD_GROUP(26)
  HANDLE_THREAD_GROUP(28)
  HANDLE_THREAD_GROUP(30)
  HANDLE_THREAD_GROUP(32)
#undef HANDLE_THREAD_GROUP
  else {
    return cudaErrorInvalidValue;
  }
}

template <typename T, typename SourceType, typename Context, typename Load>
struct DispatchLogsumexpWarpVecSize {
  cudaError_t operator()(const Context& dev_ctx,
                         const int64_t num_row,
                         const int64_t num_col,
                         Load load,
                         SourceType* out) {
    if (num_col % 2 == 0) {
      return DispatchLogsumexpWarpCols<T, SourceType, Context, Load, 2>(
          dev_ctx, num_row, num_col, load, out);
    } else {
      return DispatchLogsumexpWarpCols<T, SourceType, Context, Load, 1>(
          dev_ctx, num_row, num_col, load, out);
    }
  }
};

template <typename T, typename SourceType, typename Context, typename Load>
inline cudaError_t DispatchLogsumexpWarp(const Context& dev_ctx,
                                         const int64_t num_row,
                                         const int64_t num_col,
                                         Load load,
                                         SourceType* out) {
  return DispatchLogsumexpWarpVecSize<T, SourceType, Context, Load>()(
      dev_ctx, num_row, num_col, load, out);
}
}  // namespace funcs
}  // namespace phi
