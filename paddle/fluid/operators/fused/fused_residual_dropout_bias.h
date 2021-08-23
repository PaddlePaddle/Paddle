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

#include <cooperative_groups.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <iostream>
#include <memory>

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

const int VecSize = 4;

namespace paddle {
namespace operators {

namespace platform = paddle::platform;

inline std::pair<uint32_t, uint32_t> GetResidualDropoutThreads(
    const platform::CUDADeviceContext &ctx, const uint64_t n) {
  const uint64_t tmp_n = n / VecSize;
  int threads = std::max(
      (uint64_t)32, std::min(tmp_n, (uint64_t)ctx.GetMaxThreadsPerBlock()));
  int blocks = std::max((uint64_t)1, (tmp_n + threads - 1) / threads);
  return std::pair<uint32_t, uint32_t>{threads, blocks};
}

inline std::pair<dim3, dim3> GetResidualDropoutBiasThreads(
    const platform::CUDADeviceContext &ctx, const uint32_t rows,
    const uint32_t cols) {
  const uint32_t tmp_cols = cols / VecSize;
  int threads = std::max(
      (uint32_t)32, std::min(tmp_cols, (uint32_t)ctx.GetMaxThreadsPerBlock()));
  int blocks_x = std::max((uint32_t)1, (tmp_cols + threads - 1) / threads);
  int blocks_y = std::max((uint32_t)1, rows);
  dim3 block_dim(threads, 1, 1);
  dim3 grid_dim(blocks_x, blocks_y, 1);
  return std::pair<dim3, dim3>{block_dim, grid_dim};
}

/********Forward**************/
// aligned vector generates vectorized load/store on CUDA
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize(const T *pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  }
  return 1;
}

/**
 * dst = residual + dropout(src + bias);
 */
template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutBias(const size_t rows, const size_t cols,
                                         uint64_t seed,
                                         const float dropout_prob,
                                         const bool is_upscale_in_train,
                                         const T *src, const T *residual,
                                         const T *bias, MaskType *mask_data,
                                         T *dst, uint64_t increment) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  const int tmp_cols = cols / VecSize * VecSize;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < tmp_cols;
         i += blockDim.x * gridDim.x * VecSize) {
      float4 rand = curand_uniform4(&state);
      float *rand_data = &(rand.x);
      MaskType mask[VecSize];
      T bias_vec[VecSize];
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        mask[j] = (MaskType)(rand_data[j] > dropout_prob);
        bias_vec[j] = bias != nullptr ? bias[i + j] : static_cast<T>(0);
      }
#pragma unroll
      for (int j = 0; j < VecSize; j++) {
        mask_data[r * cols + i + j] = mask[j];
      }

      if (is_upscale_in_train) {
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          dst[r * cols + i + j] = (src[r * cols + i + j] + bias_vec[j]) *
                                      static_cast<T>(mask[j]) * factor +
                                  residual[r * cols + i + j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          dst[r * cols + i + j] =
              (src[r * cols + i + j] + bias_vec[j]) * static_cast<T>(mask[j]) +
              residual[r * cols + i + j];
        }
      }
    }

    int high_index = tmp_cols + col_id;
    if (high_index < cols) {
      float4 rand = curand_uniform4(&state);
      float *rand_data = &(rand.x);
      int k = 0;
      if (is_upscale_in_train) {
        for (int i = high_index; i < cols; i++) {
          MaskType m = (MaskType)(rand_data[k++] > dropout_prob);
          mask_data[r * cols + i] = m;
          dst[r * cols + i] =
              (src[r * cols + i] +
               (bias != nullptr ? bias[i] : static_cast<T>(0.0))) *
                  static_cast<T>(m) * factor +
              residual[r * cols + i];
        }
      } else {
        for (int i = high_index; i < cols; i++) {
          MaskType m = (MaskType)(rand_data[k++] > dropout_prob);
          mask_data[r * cols + i] = m;
          dst[r * cols + i] =
              (src[r * cols + i] +
               (bias != nullptr ? bias[i] : static_cast<T>(0.0))) *
                  static_cast<T>(m) +
              residual[r * cols + i];
        }
      }
    }
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutBiasVec(const size_t rows,
                                            const size_t cols, uint64_t seed,
                                            const float dropout_prob,
                                            const bool is_upscale_in_train,
                                            const T *src, const T *residual,
                                            const T *bias, MaskType *mask_data,
                                            T *dst, uint64_t increment) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T dest;
  MaskType mask;
  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < cols;
         i += blockDim.x * gridDim.x * VecSize) {
      T src_vec[VecSize];
      T residual_vec[VecSize];
      T bias_vec[VecSize];
#pragma unroll
      for (int ii = 0; ii < VecSize; ii++) {
        bias_vec[ii] = static_cast<T>(0);
      }
      LoadT *value = reinterpret_cast<LoadT *>(&src_vec);
      LoadT *residual_value = reinterpret_cast<LoadT *>(&residual_vec);
      *value = *reinterpret_cast<const LoadT *>(&src[r * cols + i]);
      *residual_value =
          *reinterpret_cast<const LoadT *>(&residual[r * cols + i]);

      LoadT *bias_value =
          bias != nullptr ? reinterpret_cast<LoadT *>(&bias_vec) : nullptr;
      if (bias != nullptr)
        *bias_value = *reinterpret_cast<const LoadT *>(&bias[i]);

      float4 rand = curand_uniform4(&state);
      T dest_vec[VecSize];
      MaskType mask_vec[VecSize];

#pragma unroll
      for (int ii = 0; ii < VecSize; ii++) {
        mask_vec[ii] = (MaskType)((&rand.x)[ii] >= dropout_prob);
      }

      if (is_upscale_in_train) {
#pragma unroll
        for (int ii = 0; ii < VecSize; ii++) {
          dest_vec[ii] = (src_vec[ii] + bias_vec[ii]) *
                             static_cast<T>(mask_vec[ii]) * factor +
                         residual_vec[ii];
        }
      } else {
#pragma unroll
        for (int ii = 0; ii < VecSize; ii++) {
          dest_vec[ii] =
              (src_vec[ii] + bias_vec[ii]) * static_cast<T>(mask_vec[ii]) +
              residual_vec[ii];
        }
      }
      *(reinterpret_cast<LoadT *>(&dst[r * cols + i])) =
          *reinterpret_cast<LoadT *>(&dest_vec[0]);
      *(reinterpret_cast<MaskLoadT *>(&mask_data[r * cols + i])) =
          *reinterpret_cast<MaskLoadT *>(&mask_vec[0]);
    }
  }
}

template <typename T, int VecSize>
__global__ void FusedResidualDropoutBiasTest(const size_t rows,
                                             const size_t cols,
                                             const float dropout_prob,
                                             const bool is_upscale_in_train,
                                             const T *src, const T *residual,
                                             const T *bias, T *dst) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;

  T factor = static_cast<T>(1.0f - dropout_prob);
  const int tmp_cols = cols / VecSize * VecSize;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < tmp_cols;
         i += blockDim.x * gridDim.x * VecSize) {
      if (is_upscale_in_train) {
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          dst[r * cols + i + j] =
              (src[r * cols + i + j] +
               (bias != nullptr ? bias[i + j] : static_cast<T>(0.0))) +
              residual[r * cols + i + j];
        }
      } else {
#pragma unroll
        for (int j = 0; j < VecSize; j++) {
          dst[r * cols + i + j] =
              (src[r * cols + i + j] +
               (bias != nullptr ? bias[i + j] : static_cast<T>(0.0))) *
                  factor +
              residual[r * cols + i + j];
        }
      }
    }

    int high_index = tmp_cols + col_id;
    if (high_index < cols) {
      if (is_upscale_in_train) {
        for (int i = high_index; i < cols; i++) {
          dst[r * cols + i] =
              (src[r * cols + i] +
               (bias != nullptr ? bias[i] : static_cast<T>(0.0))) +
              residual[r * cols + i];
        }
      } else {
        for (int i = high_index; i < cols; i++) {
          dst[r * cols + i] =
              (src[r * cols + i] +
               (bias != nullptr ? bias[i] : static_cast<T>(0.0))) *
                  factor +
              residual[r * cols + i];
        }
      }
    }
  }
}

/**
 * dst = residual + dropout(src + bias);
 */
template <typename T, typename MaskType>
void LaunchResidualDropoutBias(const uint32_t rows, const uint32_t cols,
                               const int increment, uint64_t seed,
                               const float dropout_prob,
                               bool is_upscale_in_train, const T *src,
                               const T *residual, const T *bias,
                               MaskType *mask_data, T *dst,
                               const platform::CUDADeviceContext &ctx) {
  if (std::abs(dropout_prob - 1.0) < 1e-5) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(dst, residual, rows * cols * sizeof(T),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemsetAsync(
        mask_data, 0, rows * cols * sizeof(MaskType), ctx.stream()));
    return;
  }

  auto threads = GetResidualDropoutBiasThreads(ctx, rows, cols);
  if (cols % VecSize != 0)
    FusedResidualDropoutBias<
        T, uint8_t,
        VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, src, residual,
        bias, mask_data, dst, increment);
  else
    FusedResidualDropoutBiasVec<
        T, uint8_t,
        VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, src, residual,
        bias, mask_data, dst, increment);
}

template <typename T>
void LaunchResidualDropoutBiasTest(const uint32_t rows, const uint32_t cols,
                                   const float dropout_prob,
                                   bool is_upscale_in_train, const T *src,
                                   const T *residual, const T *bias, T *dst,
                                   const platform::CUDADeviceContext &ctx) {
  if (std::abs(dropout_prob - 1.0) < 1e-5) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(dst, residual, rows * cols * sizeof(T),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
    return;
  }
  auto threads = GetResidualDropoutBiasThreads(ctx, rows, cols);
  FusedResidualDropoutBiasTest<
      T, VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
      rows, cols, dropout_prob, is_upscale_in_train, src, residual, bias, dst);
}

/********Backward**************/
template <typename T, typename MaskType>
__global__ void FusedResidualDropoutGrad(const T *dout, const MaskType *mask,
                                         const T factor, const int64_t size,
                                         T *dx, bool is_upscale_in_train) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  int tmp_size = size / VecSize * VecSize;
  for (int i = idx * VecSize; i < tmp_size;
       i += blockDim.x * gridDim.x * VecSize) {
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      dx[i + ii] = dout[i + ii] * static_cast<T>(mask[i + ii]) * factor;
    }
  }

  int high_index = tmp_size + idx;
  if (size > high_index) {
    for (int i = high_index; i < size; i++) {
      if (is_upscale_in_train)
        dx[i] = dout[i] * static_cast<T>(mask[i]) * factor;
      else
        dx[i] = dout[i] * static_cast<T>(mask[i]);
    }
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutGradVec(const T *dout, const MaskType *mask,
                                            const T factor, const int64_t size,
                                            T *dx, bool is_upscale_in_train) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    T dout_vec[VecSize];
    MaskType mask_vec[VecSize];
    LoadT *dout_value = reinterpret_cast<LoadT *>(&dout_vec);
    MaskLoadT *mask_value = reinterpret_cast<MaskLoadT *>(&mask_vec);
    *dout_value = *reinterpret_cast<const LoadT *>(&dout[i]);
    *mask_value = *reinterpret_cast<const MaskLoadT *>(&mask[i]);

    T dx_vec[VecSize];
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      dx_vec[ii] = dout_vec[ii] * static_cast<T>(mask_vec[ii]) * factor;
    }
    *(reinterpret_cast<LoadT *>(&dx[i])) =
        *reinterpret_cast<LoadT *>(&dx_vec[0]);
  }
}

template <typename T, typename MaskType, int BSX, int BSY>
__global__ void FusedResidualDropoutBiasGrad(
    const T *dout, const MaskType *mask, const T factor, const int64_t rows,
    const int64_t cols, T *dx, T *dbias, bool is_upscale_in_train) {
  int64_t col_id = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ T cache[BSX][BSY];
  T tmp_sum = static_cast<T>(0);
  if (col_id < cols) {
    for (int row_id = threadIdx.y; row_id < rows; row_id += blockDim.y) {
      int index = row_id * cols + col_id;
      T out_value = dout[index];
      if (is_upscale_in_train)
        dx[index] = out_value * static_cast<T>(mask[index]) * factor;
      else
        dx[index] = out_value * static_cast<T>(mask[index]);
      tmp_sum += out_value;
    }
  }
  cache[threadIdx.x][threadIdx.y] = tmp_sum;
  __syncthreads();

  // reduce sum
  // TODO(zhangkaihuo) : Replace with ModuleAPI
  T sum = static_cast<T>(0);
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid / BSY;
  int y = tid & (BSY - 1);

  int s = BSY / 2;
  while (s > 0) {
    if (y < s) {
      cache[x][y] += cache[x][y + s];
    }
    s /= 2;
    __syncthreads();
  }

  if (threadIdx.y == 0 && col_id < cols) {
    dbias[col_id] = cache[threadIdx.x][0];
  }
}

template <typename T, typename MaskType, int BSX, int BSY, int VecSize>
__global__ void FusedResidualDropoutBiasGradVec(
    const T *dout, const MaskType *mask, const T factor, const int64_t rows,
    const int64_t cols, T *dx, T *dbias, bool is_upscale_in_train) {
  int64_t col_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;

  T tmp_sum[VecSize] = {static_cast<T>(0)};
  if (col_id * 4 < cols) {
    for (int row_id = threadIdx.y; row_id < rows; row_id += blockDim.y) {
      int index = row_id * cols + col_id * 4;
      T out_vec[VecSize];
      MaskType mask_vec[VecSize];
      T dx_vec[VecSize];
      LoadT *out_value = reinterpret_cast<LoadT *>(&out_vec);
      MaskLoadT *mask_value = reinterpret_cast<MaskLoadT *>(&mask_vec);
      LoadT *dx_value = reinterpret_cast<LoadT *>(&dx_vec);
      *out_value = *reinterpret_cast<const LoadT *>(&dout[index]);
      *mask_value = *reinterpret_cast<const MaskLoadT *>(&mask[index]);

      if (is_upscale_in_train) {
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          dx_vec[i] = out_vec[i] * static_cast<T>(mask_vec[i]) * factor;
          tmp_sum[i] += out_vec[i];
        }
      } else {
#pragma unroll
        for (int i = 0; i < VecSize; i++) {
          dx_vec[i] = out_vec[i] * static_cast<T>(mask_vec[i]);
          tmp_sum[i] += out_vec[i];
        }
      }

      *(reinterpret_cast<LoadT *>(&dx[index])) =
          *reinterpret_cast<LoadT *>(&dx_vec[0]);
    }
  }

  __shared__ T cache[BSX * VecSize][BSY];
  for (int i = 0; i < VecSize; i++)
    cache[threadIdx.x * VecSize + i][threadIdx.y] = tmp_sum[i];
  __syncthreads();

  // reduce sum
  // TODO(zhangkaihuo) : Replace with ModuleAPI
  T sum = static_cast<T>(0);
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid / BSY;
  int y = tid & (BSY - 1);

  int s = BSY / 2;
  while (s > 0) {
    if (y < s) {
      for (int i = 0; i < VecSize; i++) {
        cache[x * VecSize + i][y] += cache[x * VecSize + i][y + s];
      }
    }
    s /= 2;
    __syncthreads();
  }

  if (threadIdx.y == 0 && col_id * VecSize < cols) {
    for (int i = 0; i < VecSize; i++)
      dbias[col_id * VecSize + i] = cache[threadIdx.x * VecSize + i][0];
  }
}

template <typename T, typename MaskType>
void LaunchResidualDropoutBiasGrad(const T *dout, const MaskType *mask,
                                   const float dropout_prob,
                                   const bool is_upscale_in_train,
                                   const uint32_t rows, const uint32_t cols,
                                   T *dx, T *dbias,
                                   const platform::CUDADeviceContext &ctx) {
  const T zero = static_cast<T>(0.0);
  auto factor = dropout_prob == static_cast<float>(1.0)
                    ? zero
                    : static_cast<T>(1.0 / (1.0 - dropout_prob));

  if (dbias != nullptr) {
    if (cols % 4 == 0) {
      auto threads = std::min(cols / VecSize, static_cast<uint32_t>(8));
      auto blocks = std::max((uint32_t)1,
                             std::min((cols / VecSize + threads - 1) / threads,
                                      (uint32_t)ctx.GetSMCount()));
      dim3 block_dim(threads, 128, 1);
      dim3 grid_dim(blocks, 1, 1);
      FusedResidualDropoutBiasGradVec<
          T, MaskType, 8, 128,
          VecSize><<<grid_dim, block_dim, 0, ctx.stream()>>>(
          dout, mask, factor, rows, cols, dx, dbias, is_upscale_in_train);

    } else {
      auto threads = std::min(cols, static_cast<uint32_t>(8));
      auto blocks = std::max(
          (uint32_t)1,
          std::min((cols + threads - 1) / threads, (uint32_t)ctx.GetSMCount()));
      dim3 block_dim(threads, 128, 1);
      dim3 grid_dim(blocks, 1, 1);
      FusedResidualDropoutBiasGrad<
          T, MaskType, 8, 128><<<grid_dim, block_dim, 0, ctx.stream()>>>(
          dout, mask, factor, rows, cols, dx, dbias, is_upscale_in_train);
    }
  } else {
    const uint64_t n = rows * cols;
    auto threads = GetResidualDropoutThreads(ctx, n);
    if (n % 4 == 0) {
      FusedResidualDropoutGradVec<
          T, MaskType,
          VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
          dout, mask, factor, n, dx, is_upscale_in_train);
    } else {
      FusedResidualDropoutGrad<
          T, MaskType><<<threads.second, threads.first, 0, ctx.stream()>>>(
          dout, mask, factor, n, dx, is_upscale_in_train);
    }
  }
}

}  // namespace operators
}  // namespace paddle
