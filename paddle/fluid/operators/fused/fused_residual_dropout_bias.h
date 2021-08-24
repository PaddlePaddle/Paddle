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

#pragma once

#include "paddle/fluid/operators/fused/fused_dropout.h"

namespace paddle {
namespace operators {

namespace platform = paddle::platform;
namespace cg = cooperative_groups;

/********Forward**************/
/**
 * @brief dst = residual + dropout(src + bias);
 * the src, residual, mask and dst shape is (rows, cols)
 * the bias shape is (1, cols)
 */
template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutBiasVec(const size_t rows,
                                            const size_t cols, uint64_t seed,
                                            const float dropout_prob,
                                            const bool is_upscale_in_train,
                                            const T *src, const T *residual,
                                            const T *bias, MaskType *mask,
                                            T *dst, uint64_t increment) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }
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
      // vectorize load data from global
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

#pragma unroll
      for (int ii = 0; ii < VecSize; ii++) {
        dest_vec[ii] = (src_vec[ii] + bias_vec[ii]) *
                           static_cast<T>(mask_vec[ii]) * factor +
                       residual_vec[ii];
      }

      // store result to global
      *(reinterpret_cast<LoadT *>(&dst[r * cols + i])) =
          *reinterpret_cast<LoadT *>(&dest_vec[0]);
      *(reinterpret_cast<MaskLoadT *>(&mask[r * cols + i])) =
          *reinterpret_cast<MaskLoadT *>(&mask_vec[0]);
    }
  }
}

/**
 * @brief for dropout's param is_test = true
 * the src, residual and dst shape is (rows, cols)
 * the bias shape is (1, cols)
 */
template <typename T, int VecSize>
__global__ void FusedResidualDropoutBiasIsTest(const size_t rows,
                                               const size_t cols,
                                               const float dropout_prob,
                                               const bool is_upscale_in_train,
                                               const T *src, const T *residual,
                                               const T *bias, T *dst) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;

  T factor = static_cast<T>(1.0f - dropout_prob);
  if (is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }

  using LoadT = AlignedVector<T, VecSize>;

  const int tmp_cols = cols / VecSize * VecSize;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < tmp_cols;
         i += blockDim.x * gridDim.x * VecSize) {
      T src_vec[VecSize];
      T residual_vec[VecSize];
      T bias_vec[VecSize];
#pragma unroll
      for (int ii = 0; ii < VecSize; ii++) {
        bias_vec[ii] = static_cast<T>(0);
      }
      // vectorize load data from global
      LoadT *value = reinterpret_cast<LoadT *>(&src_vec);
      LoadT *residual_value = reinterpret_cast<LoadT *>(&residual_vec);
      *value = *reinterpret_cast<const LoadT *>(&src[r * cols + i]);
      *residual_value =
          *reinterpret_cast<const LoadT *>(&residual[r * cols + i]);

      LoadT *bias_value =
          bias != nullptr ? reinterpret_cast<LoadT *>(&bias_vec) : nullptr;
      if (bias != nullptr)
        *bias_value = *reinterpret_cast<const LoadT *>(&bias[i]);

      T dest_vec[VecSize];
#pragma unroll
      for (int ii = 0; ii < VecSize; ii++) {
        dest_vec[ii] = (src_vec[ii] + bias_vec[ii]) * factor + residual_vec[ii];
      }

      // store result to global
      *(reinterpret_cast<LoadT *>(&dst[r * cols + i])) =
          *reinterpret_cast<LoadT *>(&dest_vec[0]);
    }
  }
}

/**
 * @brief dst = residual + dropout(src + bias);
 */
template <typename T, typename MaskType>
void LaunchResidualDropoutBias(const uint32_t rows, const uint32_t cols,
                               const int increment, uint64_t seed,
                               const float dropout_prob,
                               bool is_upscale_in_train, const T *src,
                               const T *residual, const T *bias,
                               MaskType *mask_data, T *dst,
                               const platform::CUDADeviceContext &ctx) {
  // dropout_prob == 1.0f
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(dst, residual, rows * cols * sizeof(T),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemsetAsync(
        mask_data, 0, rows * cols * sizeof(MaskType), ctx.stream()));
    return;
  }

  const int VecSize = 4;
  auto threads = Get1DBlocksAnd2DGrids(ctx, rows, cols);
  if (cols % VecSize != 0)
    FusedResidualDropoutBiasVec<
        T, uint8_t, 1><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, src, residual,
        bias, mask_data, dst, increment);
  else
    FusedResidualDropoutBiasVec<
        T, uint8_t,
        VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, src, residual,
        bias, mask_data, dst, increment);
}

/**
 *@brief to launch kernel FusedResidualDropoutBiasIsTest
 */
template <typename T>
void LaunchResidualDropoutBiasIsTest(const uint32_t rows, const uint32_t cols,
                                     const float dropout_prob,
                                     bool is_upscale_in_train, const T *src,
                                     const T *residual, const T *bias, T *dst,
                                     const platform::CUDADeviceContext &ctx) {
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpyAsync(dst, residual, rows * cols * sizeof(T),
                        cudaMemcpyDeviceToDevice, ctx.stream()));
    return;
  }
  const int VecSize = 4;
  auto threads = Get1DBlocksAnd2DGrids(ctx, rows, cols);
  if (cols % VecSize != 0)
    FusedResidualDropoutBiasIsTest<
        T, 1><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, dropout_prob, is_upscale_in_train, src, residual, bias,
        dst);
  else
    FusedResidualDropoutBiasIsTest<
        T, VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
        rows, cols, dropout_prob, is_upscale_in_train, src, residual, bias,
        dst);
}

/********Backward**************/
/*
 * @brief calculate the grad of no bias
 */
template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutGradVec(const T *dout, const MaskType *mask,
                                            const T factor, const int64_t size,
                                            T *dx) {
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

template <typename U>
static __forceinline__ __device__ U WarpReduceSum(U val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  const int warpSize = 32;
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += paddle::platform::CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

/**
 * blocks(128 * 8)
 * 1. calculate the dx and reduce total rows to 128 rows
 * 2. save 128*8 temporary sum in 8*128 shared memory
 * 3. reduce the sum of 128 rows data by 8*VecSize warps
 */
template <typename T, typename MaskType, int BSX, int BSY, int VecSize>
__global__ void FusedResidualDropoutBiasGradVec(
    const T *dout, const MaskType *mask, const T factor, const int64_t rows,
    const int64_t cols, T *dx, T *dbias) {
  int64_t col_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;

  T tmp_sum[VecSize] = {static_cast<T>(0)};
  // calculate the dx and temporary sum
  if (col_id * VecSize < cols) {
    for (int row_id = threadIdx.y; row_id < rows; row_id += blockDim.y) {
      int index = row_id * cols + col_id * VecSize;
      T out_vec[VecSize];
      MaskType mask_vec[VecSize];
      T dx_vec[VecSize];
      LoadT *out_value = reinterpret_cast<LoadT *>(&out_vec);
      MaskLoadT *mask_value = reinterpret_cast<MaskLoadT *>(&mask_vec);
      LoadT *dx_value = reinterpret_cast<LoadT *>(&dx_vec);
      *out_value = *reinterpret_cast<const LoadT *>(&dout[index]);
      *mask_value = *reinterpret_cast<const MaskLoadT *>(&mask[index]);

#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        dx_vec[i] = out_vec[i] * static_cast<T>(mask_vec[i]) * factor;
        tmp_sum[i] += out_vec[i];
      }

      *(reinterpret_cast<LoadT *>(&dx[index])) =
          *reinterpret_cast<LoadT *>(&dx_vec[0]);
    }
  }

  // save temporary sum to cache and do transpose
  __shared__ T cache[BSX * VecSize][BSY];
  for (int i = 0; i < VecSize; i++)
    cache[threadIdx.x * VecSize + i][threadIdx.y] = tmp_sum[i];
  __syncthreads();

  // reduce sum
  T sum = static_cast<T>(0);
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int x = tid >> 5;  // warp id
  int y = tid & 31;  // thread id on warp 0~31

  // need BSX * VecSize warps
  if (x < BSX * VecSize) {
// reduce 128 to 32
#pragma unroll
    for (int i = 0; i < (BSY >> 5); i++) {
      sum += cache[x][y + i * 32];
    }
  }

  // reduce 32 to 1
  sum = WarpReduceSum(sum);

  // save sum to dbias
  int bias_id = blockIdx.x * blockDim.x * VecSize + x;
  if (y == 0 && x < VecSize * BSX && bias_id < cols) {
    dbias[bias_id] = sum;
  }
}

/**
 * @brief to launch kernel FusedResidualDropoutBiasGradVec
 */
template <typename T, typename MaskType>
void LaunchResidualDropoutBiasGrad(const T *dout, const MaskType *mask,
                                   const float dropout_prob,
                                   const bool is_upscale_in_train,
                                   const uint32_t rows, const uint32_t cols,
                                   T *dx, T *dbias,
                                   const platform::CUDADeviceContext &ctx) {
  const T zero = static_cast<T>(0.0f);
  auto factor = dropout_prob == static_cast<float>(1.0f)
                    ? zero
                    : static_cast<T>(1.0f / (1.0f - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }

  const int VecSize = 4;
  if (dbias != nullptr) {
    int real_vec_size = VecSize;
    if (cols % VecSize != 0) real_vec_size = 1;
    auto threads = std::min(cols / real_vec_size, static_cast<uint32_t>(8));
    auto blocks = std::max(
        (uint32_t)1, std::min((cols / real_vec_size + threads - 1) / threads,
                              (uint32_t)ctx.GetSMCount()));
    dim3 block_dim(threads, 128, 1);
    dim3 grid_dim(blocks, 1, 1);

    if (cols % VecSize == 0) {
      FusedResidualDropoutBiasGradVec<
          T, MaskType, 8, 128,
          VecSize><<<grid_dim, block_dim, 0, ctx.stream()>>>(
          dout, mask, factor, rows, cols, dx, dbias);
    } else {
      FusedResidualDropoutBiasGradVec<
          T, MaskType, 8, 128, 1><<<grid_dim, block_dim, 0, ctx.stream()>>>(
          dout, mask, factor, rows, cols, dx, dbias);
    }
  } else {
    const uint64_t n = rows * cols;
    auto threads = Get1DThreadsAndBlocks(ctx, n);
    if (n % VecSize == 0) {
      FusedResidualDropoutGradVec<
          T, MaskType,
          VecSize><<<threads.second, threads.first, 0, ctx.stream()>>>(
          dout, mask, factor, n, dx);
    } else {
      FusedResidualDropoutGradVec<
          T, MaskType, 1><<<threads.second, threads.first, 0, ctx.stream()>>>(
          dout, mask, factor, n, dx);
    }
  }
}

}  // namespace operators
}  // namespace paddle
