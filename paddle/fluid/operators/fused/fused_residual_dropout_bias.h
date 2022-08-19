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

#include "paddle/fluid/operators/fused/fused_dropout_common.h"

namespace paddle {
namespace operators {

/**
 * @brief The fused function called by every thread
 * VecSize can be 1, 2, 4 or 8
 */
template <typename T,
          typename MaskType,
          int VecSize,
          bool ComputeLayerNorm,
          bool Activation,
          typename Functor>
__forceinline__ __device__ void FusedResidualDropoutBiasOneThread(
    const int row_id,
    const int col_id,
    const int cols,
    curandStatePhilox4_32_10_t *state,
    const float dropout_prob,
    const T factor,
    const T *__restrict__ src,
    const T *__restrict__ residual,
    const T *__restrict__ bias,
    T *dst,
    MaskType *mask,
    const bool is_test,
    typename details::MPTypeTrait<T>::Type *mean_val,
    typename details::MPTypeTrait<T>::Type *var_val,
    Functor act_func) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using MaskStoreT = phi::AlignedVector<MaskType, VecSize>;
  using U = typename details::MPTypeTrait<T>::Type;

  LoadT src_vec;
  LoadT residual_vec;
  LoadT bias_vec;
#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    bias_vec[ii] = static_cast<T>(0);
    residual_vec[ii] = static_cast<T>(0);
  }
  // vectorize load data from global
  phi::Load<T, VecSize>(&src[row_id * cols + col_id], &src_vec);
  if (residual) {
    phi::Load<T, VecSize>(&residual[row_id * cols + col_id], &residual_vec);
  }

  if (bias) {
    phi::Load<T, VecSize>(&bias[col_id], &bias_vec);
  }

  MaskStoreT mask_vec;
  if (!is_test) {
    float rand[VecSize];
    RandVec<VecSize>(state, rand);
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(rand[ii] >= dropout_prob);
    }
  } else {
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(1);
    }
  }

  StoreT dest_vec;

#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    T tmp = src_vec[ii] + bias_vec[ii];
    if (Activation) {
      tmp = act_func(tmp);
    }
    dest_vec[ii] =
        tmp * static_cast<T>(mask_vec[ii]) * factor + residual_vec[ii];
    if (ComputeLayerNorm) {
      U tmp = static_cast<U>(dest_vec[ii]);
      *mean_val += tmp;
      *var_val += (tmp * tmp);
    }
  }

  // store result to global
  phi::Store<T, VecSize>(dest_vec, &dst[row_id * cols + col_id]);
  if (!is_test) {
    phi::Store<MaskType, VecSize>(mask_vec, &mask[row_id * cols + col_id]);
  }
}

template <typename T,
          typename MaskType,
          int VecSize,
          bool ComputeLayerNorm,
          bool Activation,
          typename Functor>
__forceinline__ __device__ void FusedResidualDropoutBiasOneThreadDQ(
    const int row_id,
    const int col_id,
    const int cols,
    curandStatePhilox4_32_10_t *state,
    const float dropout_prob,
    const T factor,
    const int32_t *__restrict__ src,
    const T *__restrict__ residual,
    const T *__restrict__ bias,
    T *dst,
    MaskType *mask,
    const bool is_test,
    typename details::MPTypeTrait<T>::Type *mean_val,
    typename details::MPTypeTrait<T>::Type *var_val,
    Functor act_func,
    const float *__restrict__ quant_out_scale_data,
    const int quant_layer_offset) {
  using LoadQ = phi::AlignedVector<int32_t, VecSize>;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadFloat = phi::AlignedVector<float, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;

  using MaskStoreT = phi::AlignedVector<MaskType, VecSize>;
  using U = typename details::MPTypeTrait<T>::Type;

  LoadQ src_vec;
  LoadT residual_vec;
  LoadT bias_vec;
  LoadFloat quant_out_scale_vec;
#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    bias_vec[ii] = static_cast<T>(0);
    residual_vec[ii] = static_cast<T>(0);
  }
  // vectorize load data from global
  phi::Load<int32_t, VecSize>(&src[row_id * cols + col_id], &src_vec);
  phi::Load<float, VecSize>(&quant_out_scale_data[quant_layer_offset + col_id],
                            &quant_out_scale_vec);
  if (residual) {
    phi::Load<T, VecSize>(&residual[row_id * cols + col_id], &residual_vec);
  }

  if (bias) {
    phi::Load<T, VecSize>(&bias[col_id], &bias_vec);
  }

  MaskStoreT mask_vec;
  if (!is_test) {
    float rand[VecSize];
    RandVec<VecSize>(state, rand);
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(rand[ii] >= dropout_prob);
    }
  } else {
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(1);
    }
  }

  StoreT dest_vec;

#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    T tmp0 = static_cast<T>(static_cast<float>(src_vec[ii]) *
                            quant_out_scale_vec[ii]);
    T tmp = tmp0 + bias_vec[ii];
    if (Activation) {
      tmp = act_func(tmp);
    }
    dest_vec[ii] =
        tmp * static_cast<T>(mask_vec[ii]) * factor + residual_vec[ii];
    if (ComputeLayerNorm) {
      U tmp = static_cast<U>(dest_vec[ii]);
      *mean_val += tmp;
      *var_val += (tmp * tmp);
    }
  }

  // store result to global
  phi::Store<T, VecSize>(dest_vec, &dst[row_id * cols + col_id]);
  if (!is_test) {
    phi::Store<MaskType, VecSize>(mask_vec, &mask[row_id * cols + col_id]);
  }
}

template <typename T,
          typename MaskType,
          int VecSize,
          bool ComputeLayerNorm,
          bool Activation,
          typename Functor>
__forceinline__ __device__ void FusedResidualDropoutBiasOneThreadQDQ(
    const int row_id,
    const int col_id,
    const int cols,
    curandStatePhilox4_32_10_t *state,
    const float dropout_prob,
    const T factor,
    const int32_t *__restrict__ src,
    const T *__restrict__ residual,
    const T *__restrict__ bias,
    int8_t *dst,
    MaskType *mask,
    const bool is_test,
    typename details::MPTypeTrait<T>::Type *mean_val,
    typename details::MPTypeTrait<T>::Type *var_val,
    Functor act_func,
    const float *__restrict__ quant_out_scale_data,
    const int quant_layer_offset,
    const float quant_in_scale_data) {
  using LoadQ = phi::AlignedVector<int32_t, VecSize>;
  using LoadFloat = phi::AlignedVector<float, VecSize>;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using StoreQ = phi::AlignedVector<int8_t, VecSize>;
  using MaskStoreT = phi::AlignedVector<MaskType, VecSize>;
  using U = typename details::MPTypeTrait<T>::Type;

  LoadQ src_vec;
  LoadT residual_vec;
  LoadT bias_vec;
  LoadFloat quant_out_scale_vec;
#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    bias_vec[ii] = static_cast<T>(0);
    residual_vec[ii] = static_cast<T>(0);
  }
  // vectorize load data from global
  phi::Load<int32_t, VecSize>(&src[row_id * cols + col_id], &src_vec);
  phi::Load<float, VecSize>(&quant_out_scale_data[quant_layer_offset + col_id],
                            &quant_out_scale_vec);
  if (residual) {
    phi::Load<T, VecSize>(&residual[row_id * cols + col_id], &residual_vec);
  }

  if (bias) {
    phi::Load<T, VecSize>(&bias[col_id], &bias_vec);
  }

  MaskStoreT mask_vec;
  if (!is_test) {
    float rand[VecSize];
    RandVec<VecSize>(state, rand);
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(rand[ii] >= dropout_prob);
    }
  } else {
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      mask_vec[ii] = static_cast<MaskType>(1);
    }
  }

  StoreT dest_vec;
  StoreQ dest_vec_int8;

#pragma unroll
  for (int ii = 0; ii < VecSize; ii++) {
    T tmp0 = static_cast<T>(static_cast<float>(src_vec[ii]) *
                            quant_out_scale_vec[ii]);
    T tmp = tmp0 + bias_vec[ii];
    if (Activation) {
      tmp = act_func(tmp);
    }
    dest_vec[ii] =
        tmp * static_cast<T>(mask_vec[ii]) * factor + residual_vec[ii];
    if (ComputeLayerNorm) {
      U tmp = static_cast<U>(dest_vec[ii]);
      *mean_val += tmp;
      *var_val += (tmp * tmp);
    }
    dest_vec_int8[ii] =
        __float2int_rn(static_cast<float>(dest_vec[ii]) * quant_in_scale_data);
  }

  // store result to global
  phi::Store<int8_t, VecSize>(dest_vec_int8, &dst[row_id * cols + col_id]);
  if (!is_test) {
    phi::Store<MaskType, VecSize>(mask_vec, &mask[row_id * cols + col_id]);
  }
}

/**
 * @brief dst = residual + dropout(src + bias);
 * the src, residual, mask and dst shape is (rows, cols)
 * the bias shape is (1, cols)
 * is_test: only used in inference
 * mask: can be null if is_test=true
 */
template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutBias(const size_t rows,
                                         const size_t cols,
                                         uint64_t seed,
                                         const float dropout_prob,
                                         const bool is_upscale_in_train,
                                         const T *__restrict__ src,
                                         const T *__restrict__ residual,
                                         const T *__restrict__ bias,
                                         MaskType *mask,
                                         T *dst,
                                         uint64_t increment,
                                         const bool is_test) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
  const T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);
  phi::funcs::ReluFunctor<T> relu;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < cols;
         i += blockDim.x * gridDim.x * VecSize) {
      FusedResidualDropoutBiasOneThread<T,
                                        MaskType,
                                        VecSize,
                                        false,
                                        false,
                                        phi::funcs::ReluFunctor<T>>(
          r,
          i,
          cols,
          &state,
          dropout_prob,
          factor,
          src,
          residual,
          bias,
          dst,
          mask,
          is_test,
          nullptr,
          nullptr,
          relu);
    }
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutBiasDQ(
    const size_t rows,
    const size_t cols,
    uint64_t seed,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const int32_t *__restrict__ src,
    const T *__restrict__ residual,
    const T *__restrict__ bias,
    MaskType *mask,
    T *dst,
    uint64_t increment,
    const bool is_test,
    const float *__restrict__ quant_out_scale_data,
    const int quant_layer_offset) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
  const T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);
  phi::funcs::ReluFunctor<T> relu;
  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < cols;
         i += blockDim.x * gridDim.x * VecSize) {
      FusedResidualDropoutBiasOneThreadDQ<T,
                                          MaskType,
                                          VecSize,
                                          false,
                                          false,
                                          phi::funcs::ReluFunctor<T>>(
          r,
          i,
          cols,
          &state,
          dropout_prob,
          factor,
          src,
          residual,
          bias,
          dst,
          mask,
          is_test,
          nullptr,
          nullptr,
          relu,
          quant_out_scale_data,
          quant_layer_offset);
    }
  }
}

/**
 * @brief dst = residual + dropout(src + bias);
 */
template <typename T, typename MaskType>
void LaunchResidualDropoutBias(const uint32_t rows,
                               const uint32_t cols,
                               const int increment,
                               uint64_t seed,
                               const float dropout_prob,
                               const bool is_test,
                               bool is_upscale_in_train,
                               const T *src,
                               const T *residual,
                               const T *bias,
                               MaskType *mask_data,
                               T *dst,
                               const phi::GPUContext &ctx) {
  // dropout_prob == 1.0f
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    if (residual == dst) return;
    if (residual) {
      memory::Copy(ctx.GetPlace(),
                   dst,
                   ctx.GetPlace(),
                   residual,
                   rows * cols * sizeof(T),
                   ctx.stream());
    } else {
      SetZero<T>(ctx, dst, rows * cols);
    }
    if (!is_test) {
      SetZero<MaskType>(ctx, mask_data, rows * cols);
    }
    return;
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  const int real_vec_size = cols % VecSize == 0 ? VecSize : 1;
  auto config = Get1DBlocksAnd2DGrids(ctx, rows, cols, real_vec_size);
  if (cols % VecSize == 0) {
    FusedResidualDropoutBias<T, uint8_t, VecSize>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            src,
            residual,
            bias,
            mask_data,
            dst,
            increment,
            is_test);
  } else {
    FusedResidualDropoutBias<T, uint8_t, 1>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            src,
            residual,
            bias,
            mask_data,
            dst,
            increment,
            is_test);
  }
}

template <typename T, typename MaskType>
void LaunchResidualDropoutBiasDQ(const uint32_t rows,
                                 const uint32_t cols,
                                 const int increment,
                                 uint64_t seed,
                                 const float dropout_prob,
                                 const bool is_test,
                                 bool is_upscale_in_train,
                                 const int32_t *src,
                                 const T *residual,
                                 const T *bias,
                                 MaskType *mask_data,
                                 T *dst,
                                 const phi::GPUContext &ctx,
                                 const float *quant_out_scale_data,
                                 const int quant_layer_offset) {
  // dropout_prob == 1.0f
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    if (residual == dst) return;
    if (residual) {
      memory::Copy(ctx.GetPlace(),
                   dst,
                   ctx.GetPlace(),
                   residual,
                   rows * cols * sizeof(T),
                   ctx.stream());
    } else {
      SetZero<T>(ctx, dst, rows * cols);
    }
    if (!is_test) {
      SetZero<MaskType>(ctx, mask_data, rows * cols);
    }
    return;
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  const int real_vec_size = cols % VecSize == 0 ? VecSize : 1;
  auto config = Get1DBlocksAnd2DGrids(ctx, rows, cols, real_vec_size);
  if (cols % VecSize == 0) {
    FusedResidualDropoutBiasDQ<T, uint8_t, VecSize>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            src,
            residual,
            bias,
            mask_data,
            dst,
            increment,
            is_test,
            quant_out_scale_data,
            quant_layer_offset);
  } else {
    FusedResidualDropoutBiasDQ<T, uint8_t, 1>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            rows,
            cols,
            seed,
            dropout_prob,
            is_upscale_in_train,
            src,
            residual,
            bias,
            mask_data,
            dst,
            increment,
            is_test,
            quant_out_scale_data,
            quant_layer_offset);
  }
}

/*
 * @brief calculate the grad of no bias
 */
template <typename T, typename MaskType, int VecSize>
__global__ void FusedResidualDropoutGrad(const T *dout,
                                         const MaskType *mask,
                                         const T factor,
                                         const int64_t size,
                                         T *dx) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    LoadT dout_vec;
    MaskLoadT mask_vec;
    phi::Load<T, VecSize>(&dout[i], &dout_vec);
    phi::Load<MaskType, VecSize>(&mask[i], &mask_vec);

    StoreT dx_vec;
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      dx_vec[ii] = dout_vec[ii] * static_cast<T>(mask_vec[ii]) * factor;
    }
    phi::Store<T, VecSize>(dx_vec, &dx[i]);
  }
}

/**
 * blocks(128 * 8)
 * 1. calculate the dx and reduce total rows to 128 rows
 * 2. save 128*8 temporary sum in 8*128 shared memory
 * 3. reduce the sum of 128 rows data by 8*VecSize warps
 */
template <typename T,
          typename MaskType,
          int BlockSizeX,
          int BlockSizeY,
          int VecSize>
__global__ void FusedResidualDropoutBiasGrad(const T *dout,
                                             const MaskType *mask,
                                             const T factor,
                                             const int64_t rows,
                                             const int64_t cols,
                                             T *dx,
                                             T *dbias) {
  int64_t col_id = blockIdx.x * blockDim.x + threadIdx.x;

  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;

  T tmp_sum[VecSize] = {static_cast<T>(0)};
  // calculate the dx and temporary sum
  if (col_id * VecSize < cols) {
    for (int row_id = threadIdx.y; row_id < rows; row_id += blockDim.y) {
      int index = row_id * cols + col_id * VecSize;
      LoadT out_vec;
      MaskLoadT mask_vec;
      StoreT dx_vec;
      phi::Load<T, VecSize>(&dout[index], &out_vec);
      phi::Load<MaskType, VecSize>(&mask[index], &mask_vec);

#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        dx_vec[i] = out_vec[i] * static_cast<T>(mask_vec[i]) * factor;
        tmp_sum[i] += out_vec[i];
      }

      phi::Store<T, VecSize>(dx_vec, &dx[index]);
    }
  }

  CalculateDBias<T, VecSize, BlockSizeX, BlockSizeY>(tmp_sum, dbias, cols);
}

/**
 * @brief to launch kernel FusedResidualDropoutBiasGradVec
 */
template <typename T, typename MaskType>
void LaunchResidualDropoutBiasGrad(const T *dout,
                                   const MaskType *mask,
                                   const float dropout_prob,
                                   const bool is_upscale_in_train,
                                   const uint32_t rows,
                                   const uint32_t cols,
                                   T *dx,
                                   T *dbias,
                                   const phi::GPUContext &ctx) {
  const T zero = static_cast<T>(0.0f);
  auto factor = dropout_prob == static_cast<float>(1.0f)
                    ? zero
                    : static_cast<T>(1.0f / (1.0f - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  int real_vec_size = cols % VecSize == 0 ? VecSize : 1;
  if (dbias != nullptr) {
    const auto threads = 8;
    auto blocks = std::max(static_cast<uint32_t>(1),
                           (cols / real_vec_size + threads - 1) / threads);
    dim3 block_dim(threads, 128, 1);
    dim3 grid_dim(blocks, 1, 1);
    if (cols % VecSize == 0) {
      FusedResidualDropoutBiasGrad<T, MaskType, 8, 128, VecSize>
          <<<grid_dim, block_dim, 0, ctx.stream()>>>(
              dout, mask, factor, rows, cols, dx, dbias);
    } else {
      FusedResidualDropoutBiasGrad<T, MaskType, 8, 128, 1>
          <<<grid_dim, block_dim, 0, ctx.stream()>>>(
              dout, mask, factor, rows, cols, dx, dbias);
    }
  } else {
    const uint64_t n = rows * cols;
    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx, n / real_vec_size);
    if (n % VecSize == 0) {
      FusedResidualDropoutGrad<T, MaskType, VecSize>
          <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
              dout, mask, factor, n, dx);
    } else {
      FusedResidualDropoutGrad<T, MaskType, 1>
          <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
              dout, mask, factor, n, dx);
    }
  }
}

}  // namespace operators
}  // namespace paddle
