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

#include "paddle/fluid/operators/fused/fused_residual_dropout_bias.h"

namespace paddle {
namespace operators {

template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;
template <typename T>
using LayerNormParamType = typename CudnnDataType<T>::BatchNormParamType;

template <typename T, typename U, bool ScaleBiasWithSameTypeX>
using LayerNormScaleBiasT =
    typename std::conditional<ScaleBiasWithSameTypeX, T, U>::type;

/**
 * @brief fused add_bias, dropout, add residual and leyer_norm into one
 * operators. Currently only support forward
 */

template <typename T, int VecSize, typename U,
          bool ScaleBiasWithSameTypeX = false>
__device__ void CalcLayernormY(
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *bias, const T *x,
    T *y, const int row_id, const int col_id, const int cols,
    const LayerNormParamType<T> mean_val, const LayerNormParamType<T> invvar) {
  using LoadT = platform::AlignedVector<T, VecSize>;
  using StoreT = platform::AlignedVector<T, VecSize>;
  using LoadU = platform::AlignedVector<U, VecSize>;
  using LoadScaleOrBias =
      platform::AlignedVector<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,
                              VecSize>;
  for (int i = col_id * VecSize; i < cols; i += blockDim.x * VecSize) {
    LoadScaleOrBias scale_vec;
    LoadScaleOrBias bias_vec;
    LoadT x_vec;
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      scale_vec[ii] =
          static_cast<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>>(1);
      bias_vec[ii] =
          static_cast<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>>(0);
    }
    // vectorize load data from global
    platform::Load<T, VecSize>(&x[row_id * cols + i], &x_vec);

    if (scale != nullptr) {
      platform::Load<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,
                     VecSize>(&scale[i], &scale_vec);
    }
    if (bias != nullptr) {
      platform::Load<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,
                     VecSize>(&bias[i], &bias_vec);
    }

    StoreT y_vec;
    for (int ii = 0; ii < VecSize; ii++) {
      y_vec[ii] =
          static_cast<T>(static_cast<U>(scale_vec[ii]) *
                             (static_cast<U>(x_vec[ii]) - mean_val) * invvar +
                         static_cast<U>(bias_vec[ii]));
    }
    platform::Store<T, VecSize>(y_vec, &y[row_id * cols + i]);
  }
}

/**
 * @brief layernorm(residual + dropout(src + bias));
 * @param
 * rows: batch_size * seq_len
 * cols: feature_size or hidden_size
 * src: [rows, cols], inputs
 * bias: [cols], linear bias, can be null
 * residual:[rows, cols]
 * mask: [rows, cols], dropout result
 * dst: [rows, cols], residual + dropout(src+bias)
 * layernorm_dst: [rows, cols], layernorm result
 * layernorm_bias: [cols], layernorm bias, can be null
 * scale: [cols]: layernorm scale, can be null
 * means: [rows]: layernorm means
 * vars: [rows]: layernorm vars
 */
template <typename T, typename MaskType, int VecSize, typename U,
          bool ScaleBiasWithSameTypeX = false>
__global__ void FusedLayernormResidualDropoutBias(
    const size_t rows, const size_t cols, uint64_t seed,
    const float dropout_prob, const bool is_upscale_in_train,
    const bool is_test, const uint64_t increment, const float epsilon,
    const T *src, const T *residual, const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask, T *dst, T *layernorm_dst, LayerNormParamType<T> *mean,
    LayerNormParamType<T> *var) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  __shared__ U mean_share;
  __shared__ U var_share;
  __shared__ U shared_mean[32];
  __shared__ U shared_var[32];

  math::ReluFunctor<T> relu;
  U mean_val = 0;
  U var_val = 0;
  for (int i = col_id * VecSize; i < cols; i += blockDim.x * VecSize) {
    FusedResidualDropoutBiasOneThread<T, MaskType, VecSize, true, false,
                                      math::ReluFunctor<T>>(
        row_id, i, cols, &state, dropout_prob, factor, src, residual, bias, dst,
        mask, is_test, &mean_val, &var_val, relu);
  }

  mean_val = BlockReduceSum<U>(mean_val, shared_mean);
  var_val = BlockReduceSum<U>(var_val, shared_var);
  if (threadIdx.x == 0) {
    auto scale = static_cast<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>>(
        static_cast<float>(1.) / static_cast<float>(cols));
    auto tmp = mean_val * static_cast<U>(scale);
    mean[row_id] = mean_share = static_cast<U>(tmp);
    var_share = static_cast<U>(var_val * static_cast<U>(scale) -
                               mean_share * mean_share);
    var_share = var_share > U(0) ? var_share : U(0);
    var[row_id] = var_share;
  }
  __syncthreads();

  mean_val = mean_share;
  U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // calculate layernorm_dst
  CalcLayernormY<T, VecSize, U, ScaleBiasWithSameTypeX>(
      scale, layernorm_bias, dst, layernorm_dst, row_id, col_id, cols, mean_val,
      invvar);
}

/**
 * @brief layernorm(residual + dropout(src + bias));
 * @param
 * rows: batch_size * seq_len
 * cols: feature_size or hidden_size
 * src: [rows, cols], inputs
 * bias: [cols], linear bias, can be null
 * residual:[rows, cols]
 * mask: [rows, cols], dropout result, can be null if is_test = true
 * dst: [rows, cols], residual + dropout(src+bias)
 * layernorm_dst: [rows, cols], layernorm result
 * layernorm_bias: [cols], layernorm bias, can be null
 * scale: [cols]: layernorm scale, can be null
 * means: [rows]: layernorm means
 * vars: [rows]: layernorm vars
 */
template <typename T, typename MaskType, typename U,
          bool ScaleBiasWithSameTypeX = false>
void LaunchLayernormResidualDropoutBias(
    const uint32_t rows, const uint32_t cols, const int increment,
    uint64_t seed, const float dropout_prob, const float epsilon,
    const bool is_upscale_in_train, const bool is_test, const T *src,
    const T *residual, const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask_data, T *dst, T *layernorm_dst, LayerNormParamType<T> *mean,
    LayerNormParamType<T> *var, const platform::CUDADeviceContext &ctx) {
  // dropout_prob == 1.0f
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    auto cuda_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
    memory::Copy(cuda_place, dst, cuda_place, residual, rows * cols * sizeof(T),
                 ctx.stream());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
        mask_data, 0, rows * cols * sizeof(MaskType), ctx.stream()));

    // call layernorm forward
    switch (GetDesiredBlockDim(cols)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<
              T, U, kBlockDim,
              ScaleBiasWithSameTypeX><<<rows, kBlockDim, 0, ctx.stream()>>>(
              dst, scale, layernorm_bias, layernorm_dst, mean, var, epsilon,
              cols));
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Product from begin_norm_axis to end must be larger than 1"));
        break;
    }

    return;
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  if (cols % VecSize != 0) {
    int blockDim = GetDesiredBlockDim(cols);
    FusedLayernormResidualDropoutBias<
        T, uint8_t, 1, U,
        ScaleBiasWithSameTypeX><<<rows, blockDim, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, is_test, increment,
        epsilon, src, residual, bias, scale, layernorm_bias, mask_data, dst,
        layernorm_dst, mean, var);
  } else {
    int blockDim = GetDesiredBlockDim(cols / VecSize);
    FusedLayernormResidualDropoutBias<
        T, uint8_t, VecSize, U,
        ScaleBiasWithSameTypeX><<<rows, blockDim, 0, ctx.stream()>>>(
        rows, cols, seed, dropout_prob, is_upscale_in_train, is_test, increment,
        epsilon, src, residual, bias, scale, layernorm_bias, mask_data, dst,
        layernorm_dst, mean, var);
  }
}

}  // namespace operators
}  // namespace paddle
