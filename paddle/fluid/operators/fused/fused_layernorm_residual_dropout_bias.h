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

#define LN_NUM_COLS 1024

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

template <typename T,
          int VecSize,
          typename U,
          bool ScaleBiasWithSameTypeX = false>
__device__ void CalcLayernormY(
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *bias,
    const T *x,
    T *y,
    const int row_id,
    const int col_id,
    const int cols,
    const LayerNormParamType<T> mean_val,
    const LayerNormParamType<T> invvar) {
  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using LoadU = phi::AlignedVector<U, VecSize>;
  using LoadScaleOrBias =
      phi::AlignedVector<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,
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
    phi::Load<T, VecSize>(&x[row_id * cols + i], &x_vec);

    if (scale != nullptr) {
      phi::Load<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>, VecSize>(
          &scale[i], &scale_vec);
    }
    if (bias != nullptr) {
      phi::Load<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>, VecSize>(
          &bias[i], &bias_vec);
    }

    StoreT y_vec;
    for (int ii = 0; ii < VecSize; ii++) {
      y_vec[ii] =
          static_cast<T>(static_cast<U>(scale_vec[ii]) *
                             (static_cast<U>(x_vec[ii]) - mean_val) * invvar +
                         static_cast<U>(bias_vec[ii]));
    }
    phi::Store<T, VecSize>(y_vec, &y[row_id * cols + i]);
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
template <typename T,
          typename MaskType,
          int VecSize,
          typename U,
          bool ScaleBiasWithSameTypeX = false,
          bool HasDropout = true>
__global__ void FusedLayernormResidualDropoutBias(
    const size_t rows,
    const size_t cols,
    uint64_t seed,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const bool is_test,
    const uint64_t increment,
    const float epsilon,
    const T *src,
    const T *residual,
    const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask,
    T *dst,
    T *layernorm_dst,
    LayerNormParamType<T> *mean,
    LayerNormParamType<T> *var) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;
  int idx = row_id * cols + col_id;
  curandStatePhilox4_32_10_t state;
  if (HasDropout) {
    curand_init(seed, idx, increment, &state);
  }

  T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  __shared__ U mean_share;
  __shared__ U var_share;
  __shared__ U shared_mean[32];
  __shared__ U shared_var[32];

  phi::funcs::ReluFunctor<T> relu;
  U mean_val = 0;
  U var_val = 0;
  for (int i = col_id * VecSize; i < cols; i += blockDim.x * VecSize) {
    FusedResidualDropoutBiasOneThread<T,
                                      MaskType,
                                      VecSize,
                                      true,
                                      false,
                                      phi::funcs::ReluFunctor<T>,
                                      T,
                                      T,
                                      HasDropout>(row_id,
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
                                                  &mean_val,
                                                  &var_val,
                                                  relu);
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
  CalcLayernormY<T, VecSize, U, ScaleBiasWithSameTypeX>(scale,
                                                        layernorm_bias,
                                                        dst,
                                                        layernorm_dst,
                                                        row_id,
                                                        col_id,
                                                        cols,
                                                        mean_val,
                                                        invvar);
}

template <typename T,
          typename MaskType,
          int VecSize,
          typename U,
          bool ScaleBiasWithSameTypeX = false>
void LaunchFusedLayernormResidualDropoutBiasCUDAKernel(
    int grid_dim,
    int block_dim,
    gpuStream_t stream,
    const size_t rows,
    const size_t cols,
    uint64_t seed,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const bool is_test,
    const uint64_t increment,
    const float epsilon,
    const T *src,
    const T *residual,
    const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask,
    T *dst,
    T *layernorm_dst,
    LayerNormParamType<T> *mean,
    LayerNormParamType<T> *var) {
  if (dropout_prob != 0.0f) {
    FusedLayernormResidualDropoutBias<T,
                                      MaskType,
                                      VecSize,
                                      U,
                                      ScaleBiasWithSameTypeX,
                                      true>
        <<<grid_dim, block_dim, 0, stream>>>(rows,
                                             cols,
                                             seed,
                                             dropout_prob,
                                             is_upscale_in_train,
                                             is_test,
                                             increment,
                                             epsilon,
                                             src,
                                             residual,
                                             bias,
                                             scale,
                                             layernorm_bias,
                                             mask,
                                             dst,
                                             layernorm_dst,
                                             mean,
                                             var);
  } else {
    FusedLayernormResidualDropoutBias<T,
                                      MaskType,
                                      VecSize,
                                      U,
                                      ScaleBiasWithSameTypeX,
                                      false>
        <<<grid_dim, block_dim, 0, stream>>>(rows,
                                             cols,
                                             seed,
                                             dropout_prob,
                                             is_upscale_in_train,
                                             is_test,
                                             increment,
                                             epsilon,
                                             src,
                                             residual,
                                             bias,
                                             scale,
                                             layernorm_bias,
                                             mask,
                                             dst,
                                             layernorm_dst,
                                             mean,
                                             var);
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
 */
template <typename T,
          typename MaskType,
          int VecSize,
          typename U,
          bool ScaleBiasWithSameTypeX = false>
__global__ void FusedLayernormResidualDropoutBiasInfer(
    const size_t rows,
    const size_t cols,
    uint64_t seed,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const bool is_test,
    const uint64_t increment,
    const float epsilon,
    const T *src,
    const T *residual,
    const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask,
    T *dst,
    T *layernorm_dst) {
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

  phi::funcs::ReluFunctor<T> relu;
  U mean_val = 0;
  U var_val = 0;
  for (int i = col_id * VecSize; i < cols; i += blockDim.x * VecSize) {
    FusedResidualDropoutBiasOneThread<T,
                                      MaskType,
                                      VecSize,
                                      true,
                                      false,
                                      phi::funcs::ReluFunctor<T>>(row_id,
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
                                                                  &mean_val,
                                                                  &var_val,
                                                                  relu);
  }

  mean_val = BlockReduceSum<U>(mean_val, shared_mean);
  var_val = BlockReduceSum<U>(var_val, shared_var);
  if (threadIdx.x == 0) {
    auto scale = static_cast<LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>>(
        static_cast<float>(1.) / static_cast<float>(cols));
    auto tmp = mean_val * static_cast<U>(scale);
    mean_share = static_cast<U>(tmp);
    var_share = static_cast<U>(var_val * static_cast<U>(scale) -
                               mean_share * mean_share);
    var_share = var_share > U(0) ? var_share : U(0);
  }
  __syncthreads();

  mean_val = mean_share;
  U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // calculate layernorm_dst
  CalcLayernormY<T, VecSize, U, ScaleBiasWithSameTypeX>(scale,
                                                        layernorm_bias,
                                                        dst,
                                                        layernorm_dst,
                                                        row_id,
                                                        col_id,
                                                        cols,
                                                        mean_val,
                                                        invvar);
}

template <typename T,
          typename MaskType,
          int VecSize,
          typename U,
          bool ScaleBiasWithSameTypeX = false>
struct FusedLayernormResidualDropoutBiasFunctor {
  void operator()(
      const size_t rows,
      const size_t cols,
      uint64_t seed,
      const float dropout_prob,
      const bool is_upscale_in_train,
      const bool is_test,
      const uint64_t increment,
      const float epsilon,
      const T *src,
      const T *residual,
      const T *bias,
      const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
      const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
      MaskType *mask,
      T *dst,
      T *layernorm_dst,
      LayerNormParamType<T> *mean,
      LayerNormParamType<T> *var,
      cudaStream_t stream) {
    int blockDim = GetDesiredBlockDim(cols / VecSize);
    if (mean != nullptr && var != nullptr) {
      LaunchFusedLayernormResidualDropoutBiasCUDAKernel<T,
                                                        MaskType,
                                                        VecSize,
                                                        U,
                                                        ScaleBiasWithSameTypeX>(
          rows,
          blockDim,
          stream,
          rows,
          cols,
          seed,
          dropout_prob,
          is_upscale_in_train,
          is_test,
          increment,
          epsilon,
          src,
          residual,
          bias,
          scale,
          layernorm_bias,
          mask,
          dst,
          layernorm_dst,
          mean,
          var);
    } else {
      FusedLayernormResidualDropoutBiasInfer<T,
                                             MaskType,
                                             VecSize,
                                             U,
                                             ScaleBiasWithSameTypeX>
          <<<rows, blockDim, 0, stream>>>(rows,
                                          cols,
                                          seed,
                                          dropout_prob,
                                          is_upscale_in_train,
                                          is_test,
                                          increment,
                                          epsilon,
                                          src,
                                          residual,
                                          bias,
                                          scale,
                                          layernorm_bias,
                                          mask,
                                          dst,
                                          layernorm_dst);
    }
  }
};

template struct FusedLayernormResidualDropoutBiasFunctor<
    paddle::platform::float16,
    uint8_t,
    8,
    float,
    false>;

/*
 * @brief layernorm(residual + dropout(x));
 * Conditions:
 * (1) The number of cols is 768/1024/4096;
 * (2) layer_norm scale and bias is not null;
 * (3) linear bias is null;
 * @param
 * rows: batch_size * seq_len
 * cols: 1024
 * x_: [rows, cols], inputs
 * residual_:[rows, cols]
 * bias_: [cols], linear bias, can be null
 * gamma_: [cols]: layernorm scale, not null
 * beta_: [cols], layernorm bias, not null
 * mask_out_: [rows, cols], dropout result
 * residual_out_: [rows, cols], residual + dropout(src)
 * y_: [rows, cols], layernorm result
 * mean_out_: [rows]: layernorm means
 * var_out_: [rows]: layernorm vars
 */
template <bool HasDropout,
          typename T,
          typename U,
          typename ScaleT = U,
          typename MaskType = uint8_t,
          int VecSize = 8,
          int WARPS_M = 4,
          int WARPS_N = 1,
          int BYTES_PER_LDG = 16,
          int ELTS_PER_ROW = 1024,
          int THREADS_PER_WARP = 32,
          int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
          int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW,
          int ROWS_PER_CTA = WARPS_M,
          int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
          int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA,
          typename InType = T,
          typename OutType = T>
__global__ __launch_bounds__(THREADS_PER_CTA) void fused_fast_ln_fwd_kernel(
    int rows,
    int cols,
    uint64_t seed,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const bool is_test,
    const uint64_t increment,
    const float epsilon,
    const InType *__restrict__ x_ptr,
    const T *__restrict__ residual_ptr,
    const T *__restrict__ bias_ptr,
    const ScaleT *__restrict__ gamma_ptr,
    const ScaleT *__restrict__ beta_ptr,
    MaskType *__restrict__ mask_out_ptr,
    U *__restrict__ mean_out_ptr,
    U *__restrict__ var_out_ptr,
    T *__restrict__ residual_out_ptr,
    OutType *__restrict__ y_ptr,
    const float quant_last_in_scale = 1.0,
    const float *__restrict__ quant_out_scale_ptr = nullptr,
    const float quant_next_in_scale = 1.0,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0,
    const float quant_min_bound = -127.0) {
  __shared__ U smem[WARPS_M * WARPS_N];
  using Vec = phi::AlignedVector<T, VecSize>;
  using Vec_scale = phi::AlignedVector<ScaleT, VecSize>;
  using Vec_in_type = phi::AlignedVector<InType, VecSize>;
  using Vec_out_type = phi::AlignedVector<OutType, VecSize>;
  using Vec_float = phi::AlignedVector<float, VecSize>;
  using MaskStoreT = phi::AlignedVector<MaskType, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;  // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP;  // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;         // 0
  const int warp_m = warp / WARPS_N;         // 0, 1, 2, 3

  const int c = warp_n * THREADS_PER_WARP + lane;  // lane
  const int r = bidx * ROWS_PER_CTA + warp_m;      // row id

  int idx = r * ELTS_PER_ROW + c;
  curandStatePhilox4_32_10_t state;
  if (HasDropout) {
    curand_init(seed, idx, increment, &state);
  }

  T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  // bias
  Vec bias[LDGS];
  if (bias_ptr != nullptr) {
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Load<T, VecSize>(bias_ptr + col * VecSize, &bias[it]);
      col += THREADS_PER_ROW;
    }
  }

  Vec_scale gamma[LDGS];
  Vec_scale beta[LDGS];
#pragma unroll
  for (int it = 0, col = c; it < LDGS; it++) {
    phi::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    phi::Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(ELTS_PER_ROW);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
    Vec_in_type x_input[LDGS];
    Vec residual[LDGS];
    Vec_float dequant_out_scale[LDGS];

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Load<T, VecSize>(residual_ptr + row * ELTS_PER_ROW + col * VecSize,
                            &residual[it]);
      phi::Load<InType, VecSize>(x_ptr + row * ELTS_PER_ROW + col * VecSize,
                                 &x_input[it]);
      if (quant_out_scale_ptr != nullptr) {
        phi::Load<float, VecSize>(quant_out_scale_ptr + col * VecSize,
                                  &dequant_out_scale[it]);
      }
      col += THREADS_PER_ROW;
    }

    MaskStoreT mask_vec[LDGS];
    if (!is_test && HasDropout) {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
        float rand[VecSize];
        RandVec<VecSize>(&state, rand);
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
#pragma unroll
          mask_vec[it][jt] = static_cast<MaskType>(rand[jt] >= dropout_prob);
        }
      }
    } else {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          mask_vec[it][jt] = static_cast<MaskType>(1);
        }
      }
    }

    // 4 * 8
    U xf[LDGS * VecSize];
    if (bias_ptr != nullptr) {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          // dropout(x) + residual
          if (std::is_same<InType, int32_t>::value) {
            T tmp = (static_cast<T>(static_cast<float>(x_input[it][jt]) *
                                    dequant_out_scale[it][jt]) +
                     bias[it][jt]) *
                        static_cast<T>(mask_vec[it][jt]) * factor +
                    residual[it][jt];
            x[it][jt] = tmp;
            xf[it * VecSize + jt] = U(tmp);
          } else {
            x[it][jt] = (static_cast<T>(x_input[it][jt]) + bias[it][jt]) *
                            static_cast<T>(mask_vec[it][jt]) * factor +
                        residual[it][jt];
            xf[it * VecSize + jt] = U(x[it][jt]);
          }
        }
      }
    } else {
#pragma unroll
      for (int it = 0; it < LDGS; it++) {
#pragma unroll
        for (int jt = 0; jt < VecSize; jt++) {
          // dropout(x) + residual
          if (std::is_same<InType, int32_t>::value) {
            // for int32 input, we need to dequantize.
            T tmp = static_cast<T>(static_cast<float>(x_input[it][jt]) *
                                   dequant_out_scale[it][jt]) *
                        static_cast<T>(mask_vec[it][jt]) * factor +
                    residual[it][jt];
            x[it][jt] = tmp;
          } else {
            x[it][jt] = static_cast<T>(x_input[it][jt]) *
                            static_cast<T>(mask_vec[it][jt]) * factor +
                        residual[it][jt];
          }
          xf[it * VecSize + jt] = U(x[it][jt]);
        }
      }
    }

// store dropout_residual_out and mask_out
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Store<T, VecSize>(
          x[it], residual_out_ptr + row * ELTS_PER_ROW + col * VecSize);
      col += THREADS_PER_ROW;
    }
    if (!is_test && HasDropout) {
#pragma unroll
      for (int it = 0, col = c; it < LDGS; it++) {
        phi::Store<MaskType, VecSize>(
            mask_vec[it], mask_out_ptr + row * ELTS_PER_ROW + col * VecSize);
        col += THREADS_PER_ROW;
      }
    }

    U mu_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        mu_local += xf[it * VecSize + jt];
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      mu_local += __shfl_xor_sync(uint32_t(-1), mu_local, it);
    }
    if (WARPS_N > 1) {
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = mu_local;
      }
      __syncthreads();
      if (tidx % THREADS_PER_ROW == 0) {
        mu_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          mu_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m] = mu_local;
      }
      __syncthreads();
      mu_local = smem[warp_m];
    }
    mu_local *= rn;
    if (lane == 0) {
      mean_out_ptr[row] = mu_local;
    }
    U var_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        U diff = xf[it * VecSize + jt] - mu_local;
        var_local += diff * diff;
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      var_local += __shfl_xor_sync(uint32_t(-1), var_local, it);
    }
    if (WARPS_N > 1) {
      if (lane == 0) {
        smem[warp_m * WARPS_N + warp_n] = var_local;
      }
      __syncthreads();
      if (tidx % THREADS_PER_ROW == 0) {
        var_local = 0.f;
#pragma unroll
        for (int it = 0; it < WARPS_N; ++it) {
          var_local += smem[warp_m * WARPS_N + it];
        }
        smem[warp_m] = var_local;
      }
      __syncthreads();
      var_local = smem[warp_m];
    }
    U rsigma = rsqrtf(var_local * rn + epsilon);
    if (lane == 0) {
      // Note: the stored var is different for paddle(ln) and apex (fast ln).
      // var_out_ptr[row] = rsigma;
      var_out_ptr[row] = var_local * rn;
    }

    Vec_out_type x_output[LDGS];

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        // use fp16 to compute
        // ScaleT tmp = static_cast<ScaleT>(rsigma * (xf[it * VecSize + jt] -
        // mu_local));
        // x[it][jt] = gamma[it][jt] *  tmp + beta[it][jt];
        // cast to fp32 to compute
        U tmp = rsigma * (static_cast<U>(xf[it * VecSize + jt]) - mu_local);
        x[it][jt] = static_cast<T>(static_cast<U>(gamma[it][jt]) * tmp +
                                   static_cast<U>(beta[it][jt]));

        if (std::is_same<OutType, int8_t>::value)
          x_output[it][jt] = quant_helper(x[it][jt],
                                          quant_next_in_scale,
                                          quant_round_type,
                                          quant_max_bound,
                                          quant_min_bound);
      }
    }

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      if (std::is_same<OutType, int8_t>::value) {
        phi::Store<OutType, VecSize>(
            x_output[it], y_ptr + row * ELTS_PER_ROW + col * VecSize);
      } else {
        phi::Store<T, VecSize>(
            x[it],
            reinterpret_cast<T *>(y_ptr) + row * ELTS_PER_ROW + col * VecSize);
      }
      col += THREADS_PER_ROW;
    }
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
 * mask: [rows, cols], dropout result, can be null if is_test = true
 * dst: [rows, cols], residual + dropout(src+bias)
 * layernorm_dst: [rows, cols], layernorm result
 * layernorm_bias: [cols], layernorm bias, can be null
 * scale: [cols]: layernorm scale, can be null
 * means: [rows]: layernorm means
 * vars: [rows]: layernorm vars
 */
template <typename T,
          typename MaskType,
          typename U,
          bool ScaleBiasWithSameTypeX = false,
          typename InType = T,
          typename OutType = T>
void LaunchLayernormResidualDropoutBias(
    const uint32_t rows,
    const uint32_t cols,
    const int increment,
    uint64_t seed,
    const float dropout_prob,
    const float epsilon,
    const bool is_upscale_in_train,
    const bool is_test,
    const InType *src,
    const T *residual,
    const T *bias,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *layernorm_bias,
    MaskType *mask_data,
    T *dst,
    OutType *layernorm_dst,
    LayerNormParamType<T> *mean,
    LayerNormParamType<T> *var,
    const phi::GPUContext &ctx,
    const float quant_last_in_scale = 1.0,
    const float *dequant_out_scale_data = nullptr,
    const float quant_next_in_scale = 1.0,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0,
    const float quant_min_bound = -127.0) {
  // dropout_prob == 1.0f
  // NOTE(minghaoBD): OutType should be T if drop_out_rate == 1.0
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    auto cuda_place = ctx.GetPlace();
    memory::Copy(cuda_place,
                 dst,
                 cuda_place,
                 residual,
                 rows * cols * sizeof(T),
                 ctx.stream());
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
        mask_data, 0, rows * cols * sizeof(MaskType), ctx.stream()));

    // call layernorm forward
    switch (GetDesiredBlockDim(cols)) {
      FIXED_BLOCK_DIM_CASE(
          LayerNormForward<T, U, kBlockDim, ScaleBiasWithSameTypeX>
          <<<rows, kBlockDim, 0, ctx.stream()>>>(
              dst,
              scale,
              layernorm_bias,
              reinterpret_cast<T *>(layernorm_dst),
              mean,
              var,
              epsilon,
              cols));
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Product from begin_norm_axis to end must be larger than 1"));
        break;
    }

    return;
  }

#define LAUNCH_FUSED_FAST_LN_KERNEL_BASE(cols)                                 \
  case (cols): {                                                               \
    constexpr int WARPS_N = cols < 1024 ? 1 : (cols / 1024);                   \
    constexpr int WARPS_M = 4 / WARPS_N;                                       \
    const int THREADS_PER_WARP = 32;                                           \
    const int BYTES_PER_LDG = 16;                                              \
    const int VecSize = BYTES_PER_LDG / sizeof(T);                             \
    const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;          \
    const int ROWS_PER_CTA = WARPS_M;                                          \
    const int THREADS_PER_ROW = WARPS_N * THREADS_PER_WARP;                    \
    const int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW * VecSize;                \
    const int LDGS = cols / ELTS_PER_ROW_PER_CTA;                              \
    const int grid =                                                           \
        static_cast<int>(std::ceil(rows / static_cast<float>(ROWS_PER_CTA)));  \
    if (dropout_prob != 0.0f) {                                                \
      fused_fast_ln_fwd_kernel<                                                \
          true,                                                                \
          T,                                                                   \
          U,                                                                   \
          LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,                   \
          uint8_t,                                                             \
          VecSize,                                                             \
          WARPS_M,                                                             \
          WARPS_N,                                                             \
          BYTES_PER_LDG,                                                       \
          cols,                                                                \
          THREADS_PER_WARP,                                                    \
          THREADS_PER_ROW,                                                     \
          THREADS_PER_CTA,                                                     \
          ROWS_PER_CTA,                                                        \
          ELTS_PER_ROW_PER_CTA,                                                \
          LDGS,                                                                \
          InType,                                                              \
          OutType>                                                             \
          <<<grid, THREADS_PER_CTA, 0, ctx.stream()>>>(rows,                   \
                                                       cols,                   \
                                                       seed,                   \
                                                       dropout_prob,           \
                                                       is_upscale_in_train,    \
                                                       is_test,                \
                                                       increment,              \
                                                       epsilon,                \
                                                       src,                    \
                                                       residual,               \
                                                       bias,                   \
                                                       scale,                  \
                                                       layernorm_bias,         \
                                                       mask_data,              \
                                                       mean,                   \
                                                       var,                    \
                                                       dst,                    \
                                                       layernorm_dst,          \
                                                       quant_last_in_scale,    \
                                                       dequant_out_scale_data, \
                                                       quant_next_in_scale,    \
                                                       quant_round_type,       \
                                                       quant_max_bound,        \
                                                       quant_min_bound);       \
    } else {                                                                   \
      fused_fast_ln_fwd_kernel<                                                \
          false,                                                               \
          T,                                                                   \
          U,                                                                   \
          LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,                   \
          uint8_t,                                                             \
          VecSize,                                                             \
          WARPS_M,                                                             \
          WARPS_N,                                                             \
          BYTES_PER_LDG,                                                       \
          cols,                                                                \
          THREADS_PER_WARP,                                                    \
          THREADS_PER_ROW,                                                     \
          THREADS_PER_CTA,                                                     \
          ROWS_PER_CTA,                                                        \
          ELTS_PER_ROW_PER_CTA,                                                \
          LDGS,                                                                \
          InType,                                                              \
          OutType>                                                             \
          <<<grid, THREADS_PER_CTA, 0, ctx.stream()>>>(rows,                   \
                                                       cols,                   \
                                                       seed,                   \
                                                       dropout_prob,           \
                                                       is_upscale_in_train,    \
                                                       is_test,                \
                                                       increment,              \
                                                       epsilon,                \
                                                       src,                    \
                                                       residual,               \
                                                       bias,                   \
                                                       scale,                  \
                                                       layernorm_bias,         \
                                                       mask_data,              \
                                                       mean,                   \
                                                       var,                    \
                                                       dst,                    \
                                                       layernorm_dst,          \
                                                       quant_last_in_scale,    \
                                                       dequant_out_scale_data, \
                                                       quant_next_in_scale,    \
                                                       quant_round_type,       \
                                                       quant_max_bound,        \
                                                       quant_min_bound);       \
    }                                                                          \
  } break

#define LAUNCH_FUSED_FAST_LN_KERNEL       \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(768);  \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(1024); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(1280); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(1536); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(1792); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(2048); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(3072); \
  LAUNCH_FUSED_FAST_LN_KERNEL_BASE(4096)

  bool can_call_fast_ln_kernel = false;
  if (((cols >= 768 && cols <= 2048 && cols % 256 == 0) || cols == 3072 ||
       cols == 4096) &&
      scale != nullptr && layernorm_bias != nullptr) {
    can_call_fast_ln_kernel = true;
  }
  VLOG(6) << "can_call_fast_ln_kernel = " << can_call_fast_ln_kernel;

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  if (cols % VecSize != 0) {
    int blockDim = GetDesiredBlockDim(cols);
    LaunchFusedLayernormResidualDropoutBiasCUDAKernel<T,
                                                      uint8_t,
                                                      1,
                                                      U,
                                                      ScaleBiasWithSameTypeX>(
        rows,
        blockDim,
        ctx.stream(),
        rows,
        cols,
        seed,
        dropout_prob,
        is_upscale_in_train,
        is_test,
        increment,
        epsilon,
        reinterpret_cast<const T *>(src),
        residual,
        bias,
        scale,
        layernorm_bias,
        mask_data,
        dst,
        reinterpret_cast<T *>(layernorm_dst),
        mean,
        var);
  } else {
    if (can_call_fast_ln_kernel) {
      switch (cols) {
        LAUNCH_FUSED_FAST_LN_KERNEL;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Only when column is equal to 768/1024/4096 is supported for "
              "now"));
          break;
      }
    } else {
      int blockDim = GetDesiredBlockDim(cols / VecSize);
      LaunchFusedLayernormResidualDropoutBiasCUDAKernel<T,
                                                        uint8_t,
                                                        VecSize,
                                                        U,
                                                        ScaleBiasWithSameTypeX>(
          rows,
          blockDim,
          ctx.stream(),
          rows,
          cols,
          seed,
          dropout_prob,
          is_upscale_in_train,
          is_test,
          increment,
          epsilon,
          reinterpret_cast<const T *>(src),
          residual,
          bias,
          scale,
          layernorm_bias,
          mask_data,
          dst,
          reinterpret_cast<T *>(layernorm_dst),
          mean,
          var);
    }
  }
}

template <typename T,
          typename U,
          typename MaskType,
          bool ScaleBiasWithSameTypeX = false>
void LaunchLayernormResidualDropoutGrad(
    const phi::GPUContext &dev_ctx,
    const uint32_t rows,
    const uint32_t cols,
    const float epsilon,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const T *d_out,
    const T *layernorm_src,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormParamType<T> *mean,
    const LayerNormParamType<T> *var,
    const MaskType *mask_data,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_layernorm_bias,
    T *d_residual,
    T *d_dropout_src) {
  const T zero = static_cast<T>(0.0f);
  auto factor = dropout_prob == static_cast<float>(1.0f)
                    ? zero
                    : static_cast<T>(1.0f / (1.0f - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }
  ln_bwd_fast_kernel_driver<T,
                            U,
                            LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>,
                            MaskType>(dev_ctx,
                                      rows,
                                      cols,
                                      epsilon,
                                      layernorm_src,
                                      scale,
                                      mean,
                                      var,
                                      d_out,
                                      d_residual,
                                      d_scale,
                                      d_layernorm_bias,
                                      mask_data,
                                      factor,
                                      d_dropout_src);
}

}  // namespace operators
}  // namespace paddle
