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

template <typename T, int VecSize, typename U,
          bool ScaleBiasWithSameTypeX = false>
__device__ void CalcLayernormY(
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *bias, const T *x,
    T *y, const int row_id, const int col_id, const int cols,
    const LayerNormParamType<T> mean_val, const LayerNormParamType<T> invvar) {
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

  phi::funcs::ReluFunctor<T> relu;
  U mean_val = 0;
  U var_val = 0;
  for (int i = col_id * VecSize; i < cols; i += blockDim.x * VecSize) {
    FusedResidualDropoutBiasOneThread<T, MaskType, VecSize, true, false,
                                      phi::funcs::ReluFunctor<T>>(
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

/*
* @brief layernorm(residual + dropout(x));
 * Conditions:
 * (1) The number of cols is 1024;
 * (2) layer_norm scale and bias is not null;
 * (3) linear bias is null;
 * @param
 * rows: batch_size * seq_len
 * cols: 1024
 * x_: [rows, cols], inputs
 * residual_:[rows, cols]
 * gamma_: [cols]: layernorm scale, not null
 * beta_: [cols], layernorm bias, not null
 * mask_out_: [rows, cols], dropout result
 * residual_out_: [rows, cols], residual + dropout(src)
 * y_: [rows, cols], layernorm result
 * mean_out_: [rows]: layernorm means
 * var_out_: [rows]: layernorm vars
*/
template <
    typename T, typename U, typename ScaleT = U, typename MaskType = uint8_t,
    int VecSize = 8, int WARPS_M = 4, int WARPS_N = 1, int BYTES_PER_LDG = 16,
    int ELTS_PER_ROW = 1024, int THREADS_PER_WARP = 32,
    int THREADS_PER_ROW = WARPS_N *THREADS_PER_WARP,
    int THREADS_PER_CTA = WARPS_M *THREADS_PER_ROW, int ROWS_PER_CTA = WARPS_M,
    int ELTS_PER_ROW_PER_CTA = THREADS_PER_ROW *VecSize,
    int LDGS = ELTS_PER_ROW / ELTS_PER_ROW_PER_CTA>
__global__ __launch_bounds__(THREADS_PER_CTA) void fused_ln_fwd_1024_kernel(
    int rows, int cols, uint64_t seed, const float dropout_prob,
    const bool is_upscale_in_train, const bool is_test,
    const uint64_t increment, const float epsilon, const T *__restrict__ x_ptr,
    const T *__restrict__ residual_ptr, const ScaleT *__restrict__ gamma_ptr,
    const ScaleT *__restrict__ beta_ptr, MaskType *__restrict__ mask_out_ptr,
    U *__restrict__ mean_out_ptr, U *__restrict__ var_out_ptr,
    T *__restrict__ residual_out_ptr, T *__restrict__ y_ptr) {
  using Vec = phi::AlignedVector<T, VecSize>;
  using Vec_scale = phi::AlignedVector<ScaleT, VecSize>;
  using MaskStoreT = phi::AlignedVector<MaskType, VecSize>;

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;  // 0, 1, ..., 31
  const int warp = tidx / THREADS_PER_WARP;  // 0, 1, 2, 3
  const int warp_n = warp % WARPS_N;         // 0
  const int warp_m = warp / WARPS_N;         // 0, 1, 2, 3

  const int c = warp_n * THREADS_PER_WARP + lane;  // lane
  const int r = bidx * ROWS_PER_CTA + warp_m;      // row id

  int idx = r * LN_NUM_COLS + c;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  T factor = GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  Vec_scale gamma[LDGS];
  Vec_scale beta[LDGS];
#pragma unroll
  for (int it = 0, col = c; it < LDGS; it++) {
    phi::Load<ScaleT, VecSize>(gamma_ptr + col * VecSize, &gamma[it]);
    phi::Load<ScaleT, VecSize>(beta_ptr + col * VecSize, &beta[it]);
    col += THREADS_PER_ROW;
  }

  constexpr U rn = 1.f / U(LN_NUM_COLS);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
    Vec residual[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Load<T, VecSize>(x_ptr + row * LN_NUM_COLS + col * VecSize, &x[it]);
      phi::Load<T, VecSize>(residual_ptr + row * LN_NUM_COLS + col * VecSize,
                            &residual[it]);
      col += THREADS_PER_ROW;
    }

    MaskStoreT mask_vec[LDGS];
    if (!is_test) {
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
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < VecSize; jt++) {
        // dropout(x) + residual
        x[it][jt] = x[it][jt] * static_cast<T>(mask_vec[it][jt]) * factor +
                    residual[it][jt];
        xf[it * VecSize + jt] = U(x[it][jt]);
      }
    }

// store dropout_residual_out and mask_out
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Store<T, VecSize>(
          x[it], residual_out_ptr + row * LN_NUM_COLS + col * VecSize);
      phi::Store<MaskType, VecSize>(
          mask_vec[it], mask_out_ptr + row * LN_NUM_COLS + col * VecSize);
      col += THREADS_PER_ROW;
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
    U rsigma = rsqrtf(var_local * rn + epsilon);
    if (lane == 0) {
      // Note: the stored var is different for paddle(ln) and apex (fast ln).
      // var_out_ptr[row] = rsigma;
      var_out_ptr[row] = var_local * rn;
    }

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
      }
    }

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      phi::Store<T, VecSize>(x[it], y_ptr + row * LN_NUM_COLS + col * VecSize);
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
    auto cuda_place = ctx.GetPlace();
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

  bool can_call_1024_kernel = false;
  if (cols == 1024 && scale != nullptr && layernorm_bias != nullptr &&
      bias == nullptr) {
    can_call_1024_kernel = true;
  }
  VLOG(6) << "can_call_1024_kernel = " << can_call_1024_kernel;

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
    if (can_call_1024_kernel) {
      const int WARPS_M = 4;
      const int WARPS_N = 1;
      const int THREADS_PER_WARP = 32;
      const int BYTES_PER_LDG = 16;
      const int VecSize = BYTES_PER_LDG / sizeof(T);

      const int THREADS_PER_CTA = WARPS_N * THREADS_PER_WARP * WARPS_M;
      const int ROWS_PER_CTA = WARPS_M;

      // Note: the grid can not exceed max_grid of the gpu.
      const int grid =
          static_cast<int>(std::ceil(rows / static_cast<float>(ROWS_PER_CTA)));
      fused_ln_fwd_1024_kernel<
          T, U, LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>, uint8_t,
          VecSize, WARPS_M, WARPS_N,
          BYTES_PER_LDG><<<grid, THREADS_PER_CTA, 0, ctx.stream()>>>(
          rows, cols, seed, dropout_prob, is_upscale_in_train, is_test,
          increment, epsilon, src, residual, scale, layernorm_bias, mask_data,
          mean, var, dst, layernorm_dst);
    } else {
      int blockDim = GetDesiredBlockDim(cols / VecSize);
      FusedLayernormResidualDropoutBias<
          T, uint8_t, VecSize, U,
          ScaleBiasWithSameTypeX><<<rows, blockDim, 0, ctx.stream()>>>(
          rows, cols, seed, dropout_prob, is_upscale_in_train, is_test,
          increment, epsilon, src, residual, bias, scale, layernorm_bias,
          mask_data, dst, layernorm_dst, mean, var);
    }
  }
}

template <typename T, typename U, typename MaskType,
          bool ScaleBiasWithSameTypeX = false>
void LaunchLayernormResidualDropoutGrad(
    const platform::CUDADeviceContext &dev_ctx, const uint32_t rows,
    const uint32_t cols, const float epsilon, const float dropout_prob,
    const bool is_upscale_in_train, const T *d_out, const T *layernorm_src,
    const LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *scale,
    const LayerNormParamType<T> *mean, const LayerNormParamType<T> *var,
    const MaskType *mask_data,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_scale,
    LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX> *d_layernorm_bias,
    T *d_residual, T *d_dropout_src) {
  const T zero = static_cast<T>(0.0f);
  auto factor = dropout_prob == static_cast<float>(1.0f)
                    ? zero
                    : static_cast<T>(1.0f / (1.0f - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }
  ln_bwd_1024_kernel_driver<
      T, U, LayerNormScaleBiasT<T, U, ScaleBiasWithSameTypeX>, MaskType>(
      dev_ctx, rows, cols, epsilon, layernorm_src, scale, mean, var, d_out,
      d_residual, d_scale, d_layernorm_bias, mask_data, factor, d_dropout_src);
}

}  // namespace operators
}  // namespace paddle
