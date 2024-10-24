// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES  // use M_2_SQRTPI on Windows
#endif
#include "paddle/phi/kernels/fusion/gpu/fused_bias_act_utils.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_common.h"
#include "paddle/phi/kernels/fusion/gpu/fused_residual_dropout_bias.h"
#include "paddle/phi/kernels/gpu/gelu_funcs.h"

namespace phi {
namespace fusion {
template <typename T>
struct LayerNormParamTypeGeluFunctor {
  inline __host__ __device__ T operator()(const T x) const {
    using U = phi::funcs::LayerNormParamType<T>;
    const U casted_x = static_cast<U>(x);
    const U temp = erf(casted_x * static_cast<U>(M_SQRT1_2));
    const U out = (casted_x * static_cast<U>(0.5) * (static_cast<U>(1) + temp));
    return static_cast<T>(out);
  }
};

/**
 *@brief the gelu grad functor
 */
template <typename T>
struct GeluGradFunctor {
  inline __host__ __device__ T UseOut(const T x) const {
    using U = phi::funcs::LayerNormParamType<T>;
    auto casted_x = static_cast<U>(x);

    auto first =
        static_cast<U>(0.5) *
        (static_cast<U>(1) + erf(casted_x * static_cast<U>(M_SQRT1_2)));

    auto second = static_cast<U>(0.5 * M_2_SQRTPI * M_SQRT1_2) * casted_x *
                  exp(-static_cast<U>(0.5) * casted_x * casted_x);
    return static_cast<T>((first + second));
  }
};

/**
 * @brief dst = dropout(activation(src + bias));
 * the src, mask and dst shape is (rows, cols)
 * the bias shape is (1, cols)
 */
template <typename T,
          typename MaskType,
          int VecSize,
          typename Functor,
          typename InType = T,
          typename OutType = T>
__global__ void FusedDropoutActBias(
    Functor act,
    const uint64_t seed,
    const uint64_t rows,
    const uint64_t cols,
    const int increment,
    const float dropout_prob,
    const bool is_upscale_in_train,
    const bool is_test,
    const InType *__restrict__ src,
    const T *__restrict__ bias,
    OutType *dst,
    MaskType *mask,
    const float quant_last_in_scale = 1.0,
    const float *dequant_out_scale_data = nullptr,
    const float quant_next_in_scale = 1.0,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0,
    const float quant_min_bound = -127.0) {
  int col_id = blockDim.x * blockIdx.x + threadIdx.x;
  int row_id = blockIdx.y;
  int idx = row_id * cols + col_id;

  GPURAND(StatePhilox4_32_10_t) state;
  GPURAND(_init)(seed, idx, increment, &state);

  const T factor =
      phi::fusion::GetFactor<T>(dropout_prob, is_upscale_in_train, is_test);

  for (int r = row_id; r < rows; r += blockDim.y * gridDim.y) {
    for (int i = col_id * VecSize; i < cols;
         i += blockDim.x * gridDim.x * VecSize) {
      phi::fusion::FusedResidualDropoutBiasOneThread<T,
                                                     MaskType,
                                                     VecSize,
                                                     false,
                                                     true,
                                                     Functor,
                                                     InType,
                                                     OutType>(
          r,
          i,
          cols,
          &state,
          dropout_prob,
          factor,
          src,
          nullptr,
          bias,
          dst,
          mask,
          is_test,
          nullptr,
          nullptr,
          act,
          1.0, /*Since Dropout Act bias do not use residual alpha, we set 1.0*/
          quant_last_in_scale,
          dequant_out_scale_data,
          quant_next_in_scale,
          quant_round_type,
          quant_max_bound,
          quant_min_bound);
    }
  }
}

template <typename T,
          int VecSize,
          typename Functor,
          typename InType = T,
          typename OutType = T>
__global__ void FusedActBias(Functor act,
                             const uint64_t elem_cnt,
                             const uint64_t cols,
                             const InType *__restrict__ src,
                             const T *__restrict__ bias,
                             OutType *dst,
                             const float quant_last_in_scale = 1.0,
                             const float *dequant_out_scale_data = nullptr,
                             const float quant_next_in_scale = 1.0,
                             const int quant_round_type = 1,
                             const float quant_max_bound = 127.0,
                             const float quant_min_bound = -127.0) {
  const int32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = phi::AlignedVector<T, VecSize>;
  using LoadInType = phi::AlignedVector<InType, VecSize>;
  using LoadFloat = phi::AlignedVector<float, VecSize>;
  using StoreOutType = phi::AlignedVector<OutType, VecSize>;

  LoadInType src_vec;
  LoadT bias_vec;
  StoreOutType out_vec;
  LoadFloat dequant_out_scale_vec;
  for (int32_t idx = global_thread_idx * VecSize,
               step = blockDim.x * gridDim.x * VecSize;
       idx < elem_cnt;
       idx += step) {
    const int32_t col_idx = idx % cols;
    phi::Load<InType, VecSize>(&src[idx], &src_vec);
    phi::Load<float, VecSize>(&dequant_out_scale_data[col_idx],
                              &dequant_out_scale_vec);
    if (bias) {
      phi::Load<T, VecSize>(&bias[col_idx], &bias_vec);
    }
#pragma unroll
    for (int32_t unroll_idx = 0; unroll_idx < VecSize; unroll_idx++) {
      T tmp;
      if (std::is_same<InType, int32_t>::value) {
        tmp = static_cast<T>(static_cast<float>(src_vec[unroll_idx]) *
                             dequant_out_scale_vec[unroll_idx]);
        if (bias) {
          tmp = static_cast<T>(act(tmp + bias_vec[unroll_idx]));
        } else {
          tmp = static_cast<T>(act(tmp));
        }
        out_vec[unroll_idx] = phi::funcs::quant_helper(tmp,
                                                       quant_next_in_scale,
                                                       quant_round_type,
                                                       quant_max_bound,
                                                       quant_min_bound);
      } else {
        if (bias) {
          out_vec[unroll_idx] = static_cast<OutType>(
              act(static_cast<T>(src_vec[unroll_idx]) + bias_vec[unroll_idx]));
        } else {
          out_vec[unroll_idx] =
              static_cast<OutType>(act(static_cast<T>(src_vec[unroll_idx])));
        }
      }
    }
    phi::Store<OutType, VecSize>(out_vec, &dst[idx]);
  }
}

/**
 * @brief dst = dropout(activation(src + bias));
 */
template <typename T,
          typename MaskType,
          typename Functor,
          typename InType = T,
          typename OutType = T>
void LaunchDropoutActBias(Functor act_functor,
                          const uint64_t seed,
                          const uint32_t rows,
                          const uint32_t cols,
                          const int increment,
                          const float dropout_prob,
                          const bool is_upscale_in_train,
                          const bool is_test,
                          const InType *src,
                          const T *bias,
                          OutType *dst,
                          MaskType *mask_data,
                          const phi::GPUContext &ctx,
                          const float quant_last_in_scale = 1.0,
                          const float *dequant_out_scale_data = nullptr,
                          const float quant_next_in_scale = 1.0,
                          const int quant_round_type = 1,
                          const float quant_max_bound = 127.0,
                          const float quant_min_bound = -127.0) {
  // dropout_prob == 1.0f
  if (std::abs(dropout_prob - 1.0f) < 1e-5) {
    phi::fusion::SetZero<T>(ctx, reinterpret_cast<T *>(dst), rows * cols);
    phi::fusion::SetZero<MaskType>(ctx, mask_data, rows * cols);
    return;
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  const int real_vec_size = cols % VecSize == 0 ? VecSize : 1;
  const auto config =
      phi::fusion::Get1DBlocksAnd2DGrids(ctx, rows, cols, real_vec_size);
  if (cols % VecSize == 0) {
    if (is_test) {
      const int32_t elem_cnt = rows * cols;
      const int32_t pack_num = elem_cnt / VecSize;
      const int32_t tmp_cols = cols / VecSize;
      int block_size =
          std::max(static_cast<int32_t>(32), std::min(tmp_cols, 128));
      const int grid_size = std::max(static_cast<int32_t>(1),
                                     (pack_num + block_size - 1) / block_size);
      FusedActBias<T, VecSize, Functor, InType, OutType>
          <<<grid_size, block_size, 0, ctx.stream()>>>(act_functor,
                                                       elem_cnt,
                                                       cols,
                                                       src,
                                                       bias,
                                                       dst,
                                                       quant_last_in_scale,
                                                       dequant_out_scale_data,
                                                       quant_next_in_scale);
    } else {
      FusedDropoutActBias<T, MaskType, VecSize, Functor, InType, OutType>
          <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
              act_functor,
              seed,
              rows,
              cols,
              increment,
              dropout_prob,
              is_upscale_in_train,
              is_test,
              src,
              bias,
              dst,
              mask_data,
              quant_last_in_scale,
              dequant_out_scale_data,
              quant_next_in_scale);
    }
  } else {
    FusedDropoutActBias<T, MaskType, 1, Functor, InType, OutType>
        <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
            act_functor,
            seed,
            rows,
            cols,
            increment,
            dropout_prob,
            is_upscale_in_train,
            is_test,
            src,
            bias,
            dst,
            mask_data,
            quant_last_in_scale,
            dequant_out_scale_data,
            quant_next_in_scale);
  }
}

/*
 * @brief calculate the grad of no bias
 */
template <typename T, typename MaskType, int VecSize, typename Functor>
__global__ void FusedDropoutActGrad(Functor act_grad,
                                    const T *dout,
                                    const MaskType *mask,
                                    const T *src,
                                    const T factor,
                                    const int64_t size,
                                    T *dx) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = phi::AlignedVector<T, VecSize>;
  using StoreT = phi::AlignedVector<T, VecSize>;
  using MaskLoadT = phi::AlignedVector<MaskType, VecSize>;
  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    LoadT dout_vec;
    LoadT src_vec;
    MaskLoadT mask_vec;

    phi::Load<T, VecSize>(&dout[i], &dout_vec);
    phi::Load<MaskType, VecSize>(&mask[i], &mask_vec);
    phi::Load<T, VecSize>(&src[i], &src_vec);

    StoreT dx_vec;
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      T tmp = dout_vec[ii] * static_cast<T>(mask_vec[ii]) * factor;
      dx_vec[ii] = tmp * act_grad.UseOut(src_vec[ii]);
    }
    phi::Store<T, VecSize>(dx_vec, &dx[i]);
  }
}

/**
 * blocks(128 * 8)
 * 1. calculate the dx and reduce total rows to 128 rows
 * 2. save 128*8 temporary sum in 8*128 shared memory
 * 3. reduce the sum of 128 cols data by 8*VecSize warps
 */
template <typename T,
          typename MaskType,
          int BlockSizeX,
          int BlockSizeY,
          int VecSize,
          typename Functor,
          int THREADS_PER_CTA = BlockSizeX *BlockSizeY>
__global__ __launch_bounds__(THREADS_PER_CTA) void FusedDropoutActBiasGrad(
    Functor act_grad,
    const T *dout,
    const MaskType *mask,
    const T *src,
    const T *bias,
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
      LoadT dout_vec;
      LoadT src_vec;
      LoadT bias_vec;
      MaskLoadT mask_vec;

      phi::Load<T, VecSize>(&dout[index], &dout_vec);
      phi::Load<T, VecSize>(&src[index], &src_vec);
      phi::Load<MaskType, VecSize>(&mask[index], &mask_vec);
      phi::Load<T, VecSize>(&bias[col_id * VecSize], &bias_vec);

      StoreT dx_vec;
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        T val;
        T tmp = dout_vec[i] * static_cast<T>(mask_vec[i]) * factor;
        val = tmp * act_grad.UseOut(src_vec[i] + bias_vec[i]);
        dx_vec[i] = val;
        tmp_sum[i] += val;
      }
      phi::Store<T, VecSize>(dx_vec, &dx[index]);
    }
  }

  phi::fusion::CalculateDBias<T, VecSize, BlockSizeX, BlockSizeY>(
      tmp_sum, dbias, cols);
}

/**
 * @brief to launch kernel FusedResidualDropoutBiasGradVec
 */
template <typename T, typename MaskType, typename Functor>
void LaunchDropoutActBiasGrad(Functor act_functor,
                              const T *dout,
                              const MaskType *mask,
                              const T *src,
                              const T *bias,
                              const float dropout_prob,
                              const bool is_upscale_in_train,
                              const uint32_t rows,
                              const uint32_t cols,
                              T *dx,
                              T *dbias,
                              const phi::GPUContext &ctx) {
  const T zero = static_cast<T>(0.0);
  auto factor = dropout_prob == static_cast<float>(1.0f)
                    ? zero
                    : static_cast<T>(1.0 / (1.0 - dropout_prob));
  if (!is_upscale_in_train) {
    factor = static_cast<T>(1.0f);
  }

  const int VecSize = MAX_CACHE_BYTES / sizeof(T);
  int real_vec_size = cols % VecSize == 0 ? VecSize : 1;

  if (dbias != nullptr) {
    const auto threads = 8;
    const auto blocks =
        std::max(static_cast<uint32_t>(1),
                 (cols / real_vec_size + threads - 1) / threads);
    dim3 block_dim(threads, 128, 1);
    dim3 grid_dim(blocks, 1, 1);
    if (cols % VecSize == 0) {
      FusedDropoutActBiasGrad<T, MaskType, 8, 128, VecSize, Functor>
          <<<grid_dim, block_dim, 0, ctx.stream()>>>(act_functor,
                                                     dout,
                                                     mask,
                                                     src,
                                                     bias,
                                                     factor,
                                                     rows,
                                                     cols,
                                                     dx,
                                                     dbias);
    } else {
      FusedDropoutActBiasGrad<T, MaskType, 8, 128, 1, Functor>
          <<<grid_dim, block_dim, 0, ctx.stream()>>>(act_functor,
                                                     dout,
                                                     mask,
                                                     src,
                                                     bias,
                                                     factor,
                                                     rows,
                                                     cols,
                                                     dx,
                                                     dbias);
    }
  } else {
    const uint64_t n = rows * cols;
    phi::backends::gpu::GpuLaunchConfig config =
        phi::backends::gpu::GetGpuLaunchConfig1D(ctx, n / real_vec_size);
    if (n % VecSize == 0) {
      FusedDropoutActGrad<T, MaskType, VecSize, Functor>
          <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
              act_functor, dout, mask, src, factor, n, dx);
    } else {
      FusedDropoutActGrad<T, MaskType, 1, Functor>
          <<<config.block_per_grid, config.thread_per_block, 0, ctx.stream()>>>(
              act_functor, dout, mask, src, factor, n, dx);
    }
  }
}

}  // namespace fusion
}  // namespace phi
