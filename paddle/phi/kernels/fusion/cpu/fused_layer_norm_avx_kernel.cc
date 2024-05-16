// Copyright (c) 2024 PaddlePaddle Authors All Rights Reserved.
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

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace fusion {

template <typename T>
void ResidualBiasSumFunc(const T* x_data,
                         const T* residual_data,
                         const T* bias_data,
                         const float residual_alpha,
                         const int rows,
                         const int cols,
                         const int iStride,
                         const int oStride,
                         T* out_data) {
  __m512 vresidual_alpha = _mm512_set1_ps(residual_alpha);
  const T* pb = bias_data;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int r = 0; r < rows; ++r) {
    const T* px = x_data + r * iStride;
    const T* pr = residual_data ? residual_data + r * iStride : nullptr;
    T* py = out_data + r * oStride;
    for (int col = 0; col < cols; col += 16) {
      int remain = cols - col;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      // residual*alpha + bias + x
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      if (residual_data) {
        __m512 residual_vx = _mm512_maskz_loadu_ps(mask, pr + col);
        residual_vx = _mm512_mul_ps(residual_vx, vresidual_alpha);
        vx = _mm512_mask_add_ps(vx, mask, vx, residual_vx);
      }
      if (bias_data) {
        __m512 vb = _mm512_maskz_loadu_ps(mask, pb + col);
        vx = _mm512_mask_add_ps(vx, mask, vx, vb);
      }
      _mm512_mask_storeu_ps(py + col, mask, vx);
    }
  }
}

template <typename T>
void LayerNormFunc(const T* x_data,
                   const T* residual_data,
                   const T* bias_data,
                   const T* norm_weight_data,
                   const T* norm_bias_data,
                   const float epsilon,
                   const float residual_alpha,
                   const int rows,
                   const int cols,
                   const int iStride,
                   const int oStride,
                   T* out_data,
                   T* residual_out_data,
                   T* mean_out,
                   T* var_out) {
  auto size = cols;
  __m512 vresidual_alpha = _mm512_set1_ps(residual_alpha);
  __m512 vgamma = _mm512_set1_ps(1);
  __m512 vbeta = _mm512_set1_ps(0);
  const T* pb = bias_data;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int r = 0; r < rows; ++r) {
    const T* px = x_data + r * iStride;
    const T* pr = residual_data ? residual_data + r * iStride : nullptr;
    T* pr_out = residual_out_data ? residual_out_data + r * oStride : nullptr;
    T* py = out_data + r * oStride;

    T sum = 0;
    T squareSum = 0;

    __m512 vsum = _mm512_set1_ps(0);
    __m512 vsqare = _mm512_set1_ps(0);
    for (int col = 0; col < size; col += 16) {
      int remain = size - col;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      // SUM(x)
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      if (residual_data) {
        __m512 residual_vx = _mm512_maskz_loadu_ps(mask, pr + col);
        residual_vx = _mm512_mul_ps(residual_vx, vresidual_alpha);
        vx = _mm512_mask_add_ps(vx, mask, vx, residual_vx);
        if (bias_data) {
          __m512 vb = _mm512_maskz_loadu_ps(mask, pb + col);
          vx = _mm512_mask_add_ps(vx, mask, vx, vb);
        }
        _mm512_mask_storeu_ps(pr_out + col, mask, vx);
      }
      vsum = _mm512_add_ps(vsum, vx);

      // SUM(x*x)
      __m512 tmp = _mm512_mul_ps(vx, vx);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }

    sum = _mm512_reduce_add_ps(vsum);
    squareSum = _mm512_reduce_add_ps(vsqare);

    // Mean
    T mean = sum / size;
    mean_out[r] = mean;
    __m512 vmean = _mm512_set1_ps(mean);

    // Variance
    T var = 1 / sqrt(squareSum / size - mean * mean + epsilon);
    var_out[r] = var;
    __m512 vvar = _mm512_set1_ps(var);

    for (int col = 0; col < size; col += 16) {
      int remain = size - col;
      __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);

      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      if (residual_data) {
        __m512 residual_vx = _mm512_maskz_loadu_ps(mask, pr + col);
        residual_vx = _mm512_mul_ps(residual_vx, vresidual_alpha);
        vx = _mm512_mask_add_ps(vx, mask, vx, residual_vx);
        if (bias_data) {
          __m512 vb = _mm512_maskz_loadu_ps(mask, pb + col);
          vx = _mm512_mask_add_ps(vx, mask, vx, vb);
        }
      }
      if (norm_weight_data) {
        vgamma = _mm512_maskz_loadu_ps(mask, norm_weight_data + col);
      }
      if (norm_bias_data) {
        vbeta = _mm512_maskz_loadu_ps(mask, norm_bias_data + col);
      }
      // (vx - vmean) * vgamma * vvar + vbeta
      vx = _mm512_mask_sub_ps(vx, mask, vx, vmean);
      vx = _mm512_mask_mul_ps(vx, mask, vx, vgamma);
      vx = _mm512_mask_mul_ps(vx, mask, vx, vvar);
      __m512 vy = _mm512_mask_add_ps(vx, mask, vx, vbeta);
      _mm512_mask_storeu_ps(py + col, mask, vy);
    }
  }
}

template <typename T, typename Context>
void FusedLayerNormAvxKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const paddle::optional<DenseTensor>& bias,
                             const paddle::optional<DenseTensor>& residual,
                             const paddle::optional<DenseTensor>& norm_weight,
                             const paddle::optional<DenseTensor>& norm_bias,
                             const float epsilon,
                             const float residual_alpha,
                             const int begin_norm_axis,
                             const float quant_scale,
                             const int quant_round_type,
                             const float quant_max_bound,
                             const float quant_min_bound,
                             DenseTensor* out,
                             DenseTensor* residual_out,
                             DenseTensor* mean,
                             DenseTensor* variance) {
  if (quant_scale > 0.0f) {
    PD_THROW("NOT supported quant int8. ");
  }
  const auto x_dims = x.dims();
  auto matrix_dim = common::flatten_to_2d(x_dims, begin_norm_axis);
  T* out_data = dev_ctx.template Alloc<T>(out);
  T* mean_out = dev_ctx.template Alloc<T>(mean);
  T* var_out = dev_ctx.template Alloc<T>(variance);

  const T* x_data = x.data<T>();
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  const T* residual_data = residual ? residual.get().data<T>() : nullptr;
  const T* norm_weight_data =
      norm_weight ? norm_weight.get().data<T>() : nullptr;
  const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;
  T* residual_out_data =
      residual ? dev_ctx.template Alloc<T>(residual_out) : nullptr;

  int32_t rows = static_cast<int32_t>(matrix_dim[0]);
  int32_t cols = static_cast<int32_t>(matrix_dim[1]);

  auto iStride = cols;
  auto oStride = cols;
  if (!norm_weight && !norm_bias_data) {
    ResidualBiasSumFunc(x_data,
                        residual_data,
                        bias_data,
                        residual_alpha,
                        rows,
                        cols,
                        iStride,
                        oStride,
                        out_data);
  } else {
    LayerNormFunc(x_data,
                  residual_data,
                  bias_data,
                  norm_weight_data,
                  norm_bias_data,
                  epsilon,
                  residual_alpha,
                  rows,
                  cols,
                  iStride,
                  oStride,
                  out_data,
                  residual_out_data,
                  mean_out,
                  var_out);
  }
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_bias_residual_layernorm,
                   CPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedLayerNormAvxKernel,
                   float) {}
