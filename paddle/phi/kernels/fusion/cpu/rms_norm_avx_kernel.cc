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

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void RmsNormAvxKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const paddle::optional<DenseTensor>& bias,
                      const paddle::optional<DenseTensor>& residual,
                      const DenseTensor& norm_weight,
                      const paddle::optional<DenseTensor>& norm_bias,
                      const float epsilon,
                      const int begin_norm_axis,
                      const float quant_scale,
                      const int quant_round_type,
                      const float quant_max_bound,
                      const float quant_min_bound,
                      DenseTensor* out,
                      DenseTensor* residual_out,
                      DenseTensor* inv_var) {
  if (quant_scale > 0.0f) {
    PD_THROW("NOT supported quant int8. ");
  }

  const T* x_data = x.data<T>();
  int32_t rows = 1;
  int32_t cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }
  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  int size = cols;
  auto istride = cols;
  auto ostride = cols;
  const T* norm_weight_data = norm_weight.data<T>();
  const T* norm_bias_data = norm_bias ? norm_bias.get().data<T>() : nullptr;
  const T* residual_data = residual ? residual.get().data<T>() : nullptr;
  const T* bias_data = bias ? bias.get().data<T>() : nullptr;
  T* out_data = dev_ctx.template Alloc<T>(out);
  T* residual_out_data =
      residual ? dev_ctx.template Alloc<T>(residual_out) : nullptr;

  __m512 vb = _mm512_setzero_ps();
  const T* pb = bias_data;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int r = 0; r < rows; ++r) {
    const T* px = x_data + r * istride;
    const T* pr = residual ? residual_data + r * istride : nullptr;
    T* pr_out = residual ? residual_out_data + r * ostride : nullptr;
    T* py = out_data + r * ostride;

    T squareSum = 0;

    __m512 vsqare = _mm512_set1_ps(0);

    int col = 0;
    for (; col + 15 < size; col += 16) {
      // SUM(x*x)
      __m512 vx = _mm512_loadu_ps(px + col);
      if (residual) {
        __m512 residual_vx = _mm512_loadu_ps(pr + col);
        vx = _mm512_add_ps(vx, residual_vx);
        if (bias) {
          __m512 vb = _mm512_loadu_ps(pb + col);
          vx = _mm512_add_ps(vx, vb);
        }
        _mm512_storeu_ps(pr_out + col, vx);
      }
      __m512 tmp = _mm512_mul_ps(vx, vx);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }
    if (col < size) {
      __mmask16 mask = (1 << (size - col)) - 1;
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      if (residual) {
        __m512 residual_vx = _mm512_maskz_loadu_ps(mask, pr + col);
        vx = _mm512_mask_add_ps(vx, mask, vx, residual_vx);
        if (bias) {
          __m512 vb = _mm512_maskz_loadu_ps(mask, pb + col);
          vx = _mm512_mask_add_ps(vx, mask, vx, vb);
        }
        _mm512_mask_storeu_ps(pr_out + col, mask, vx);
      }
      __m512 tmp = _mm512_mul_ps(vx, vx);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }

    squareSum = _mm512_reduce_add_ps(vsqare);

    // Variance
    T var = 1 / sqrt(squareSum / size + epsilon);
    __m512 vvar = _mm512_set1_ps(var);

    for (col = 0; col + 15 < size; col += 16) {
      __m512 vx = _mm512_loadu_ps(px + col);
      if (residual) {
        __m512 residual_vx = _mm512_loadu_ps(pr + col);
        vx = _mm512_add_ps(vx, residual_vx);
        if (bias) {
          __m512 vb = _mm512_loadu_ps(pb + col);
          vx = _mm512_add_ps(vx, vb);
        }
      }
      __m512 vw = _mm512_loadu_ps(norm_weight_data + col);
      if (norm_bias_data) {
        vb = _mm512_loadu_ps(norm_bias_data + col);
      }

      // vy = vx * vvar * vw + vb
      vx = _mm512_mul_ps(vx, vvar);
      vx = _mm512_mul_ps(vx, vw);
      __m512 vy = _mm512_add_ps(vx, vb);
      _mm512_storeu_ps(py + col, vy);
    }
    if (col < size) {
      __mmask16 mask = (1 << (size - col)) - 1;
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      if (residual) {
        __m512 residual_vx = _mm512_maskz_loadu_ps(mask, pr + col);
        vx = _mm512_mask_add_ps(vx, mask, vx, residual_vx);
        if (bias) {
          __m512 vb = _mm512_maskz_loadu_ps(mask, pb + col);
          vx = _mm512_mask_add_ps(vx, mask, vx, vb);
        }
      }
      __m512 vw = _mm512_maskz_loadu_ps(mask, norm_weight_data + col);
      if (norm_bias_data) {
        vb = _mm512_maskz_loadu_ps(mask, norm_bias_data + col);
      }
      // vx * vvar * vw + vb
      vx = _mm512_mask_mul_ps(vx, mask, vx, vvar);
      vx = _mm512_mask_mul_ps(vx, mask, vx, vw);
      __m512 vy = _mm512_mask_add_ps(vx, mask, vx, vb);
      _mm512_mask_storeu_ps(py + col, mask, vy);
    }
  }  // end for rows
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    rms_norm, CPU, ALL_LAYOUT, phi::fusion::RmsNormAvxKernel, float) {}
