/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "glog/logging.h"
#include "xft/common/my_types.h"

namespace phi {
namespace fusion {
template <typename T, typename Context>
void FusedXFTRmsNormKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& norm_weight,
                           const float epsilon,
                           const int begin_norm_axis,
                           int istride,
                           int ostride,
                           DenseTensor* out) {
  const float* x_data = x.data<float>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  const T* norm_weight_data = norm_weight.data<T>();
  // x(batch_size,seq_len,hidden_size)
  int rows = 1;
  int cols = 1;
  for (int i = 0; i < begin_norm_axis; i++) {
    rows *= x.dims()[i];
  }
  for (int i = begin_norm_axis; i < x.dims().size(); i++) {
    cols *= x.dims()[i];
  }

  int size = cols;
  if (istride == -1) istride = cols;
  if (ostride == -1) ostride = cols;

#pragma omp parallel for
  for (int r = 0; r < rows; ++r) {
    const T* px = x_data + r * istride;
    T* py = out_data + r * ostride;

    float squareSum = 0;

    __m512 vsqare = _mm512_set1_ps(0);

    int col = 0;
    for (; col + 15 < size; col += 16) {
      // SUM(x*x)
      __m512 vx = _mm512_loadu_ps(px + col);
      __m512 tmp = _mm512_mul_ps(vx, vx);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }
    if (col < size) {
      __mmask16 mask = (1 << (size - col)) - 1;
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      __m512 tmp = _mm512_mul_ps(vx, vx);
      vsqare = _mm512_add_ps(vsqare, tmp);
    }

    squareSum = _mm512_reduce_add_ps(vsqare);

    // Variance
    float var = 1 / sqrt(squareSum / size + epsilon);
    __m512 vvar = _mm512_set1_ps(var);

    for (col = 0; col + 15 < size; col += 16) {
      __m512 vx = _mm512_loadu_ps(px + col);
      __m512 vw = _mm512_loadu_ps(norm_weight_data + col);
      __m512 vy = vx * vvar * vw;
      _mm512_storeu_ps(py + col, vy);
    }
    if (col < size) {
      __mmask16 mask = (1 << (size - col)) - 1;
      __m512 vx = _mm512_maskz_loadu_ps(mask, px + col);
      __m512 vw = _mm512_maskz_loadu_ps(mask, norm_weight_data + col);
      __m512 vy = vx * vvar * vw;
      _mm512_mask_storeu_ps(py + col, mask, vy);
    }
  }  // end for rows
}
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(
    xft_rms_norm, CPU, ALL_LAYOUT, phi::fusion::FusedXFTRmsNormKernel, float) {}
