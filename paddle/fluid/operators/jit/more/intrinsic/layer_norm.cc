/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/more/intrinsic/layer_norm.h"

#include <limits>

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace intrinsic {

void LayerNorm(float* x,
               float* out,
               float* mean,
               float* var,
               const float* scale,
               const float* bias,
               int height,
               const float epsilon,
               int right) {
  int block = YMM_FLOAT_BLOCK;
  const int rest = right % block;
  const int end = right - rest;
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel
  {
#endif
    __m256 sum;
    __m256 mean_vec, var_vec;
    __m128 hi, lo;
    __m256 tmp;
    size_t offset;
    size_t j;
    __m256 reverse_num_vec =
        _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(right));
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    int rest_mask =
        ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
    __m256i mask_vec = _mm256_set_epi32(rest_mask & 0x80 ? 0xffffffff : 0,
                                        rest_mask & 0x40 ? 0xffffffff : 0,
                                        rest_mask & 0x20 ? 0xffffffff : 0,
                                        rest_mask & 0x10 ? 0xffffffff : 0,
                                        rest_mask & 0x8 ? 0xffffffff : 0,
                                        rest_mask & 0x4 ? 0xffffffff : 0,
                                        rest_mask & 0x2 ? 0xffffffff : 0,
                                        rest_mask & 0x1 ? 0xffffffff : 0);

#ifdef PADDLE_WITH_MKLML
#pragma omp for
#endif
    for (int i = 0; i < height; ++i) {
      offset = i * right;

      /* get mean */
      sum = _mm256_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)x + j));
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_loadu_ps((const float*)x + j);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(),
                               tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
      }
      hi = _mm256_extractf128_ps(sum, 1);
      lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum,
          _mm256_insertf128_ps(
              _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      mean_vec = _mm256_mul_ps(sum, reverse_num_vec);
      mean[i] = *reinterpret_cast<float*>(&mean_vec);

      /* get variance */
      sum = _mm256_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_mul_ps(tmp, tmp);
        sum = _mm256_add_ps(sum, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_mul_ps(tmp, tmp);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(),
                               tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
      }
      hi = _mm256_extractf128_ps(sum, 1);
      lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum,
          _mm256_insertf128_ps(
              _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      var_vec = _mm256_mul_ps(sum, reverse_num_vec);
      var[i] = *reinterpret_cast<float*>(&var_vec);

      /* get x_norm and calculate output*/
      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
      }

      if (scale) {
        if (rest != 0) {
          j = offset + right - block;
          tmp = _mm256_loadu_ps((const float*)out + j);
        }
        for (j = offset; j < end + offset; j += block) {
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_mul_ps(_mm256_loadu_ps((const float*)out + j),
                            _mm256_loadu_ps((const float*)scale + j - offset)));
        }
        if (rest != 0) {
          j = offset + right - block;
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_mul_ps(tmp,
                            _mm256_loadu_ps((const float*)scale + j - offset)));
        }
      }

      if (bias) {
        if (rest != 0) {
          j = offset + right - block;
          tmp = _mm256_loadu_ps((const float*)out + j);
        }
        for (j = offset; j < end + offset; j += block) {
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_add_ps(_mm256_loadu_ps((const float*)out + j),
                            _mm256_loadu_ps((const float*)bias + j - offset)));
        }
        if (rest != 0) {
          j = offset + right - block;
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_add_ps(tmp,
                            _mm256_loadu_ps((const float*)bias + j - offset)));
        }
      }
    }
#ifdef PADDLE_WITH_MKLML
  }
#endif
}

bool LayerNormKernel::CanBeUsed(const int& d) const {
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) &&
         d >= YMM_FLOAT_BLOCK;
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace intrinsic = paddle::operators::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kLayerNorm, intrinsic, intrinsic::LayerNormKernel);
