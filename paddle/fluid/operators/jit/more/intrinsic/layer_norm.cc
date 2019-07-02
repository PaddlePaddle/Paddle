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
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace intrinsic {

void LayerNorm_impl_avx(float* x, float* out, float* mean, float* var,
                        const float* scale, const float* bias, int height,
                        const float epsilon, int right) {
#ifdef __AVX__
  __m256 sum;
  __m256 mean_vec, var_vec;
  __m128 hi, lo;
  __m256 tmp;
  size_t offset;
  size_t j;
  int block = YMM_FLOAT_BLOCK;
  const int rest = right % block;
  const int end = right - rest;

  __m256 reverse_num_vec =
      _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(right));
  __m256 epsilon_vec = _mm256_set1_ps(epsilon);
  int rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
  __m256i mask_vec = _mm256_set_epi32(
      rest_mask & 0x80 ? 0xffffffff : 0, rest_mask & 0x40 ? 0xffffffff : 0,
      rest_mask & 0x20 ? 0xffffffff : 0, rest_mask & 0x10 ? 0xffffffff : 0,
      rest_mask & 0x8 ? 0xffffffff : 0, rest_mask & 0x4 ? 0xffffffff : 0,
      rest_mask & 0x2 ? 0xffffffff : 0, rest_mask & 0x1 ? 0xffffffff : 0);

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
      tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                             *(__m256*)&mask_vec);  // NOLINT
      sum = _mm256_add_ps(sum, tmp);
    }
    hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm256_extractf128_ps(sum, 0);
    sum = _mm256_add_ps(
        sum, _mm256_insertf128_ps(
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
      tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                             *(__m256*)&mask_vec);  // NOLINT
      sum = _mm256_add_ps(sum, tmp);
    }
    hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm256_extractf128_ps(sum, 0);
    sum = _mm256_add_ps(
        sum, _mm256_insertf128_ps(
                 _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    var_vec = _mm256_mul_ps(sum, reverse_num_vec);
    var[i] = *reinterpret_cast<float*>(&var_vec);

    /* get x_norm and calculate output*/
    for (j = offset; j < end + offset; j += block) {
      tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm256_div_ps(tmp,
                          _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
      _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm256_div_ps(tmp,
                          _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
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
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j,
                         _mm256_add_ps(tmp, _mm256_loadu_ps((const float*)bias +
                                                            j - offset)));
      }
    }
  }
#endif
}

void LayerNorm_impl_avx512f(float* x, float* out, float* mean, float* var,
                            const float* scale, const float* bias, int height,
                            const float epsilon, int right) {
#ifdef __AVX512F__
  __m512 sum;
  __m512 mean_vec, var_vec;
  __m512 hi, lo;
  __m512 tmp;
  size_t offset;
  size_t j;
  int block = ZMM_FLOAT_BLOCK;
  const int rest = right % block;
  const int end = right - rest;

  __m512 reverse_num_vec =
      _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_set1_ps(right));
  __m512 epsilon_vec = _mm512_set1_ps(epsilon);
  __mmask16 rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 16 - (block - rest))))) & 0x0ffff;

  for (int i = 0; i < height; ++i) {
    offset = i * right;

    /* get mean */
    sum = _mm512_setzero_ps();
    for (j = offset; j < end + offset; j += block) {
      sum = _mm512_add_ps(sum, _mm512_loadu_ps((const float*)x + j));
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = _mm512_loadu_ps((const float*)x + j);
      tmp = _mm512_mask_blend_ps(rest_mask, _mm512_setzero_ps(), tmp);
      sum = _mm512_add_ps(sum, tmp);
    }
    hi = _mm512_maskz_compress_ps(0xaaaa, sum);
    lo = _mm512_maskz_compress_ps(0x5555, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0xaa, sum);
    lo = _mm512_maskz_compress_ps(0x55, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0xa, sum);
    lo = _mm512_maskz_compress_ps(0x5, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0x2, sum);
    lo = _mm512_maskz_compress_ps(0x1, sum);
    sum = _mm512_add_ps(hi, lo);
    sum = _mm512_broadcastss_ps(_mm512_extractf32x4_ps(sum, 0));
    mean_vec = _mm512_mul_ps(sum, reverse_num_vec);
    mean[i] = *reinterpret_cast<float*>(&mean_vec);

    /* get variance */
    sum = _mm512_setzero_ps();
    for (j = offset; j < end + offset; j += block) {
      tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm512_mul_ps(tmp, tmp);
      sum = _mm512_add_ps(sum, tmp);
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm512_mul_ps(tmp, tmp);
      tmp = _mm512_mask_blend_ps(rest_mask, _mm512_setzero_ps(), tmp);
      sum = _mm512_add_ps(sum, tmp);
    }
    hi = _mm512_maskz_compress_ps(0xaaaa, sum);
    lo = _mm512_maskz_compress_ps(0x5555, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0xaa, sum);
    lo = _mm512_maskz_compress_ps(0x55, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0xa, sum);
    lo = _mm512_maskz_compress_ps(0x5, sum);
    sum = _mm512_add_ps(hi, lo);
    hi = _mm512_maskz_compress_ps(0x2, sum);
    lo = _mm512_maskz_compress_ps(0x1, sum);
    sum = _mm512_add_ps(hi, lo);
    sum = _mm512_broadcastss_ps(_mm512_extractf32x4_ps(sum, 0));
    var_vec = _mm512_mul_ps(sum, reverse_num_vec);
    var[i] = *reinterpret_cast<float*>(&var_vec);

    /* get x_norm and calculate output*/
    for (j = offset; j < end + offset; j += block) {
      tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm512_div_ps(tmp,
                          _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
      _mm512_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
    }
    if (rest != 0) {
      j = offset + right - block;
      tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
      tmp = _mm512_div_ps(tmp,
                          _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
      _mm512_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
    }

    if (scale) {
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_loadu_ps((const float*)out + j);
      }
      for (j = offset; j < end + offset; j += block) {
        _mm512_storeu_ps(
            reinterpret_cast<float*>(out) + j,
            _mm512_mul_ps(_mm512_loadu_ps((const float*)out + j),
                          _mm512_loadu_ps((const float*)scale + j - offset)));
      }
      if (rest != 0) {
        j = offset + right - block;
        _mm512_storeu_ps(
            reinterpret_cast<float*>(out) + j,
            _mm512_mul_ps(tmp,
                          _mm512_loadu_ps((const float*)scale + j - offset)));
      }
    }

    if (bias) {
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_loadu_ps((const float*)out + j);
      }
      for (j = offset; j < end + offset; j += block) {
        _mm512_storeu_ps(
            reinterpret_cast<float*>(out) + j,
            _mm512_add_ps(_mm512_loadu_ps((const float*)out + j),
                          _mm512_loadu_ps((const float*)bias + j - offset)));
      }
      if (rest != 0) {
        j = offset + right - block;
        _mm512_storeu_ps(reinterpret_cast<float*>(out) + j,
                         _mm512_add_ps(tmp, _mm512_loadu_ps((const float*)bias +
                                                            j - offset)));
      }
    }
  }
#else
  LayerNorm_impl_avx(x, out, mean, var, scale, bias, height, epsilon, right);
#endif
}

void LayerNorm(float* x, float* out, float* mean, float* var,
               const float* scale, const float* bias, int height,
               const float epsilon, int right) {
  if (platform::MayIUse(platform::avx512f) && right >= ZMM_FLOAT_BLOCK) {
    LayerNorm_impl_avx512f(x, out, mean, var, scale, bias, height, epsilon,
                           right);
  } else {
    LayerNorm_impl_avx(x, out, mean, var, scale, bias, height, epsilon, right);
  }
}

bool LayerNormKernel::CanBeUsed(const int& d) const {
#ifdef __AVX__
  return platform::MayIUse(platform::avx) && d >= YMM_FLOAT_BLOCK;
#else
  return false;
#endif
}

void LayerNormGrad_impl_avx(const float* d_y, float* d_x, const float* x,
                            const float* mean, const float* var,
                            const float* scale, float* d_scale, float* d_bias,
                            float* temp, float* temp_norm, int height,
                            const float epsilon, int right) {
#ifdef __AVX__
  size_t offset;
  size_t j;
  int block = YMM_FLOAT_BLOCK;

  __m256 sum;
  __m256 mean_vec, var_vec;
  __m128 hi, lo;
  __m256 tmp, last_vec;

  const int rest = right % block;
  const int end = right - rest;

  __m256 reverse_num_vec =
      _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(right));
  __m256 epsilon_vec = _mm256_set1_ps(epsilon);
  int rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
  __m256i mask_vec = _mm256_set_epi32(
      rest_mask & 0x80 ? 0xffffffff : 0, rest_mask & 0x40 ? 0xffffffff : 0,
      rest_mask & 0x20 ? 0xffffffff : 0, rest_mask & 0x10 ? 0xffffffff : 0,
      rest_mask & 0x8 ? 0xffffffff : 0, rest_mask & 0x4 ? 0xffffffff : 0,
      rest_mask & 0x2 ? 0xffffffff : 0, rest_mask & 0x1 ? 0xffffffff : 0);

  for (int i = 0; i < height; ++i) {
    offset = i * right;
    if (d_scale || d_x) {
      // get x_norm
      mean_vec = _mm256_set1_ps(mean[i]);
      var_vec = _mm256_set1_ps(var[i]);

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(temp_norm) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(temp_norm) + j, tmp);
      }
    }

    if (d_bias) {
      if (rest != 0) {
        j = offset + right - block;
        if (i != 0) {
          last_vec =
              _mm256_loadu_ps(reinterpret_cast<float*>(d_bias) + j - offset);
        }
      }
      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_loadu_ps((const float*)d_y + j);
        if (i != 0) {
          tmp = _mm256_add_ps(
              _mm256_loadu_ps(reinterpret_cast<float*>(d_bias) + j - offset),
              tmp);
        }
        _mm256_storeu_ps(reinterpret_cast<float*>(d_bias) + j - offset, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_loadu_ps((const float*)d_y + j);
        if (i != 0) {
          tmp = _mm256_add_ps(last_vec, tmp);
        }
        _mm256_storeu_ps(reinterpret_cast<float*>(d_bias) + j - offset, tmp);
      }
    }

    if (d_scale) {
      if (rest != 0) {
        j = offset + right - block;
        if (i != 0) {
          last_vec =
              _mm256_loadu_ps(reinterpret_cast<float*>(d_scale) + j - offset);
        }
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_mul_ps(_mm256_loadu_ps((const float*)d_y + j),
                            _mm256_loadu_ps((const float*)temp_norm + j));
        if (i != 0) {
          tmp = _mm256_add_ps(
              _mm256_loadu_ps(reinterpret_cast<float*>(d_scale) + j - offset),
              tmp);
        }
        _mm256_storeu_ps(reinterpret_cast<float*>(d_scale) + j - offset, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_mul_ps(_mm256_loadu_ps((const float*)d_y + j),
                            _mm256_loadu_ps((const float*)temp_norm + j));
        if (i != 0) {
          tmp = _mm256_add_ps(last_vec, tmp);
        }
        _mm256_storeu_ps(reinterpret_cast<float*>(d_scale) + j - offset, tmp);
      }
    }

    if (d_x) {
      if (d_scale) {
        // dy_dx
        for (j = offset; j < end + offset; j += block) {
          tmp =
              _mm256_mul_ps(_mm256_loadu_ps((const float*)d_y + j),
                            _mm256_loadu_ps((const float*)scale + j - offset));
          _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
        }
        if (rest != 0) {
          j = offset + right - block;
          tmp =
              _mm256_mul_ps(_mm256_loadu_ps((const float*)d_y + j),
                            _mm256_loadu_ps((const float*)scale + j - offset));
          _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
        }
      }

      // dy_dmean_dx && dy_var_dx
      sum = _mm256_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)d_x + j));
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_loadu_ps((const float*)d_x + j);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
      }
      hi = _mm256_extractf128_ps(sum, 1);
      lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum, _mm256_insertf128_ps(
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      mean_vec = _mm256_mul_ps(sum, reverse_num_vec);

      // dy_dmean_dx && dy_var_dx
      if (rest != 0) {
        j = offset + right - block;
        last_vec = _mm256_loadu_ps(reinterpret_cast<float*>(d_x) + j);
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_mul_ps(_mm256_loadu_ps((const float*)d_x + j),
                            _mm256_loadu_ps((const float*)temp_norm + j));
        _mm256_storeu_ps(reinterpret_cast<float*>(temp) + j, tmp);
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)d_x + j), mean_vec);
        _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_mul_ps(last_vec,
                            _mm256_loadu_ps((const float*)temp_norm + j));
        _mm256_storeu_ps(reinterpret_cast<float*>(temp) + j, tmp);
        tmp = _mm256_sub_ps(last_vec, mean_vec);
        _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      // dy_var_dx
      sum = _mm256_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)temp + j));
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_loadu_ps((const float*)temp + j);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
      }
      hi = _mm256_extractf128_ps(sum, 1);
      lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum, _mm256_insertf128_ps(
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      mean_vec = _mm256_mul_ps(sum, reverse_num_vec);

      var_vec = _mm256_set1_ps(var[i]);

      // dy_dmean_dx && dy_var_dx
      if (rest != 0) {
        j = offset + right - block;
        last_vec = _mm256_loadu_ps(reinterpret_cast<float*>(d_x) + j);
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm256_mul_ps(_mm256_loadu_ps((const float*)temp_norm + j),
                            mean_vec);
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)d_x + j), tmp);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm256_mul_ps(_mm256_loadu_ps((const float*)temp_norm + j),
                            mean_vec);
        tmp = _mm256_sub_ps(last_vec, tmp);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
    }
  }
#endif
}

void LayerNormGrad_impl_avx512f(const float* d_y, float* d_x, const float* x,
                                const float* mean, const float* var,
                                const float* scale, float* d_scale,
                                float* d_bias, float* temp, float* temp_norm,
                                int height, const float epsilon, int right) {
#ifdef __AVX512F__
  size_t offset;
  size_t j;
  int block = ZMM_FLOAT_BLOCK;

  __m512 sum;
  __m512 mean_vec, var_vec;
  __m512 hi, lo;
  __m512 tmp, last_vec;

  const int rest = right % block;
  const int end = right - rest;

  __m512 reverse_num_vec =
      _mm512_div_ps(_mm512_set1_ps(1.0), _mm512_set1_ps(right));
  __m512 epsilon_vec = _mm512_set1_ps(epsilon);
  __mmask16 rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 16 - (block - rest))))) & 0x0ffff;

  for (int i = 0; i < height; ++i) {
    offset = i * right;
    if (d_scale || d_x) {
      // get x_norm
      mean_vec = _mm512_set1_ps(mean[i]);
      var_vec = _mm512_set1_ps(var[i]);

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm512_div_ps(
            tmp, _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
        _mm512_storeu_ps(reinterpret_cast<float*>(temp_norm) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm512_div_ps(
            tmp, _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
        _mm512_storeu_ps(reinterpret_cast<float*>(temp_norm) + j, tmp);
      }
    }

    if (d_bias) {
      if (rest != 0) {
        j = offset + right - block;
        if (i != 0) {
          last_vec =
              _mm512_loadu_ps(reinterpret_cast<float*>(d_bias) + j - offset);
        }
      }
      for (j = offset; j < end + offset; j += block) {
        tmp = _mm512_loadu_ps((const float*)d_y + j);
        if (i != 0) {
          tmp = _mm512_add_ps(
              _mm512_loadu_ps(reinterpret_cast<float*>(d_bias) + j - offset),
              tmp);
        }
        _mm512_storeu_ps(reinterpret_cast<float*>(d_bias) + j - offset, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_loadu_ps((const float*)d_y + j);
        if (i != 0) {
          tmp = _mm512_add_ps(last_vec, tmp);
        }
        _mm512_storeu_ps(reinterpret_cast<float*>(d_bias) + j - offset, tmp);
      }
    }

    if (d_scale) {
      if (rest != 0) {
        j = offset + right - block;
        if (i != 0) {
          last_vec =
              _mm512_loadu_ps(reinterpret_cast<float*>(d_scale) + j - offset);
        }
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm512_mul_ps(_mm512_loadu_ps((const float*)d_y + j),
                            _mm512_loadu_ps((const float*)temp_norm + j));
        if (i != 0) {
          tmp = _mm512_add_ps(
              _mm512_loadu_ps(reinterpret_cast<float*>(d_scale) + j - offset),
              tmp);
        }
        _mm512_storeu_ps(reinterpret_cast<float*>(d_scale) + j - offset, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_mul_ps(_mm512_loadu_ps((const float*)d_y + j),
                            _mm512_loadu_ps((const float*)temp_norm + j));
        if (i != 0) {
          tmp = _mm512_add_ps(last_vec, tmp);
        }
        _mm512_storeu_ps(reinterpret_cast<float*>(d_scale) + j - offset, tmp);
      }
    }

    if (d_x) {
      if (d_scale) {
        // dy_dx
        for (j = offset; j < end + offset; j += block) {
          tmp =
              _mm512_mul_ps(_mm512_loadu_ps((const float*)d_y + j),
                            _mm512_loadu_ps((const float*)scale + j - offset));
          _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
        }
        if (rest != 0) {
          j = offset + right - block;
          tmp =
              _mm512_mul_ps(_mm512_loadu_ps((const float*)d_y + j),
                            _mm512_loadu_ps((const float*)scale + j - offset));
          _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
        }
      }

      // dy_dmean_dx && dy_var_dx
      sum = _mm512_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        sum = _mm512_add_ps(sum, _mm512_loadu_ps((const float*)d_x + j));
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_loadu_ps((const float*)d_x + j);
        tmp = _mm512_mask_blend_ps(rest_mask, _mm512_setzero_ps(), tmp);
        sum = _mm512_add_ps(sum, tmp);
      }
      hi = _mm512_maskz_compress_ps(0xaaaa, sum);
      lo = _mm512_maskz_compress_ps(0x5555, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0xaa, sum);
      lo = _mm512_maskz_compress_ps(0x55, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0xa, sum);
      lo = _mm512_maskz_compress_ps(0x5, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0x2, sum);
      lo = _mm512_maskz_compress_ps(0x1, sum);
      sum = _mm512_add_ps(hi, lo);
      sum = _mm512_broadcastss_ps(_mm512_extractf32x4_ps(sum, 0));
      mean_vec = _mm512_mul_ps(sum, reverse_num_vec);

      // dy_dmean_dx && dy_var_dx
      if (rest != 0) {
        j = offset + right - block;
        last_vec = _mm512_loadu_ps(reinterpret_cast<float*>(d_x) + j);
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm512_mul_ps(_mm512_loadu_ps((const float*)d_x + j),
                            _mm512_loadu_ps((const float*)temp_norm + j));
        _mm512_storeu_ps(reinterpret_cast<float*>(temp) + j, tmp);
        tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)d_x + j), mean_vec);
        _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_mul_ps(last_vec,
                            _mm512_loadu_ps((const float*)temp_norm + j));
        _mm512_storeu_ps(reinterpret_cast<float*>(temp) + j, tmp);
        tmp = _mm512_sub_ps(last_vec, mean_vec);
        _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      // dy_var_dx
      sum = _mm512_setzero_ps();
      for (j = offset; j < end + offset; j += block) {
        sum = _mm512_add_ps(sum, _mm512_loadu_ps((const float*)temp + j));
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_loadu_ps((const float*)temp + j);
        tmp = _mm512_mask_blend_ps(rest_mask, _mm512_setzero_ps(), tmp);
        sum = _mm512_add_ps(sum, tmp);
      }
      hi = _mm512_maskz_compress_ps(0xaaaa, sum);
      lo = _mm512_maskz_compress_ps(0x5555, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0xaa, sum);
      lo = _mm512_maskz_compress_ps(0x55, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0xa, sum);
      lo = _mm512_maskz_compress_ps(0x5, sum);
      sum = _mm512_add_ps(hi, lo);
      hi = _mm512_maskz_compress_ps(0x2, sum);
      lo = _mm512_maskz_compress_ps(0x1, sum);
      sum = _mm512_add_ps(hi, lo);
      sum = _mm512_broadcastss_ps(_mm512_extractf32x4_ps(sum, 0));
      mean_vec = _mm512_mul_ps(sum, reverse_num_vec);

      var_vec = _mm512_set1_ps(var[i]);

      // dy_dmean_dx && dy_var_dx
      if (rest != 0) {
        j = offset + right - block;
        last_vec = _mm512_loadu_ps(reinterpret_cast<float*>(d_x) + j);
      }

      for (j = offset; j < end + offset; j += block) {
        tmp = _mm512_mul_ps(_mm512_loadu_ps((const float*)temp_norm + j),
                            mean_vec);
        tmp = _mm512_sub_ps(_mm512_loadu_ps((const float*)d_x + j), tmp);
        tmp = _mm512_div_ps(
            tmp, _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
        _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
      if (rest != 0) {
        j = offset + right - block;
        tmp = _mm512_mul_ps(_mm512_loadu_ps((const float*)temp_norm + j),
                            mean_vec);
        tmp = _mm512_sub_ps(last_vec, tmp);
        tmp = _mm512_div_ps(
            tmp, _mm512_sqrt_ps(_mm512_add_ps(var_vec, epsilon_vec)));
        _mm512_storeu_ps(reinterpret_cast<float*>(d_x) + j, tmp);
      }
    }
  }
#else
  LayerNormGrad_impl_avx(d_y, d_x, x, mean, var, scale, d_scale, d_bias, temp,
                         temp_norm, height, epsilon, right);
#endif
}

void LayerNormGrad(const float* d_y, float* d_x, const float* x,
                   const float* mean, const float* var, const float* scale,
                   float* d_scale, float* d_bias, float* temp, float* temp_norm,
                   int height, const float epsilon, int right) {
  if (platform::MayIUse(platform::avx512f) && right >= ZMM_FLOAT_BLOCK) {
    LayerNormGrad_impl_avx512f(d_y, d_x, x, mean, var, scale, d_scale, d_bias,
                               temp, temp_norm, height, epsilon, right);
  } else {
    LayerNormGrad_impl_avx(d_y, d_x, x, mean, var, scale, d_scale, d_bias, temp,
                           temp_norm, height, epsilon, right);
  }
}

bool LayerNormGradKernel::CanBeUsed(const int& d) const {
#ifdef __AVX__
  return platform::MayIUse(platform::avx) && d >= YMM_FLOAT_BLOCK;
#else
  return false;
#endif
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace intrinsic = paddle::operators::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kLayerNorm, intrinsic, intrinsic::LayerNormKernel);
REGISTER_JITKERNEL_MORE(kLayerNormGrad, intrinsic,
                        intrinsic::LayerNormGradKernel);
