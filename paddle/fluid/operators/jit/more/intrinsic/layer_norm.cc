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

void LayerNorm(float* x, float* out, float* mean, float* var,
               const float* scale, const float* bias, int height,
               const float epsilon, int right) {
  constexpr int block = YMM_FLOAT_BLOCK;
  const int rest = right % block;
  const int num = right / block;

  __m256 epsilon_vec = _mm256_set1_ps(epsilon);
  __m256 reverse_num_vec =
      _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(right));
  int rest_mask =
      ((-1) & (~((~0U) >> (sizeof(int) * 8 - (block - rest))))) & 0x0ff;
  __m256i mask_vec = _mm256_set_epi32(
      rest_mask & 0x80 ? 0xffffffff : 0, rest_mask & 0x40 ? 0xffffffff : 0,
      rest_mask & 0x20 ? 0xffffffff : 0, rest_mask & 0x10 ? 0xffffffff : 0,
      rest_mask & 0x8 ? 0xffffffff : 0, rest_mask & 0x4 ? 0xffffffff : 0,
      rest_mask & 0x2 ? 0xffffffff : 0, rest_mask & 0x1 ? 0xffffffff : 0);

  void* buf = NULL;
  size_t mean_buf_len = block * height;
  size_t var_buf_len = mean_buf_len;
  size_t rest_buf_len = mean_buf_len;

  buf = malloc(sizeof(float) * (mean_buf_len + var_buf_len + rest_buf_len));
  if (buf != NULL) {
    float* rest_buf = NULL;
    float* mean_buf = NULL;
    float* var_buf = NULL;
    mean_buf = reinterpret_cast<float*>(buf);
    var_buf = mean_buf + mean_buf_len;
    rest_buf = var_buf + var_buf_len;

    memset(mean_buf, 0, mean_buf_len * sizeof(float));
    memset(var_buf, 0, var_buf_len * sizeof(float));

/* get mean */
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < num; j++) {
        __m256 sum = _mm256_loadu_ps((const float*)mean_buf + i * block);
        sum = _mm256_add_ps(
            sum, _mm256_loadu_ps((const float*)x + i * right + j * block));
        _mm256_storeu_ps(reinterpret_cast<float*>(mean_buf) + i * block, sum);
      }
    }

    if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < height; ++i) {
        int j = (i + 1) * right - block;
        __m256 sum = _mm256_loadu_ps((const float*)mean_buf + i * block);
        __m256 tmp = _mm256_loadu_ps((const float*)x + j);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
        _mm256_storeu_ps(reinterpret_cast<float*>(mean_buf) + i * block, sum);
      }
    }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < height; ++i) {
      __m256 sum = _mm256_loadu_ps((const float*)mean_buf + i * block);
      __m128 hi = _mm256_extractf128_ps(sum, 1);
      __m128 lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum, _mm256_insertf128_ps(
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      __m256 mean_vec = _mm256_mul_ps(sum, reverse_num_vec);
      mean[i] = *reinterpret_cast<float*>(&mean_vec);
      _mm256_storeu_ps(reinterpret_cast<float*>(mean_buf) + i * block,
                       mean_vec);
    }

/* get variance */
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < num; j++) {
        __m256 mean_vec = _mm256_loadu_ps((const float*)mean_buf + i * block);
        __m256 sum = _mm256_loadu_ps((const float*)var_buf + i * block);
        __m256 tmp = _mm256_sub_ps(
            _mm256_loadu_ps((const float*)x + i * right + j * block), mean_vec);
        tmp = _mm256_mul_ps(tmp, tmp);
        sum = _mm256_add_ps(sum, tmp);

        _mm256_storeu_ps(reinterpret_cast<float*>(var_buf) + i * block, sum);
      }
    }

    if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < height; ++i) {
        int j = (i + 1) * right - block;
        __m256 mean_vec = _mm256_loadu_ps((const float*)mean_buf + i * block);
        __m256 sum = _mm256_loadu_ps((const float*)var_buf + i * block);
        __m256 tmp =
            _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_mul_ps(tmp, tmp);
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp,
                               *(__m256*)&mask_vec);  // NOLINT
        sum = _mm256_add_ps(sum, tmp);
        _mm256_storeu_ps(reinterpret_cast<float*>(var_buf) + i * block, sum);
      }
    }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
    for (int i = 0; i < height; ++i) {
      __m256 sum = _mm256_loadu_ps((const float*)var_buf + i * block);
      __m128 hi = _mm256_extractf128_ps(sum, 1);
      __m128 lo = _mm256_extractf128_ps(sum, 0);
      sum = _mm256_add_ps(
          sum, _mm256_insertf128_ps(
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));
      sum = _mm256_hadd_ps(sum, sum);
      sum = _mm256_hadd_ps(sum, sum);
      __m256 var_vec = _mm256_mul_ps(sum, reverse_num_vec);
      var[i] = *reinterpret_cast<float*>(&var_vec);
      _mm256_storeu_ps(reinterpret_cast<float*>(var_buf) + i * block, var_vec);
    }

/* get x_norm and calculate output*/
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < num; j++) {
        __m256 mean_vec = _mm256_loadu_ps((const float*)mean_buf + i * block);
        __m256 var_vec = _mm256_loadu_ps((const float*)var_buf + i * block);
        __m256 tmp = _mm256_sub_ps(
            _mm256_loadu_ps((const float*)x + i * right + j * block), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + i * right + j * block,
                         tmp);
      }
    }

    if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < height; ++i) {
        int j = (i + 1) * right - block;
        __m256 mean_vec = _mm256_loadu_ps((const float*)mean_buf + i * block);
        __m256 var_vec = _mm256_loadu_ps((const float*)var_buf + i * block);
        __m256 tmp =
            _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);
        tmp = _mm256_div_ps(
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);
      }
    }

    if (scale) {
      if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < height; ++i) {
          int j = (i + 1) * right - block;
          __m256 tmp = _mm256_loadu_ps((const float*)out + j);
          _mm256_storeu_ps(reinterpret_cast<float*>(rest_buf) + i * block, tmp);
        }
      }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < num; j++) {
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + i * right + j * block,
              _mm256_mul_ps(
                  _mm256_loadu_ps((const float*)out + i * right + j * block),
                  _mm256_loadu_ps((const float*)scale + j * block)));
        }
      }

      if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < height; ++i) {
          int j = (i + 1) * right - block;
          __m256 tmp = _mm256_loadu_ps((const float*)rest_buf + i * block);
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_mul_ps(tmp,
                            _mm256_loadu_ps((const float*)scale + j % right)));
        }
      }
    }

    if (bias) {
      if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < height; ++i) {
          int j = (i + 1) * right - block;
          __m256 tmp = _mm256_loadu_ps((const float*)out + j);
          _mm256_storeu_ps(reinterpret_cast<float*>(rest_buf) + i * block, tmp);
        }
      }

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for collapse(2)
#endif
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < num; j++) {
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + i * right + j * block,
              _mm256_add_ps(
                  _mm256_loadu_ps((const float*)out + i * right + j * block),
                  _mm256_loadu_ps((const float*)bias + j * block)));
        }
      }

      if (rest != 0) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < height; ++i) {
          int j = (i + 1) * right - block;
          __m256 tmp = _mm256_loadu_ps((const float*)rest_buf + i * block);
          _mm256_storeu_ps(
              reinterpret_cast<float*>(out) + j,
              _mm256_add_ps(tmp,
                            _mm256_loadu_ps((const float*)bias + j % right)));
        }
      }
    }

    free(buf);
  }
}

bool LayerNormKernel::CanBeUsed(const int& d) const {
  return platform::MayIUse(platform::avx) && d >= YMM_FLOAT_BLOCK;
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace intrinsic = paddle::operators::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kLayerNorm, intrinsic, intrinsic::LayerNormKernel);
