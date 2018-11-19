/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/math/jit_kernel.h"
#include <math.h>
#include <limits>
#include <string>
#include "paddle/fluid/operators/math/jit_kernel_macro.h"
#ifdef __AVX__
#include <immintrin.h>
#endif

namespace paddle {
namespace operators {
namespace math {
namespace jitkernel {

namespace jit = platform::jit;

/* Layer Norm JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class LayerNormKernelImpl : public LayerNormKernel<T> {
 public:
  explicit LayerNormKernelImpl(int right) : LayerNormKernel<T>() {
    this->num_ = right;
  }

  void Compute(T* x, T* out, T* mean, T* var, const T* scale, const T* bias,
               int height, const float epsilon) const override {
    // get mean
    for (int i = 0; i < height; i++) {
      T sum = 0.0;
      int offset = i * this->num_;
      for (int j = 0; j < this->num_; j++) {
        sum += x[offset + j];
      }
      mean[i] = sum / this->num_;
    }

    // get variance
    for (int i = 0; i < height; i++) {
      T sum = 0.0;
      int offset = i * this->num_;
      for (int j = 0; j < this->num_; j++) {
        sum += (x[offset + j] - mean[i]) * (x[offset + j] - mean[i]);
      }
      var[i] = sum / this->num_;
    }

    for (int i = 0; i < height; i++) {
      int offset = i * this->num_;
      T sqrt_var = sqrt(var[i] + (T)epsilon);
      for (int j = 0; j < this->num_; j++) {
        out[offset + j] = (x[offset + j] - mean[i]) / sqrt_var;
      }
    }
    if (scale) {
      for (int i = 0; i < height; i++) {
        int offset = i * this->num_;
        for (int j = 0; j < this->num_; j++) {
          out[offset + j] *= scale[j];
        }
      }
    }

    if (bias) {
      for (int i = 0; i < height; i++) {
        int offset = i * this->num_;
        for (int j = 0; j < this->num_; j++) {
          out[offset + j] += bias[j];
        }
      }
    }
  }
};

#define INTRIAVX_FLOAT(isa, block)                                             \
  template <>                                                                  \
  LayerNormKernelImpl<float, isa, block>::LayerNormKernelImpl(int right)       \
      : LayerNormKernel<float>() {                                             \
    this->num_ = right;                                                        \
    this->rest_ = this->num_ % YMM_FLOAT_BLOCK;                                \
    this->end_ = this->num_ - this->rest_;                                     \
  }                                                                            \
  template <>                                                                  \
  void LayerNormKernelImpl<float, jit::avx, block>::Compute(                   \
      float* x, float* out, float* mean, float* var, const float* scale,       \
      const float* bias, int height, const float epsilon) const {              \
    __m256 sum;                                                                \
    __m256 mean_vec, var_vec;                                                  \
    __m128 hi, lo;                                                             \
    __m256 tmp;                                                                \
    size_t offset;                                                             \
    size_t j;                                                                  \
    __m256 reverse_num_vec =                                                   \
        _mm256_div_ps(_mm256_set1_ps(1.0), _mm256_set1_ps(this->num_));        \
    __m256 epsilon_vec = _mm256_set1_ps(epsilon);                              \
    int rest_mask =                                                            \
        ((-1) & (~((~0U) >> (sizeof(int) * 8 - (YMM_FLOAT_BLOCK - rest_))))) & \
        0x0ff;                                                                 \
    __m256i mask_vec = _mm256_set_epi32(                                       \
        rest_mask & 0x80 ? 0xffffffff : 0, rest_mask & 0x40 ? 0xffffffff : 0,  \
        rest_mask & 0x20 ? 0xffffffff : 0, rest_mask & 0x10 ? 0xffffffff : 0,  \
        rest_mask & 0x8 ? 0xffffffff : 0, rest_mask & 0x4 ? 0xffffffff : 0,    \
        rest_mask & 0x2 ? 0xffffffff : 0, rest_mask & 0x1 ? 0xffffffff : 0);   \
                                                                               \
    for (int i = 0; i < height; ++i) {                                         \
      offset = i * this->num_;                                                 \
                                                                               \
      /* get mean */                                                           \
      sum = _mm256_setzero_ps();                                               \
      for (j = offset; j < end_ + offset; j += block) {                        \
        sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float*)x + j));        \
      }                                                                        \
      if (rest_ != 0) {                                                        \
        j = offset + this->num_ - block;                                       \
        tmp = _mm256_loadu_ps((const float*)x + j);                            \
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp, (__m256)mask_vec);    \
        sum = _mm256_add_ps(sum, tmp);                                         \
      }                                                                        \
      hi = _mm256_extractf128_ps(sum, 1);                                      \
      lo = _mm256_extractf128_ps(sum, 0);                                      \
      sum = _mm256_add_ps(                                                     \
          sum, _mm256_insertf128_ps(                                           \
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));  \
      sum = _mm256_hadd_ps(sum, sum);                                          \
      sum = _mm256_hadd_ps(sum, sum);                                          \
      mean_vec = _mm256_mul_ps(sum, reverse_num_vec);                          \
      mean[i] = *reinterpret_cast<float*>(&mean_vec);                          \
                                                                               \
      /* get variance */                                                       \
      sum = _mm256_setzero_ps();                                               \
      for (j = offset; j < end_ + offset; j += block) {                        \
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);   \
        tmp = _mm256_mul_ps(tmp, tmp);                                         \
        sum = _mm256_add_ps(sum, tmp);                                         \
      }                                                                        \
      if (rest_ != 0) {                                                        \
        j = offset + this->num_ - block;                                       \
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);   \
        tmp = _mm256_mul_ps(tmp, tmp);                                         \
        tmp = _mm256_blendv_ps(_mm256_setzero_ps(), tmp, (__m256)mask_vec);    \
        sum = _mm256_add_ps(sum, tmp);                                         \
      }                                                                        \
      hi = _mm256_extractf128_ps(sum, 1);                                      \
      lo = _mm256_extractf128_ps(sum, 0);                                      \
      sum = _mm256_add_ps(                                                     \
          sum, _mm256_insertf128_ps(                                           \
                   _mm256_insertf128_ps(_mm256_setzero_ps(), hi, 0), lo, 1));  \
      sum = _mm256_hadd_ps(sum, sum);                                          \
      sum = _mm256_hadd_ps(sum, sum);                                          \
      var_vec = _mm256_mul_ps(sum, reverse_num_vec);                           \
      var[i] = *reinterpret_cast<float*>(&var_vec);                            \
                                                                               \
      /* get x_norm and calculate output*/                                     \
      for (j = offset; j < end_ + offset; j += block) {                        \
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);   \
        tmp = _mm256_div_ps(                                                   \
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));         \
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);              \
      }                                                                        \
      if (rest_ != 0) {                                                        \
        j = offset + num_ - block;                                             \
        tmp = _mm256_sub_ps(_mm256_loadu_ps((const float*)x + j), mean_vec);   \
        tmp = _mm256_div_ps(                                                   \
            tmp, _mm256_sqrt_ps(_mm256_add_ps(var_vec, epsilon_vec)));         \
        _mm256_storeu_ps(reinterpret_cast<float*>(out) + j, tmp);              \
      }                                                                        \
                                                                               \
      if (scale) {                                                             \
        if (rest_ != 0) {                                                      \
          j = offset + this->num_ - block;                                     \
          tmp = _mm256_loadu_ps((const float*)out + j);                        \
        }                                                                      \
        for (j = offset; j < end_ + offset; j += block) {                      \
          _mm256_storeu_ps(                                                    \
              reinterpret_cast<float*>(out) + j,                               \
              _mm256_mul_ps(                                                   \
                  _mm256_loadu_ps((const float*)out + j),                      \
                  _mm256_loadu_ps((const float*)scale + j - offset)));         \
        }                                                                      \
        if (rest_ != 0) {                                                      \
          j = offset + this->num_ - block;                                     \
          _mm256_storeu_ps(                                                    \
              reinterpret_cast<float*>(out) + j,                               \
              _mm256_mul_ps(                                                   \
                  tmp, _mm256_loadu_ps((const float*)scale + j - offset)));    \
        }                                                                      \
      }                                                                        \
                                                                               \
      if (bias) {                                                              \
        if (rest_ != 0) {                                                      \
          j = offset + this->num_ - block;                                     \
          tmp = _mm256_loadu_ps((const float*)out + j);                        \
        }                                                                      \
        for (j = offset; j < end_ + offset; j += block) {                      \
          _mm256_storeu_ps(                                                    \
              reinterpret_cast<float*>(out) + j,                               \
              _mm256_add_ps(                                                   \
                  _mm256_loadu_ps((const float*)out + j),                      \
                  _mm256_loadu_ps((const float*)bias + j - offset)));          \
        }                                                                      \
        if (rest_ != 0) {                                                      \
          j = offset + this->num_ - block;                                     \
          _mm256_storeu_ps(                                                    \
              reinterpret_cast<float*>(out) + j,                               \
              _mm256_add_ps(                                                   \
                  tmp, _mm256_loadu_ps((const float*)bias + j - offset)));     \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

#ifdef __AVX__
INTRIAVX_FLOAT(jit::avx, kEQ8);
INTRIAVX_FLOAT(jit::avx, kGT8LT16);
INTRIAVX_FLOAT(jit::avx, kEQ16);
INTRIAVX_FLOAT(jit::avx, kGT16);
#endif
#ifdef __AVX2__
INTRIAVX_FLOAT(jit::avx2, kEQ8);
INTRIAVX_FLOAT(jit::avx2, kGT8LT16);
INTRIAVX_FLOAT(jit::avx2, kEQ16);
INTRIAVX_FLOAT(jit::avx2, kGT16);
#endif

#undef INTRIAVX_FLOAT

REGISTER_JITKERNEL_DEPRECATED(layer_norm, LayerNormKernel);

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
