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

/* CRF Decode JitKernel */
template <typename T, platform::jit::cpu_isa_t isa, jit_block>
class CRFDecodeKernelImpl : public CRFDecodeKernel<T> {
 public:
  explicit CRFDecodeKernelImpl(int tag_num) : CRFDecodeKernel<T>() {
    this->num_ = tag_num;
  }
  void Compute(const int seq_len, const T* x, const T* w, T* alpha,
               int* track) const override {
    constexpr int state_trans_base_idx = 2;
    for (int i = 0; i < this->num_; ++i) {
      alpha[i] = w[i] + x[i];
    }
    for (int k = 1; k < seq_len; ++k) {
      for (int i = 0; i < this->num_; ++i) {
        T max_score = -std::numeric_limits<T>::max();
        int max_j = 0;
        for (int j = 0; j < this->num_; ++j) {
          T score = alpha[(k - 1) * this->num_ + j] +
                    w[(j + state_trans_base_idx) * this->num_ + i];
          if (score > max_score) {
            max_score = score;
            max_j = j;
          }
        }
        alpha[k * this->num_ + i] = max_score + x[k * this->num_ + i];
        track[k * this->num_ + i] = max_j;
      }
    }
  }
};

#define INIT_ALPHA(step_size)                                               \
  /* Setup the alpha initial value.*/                                       \
  int i_offset = 0;                                                         \
  int last_offset = this->rest_ - step_size;                                \
  for (int i = 0; i <= this->end_; ++i) {                                   \
    /* weights, input and alpha values. */                                  \
    __m256 w_content, x_content, alpha_content;                             \
    /* Load the relevant data into the variables from un-aligned address.*/ \
    w_content = _mm256_loadu_ps(w + i_offset);                              \
    x_content = _mm256_loadu_ps(x + i_offset);                              \
    alpha_content = _mm256_add_ps(w_content, x_content);                    \
    _mm256_storeu_ps(alpha + i_offset, alpha_content);                      \
    i_offset += step_size;                                                  \
    if (i == this->end_ - 1) {                                              \
      if (this->rest_ > 0) {                                                \
        i_offset += last_offset;                                            \
      } else {                                                              \
        break;                                                              \
      }                                                                     \
    }                                                                       \
  }

#define UPDATE_ALPHA(step_size)                                               \
  /* Update the alpha and track values. */                                    \
  __m256 x_content = _mm256_loadu_ps(x + seq_offset + this->num_ + j_offset); \
  max_score = _mm256_add_ps(max_score, x_content);                            \
  _mm256_storeu_ps(alpha + seq_offset + this->num_ + j_offset, max_score);    \
  _mm256_storeu_si256(                                                        \
      reinterpret_cast<__m256i*>(track + seq_offset + this->num_ + j_offset), \
      max_j);                                                                 \
  /* Calculate the offset of next step*/                                      \
  j_offset += step_size;                                                      \
  if (j == this->end_ - 1) {                                                  \
    if (this->rest_ > 0) {                                                    \
      j_offset += last_offset;                                                \
    } else {                                                                  \
      break;                                                                  \
    }                                                                         \
  }

#define INTRIAVX_FLOAT(block)                                                  \
  template <>                                                                  \
  CRFDecodeKernelImpl<float, jit::avx, block>::CRFDecodeKernelImpl(            \
      int tag_num)                                                             \
      : CRFDecodeKernel<float>() {                                             \
    this->num_ = tag_num;                                                      \
    this->end_ = this->num_ / YMM_FLOAT_BLOCK;                                 \
    this->rest_ = this->num_ % YMM_FLOAT_BLOCK;                                \
  }                                                                            \
  template <>                                                                  \
  void CRFDecodeKernelImpl<float, jit::avx, block>::Compute(                   \
      const int seq_len, const float* x, const float* w, float* alpha,         \
      int* track) const {                                                      \
    INIT_ALPHA(YMM_FLOAT_BLOCK)                                                \
    /* Use the column-major strategy to get the location of maximum score.*/   \
    int seq_offset = 0;                                                        \
    constexpr int state_trans_base_idx = 2;                                    \
    for (int k = 1; k < seq_len; ++k) {                                        \
      int j_offset = 0;                                                        \
      for (int j = 0; j <= this->end_; ++j) {                                  \
        /* Initialize the variables of maximum score and location.*/           \
        __m256 max_score = _mm256_set1_ps(-std::numeric_limits<float>::max()); \
        __m256i max_j = _mm256_set1_epi32(0);                                  \
        /* Calculate the offset of transition_weights.*/                       \
        int trans_offset = state_trans_base_idx * this->num_ + j_offset;       \
        for (int i = 0; i < this->num_; ++i) {                                 \
          /* Initalize the content of alpha variable with related offset.*/    \
          __m256 alpha_content = _mm256_broadcast_ss(alpha + seq_offset + i);  \
          /* Obtain the content of weights from un-aligned address.*/          \
          __m256 w_content = _mm256_loadu_ps(w + trans_offset);                \
          __m256 score_v = _mm256_add_ps(alpha_content, w_content);            \
          __m256 mask = _mm256_cmp_ps(score_v, max_score, _CMP_GT_OS);         \
          /* According to the mask value, update the index of the max_score.*/ \
          /* AVX instructions.*/                                               \
          __m128i lo_max_j = _mm256_extractf128_si256(max_j, 0);               \
          __m128i hi_max_j = _mm256_extractf128_si256(max_j, 1);               \
          __m128i lo_mask = _mm256_extractf128_si256((__m256i)mask, 0);        \
          __m128i hi_mask = _mm256_extractf128_si256((__m256i)mask, 1);        \
          lo_max_j = _mm_andnot_si128(lo_mask, lo_max_j);                      \
          hi_max_j = _mm_andnot_si128(hi_mask, hi_max_j);                      \
          lo_mask = _mm_and_si128(lo_mask, _mm_set1_epi32(i));                 \
          hi_mask = _mm_and_si128(hi_mask, _mm_set1_epi32(i));                 \
          lo_max_j = _mm_or_si128(lo_mask, lo_max_j);                          \
          hi_max_j = _mm_or_si128(hi_mask, hi_max_j);                          \
          max_j = _mm256_insertf128_si256(max_j, lo_max_j, 0);                 \
          max_j = _mm256_insertf128_si256(max_j, hi_max_j, 1);                 \
          /* AVX done*/                                                        \
          /* Update the max_score value.*/                                     \
          max_score = _mm256_max_ps(max_score, score_v);                       \
          trans_offset += this->num_;                                          \
        }                                                                      \
        UPDATE_ALPHA(YMM_FLOAT_BLOCK)                                          \
      }                                                                        \
      seq_offset += this->num_;                                                \
    }                                                                          \
  }

#define INTRIAVX2_FLOAT(isa, block)                                            \
  template <>                                                                  \
  CRFDecodeKernelImpl<float, isa, block>::CRFDecodeKernelImpl(int tag_num)     \
      : CRFDecodeKernel<float>() {                                             \
    this->num_ = tag_num;                                                      \
    this->end_ = this->num_ / YMM_FLOAT_BLOCK;                                 \
    this->rest_ = this->num_ % YMM_FLOAT_BLOCK;                                \
  }                                                                            \
  template <>                                                                  \
  void CRFDecodeKernelImpl<float, isa, block>::Compute(                        \
      const int seq_len, const float* x, const float* w, float* alpha,         \
      int* track) const {                                                      \
    INIT_ALPHA(YMM_FLOAT_BLOCK)                                                \
    /* Use the column-major strategy to get the location of maximum score.*/   \
    int seq_offset = 0;                                                        \
    constexpr int state_trans_base_idx = 2;                                    \
    for (int k = 1; k < seq_len; ++k) {                                        \
      int j_offset = 0;                                                        \
      for (int j = 0; j <= this->end_; ++j) {                                  \
        /* Initialize the variables of maximum score and location.*/           \
        __m256 max_score = _mm256_set1_ps(-std::numeric_limits<float>::max()); \
        __m256i max_j = _mm256_set1_epi32(0);                                  \
        /* Calculate the offset of transition_weights.*/                       \
        int trans_offset = state_trans_base_idx * this->num_ + j_offset;       \
        for (int i = 0; i < this->num_; ++i) {                                 \
          /* Initalize the content of alpha variable with related offset.*/    \
          __m256 alpha_content = _mm256_broadcast_ss(alpha + seq_offset + i);  \
          /* Obtain the content of weights from un-aligned address.*/          \
          __m256 w_content = _mm256_loadu_ps(w + trans_offset);                \
          __m256 score_v = _mm256_add_ps(alpha_content, w_content);            \
          __m256 mask = _mm256_cmp_ps(score_v, max_score, _CMP_GT_OS);         \
          /* According to the mask value, update the index of the max_score.*/ \
          /* AVX2 instructions.*/                                              \
          max_j = _mm256_or_si256(                                             \
              _mm256_andnot_si256((__m256i)mask, max_j),                       \
              _mm256_and_si256((__m256i)mask, _mm256_set1_epi32(i)));          \
          /* Update the max_score value.*/                                     \
          max_score = _mm256_max_ps(max_score, score_v);                       \
          trans_offset += this->num_;                                          \
        }                                                                      \
        UPDATE_ALPHA(YMM_FLOAT_BLOCK)                                          \
      }                                                                        \
      seq_offset += this->num_;                                                \
    }                                                                          \
  }

#define INTRIAVX512_FLOAT(block)                                               \
  template <>                                                                  \
  CRFDecodeKernelImpl<float, jit::avx512f, block>::CRFDecodeKernelImpl(        \
      int tag_num)                                                             \
      : CRFDecodeKernel<float>() {                                             \
    this->num_ = tag_num;                                                      \
    this->end_ = this->num_ / ZMM_FLOAT_BLOCK;                                 \
    this->rest_ = this->num_ % ZMM_FLOAT_BLOCK;                                \
  }                                                                            \
  template <>                                                                  \
  void CRFDecodeKernelImpl<float, jit::avx512f, block>::Compute(               \
      const int seq_len, const float* x, const float* w, float* alpha,         \
      int* track) const {                                                      \
    INIT_ALPHA(ZMM_FLOAT_BLOCK)                                                \
    /* Use the column-major strategy to get the location of maximum score.*/   \
    int seq_offset = 0;                                                        \
    constexpr int state_trans_base_idx = 2;                                    \
    for (int k = 1; k < seq_len; ++k) {                                        \
      int j_offset = 0;                                                        \
      for (int j = 0; j <= this->end_; ++j) {                                  \
        /* Initialize the variables of maximum score and location.*/           \
        __m512 max_score = _mm512_set1_ps(-std::numeric_limits<float>::max()); \
        __m512i max_j = _mm512_setzero_si512();                                \
        /* Calculate the offset of transition_weights.*/                       \
        int trans_offset = state_trans_base_idx * this->num_ + j_offset;       \
        for (int i = 0; i < this->num_; ++i) {                                 \
          /* Initalize the content of alpha variable with related offset.*/    \
          __m512 alpha_content = _mm512_set1_ps(*(alpha + seq_offset + i));    \
          /* Obtain the content of weights from un-aligned address.*/          \
          __m512 w_content = _mm512_loadu_ps(w + trans_offset);                \
          __m512 score_v = _mm512_add_ps(alpha_content, w_content);            \
          __mmask16 mask = _mm512_cmp_ps_mask(score_v, max_score, _CMP_GT_OS); \
          /* AVX512 instructions.*/                                            \
          max_j = _mm512_mask_set1_epi32(max_j, mask, i);                      \
          /* Update the max_score value.*/                                     \
          max_score = _mm512_max_ps(max_score, score_v);                       \
          trans_offset += this->num_;                                          \
        }                                                                      \
        /* Update the alpha and track values.*/                                \
        __m512 x_content =                                                     \
            _mm512_loadu_ps(x + seq_offset + this->num_ + j_offset);           \
        max_score = _mm512_add_ps(max_score, x_content);                       \
        _mm512_storeu_ps(alpha + seq_offset + this->num_ + j_offset,           \
                         max_score);                                           \
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(track + seq_offset +    \
                                                       this->num_ + j_offset), \
                            max_j);                                            \
        /* Calculate the offset of next step*/                                 \
        j_offset += ZMM_FLOAT_BLOCK;                                           \
        if (j == this->end_ - 1) {                                             \
          if (this->rest_ > 0) {                                               \
            j_offset += last_offset;                                           \
          } else {                                                             \
            break;                                                             \
          }                                                                    \
        }                                                                      \
      }                                                                        \
      seq_offset += this->num_;                                                \
    }                                                                          \
  }

#ifdef __AVX__
INTRIAVX_FLOAT(kEQ8);
INTRIAVX_FLOAT(kGT8LT16);
INTRIAVX_FLOAT(kEQ16);
INTRIAVX_FLOAT(kGT16);
#endif
#ifdef __AVX2__
INTRIAVX2_FLOAT(jit::avx2, kEQ8);
INTRIAVX2_FLOAT(jit::avx2, kGT8LT16);
INTRIAVX2_FLOAT(jit::avx2, kEQ16);
INTRIAVX2_FLOAT(jit::avx2, kGT16);
#endif
#ifdef __AVX512F__
INTRIAVX2_FLOAT(jit::avx512f, kEQ8);
INTRIAVX2_FLOAT(jit::avx512f, kGT8LT16);
INTRIAVX512_FLOAT(kEQ16);
INTRIAVX512_FLOAT(kGT16);
#endif

#undef INTRIAVX512_FLOAT
#undef INTRIAVX2_FLOAT
#undef INTRIAVX_FLOAT
#undef INIT_ALPHA
#undef UPDATE_ALPHA

REGISTER_JITKERNEL_DEPRECATED(crf_decode, CRFDecodeKernel);

}  // namespace jitkernel
}  // namespace math
}  // namespace operators
}  // namespace paddle
