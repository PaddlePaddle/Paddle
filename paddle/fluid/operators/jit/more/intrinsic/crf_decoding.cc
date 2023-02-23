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

#include "paddle/fluid/operators/jit/more/intrinsic/crf_decoding.h"

#include <limits>

#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/phi/backends/cpu/cpu_info.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace intrinsic {
// Note: intrinsic code is not runtime build.
// For example, if you build code on AVX, and run on AVX512 it can only use AVX

void CRFDecoding(const int seq_len,
                 const float* x,
                 const float* w,
                 float* alpha,
                 int* track,
                 int tag_num) {
#ifdef __AVX512F__
  const int step_size = ZMM_FLOAT_BLOCK;
#else
  const int step_size = YMM_FLOAT_BLOCK;
#endif
  const int end = tag_num / step_size;
  const int rest = tag_num % step_size;
  /* Setup the alpha initial value.*/
  int i_offset = 0;
  int last_offset = rest - step_size;
  for (int i = 0; i <= end; ++i) {
#ifdef __AVX512F__
    // Declare the variable for the content of weights, input and alpha values.
    __m512 w_content, x_content, alpha_content;
    // Load the relevant data into the variables from un-aligned address.
    w_content = _mm512_loadu_ps(w + i_offset);
    x_content = _mm512_loadu_ps(x + i_offset);
    alpha_content = _mm512_add_ps(w_content, x_content);
    // Save the alpha value.
    _mm512_storeu_ps(alpha + i_offset, alpha_content);
#else
    // AVX or AVX2
    // weights, input and alpha values.
    __m256 w_content, x_content, alpha_content;
    // Load the relevant data into the variables from un-aligned address.
    w_content = _mm256_loadu_ps(w + i_offset);
    x_content = _mm256_loadu_ps(x + i_offset);
    alpha_content = _mm256_add_ps(w_content, x_content);
    _mm256_storeu_ps(alpha + i_offset, alpha_content);
#endif
    i_offset += step_size;
    if (i == end - 1) {
      if (rest > 0) {
        i_offset += last_offset;
      } else {
        break;
      }
    }
  }
  // Use the column-major strategy to get the location of maximum score.
  int seq_offset = 0;
  constexpr int state_trans_base_idx = 2;
  for (int k = 1; k < seq_len; ++k) {
    int j_offset = 0;
    for (int j = 0; j <= end; ++j) {
/* Initialize the variables of maximum score and location.*/
#ifdef __AVX512F__
      __m512 max_score = _mm512_set1_ps(-std::numeric_limits<float>::max());
      __m512i max_j = _mm512_setzero_si512();
#else
      __m256 max_score = _mm256_set1_ps(-std::numeric_limits<float>::max());
      __m256i max_j = _mm256_set1_epi32(0);
#endif
      /* Calculate the offset of transition_weights.*/
      int trans_offset = state_trans_base_idx * tag_num + j_offset;
      for (int i = 0; i < tag_num; ++i) {
/* Initalize the content of alpha variable with related offset.*/
#ifdef __AVX512F__
        __m512 alpha_content = _mm512_set1_ps(*(alpha + seq_offset + i));
        /* Obtain the content of weights from un-aligned address.*/
        __m512 w_content = _mm512_loadu_ps(w + trans_offset);
        __m512 score_v = _mm512_add_ps(alpha_content, w_content);
        __mmask16 mask = _mm512_cmp_ps_mask(score_v, max_score, _CMP_GT_OS);
        /* AVX512 instructions.*/
        max_j = _mm512_mask_set1_epi32(max_j, mask, i);
        /* Update the max_score value.*/
        max_score = _mm512_max_ps(max_score, score_v);

#else
        __m256 alpha_content = _mm256_broadcast_ss(alpha + seq_offset + i);
        /* Obtain the content of weights from un-aligned address.*/
        __m256 w_content = _mm256_loadu_ps(w + trans_offset);
        __m256 score_v = _mm256_add_ps(alpha_content, w_content);
        __m256 mask = _mm256_cmp_ps(score_v, max_score, _CMP_GT_OS);
/* According to the mask value, update the index of the max_score.*/
#ifdef __AVX2__
        max_j = _mm256_or_si256(
            _mm256_andnot_si256((__m256i)mask, max_j),
            _mm256_and_si256((__m256i)mask, _mm256_set1_epi32(i)));
#else
        __m128i lo_max_j = _mm256_extractf128_si256(max_j, 0);
        __m128i hi_max_j = _mm256_extractf128_si256(max_j, 1);
        __m128i lo_mask =
            _mm256_extractf128_si256(*(__m256i*)&mask, 0);  // NOLINT
        __m128i hi_mask =
            _mm256_extractf128_si256(*(__m256i*)&mask, 1);  // NOLINT
        lo_max_j = _mm_andnot_si128(lo_mask, lo_max_j);
        hi_max_j = _mm_andnot_si128(hi_mask, hi_max_j);
        lo_mask = _mm_and_si128(lo_mask, _mm_set1_epi32(i));
        hi_mask = _mm_and_si128(hi_mask, _mm_set1_epi32(i));
        lo_max_j = _mm_or_si128(lo_mask, lo_max_j);
        hi_max_j = _mm_or_si128(hi_mask, hi_max_j);
        max_j = _mm256_insertf128_si256(max_j, lo_max_j, 0);
        max_j = _mm256_insertf128_si256(max_j, hi_max_j, 1);
#endif
        /* Update the max_score value.*/
        max_score = _mm256_max_ps(max_score, score_v);

#endif

        trans_offset += tag_num;
      }
/* Update the alpha and track values. */
#ifdef __AVX512F__
      __m512 x_content = _mm512_loadu_ps(x + seq_offset + tag_num + j_offset);
      max_score = _mm512_add_ps(max_score, x_content);
      _mm512_storeu_ps(alpha + seq_offset + tag_num + j_offset, max_score);
      _mm512_storeu_si512(
          reinterpret_cast<__m512i*>(track + seq_offset + tag_num + j_offset),
          max_j);
#else
      __m256 x_content = _mm256_loadu_ps(x + seq_offset + tag_num + j_offset);
      max_score = _mm256_add_ps(max_score, x_content);
      _mm256_storeu_ps(alpha + seq_offset + tag_num + j_offset, max_score);
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(track + seq_offset + tag_num + j_offset),
          max_j);
#endif

      /* Calculate the offset of next step*/
      j_offset += step_size;
      if (j == end - 1) {
        if (rest > 0) {
          j_offset += last_offset;
        } else {
          break;
        }
      }
    }
    seq_offset += tag_num;
  }
}

bool CRFDecodingKernel::CanBeUsed(const int& d) const {
#ifdef __AVX512F__
  constexpr int block = ZMM_FLOAT_BLOCK;
#else
  constexpr int block = YMM_FLOAT_BLOCK;
#endif
  return phi::backends::cpu::MayIUse(phi::backends::cpu::avx) && d >= block;
}

}  // namespace intrinsic
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

namespace intrinsic = paddle::operators::jit::more::intrinsic;

REGISTER_JITKERNEL_MORE(kCRFDecoding, intrinsic, intrinsic::CRFDecodingKernel);
