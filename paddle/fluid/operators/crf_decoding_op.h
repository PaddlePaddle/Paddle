/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <limits>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;
using framework::LoD;
using framework::Tensor;

template <typename DeviceContext, typename T>
class CRFDecodingOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* emission_weights = ctx.Input<LoDTensor>("Emission");
    auto* transition_weights = ctx.Input<Tensor>("Transition");
    auto* label = ctx.Input<LoDTensor>("Label");
    auto* decoded_path = ctx.Output<Tensor>("ViterbiPath");

    PADDLE_ENFORCE_EQ(emission_weights->NumLevels(), 1UL,
                      "The Input(Emission) should be a sequence.");
    auto lod = emission_weights->lod();
    PADDLE_ENFORCE(lod.size(), "Input(Emission) must be a sequence.");
    const size_t level = 0;
    const size_t seq_num = lod[level].size() - 1;

    int64_t* path = decoded_path->mutable_data<int64_t>(platform::CPUPlace());
    math::SetConstant<DeviceContext, int64_t>()(
        ctx.template device_context<DeviceContext>(), decoded_path, 0);
    for (size_t i = 0; i < seq_num; ++i) {
      int start_pos = static_cast<int>(lod[level][i]);
      int end_pos = static_cast<int>(lod[level][i + 1]);
      Tensor decoded_path_one_seq = decoded_path->Slice(start_pos, end_pos);
      Decode(emission_weights->Slice(start_pos, end_pos), *transition_weights,
             &decoded_path_one_seq);
    }

    if (label) {
      PADDLE_ENFORCE_EQ(label->NumLevels(), 1UL,
                        "The Input(Label) should be a sequence.");
      const int64_t* label_value = label->data<int64_t>();
      size_t batch_size = emission_weights->dims()[0];
      for (size_t i = 0; i < batch_size; ++i) {
        path[i] = label_value[i] == path[i] ? 1 : 0;
      }
    }
  }

 private:
  void Decode(const Tensor& emission_weights, const Tensor& transition_weights,
              Tensor* decoded_path) const {
    auto emission_dims = emission_weights.dims();
    const size_t seq_len = emission_dims[0];
    const size_t tag_num = emission_dims[1];

    const size_t state_trans_base_idx = 2;

    const T* x = emission_weights.data<T>();
    const T* w = transition_weights.data<T>();
    int64_t* path = decoded_path->data<int64_t>();

    // alpha is a memo table. An element alpha(k, v) records the score of the
    // best sequence of tags from position 1 to position k with v being the end
    // tag.
    Tensor alpha;
    T* alpha_value = alpha.mutable_data<T>(emission_dims, platform::CPUPlace());
    Tensor track;
    int* track_value =
        track.mutable_data<int>(emission_dims, platform::CPUPlace());

#ifdef __AVX__
    //Only optimize for float type.
#ifdef __AVX512F__
      size_t step_size = 16;
#else
      size_t step_size = 8;
#endif
    if (std::is_same<T, float>::value && (tag_num >= step_size)) {
      size_t steps = tag_num/step_size;
      size_t remain = tag_num%step_size;
      int last_offset = (int)remain - (int)step_size;

      //Setup the alpha initial value.
      size_t i_offset = 0;
      for (size_t i = 0; i <= steps; ++i) {
#ifdef __AVX512F__
        __m512 w_content, x_content, alpha_content;

        w_content = _mm512_loadu_ps((const float*)(w + i_offset));
        x_content = _mm512_loadu_ps((const float*)(x + i_offset));
        alpha_content = _mm512_add_ps(w_content, x_content);
        _mm512_storeu_ps((float *)(alpha_value + i_offset), alpha_content);
#else
        __m256 w_content, x_content, alpha_content;

        w_content = _mm256_loadu_ps((const float*)(w + i_offset));
        x_content = _mm256_loadu_ps((const float*)(x + i_offset));
        alpha_content = _mm256_add_ps(w_content, x_content);
        _mm256_storeu_ps((float *)(alpha_value + i_offset), alpha_content);
#endif
        i_offset += step_size;
        if ( i == steps - 1 ) {
          if (remain > 0) {
            i_offset += last_offset;
          } else {
            break;
          }
        }
      }

      //Use the column-major strategy to get the location of maximum score.
      size_t seq_offset = 0;
      for (size_t k = 1; k < seq_len; ++k) {
        size_t j_offset = 0;
        for (size_t j = 0; j <= steps; ++j) {
#ifdef __AVX512F__
          __m512 max_score = _mm512_set1_ps(
                    -std::numeric_limits<T>::max());
          __m512i max_j = _mm512_setzero_si512();
#else
          __m256 max_score = _mm256_set1_ps(
                    -std::numeric_limits<T>::max());
          __m256i max_j = _mm256_set1_epi32(0);
#endif
          size_t trans_offset = state_trans_base_idx * tag_num + j_offset;
          for (size_t i = 0; i < tag_num; ++i) {

#ifdef __AVX512F__
            __m512 alpha_content = _mm512_set1_ps(*(const float*)
                                      (alpha_value + seq_offset + i));
            __m512 w_content = _mm512_loadu_ps((const float*)(w +
                                      trans_offset));
            __m512 score_v = _mm512_add_ps(alpha_content, w_content);

            __mmask16 mask = _mm512_cmp_ps_mask(score_v, max_score, _CMP_GT_OS);

            max_j = _mm512_mask_set1_epi32(max_j, mask, i);              

            max_score = _mm512_max_ps(max_score, score_v);
            if ( i == tag_num - 1 ) {
              __m512 x_content = _mm512_loadu_ps((const float*)(x +
                                              seq_offset + tag_num + j_offset));
              max_score = _mm512_add_ps(max_score, x_content);
            }
#else
            __m256 alpha_content = _mm256_broadcast_ss((const float*)
                                      (alpha_value + seq_offset + i));
            __m256 w_content = _mm256_loadu_ps((const float*)(w +
                                      trans_offset));
            __m256 score_v = _mm256_add_ps(alpha_content, w_content);

            __m256 mask = _mm256_cmp_ps(score_v, max_score, _CMP_GT_OS);

#ifdef __AVX2__
            max_j = _mm256_or_si256(
                      _mm256_andnot_si256((__m256i)mask, max_j),
                      _mm256_and_si256((__m256i)mask, _mm256_set1_epi32(i)));
#else
            __m128i lo_max_j = _mm256_extractf128_si256(max_j, 0);
            __m128i hi_max_j = _mm256_extractf128_si256(max_j, 1);
            __m128i lo_mask = _mm256_extractf128_si256((__m256i)mask, 0);
            __m128i hi_mask = _mm256_extractf128_si256((__m256i)mask, 1);

            lo_max_j = _mm_andnot_si128(lo_mask, lo_max_j);
            hi_max_j = _mm_andnot_si128(hi_mask, hi_max_j);
            lo_mask = _mm_and_si128(lo_mask, _mm_set1_epi32(i));
            hi_mask = _mm_and_si128(hi_mask, _mm_set1_epi32(i));

            lo_max_j = _mm_or_si128(lo_mask, lo_max_j);
            hi_max_j = _mm_or_si128(hi_mask, hi_max_j);

            max_j = _mm256_insertf128_si256(max_j, lo_max_j, 0);
            max_j = _mm256_insertf128_si256(max_j, hi_max_j, 1);
#endif

            max_score = _mm256_max_ps(max_score, score_v);
            if ( i == tag_num - 1 ) {
              __m256 x_content = _mm256_loadu_ps((const float*)(x +
                                              seq_offset + tag_num + j_offset));
              max_score = _mm256_add_ps(max_score, x_content);
            }
#endif
            trans_offset += tag_num;
          }

#ifdef __AVX512F__
          _mm512_storeu_ps((float *)(alpha_value +
                            seq_offset + tag_num + j_offset), max_score);
          _mm512_storeu_si512((__m512i *)(track_value + seq_offset + tag_num +
                              j_offset), max_j);
#else
          _mm256_storeu_ps((float *)(alpha_value +
                            seq_offset + tag_num + j_offset), max_score);
          _mm256_storeu_si256((__m256i *)(track_value + seq_offset + tag_num +
                              j_offset), max_j);
#endif

          //Calculate the offset of next step
          j_offset += step_size;
          if ( j == steps - 1 ) {
            if ( remain > 0 ) {
              j_offset += last_offset;
            } else {
              break;
            }
          }
        }

        seq_offset += tag_num;
      }
    } else {
      for (size_t i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

      for (size_t k = 1; k < seq_len; ++k) {
        for (size_t i = 0; i < tag_num; ++i) {
          T max_score = -std::numeric_limits<T>::max();
          int max_j = 0;
          for (size_t j = 0; j < tag_num; ++j) {
            T score = alpha_value[(k - 1) * tag_num + j] +
                      w[(j + state_trans_base_idx) * tag_num + i];
            if (score > max_score) {
              max_score = score;
              max_j = j;
            }
          }

          alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
          track_value[k * tag_num + i] = max_j;
        }
      }
    }
#else
    for (size_t i = 0; i < tag_num; ++i) alpha_value[i] = w[i] + x[i];

    for (size_t k = 1; k < seq_len; ++k) {
      for (size_t i = 0; i < tag_num; ++i) {
        T max_score = -std::numeric_limits<T>::max();
        int max_j = 0;
        for (size_t j = 0; j < tag_num; ++j) {
          T score = alpha_value[(k - 1) * tag_num + j] +
                    w[(j + state_trans_base_idx) * tag_num + i];
          if (score > max_score) {
            max_score = score;
            max_j = j;
          }
        }

        alpha_value[k * tag_num + i] = max_score + x[k * tag_num + i];
        track_value[k * tag_num + i] = max_j;
      }
    }

#endif
    T max_score = -std::numeric_limits<T>::max();
    int max_i = 0;
    for (size_t i = 0; i < tag_num; ++i) {
      T score = alpha_value[(seq_len - 1) * tag_num + i] + w[tag_num + i];
      if (score > max_score) {
        max_score = score;
        max_i = i;
      }
    }
    path[seq_len - 1] = max_i;
    for (int k = seq_len - 1; k >= 1; --k) {
      path[k - 1] = max_i = track_value[k * tag_num + max_i];
    }
  }
};

}  // namespace operators
}  // namespace paddle
