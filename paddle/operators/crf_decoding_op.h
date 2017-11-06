/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;
using framework::LoD;
using framework::Tensor;

template <typename Place, typename T>
class CRFDecodingOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "The crf_decoding operator can only run on CPU.");

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

    int* path = decoded_path->mutable_data<int>(platform::CPUPlace());
    math::SetConstant<platform::CPUPlace, int>()(ctx.device_context(),
                                                 decoded_path, 0);
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
      const int* label_value = label->data<int>();
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
    int* path = decoded_path->data<int>();

    // alpha is a memo table. An element alpha(k, v) records the score of the
    // best sequence of tags from position 1 to position k with v being the end
    // tag.
    Tensor alpha;
    T* alpha_value = alpha.mutable_data<T>(emission_dims, platform::CPUPlace());
    Tensor track;
    int* track_value =
        track.mutable_data<int>(emission_dims, platform::CPUPlace());

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
