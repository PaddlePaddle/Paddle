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
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/pten/kernels/funcs/math_function.h"

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

    int64_t* path = decoded_path->mutable_data<int64_t>(platform::CPUPlace());
    pten::funcs::SetConstant<DeviceContext, int64_t>()(
        ctx.template device_context<DeviceContext>(), decoded_path, 0);

    bool has_length = ctx.HasInput("Length");
    if (has_length) {
      auto* length = ctx.Input<Tensor>("Length");
      const size_t seq_num = length->numel();
      const int64_t* length_data = length->data<int64_t>();
      auto in_dims = emission_weights->dims();

      Tensor emission_weights_tmp = *emission_weights;
      emission_weights_tmp.Resize({in_dims[0] * in_dims[1], in_dims[2]});

      decoded_path->Resize({in_dims[0] * in_dims[1], 1});
      for (size_t i = 0; i < seq_num; ++i) {
        if (length_data[i] == 0) continue;
        int64_t start_pos = i * in_dims[1];
        int64_t end_pos = start_pos + static_cast<int64_t>(length_data[i]);
        Tensor decoded_path_one_seq = decoded_path->Slice(start_pos, end_pos);
        Decode(emission_weights_tmp.Slice(start_pos, end_pos),
               *transition_weights, &decoded_path_one_seq);
      }
      decoded_path->Resize({in_dims[0], in_dims[1]});

      if (label) {
        const int64_t* label_value = label->data<int64_t>();
        for (size_t i = 0; i < seq_num; ++i) {
          for (int64_t j = 0; j < in_dims[1]; ++j) {
            int64_t start_pos = i * in_dims[1];
            if (j < length_data[i]) {
              path[start_pos + j] =
                  label_value[start_pos + j] == path[start_pos + j] ? 1 : 0;
            } else {
              path[start_pos + j] = 0;
            }
          }
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(emission_weights->NumLevels(), 1UL,
                        platform::errors::InvalidArgument(
                            "The Input(Emission) should be a sequence with lod "
                            "level 1. But received: lod level %u.",
                            emission_weights->NumLevels()));
      auto lod = emission_weights->lod();
      PADDLE_ENFORCE_GT(
          lod.size(), 0,
          platform::errors::InvalidArgument(
              "Input(Emission) must be a sequence. But received: lod level %u.",
              lod.size()));
      const size_t level = 0;
      const size_t seq_num = lod[level].size() - 1;

      for (size_t i = 0; i < seq_num; ++i) {
        if (lod[level][i] == lod[level][i + 1]) continue;
        int64_t start_pos = static_cast<int64_t>(lod[level][i]);
        int64_t end_pos = static_cast<int64_t>(lod[level][i + 1]);
        Tensor decoded_path_one_seq = decoded_path->Slice(start_pos, end_pos);
        Decode(emission_weights->Slice(start_pos, end_pos), *transition_weights,
               &decoded_path_one_seq);
      }
      if (label) {
        PADDLE_ENFORCE_EQ(label->NumLevels(), 1UL,
                          platform::errors::InvalidArgument(
                              "The Input(label) should be a sequence with lod "
                              "level 1. But received: lod level %u.",
                              label->NumLevels()));
        const int64_t* label_value = label->data<int64_t>();
        size_t numel = label->numel();
        for (size_t i = 0; i < numel; ++i) {
          path[i] = label_value[i] == path[i] ? 1 : 0;
        }
      }
    }
  }

 private:
  void Decode(const Tensor& emission_weights, const Tensor& transition_weights,
              Tensor* decoded_path) const {
    auto emission_dims = emission_weights.dims();
    const size_t seq_len = emission_dims[0];
    const size_t tag_num = emission_dims[1];
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
    auto ker =
        jit::KernelFuncs<jit::CRFDecodingTuple<T>, platform::CPUPlace>::Cache()
            .At(tag_num);
    ker(static_cast<int>(seq_len), x, w, alpha_value, track_value, tag_num);
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
