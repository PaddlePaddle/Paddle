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

namespace paddle {
namespace operators {

using framework::LoDTensor;
using framework::LoD;
using framework::Tensor;

template <typename Place, typename T>
class ViterbiDecodeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "The viterbi_decode operator can only run on CPU.");

    auto* emission_weights = ctx.Input<LoDTensor>("Emission");
    auto* transition_weights = ctx.Input<Tensor>("Transition");
    auto* label = ctx.Input<LoDTensor>("Label");

    auto* viterbi = ctx.Output<Tensor>("Viterbi");
    auto* viterbi_score = ctx.Output<Tensor>("ViterbiScore");

    PADDLE_ENFORCE_EQ(emission_weights->NumLevels(), 1UL,
                      "The Input(Emission) should be a sequence.");
    PADDLE_ENFORCE_EQ(label->NumLevels(), 1UL,
                      "The Input(Label) should be a sequence.");
    auto in_lod = emission_weights->lod();
    PADDLE_ENFORCE(in_lod.size(), "Input(Emission) must be a sequence.");
    const size_t level = 0;
    const size_t seq_num = in_lod[level].size() - 1;

    viterbi->mutable_data<T>(platform::CPUPlace());
    viterbi_score->mutable_data<T>(platform::CPUPlace());
    for (size_t i = 0; i < seq_num; ++i) {
      int start_pos = static_cast<int>(in_lod[level][i]);
      int end_pos = static_cast<int>(in_lod[level][i + 1]);
      const Tensor emission_one_seq =
          emission_weights->Slice(start_pos, end_pos);
      Tensor viterbi_one_seq = viterbi->Slice(start_pos, end_pos);
      Tensor score_one_seq = viterbi->Slice(start_pos, end_pos);
      Decode(emission_one_seq, *transition_weights, &viterbi_one_seq,
             &score_one_seq);
    }
  }

 private:
  void Decode(const Tensor& emission, const Tensor& transition_weights,
              Tensor* viterbi, Tensor* viterbi_score) const {}
};

}  // namespace operators
}  // namespace paddle
