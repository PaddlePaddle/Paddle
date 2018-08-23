//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceEnumerateKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int win_size = context.Attr<int>("win_size");
    int pad_value = context.Attr<int>("pad_value");

    auto in_dims = in->dims();
    auto in_lod = in->lod();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    // Generate enumerate sequence set
    auto seq_length = in_dims[0];
    auto in_data = in->data<T>();
    auto out_data = out->mutable_data<T>(context.GetPlace());
    for (int idx = 0; idx < seq_length; ++idx) {
      for (int word_idx = 0; word_idx < win_size; ++word_idx) {
        int word_pos = idx + word_idx;
        out_data[win_size * idx + word_idx] =
            word_pos < seq_length ? in_data[word_pos] : pad_value;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
