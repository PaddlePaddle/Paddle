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
    bool need_pad = context.Attr<bool>("need_pad");

    auto in_dims = in->dims();
    auto in_lod = in->lod();

    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), in_lod[0].back(),
        "The actual input data's size mismatched with LoD information.");

    // Generate enumerate sequence set
    auto lod0 = in_lod[0];
    auto in_data = in->data<T>();
    int enumerate_shift = 0;
    int offset = 0;
    if (need_pad) {
      out->Resize({in_dims[0], win_size});
      auto out_data = out->mutable_data<T>(context.GetPlace());
      out->set_lod(in->lod());
      for (size_t i = 0; i < lod0.size() - 1; ++i) {
        for (size_t idx = lod0[i]; idx < lod0[i + 1] - enumerate_shift; ++idx) {
          for (int word_idx = 0; word_idx < win_size; ++word_idx) {
            size_t word_pos = idx + word_idx;
            out_data[win_size * idx + word_idx] =
                word_pos < lod0[i + 1] ? in_data[word_pos] : pad_value;
          }
        }
      }
    } else {
      framework::LoD new_lod;
      enumerate_shift = win_size - 1;
      new_lod.emplace_back(1, 0);  // size = 1, value = 0;
      auto new_lod0 = new_lod[0];
      for (size_t i = 1; i < lod0.size(); ++i) {
          if (lod0[i] - lod0[i - 1] > enumerate_shift) {
              offset = offset + lod0[i] - lod0[i - 1] - enumerate_shift;
          }
          new_lod0.push_back(offset);
      }
      new_lod[0] = new_lod0;
      if (offset == 0) {
          win_size = 0;
      }
      out->Resize({offset, win_size});
      auto out_data = out->mutable_data<T>(context.GetPlace());

      out->set_lod(new_lod);
      for (size_t i = 0; i < lod0.size() - 1; ++i) {
        size_t start_idx = lod0[i];
        if (lod0[i + 1] - lod0[i] > enumerate_shift) {
          size_t len = lod0[i + 1] - lod0[i] - enumerate_shift;
          size_t new_start_idx = new_lod0[i];

          for (size_t j = 0; j < len; ++j) {
            size_t idx = start_idx + j;
            size_t new_idx = (new_start_idx + j) * win_size;
            for (int word_idx = 0; word_idx < win_size; ++word_idx) {
              size_t word_pos = idx + word_idx;
              size_t new_word_pos = new_idx + word_idx;
              out_data[new_word_pos] = in_data[word_pos];
            }
          }
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
