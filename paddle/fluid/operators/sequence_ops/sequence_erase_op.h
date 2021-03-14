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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SequenceEraseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto lod = in->lod();
    PADDLE_ENFORCE_EQ(
        lod.empty(), false,
        platform::errors::InvalidArgument("Input(X) Tensor of SequenceEraseOp "
                                          "does not contain LoD information."));
    PADDLE_ENFORCE_EQ(lod[lod.size() - 1].back(), (size_t)in->numel(),
                      platform::errors::InvalidArgument(
                          "The actual input size %d mismatches with the LoD "
                          "information size %d.",
                          lod[lod.size() - 1].back(), (size_t)in->numel()));
    auto tokens = ctx.Attr<std::vector<int>>("tokens");
    auto in_len = in->numel();
    auto in_dat = in->data<T>();
    auto last_lod = lod[lod.size() - 1];

    std::vector<size_t> num_erased(in_len + 1, 0);
    std::vector<size_t> out_last_lod(1, 0);
    for (size_t i = 0; i < last_lod.size() - 1; ++i) {
      size_t num_out = 0;
      for (auto j = last_lod[i] + 1; j <= last_lod[i + 1]; ++j) {
        num_erased[j] = num_erased[j - 1];
        if (std::find(tokens.begin(), tokens.end(), in_dat[j - 1]) !=
            tokens.end()) {
          num_erased[j] += 1;
        } else {
          num_out += 1;
        }
      }
      out_last_lod.push_back(out_last_lod.back() + num_out);
    }

    auto out_len = in_len - num_erased[in_len];
    out->Resize({static_cast<int64_t>(out_len), 1});
    auto out_dat = out->mutable_data<T>(ctx.GetPlace());

    for (int64_t i = 0; i < in_len; ++i) {
      if (num_erased[i] == num_erased[i + 1]) {
        out_dat[i - num_erased[i]] = in_dat[i];
      }
    }
    framework::LoD out_lod;
    for (size_t i = 0; i < lod.size() - 1; ++i) {
      out_lod.push_back(lod[i]);
    }
    out_lod.push_back(out_last_lod);
    out->set_lod(out_lod);
  }
};

}  // namespace operators
}  // namespace paddle
