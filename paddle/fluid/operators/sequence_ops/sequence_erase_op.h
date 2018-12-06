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
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_EQ(lod[0].back(), (size_t)in->numel(),
                      "The actual size mismatches with the LoD information.");
    auto tokens = ctx.Attr<std::vector<int>>("tokens");
    auto in_len = in->numel();
    auto in_dat = in->data<T>();
    auto lod0 = lod[0];

    std::vector<size_t> num_erased(in_len + 1, 0);
    std::vector<size_t> out_lod0(1, 0);
    for (size_t i = 0; i < lod0.size() - 1; ++i) {
      size_t num_out = 0;
      for (auto j = lod0[i] + 1; j <= lod0[i + 1]; ++j) {
        num_erased[j] = num_erased[j - 1];
        if (std::find(tokens.begin(), tokens.end(), in_dat[j - 1]) !=
            tokens.end()) {
          num_erased[j] += 1;
        } else {
          num_out += 1;
        }
      }
      out_lod0.push_back(out_lod0.back() + num_out);
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
    out_lod.push_back(out_lod0);
    out->set_lod(out_lod);
  }
};

}  // namespace operators
}  // namespace paddle
