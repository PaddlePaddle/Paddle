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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceEraseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::LoDTensor>("X");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto lod = in->lod();
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    // auto dims = x->dims();
    /*
    const size_t level = lod.size() - 1;
    PADDLE_ENFORCE_EQ(dims[0], static_cast<int64_t>(lod[level].back()),
                      "The first dimension of Input(X) should be equal to the "
                      "sum of all sequences' lengths.");
    PADDLE_ENFORCE_EQ(dims[0], x->numel(),
                      "The width of each timestep in Input(X) of "
                      "SequenceEraseOp should be 1.");
    out->mutable_data<T>(ctx.GetPlace());
    */
    auto tokens = ctx.Attr<std::vector<int>>("tokens");
    auto in_len = in->numel();
    auto in_dat = in->data<T>();
    auto lod0 = lod[0];
    std::vector<size_t> num_erased(in_len + 1, 0);
    for (int64_t i = 1; i < in_len + 1; ++i) {
      num_erased[i] = num_erased[i - 1];
      if (std::find(tokens.begin(), tokens.end(), in_dat[i - 1]) !=
          tokens.end()) {
        num_erased[i] += 1;
      }
    }

    std::vector<size_t> out_lod0(lod0.size(), 0);
    for (size_t i = 1; i < lod0.size(); ++i) {
      out_lod0[i] = lod0[i] - num_erased[lod0[i]];
    }

    auto out_len = in_len - num_erased[in_len];
    out->Resize({static_cast<int64_t>(out_len), 1});
    auto out_dat = out->mutable_data<T>(ctx.GetPlace());

    for (size_t i = 0; i < in_len; ++i) {
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
