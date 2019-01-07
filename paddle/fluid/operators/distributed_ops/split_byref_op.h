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
class SplitByrefOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto place = ctx.GetPlace();

    size_t row_offset = 0;
    for (size_t i = 0; i < outs.size(); ++i) {
      // NOTE: no need to call mutable_data here to allocate memory.
      auto* out = outs[i];
      VLOG(3) << "spliting by ref: " << row_offset << " " << out->dims()[0];
      *out = in->Slice(row_offset, row_offset + out->dims()[0]);
      row_offset += out->dims()[0];
    }
  }
};

}  // namespace operators
}  // namespace paddle
