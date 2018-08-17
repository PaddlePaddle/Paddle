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
#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ShapeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("Input");
    auto* out_t = ctx.Output<Tensor>("Out");
    auto out_data = out_t->mutable_data<int32_t>(platform::CPUPlace());
    auto in_dims = in_t->dims();
    for (int i = 0; i < in_dims.size(); ++i) {
      out_data[i] = in_dims[i];
    }
  }
};
}  // namespace operators
}  // namespace paddle
