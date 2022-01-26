/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class SizeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("Input");
    auto* out_t = ctx.Output<Tensor>("Out");
    auto place = ctx.GetPlace();
    auto out_data = out_t->mutable_data<int64_t>(place);
    auto cpu_place = platform::CPUPlace();
    if (place == cpu_place) {
      out_data[0] = in_t->numel();
    } else {
      Tensor cpu_tensor;
      auto cpu_data =
          cpu_tensor.mutable_data<int64_t>(out_t->dims(), cpu_place);
      cpu_data[0] = in_t->numel();
      paddle::framework::TensorCopy(cpu_tensor, place, out_t);
    }
  }
};
}  // namespace operators
}  // namespace paddle
