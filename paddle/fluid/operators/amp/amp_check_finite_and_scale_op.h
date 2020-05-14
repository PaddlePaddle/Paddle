/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/operators/isfinite_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AmpCheckFiniteAndScaleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // const auto* x = ctx.Input<framework::Tensor>("X");
    // const auto* scale = ctx.Input<framework::Tensor>("Scale");
    // auto* out = ctx.Output<framework::Tensor>("Out");
    // auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    // const T* x_data = x->data<T>();
    // const T* scale_data = scale->data<T>();
    // T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
    // bool* found_inf_data = found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    return;
  }
};

}  // namespace operators
}  // namespace paddle
