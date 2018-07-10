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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillConstantBatchSizeLikeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    if (in->lod().size() && ctx.Attr<int>("input_dim_idx") == 0) {
      // set the correct batch size for the LoDTensor.
      auto odims = out->dims();
      int output_dim_idx = ctx.Attr<int>("output_dim_idx");
      odims[output_dim_idx] = static_cast<int>(in->lod().back().size()) - 1;
      out->mutable_data<T>(odims, ctx.GetPlace());
    }
    out->mutable_data<T>(ctx.GetPlace());
    auto value = ctx.Attr<float>("value");

    math::SetConstant<DeviceContext, T> setter;
    setter(ctx.template device_context<DeviceContext>(), out,
           static_cast<T>(value));
  }
};

}  // namespace operators
}  // namespace paddle
