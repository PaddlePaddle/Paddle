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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class RealKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* out = ctx.Output<framework::Tensor>("Out");

    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<math::Real<T>>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(math::Real<T>)));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::RealFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class RealGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<math::Real<T>>();
    auto* dx_data = d_x->mutable_data<T>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::RealToComplexFunctor<T> functor(dout_data, dx_data, numel);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
