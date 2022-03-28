// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "paddle/phi/kernels/funcs/complex_functors.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class AngleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<phi::dtype::Real<T>>(
        context.GetPlace(), size_t(x->numel() * sizeof(phi::dtype::Real<T>)));

    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    phi::funcs::AngleFunctor<T> functor(x_data, out_data, numel);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class AngleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    auto* dout_data = d_out->data<phi::dtype::Real<T>>();
    auto* x_data = x->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    phi::funcs::AngleGradFunctor<T> functor(dout_data, x_data, dx_data, numel);
    for_range(functor);
  }
};
}  // namespace operators
}  // namespace paddle
