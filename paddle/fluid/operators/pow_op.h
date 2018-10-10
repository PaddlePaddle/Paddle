/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct PowFunctor {
  void operator()(const DeviceContext &context, const framework::Tensor &x,
                  float factor, framework::Tensor *out) const;
};

template <typename DeviceContext, typename T>
struct PowGradFunctor {
  void operator()(const DeviceContext &context, const framework::Tensor &x,
                  const framework::Tensor &d_out, float factor,
                  framework::Tensor *d_x) const;
};

template <typename DeviceContext, typename T>
class PowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext &context) const {
    auto &x = detail::Ref(context.Input<framework::Tensor>("X"),
                          "Cannot get input tensor X, variable name = %s",
                          context.op().Input("X"));
    auto &out = detail::Ref(context.Output<framework::Tensor>("Out"),
                            "Cannot get output tensor Out, variable name = %s",
                            context.op().Output("Out"));

    float factor = context.Attr<float>("factor");
    out.mutable_data<T>(context.GetPlace());

    auto &dev_ctx = context.template device_context<DeviceContext>();

    PowFunctor<DeviceContext, T> pow_functor;
    pow_functor(dev_ctx, x, factor, &out);
  }
};

template <typename DeviceContext, typename T>
class PowGradKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext &context) const {
    auto &x = detail::Ref(context.Input<framework::Tensor>("X"),
                          "Cannot get input tensor X, variable name = %s",
                          context.op().Input("X"));
    auto &d_out = detail::Ref(
        context.Input<framework::Tensor>(framework::GradVarName("Out")),
        "Cannot get output tensor Out, variable name = %s",
        context.op().Input(framework::GradVarName("Out")));

    auto *d_x = context.Output<framework::Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(context.GetPlace());
    float factor = context.Attr<float>("factor");
    auto &dev_ctx = context.template device_context<DeviceContext>();

    PowGradFunctor<DeviceContext, T> pow_grad_functor;
    pow_grad_functor(dev_ctx, x, d_out, factor, d_x);
  }
};

}  // namespace operators
}  // namespace paddle
