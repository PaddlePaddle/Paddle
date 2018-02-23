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
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/maxouting.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class MaxOutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    int groups = context.template Attr<int>("groups");

    math::MaxOutFunctor<Place, T> maxout_forward;
    maxout_forward(context.device_context(), *in_x, out, groups);
  }
};

template <typename Place, typename T>
class MaxOutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    int groups = context.template Attr<int>("groups");
    auto& device_ctx = context.device_context();
    math::SetConstant<Place, T> zero;
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      zero(device_ctx, in_x_grad, static_cast<T>(0.0));
      math::MaxOutGradFunctor<Place, T> maxout_backward;
      maxout_backward(context.device_context(), *in_x, in_x_grad, *out,
                      *out_grad, groups);
    }
  }
};

}  // namespace operators
}  // namespace paddle
