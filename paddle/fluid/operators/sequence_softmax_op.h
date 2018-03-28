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
#include "paddle/fluid/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceSoftmaxKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto lod = x->lod();
    auto dims = x->dims();

    const size_t level = lod.size() - 1;
    PADDLE_ENFORCE_EQ(dims[0], static_cast<int64_t>(lod[level].back()),
                      "The first dimension of Input(X) should be equal to the "
                      "sum of all sequences' lengths.");
    PADDLE_ENFORCE_EQ(dims[0], x->numel(),
                      "The width of each timestep in Input(X) of "
                      "SequenceSoftmaxOp should be 1.");

    math::SequenceSoftmaxFunctor<DeviceContext, T> functor;
    functor(ctx.template device_context<DeviceContext>(), *x, out);
  }
};

template <typename DeviceContext, typename T>
class SequenceSoftmaxGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<LoDTensor>("Out");
    auto* out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    // auto* x = ctx.Input<LoDTensor>("X");
    auto* x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    x_grad->mutable_data<T>(ctx.GetPlace());
    math::SequenceSoftmaxGradFunctor<DeviceContext, T> functor;
    functor(ctx.template device_context<DeviceContext>(), *out, *out_grad,
            x_grad);
  }
};

}  // namespace operators
}  // namespace paddle
