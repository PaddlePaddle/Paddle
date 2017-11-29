/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/unpooling.h"

namespace paddle {
namespace operators {
template <typename Place, typename T>
class UnpoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    const framework::Tensor* in_y = context.Input<framework::Tensor>("Indices");
    auto* out = context.Output<framework::Tensor>("Out");
    std::string unpooling_type = context.Attr<std::string>("unpooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    T* output_data = out->mutable_data<T>(context.GetPlace());
    if (output_data) {
      math::SetConstant<Place, T> set_zero;
      set_zero(context.device_context(), out, static_cast<T>(0));
    }
    math::Unpool2dMaxFunctor<Place, T> unpool2d_max_forward;
    unpool2d_max_forward(context.device_context(), *in_x, *in_y, out);
  }
};
template <typename Place, typename T>
class UnpoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = context.Input<framework::Tensor>("X");
    const framework::Tensor* in_y = context.Input<framework::Tensor>("Indices");
    const framework::Tensor* out = context.Input<framework::Tensor>("Out");
    const framework::Tensor* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* in_x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    std::string unpooling_type = context.Attr<std::string>("unpooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    auto& device_ctx = context.device_context();
    math::SetConstant<Place, T> zero;
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      zero(device_ctx, in_x_grad, static_cast<T>(0));
    }
    math::Unpool2dMaxGradFunctor<Place, T> unpool2d_max_backward;
    unpool2d_max_backward(context.device_context(), *in_x, *in_y, *out,
                          *out_grad, in_x_grad);
  }
};
}  // namespace operators
}  // namespace paddle
