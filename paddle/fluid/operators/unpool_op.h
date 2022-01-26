/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/unpooling.h"

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
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
    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (output_data) {
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(dev_ctx, out, static_cast<T>(0));
    }
    math::Unpool2dMaxFunctor<DeviceContext, T> unpool2d_max_forward;
    unpool2d_max_forward(dev_ctx, *in_x, *in_y, out);
  }
};
template <typename DeviceContext, typename T>
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

    auto& device_ctx = context.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> zero;

    in_x_grad->mutable_data<T>(context.GetPlace());
    zero(device_ctx, in_x_grad, static_cast<T>(0));

    math::Unpool2dMaxGradFunctor<DeviceContext, T> unpool2d_max_backward;
    unpool2d_max_backward(device_ctx, *in_x, *in_y, *out, *out_grad, in_x_grad);
  }
};

template <typename DeviceContext, typename T>
class Unpool3dKernel : public framework::OpKernel<T> {
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
    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (output_data) {
      math::SetConstant<DeviceContext, T> set_zero;
      set_zero(dev_ctx, out, static_cast<T>(0));
    }
    math::Unpool3dMaxFunctor<DeviceContext, T> unpool3d_max_forward;
    unpool3d_max_forward(dev_ctx, *in_x, *in_y, out);
  }
};

template <typename DeviceContext, typename T>
class Unpool3dGradKernel : public framework::OpKernel<T> {
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

    auto& device_ctx = context.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> zero;

    in_x_grad->mutable_data<T>(context.GetPlace());
    zero(device_ctx, in_x_grad, static_cast<T>(0));

    math::Unpool3dMaxGradFunctor<DeviceContext, T> unpool3d_max_backward;
    unpool3d_max_backward(device_ctx, *in_x, *in_y, *out, *out_grad, in_x_grad);
  }
};
}  // namespace operators
}  // namespace paddle
