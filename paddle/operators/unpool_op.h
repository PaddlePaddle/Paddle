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
#include "paddle/operators/math/unpooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class UnpoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* in_y = context.Input<Tensor>("Y");
    Tensor* out = context.Output<Tensor>("Out");
    std::string pooling_type = context.Attr<std::string>("unpooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    switch (ksize.size()) {
    case 2: {
      if (pooling_type == "max") {
        math::Unpool2d_MaxFunctor<Place, T> unpool2d_max_forward;
        unpool2d_max_forward(context.device_context(), *in_x, *in_y, out);
      }
    } break;
    default: { PADDLE_THROW("Pool op only supports 2D input."); }
    }
  }
};

template <typename Place, typename T>
class UnpoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* in_y = context.Input<Tensor>("Y");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    std::string pooling_type = context.Attr<std::string>("unpooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    auto& device_ctx = context.device_context();
    math::SetConstant<Place, T> zero;
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      zero(device_ctx, in_x_grad, static_cast<T>(0.0));
          }
    switch (ksize.size()) {
    case 2: {
    if (pooling_type == "max") {
      math::Unpool2d_MaxGradFunctor<Place, T> unpool2d_max_backward;
      unpool2d_max_backward(context.device_context(), *in_x, *in_y, in_x_grad,
                            *out, *out_grad);
      }
    } break;
    default: { PADDLE_THROW("Unpool op only supports 2D input."); }
    }
  }
};

}  // namespace operators
}  // namespace paddle
