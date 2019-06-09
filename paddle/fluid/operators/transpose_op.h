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

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
inline void TransCompute(const int dim, const DeviceContext& dev_ctx,
                         const framework::Tensor& in, framework::Tensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      math::Transpose<DeviceContext, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      math::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      math::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      math::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      math::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      math::Transpose<DeviceContext, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      PADDLE_THROW("Tensors with rank at most 6 are supported");
  }
}

template <typename DeviceContext, typename T>
class TransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    TransCompute<DeviceContext, T>(ndims, dev_ctx, *x, out, axis);
  }
};

template <typename DeviceContext, typename T>
class TransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    if (!x_grad) return;

    x_grad->mutable_data<T>(context.GetPlace());
    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    int ndims = axis.size();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    TransCompute<DeviceContext, T>(ndims, dev_ctx, *out_grad, x_grad,
                                   reversed_axis);
  }
};

}  // namespace operators
}  // namespace paddle
