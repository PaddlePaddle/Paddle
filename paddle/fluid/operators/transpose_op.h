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
#include "paddle/pten/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

enum { kTransposeMKLDNNFP32 = 1, kTransposeMKLDNNINT8 = 2 };

template <typename DeviceContext, typename T>
inline void TransCompute(const int dim, const DeviceContext& dev_ctx,
                         const framework::Tensor& in, framework::Tensor* out,
                         const std::vector<int>& axis) {
  switch (dim) {
    case 1:
      pten::funcs::Transpose<DeviceContext, T, 1> trans1;
      trans1(dev_ctx, in, out, axis);
      break;
    case 2:
      pten::funcs::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, axis);
      break;
    case 3:
      pten::funcs::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, axis);
      break;
    case 4:
      pten::funcs::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, axis);
      break;
    case 5:
      pten::funcs::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, axis);
      break;
    case 6:
      pten::funcs::Transpose<DeviceContext, T, 6> trans6;
      trans6(dev_ctx, in, out, axis);
      break;
    default:
      // for dim >= 7 situation
      pten::funcs::TransposeNormal<DeviceContext, T> trans_normal;
      trans_normal(dev_ctx, in, out, axis);
  }
}

template <typename DeviceContext, typename T>
class TransposeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.InputVar("X");
    auto* out = context.OutputVar("Out");

    const framework::Tensor* x_tensor =
        GetLoDTensorOrSelectedRowsValueFromVar(*x);
    framework::Tensor* out_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(out);

    out_tensor->mutable_data<T>(context.GetPlace());
    if (out_tensor->numel() == 0) {
      return;
    }

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    int ndims = axis.size();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    TransCompute<DeviceContext, T>(ndims, dev_ctx, *x_tensor, out_tensor, axis);
  }
};

template <typename DeviceContext, typename T>
class TransposeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_grad = context.InputVar(framework::GradVarName("Out"));
    auto* x_grad = context.OutputVar(framework::GradVarName("X"));

    if (!x_grad) {
      return;
    }
    const framework::Tensor* out_grad_tensor =
        GetLoDTensorOrSelectedRowsValueFromVar(*out_grad);
    framework::Tensor* x_grad_tensor =
        GetMutableLoDTensorOrSelectedRowsValueFromVar(x_grad);

    x_grad_tensor->mutable_data<T>(context.GetPlace());
    if (x_grad_tensor->numel() == 0) {
      return;
    }

    std::vector<int> axis = context.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);

    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    int ndims = axis.size();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    TransCompute<DeviceContext, T>(ndims, dev_ctx, *out_grad_tensor,
                                   x_grad_tensor, reversed_axis);
  }
};

}  // namespace operators
}  // namespace paddle
