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

#include "paddle/fluid/operators/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct AddFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a + b; }
};

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;

inline const Tensor* GetInputTensor(const framework::ExecutionContext& ctx,
                                    const std::string& var_name) {
  auto* var = ctx.InputVar(var_name);
  PADDLE_ENFORCE(var != nullptr, "InputVar(%s) should not be null", var_name);
  if (var->IsType<LoDTensor>()) {
    return ctx.Input<Tensor>(var_name);
  } else if (var->IsType<SelectedRows>()) {
    auto sr_data = ctx.Input<SelectedRows>(var_name);
    return &(sr_data->value());
  }
  PADDLE_THROW("Unsupported Variable(%s) Type of %s", var_name,
               var->Type().name());
  return nullptr;
}

inline Tensor* GetOutputTensor(const framework::ExecutionContext& ctx,
                               const std::string& var_name) {
  auto* var = ctx.OutputVar(var_name);
  if (var == nullptr) return nullptr;
  if (var->IsType<LoDTensor>()) {
    return ctx.Output<Tensor>(var_name);
  } else if (var->IsType<SelectedRows>()) {
    return const_cast<Tensor*>(&(ctx.Output<SelectedRows>(var_name)->value()));
  }
  PADDLE_THROW("Unsupported Variable(%s) Type of %s", var_name,
               var->Type().name());
  return nullptr;
}

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int axis = ctx.Attr<int>("axis");
    auto* x = GetInputTensor(ctx, "X");
    auto* y = GetInputTensor(ctx, "Y");
    auto* out = GetOutputTensor(ctx, "Out");

    out->mutable_data<T>(ctx.GetPlace());
    ElementwiseComputeEx<AddFunctor<T>, DeviceContext, T>(ctx, x, y, axis,
                                                          AddFunctor<T>(), out);
  }
};

template <typename T>
struct IdentityGrad {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const { return dout; }
};

template <typename DeviceContext, typename T>
class ElementwiseAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = GetInputTensor(ctx, "X");
    auto* y = GetInputTensor(ctx, "Y");
    auto* out = GetInputTensor(ctx, "Out");
    auto* dout = GetInputTensor(ctx, framework::GradVarName("Out"));
    auto* dx = GetOutputTensor(ctx, framework::GradVarName("X"));
    auto* dy = GetOutputTensor(ctx, framework::GradVarName("Y"));

    int axis = ctx.Attr<int>("axis");
    ElemwiseGradCompute<DeviceContext, T, IdentityGrad<T>, IdentityGrad<T>>(
        ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
        IdentityGrad<T>());
  }
};

}  // namespace operators
}  // namespace paddle
