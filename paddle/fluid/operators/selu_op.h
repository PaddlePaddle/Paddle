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
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"
namespace paddle {
namespace operators {

template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename DeviceContext, typename T>
class SeluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    float alpha = context.Attr<float>("alpha");
    float scale = context.Attr<float>("scale");

    auto out_ptr = out->mutable_data<T>(context.GetPlace());

    ConstEigenVectorArrayMap<T> x_e(x->data<T>(), x->numel());
    EigenVectorArrayMap<T> out_e(out_ptr, out->numel());

    out_e = scale * (x_e > 0).select(x_e, (alpha * x_e.exp() - alpha));
  }
};

template <typename DeviceContext, typename T>
class SeluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = framework::Tensor;

    auto* out = context.Input<Tensor>("Out");
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    float alpha = context.Attr<float>("alpha");
    float scale = context.Attr<float>("scale");

    auto dx_ptr = dx->mutable_data<T>(context.GetPlace());

    ConstEigenVectorArrayMap<T> out_e(out->data<T>(), out->numel());
    ConstEigenVectorArrayMap<T> dout_e(dout->data<T>(), dout->numel());
    EigenVectorArrayMap<T> dx_e(dx_ptr, dx->numel());

    const float la = scale * alpha;
    dx_e = (out_e > 0).select(scale * dout_e, dout_e * (out_e + la));
  }
};

}  // namespace operators
}  // namespace paddle
