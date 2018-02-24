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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class UnsqueezeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    auto out_dims = out->dims();

    out->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(*in, context.GetPlace(), context.device_context(),
                          out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class UnsqueezeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(context.GetPlace());

    auto in_dims = d_x->dims();
    framework::TensorCopy(*d_out, context.GetPlace(), context.device_context(),
                          d_x);
    d_x->Resize(in_dims);
  }
};

}  // namespace operators
}  // namespace paddle
