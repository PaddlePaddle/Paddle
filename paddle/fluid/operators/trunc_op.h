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

#include <math.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class TruncKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "Trunc_kernel_start\n";

    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    auto numel = x->numel();
    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    int i = 0;
    for (i = 0; i < numel; i++) {
      out_data[i] = x_data[i] - fmod(x_data[i], 1.0);
    }
  }
};

template <typename DeviceContext, typename T>
class TruncGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "Trunc_Grad_kernel_start\n";

    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    const T* dout_data = dout->data<T>();
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int numel = dx->numel();
    int i = 0;
    for (i = 0; i < numel; i++) {
      dx_data[i] = dout_data[i] * 0.0;
    }
  }
};

}  // namespace operators
}  // namespace paddle
