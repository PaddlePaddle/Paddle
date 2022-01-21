// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/pten/kernels/masked_select_grad_kernel.h"
#include "paddle/pten/kernels/masked_select_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename T>
class MaskedSelectKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto input = context.Input<framework::Tensor>("X");
    auto mask = context.Input<framework::Tensor>("Mask");
    auto out = context.Output<framework::Tensor>("Y");
    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::MaskedSelectKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, *mask, out);
  }
};

template <typename DeviceContext, typename T>
class MaskedSelectGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto out = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto mask = context.Input<framework::Tensor>("Mask");
    auto x = context.Input<framework::Tensor>("X");
    auto input = context.Input<framework::Tensor>(framework::GradVarName("Y"));

    auto& dev_ctx = context.device_context<DeviceContext>();
    pten::MaskedSelectGradKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE&>(dev_ctx),
        *input, *x, *mask, out);
  }
};

}  // namespace operators
}  // namespace paddle
