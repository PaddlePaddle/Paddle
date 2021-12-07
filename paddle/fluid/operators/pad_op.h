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

#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/padding.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class PadKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto pads = context.Attr<std::vector<int>>("paddings");
    float pad_value = context.Attr<float>("pad_value");
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    int rank = x->dims().size();
    math::PaddingFunctor<DeviceContext, T>(rank, context, pads,
                                           static_cast<T>(pad_value), *x, out);
  }
};

template <typename DeviceContext, typename T>
class PadGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto pads = context.Attr<std::vector<int>>("paddings");
    auto* d_out = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = context.Output<Tensor>(framework::GradVarName("X"));
    if (d_x == nullptr) {
      return;
    }

    d_x->mutable_data<T>(context.GetPlace());
    int rank = d_out->dims().size();
    math::PaddingGradFunctor<DeviceContext, T>(rank, context, pads, *d_out,
                                               d_x);
  }
};

}  // namespace operators
}  // namespace paddle
