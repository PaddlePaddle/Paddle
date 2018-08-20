/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sequence_padding.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
class SequencePadOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    const auto* pad_value = ctx.Input<LoDTensor>("PadValue");

    int padded_length = ctx.Attr<int>("padded_length");

    math::PaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *x, out, *pad_value,
        padded_length, 0, false, math::kBatchLengthWidth);
  }
};

template <typename DeviceContext, typename T>
class SequencePadGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    if (d_x) {
      const auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
      d_x->mutable_data<T>(ctx.GetPlace());

      int padded_length = ctx.Attr<int>("padded_length");

      math::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(), *d_out, d_x,
          padded_length, 0, false, math::kBatchLengthWidth);
    }
  }
};

}  // namespace operators
}  // namespace paddle
