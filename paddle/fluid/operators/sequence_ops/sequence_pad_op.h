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
#include "paddle/fluid/operators/math/sequence_padding.h"
#include "paddle/pten/kernels/funcs/math_function.h"

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
    auto* len_t = ctx.Output<LoDTensor>("Length");
    out->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_EQ(
        x->lod().empty(), false,
        platform::errors::NotFound("Input(X) Tensor of SequencePadOp does not "
                                   "contain LoD information."));

    const auto* pad_value = ctx.Input<LoDTensor>("PadValue");

    int padded_length = ctx.Attr<int>("padded_length");

    math::PaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), *x, out, *pad_value,
        padded_length, 0, false, math::kBatchLengthWidth);

    LoDTensor seq_len;
    seq_len.Resize(len_t->dims());
    int64_t* len_data = seq_len.mutable_data<int64_t>(platform::CPUPlace());
    for (size_t i = 1; i < x->lod()[0].size(); ++i) {
      len_data[i - 1] = x->lod()[0][i] - x->lod()[0][i - 1];
    }
    framework::TensorCopy(seq_len, ctx.GetPlace(),
                          ctx.template device_context<DeviceContext>(), len_t);
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
