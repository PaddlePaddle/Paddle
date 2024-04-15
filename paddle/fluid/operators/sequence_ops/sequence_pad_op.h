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
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sequence_padding.h"

namespace paddle {
namespace operators {

using LoD = framework::LoD;
template <typename T, typename DeviceContext>
class SequencePadOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto* len_t = ctx.Output<phi::DenseTensor>("Length");
    out->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_EQ(x->lod().empty(),
                      false,
                      phi::errors::NotFound(
                          "Input(X) phi::DenseTensor of SequencePadOp does not "
                          "contain LoD information."));

    const auto* pad_value = ctx.Input<phi::DenseTensor>("PadValue");

    int padded_length = ctx.Attr<int>("padded_length");

    phi::funcs::PaddingLoDTensorFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(),
        *x,
        out,
        *pad_value,
        padded_length,
        0,
        false,
        phi::funcs::kBatchLengthWidth);

    phi::DenseTensor seq_len;
    seq_len.Resize(len_t->dims());
    int64_t* len_data = seq_len.mutable_data<int64_t>(platform::CPUPlace());
    for (size_t i = 1; i < x->lod()[0].size(); ++i) {
      len_data[i - 1] = x->lod()[0][i] - x->lod()[0][i - 1];
    }
    framework::TensorCopy(seq_len,
                          ctx.GetPlace(),
                          ctx.template device_context<DeviceContext>(),
                          len_t);
  }
};

template <typename T, typename DeviceContext>
class SequencePadGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    if (d_x) {
      const auto* d_out =
          ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
      d_x->mutable_data<T>(ctx.GetPlace());

      int padded_length = ctx.Attr<int>("padded_length");

      phi::funcs::UnpaddingLoDTensorFunctor<DeviceContext, T>()(
          ctx.template device_context<DeviceContext>(),
          *d_out,
          d_x,
          padded_length,
          0,
          false,
          phi::funcs::kBatchLengthWidth);
    }
  }
};

}  // namespace operators
}  // namespace paddle
