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
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class SqueezeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("X");

    framework::DDim out_dims = out->dims();

    bool inplace = ctx.Attr<bool>("inplace");
    out->Resize(out_dims);
    if (!inplace) {
      out->mutable_data<T>(ctx.GetPlace());
      framework::TensorCopySync(*in, ctx.GetPlace(), out);
      out->Resize(out_dims);
    } else {
      out->ShareDataWith(*in);
      out->Resize(out_dims);
    }
  }
};

template <typename DeviceContext, typename T>
class SqueezeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    d_x->mutable_data<T>(ctx.GetPlace());
    bool inplace = ctx.Attr<bool>("inplace");

    auto in_dims = d_x->dims();
    if (!inplace) {
      framework::TensorCopy(*d_out, ctx.GetPlace(), ctx.device_context(), d_x);
      ctx.device_context().Wait();
      d_x->Resize(in_dims);
    } else {
      d_x->ShareDataWith(*d_out);
      d_x->Resize(in_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle
