/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                               const framework::DDim &in_dims, bool is_runtime);

template <typename DeviceContext, typename T>
class SqueezeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");

    auto &axes = context.Attr<std::vector<int>>("axes");
    auto x_dims = in->dims();
    auto out_dims = GetOutputShape(axes, x_dims, true);

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class SqueezeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto in_dims = ctx.Input<framework::LoDTensor>("X")->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopySync(*d_out, ctx.GetPlace(), d_x);
    d_x->Resize(in_dims);
  }
};

template <typename DeviceContext, typename T>
class Squeeze2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<framework::LoDTensor>("Out");
    auto *in = context.Input<framework::LoDTensor>("X");

    auto &axes = context.Attr<std::vector<int>>("axes");

    auto x_dims = in->dims();
    auto out_dims = GetOutputShape(axes, x_dims, true);

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class Squeeze2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    // auto in_dims = d_x->dims();

    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopySync(*d_out, ctx.GetPlace(), d_x);
    d_x->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle
