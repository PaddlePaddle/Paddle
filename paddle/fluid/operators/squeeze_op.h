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
                               const framework::DDim &in_dims,
                               bool is_runtime) {
  size_t num_squeeze_dims = squeeze_dims.size();
  std::vector<bool> should_squeeze(in_dims.size(), false);

  // Mark dimensions need to be squeezed.
  if (num_squeeze_dims == 0) {
    for (int i = 0; i < in_dims.size(); ++i) {
      if (in_dims[i] == 1) {
        should_squeeze[i] = true;
      }
    }
  } else {
    for (size_t i = 0; i < num_squeeze_dims; ++i) {
      int current = squeeze_dims[i] < 0 ? squeeze_dims[i] + in_dims.size()
                                        : squeeze_dims[i];

      PADDLE_ENFORCE_GE(
          current,
          0,
          platform::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));
      PADDLE_ENFORCE_LT(
          current,
          in_dims.size(),
          platform::errors::InvalidArgument(
              "Each axis in Attr(axes) should be in the range of [%d, %d]"
              "But current axis is:%d, input tensor's shape = [%s].",
              -in_dims.size(),
              in_dims.size() - 1,
              current,
              in_dims));

      if (!should_squeeze[current]) {
        if (is_runtime) {
          // At run time, dim of 1 is allowed to squeeze
          if (in_dims[current] == 1) {
            should_squeeze[current] = true;
          }
        } else {
          // At compile time, dim of -1 or 1 is allowed to squeeze
          if (in_dims[current] == 1 || in_dims[current] == -1) {
            should_squeeze[current] = true;
          }
        }
      }
    }
  }
  // Make output dimensions
  std::vector<int64_t> output_shape;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (!should_squeeze[i]) {
      output_shape.push_back(in_dims[i]);
    }
  }
  return common::make_ddim(output_shape);
}

template <typename DeviceContext, typename T>
class Squeeze2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *out = context.Output<phi::DenseTensor>("Out");
    auto *in = context.Input<phi::DenseTensor>("X");

    auto &axes = context.Attr<std::vector<int>>("axes");

    auto x_dims = in->dims();
    auto out_dims = GetOutputShape(axes, x_dims, true);

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in,
        context.GetPlace(),
        context.template device_context<platform::DeviceContext>(),
        out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class Squeeze2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_out = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    // auto in_dims = d_x->dims();

    auto xshape_dims = ctx.Input<phi::DenseTensor>("XShape")->dims();
    auto x_dims = common::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopySync(*d_out, ctx.GetPlace(), d_x);
    d_x->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle
