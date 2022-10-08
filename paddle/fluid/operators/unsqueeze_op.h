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
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class UnsqueezeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto axes = context.Attr<std::vector<int>>("axes");
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");
    auto x_dims = in->dims();

    bool need_resize_out_dims = false;
    if (axes.empty()) {
      auto axes_tensor_list =
          context.MultiInput<phi::DenseTensor>("AxesTensorList");
      if (axes_tensor_list.size() > 0) {
        axes = GetDataFromTensorList<int>(axes_tensor_list);
      } else if (context.HasInput("AxesTensor")) {
        auto *axes_tensor = context.Input<phi::DenseTensor>("AxesTensor");
        axes = GetDataFromTensor<int>(axes_tensor);
      }
      need_resize_out_dims = true;
    }
    framework::DDim out_dims = out->dims();
    if (need_resize_out_dims) {
      out_dims = GetOutputShape(axes, x_dims);
      out->Resize(out_dims);
    }
    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in,
        context.GetPlace(),
        context.template device_context<platform::DeviceContext>(),
        out);
    out->Resize(out_dims);
  }

  static framework::DDim GetOutputShape(const std::vector<int> unsqz_dims,
                                        const framework::DDim &in_dims) {
    int output_size = in_dims.size() + static_cast<int>(unsqz_dims.size());
    int cur_output_size = in_dims.size();
    std::vector<int64_t> output_shape(output_size, 0);

    // Validity Check: rank range.
    PADDLE_ENFORCE_LE(output_size,
                      6,
                      platform::errors::InvalidArgument(
                          "The output "
                          "tensor's rank should be less than 6."));

    for (int axis : unsqz_dims) {
      int cur = axis < 0 ? axis + cur_output_size + 1 : axis;
      // Vaildity Check: the axis bound
      PADDLE_ENFORCE_GE(
          cur,
          0,
          platform::errors::InvalidArgument("The insert dimension value should "
                                            "not be less than 0"));
      PADDLE_ENFORCE_LE(cur,
                        cur_output_size,
                        platform::errors::InvalidArgument(
                            "The insert dimension value shoule not be larger "
                            "than the dimension size of input tensor"));
      // Move old axis, and insert new axis
      for (int i = cur_output_size; i >= cur; --i) {
        if (output_shape[i] == 1) {
          // Move axis
          output_shape[i + 1] = 1;
          output_shape[i] = 0;
        }
      }
      output_shape[cur] = 1;
      // Add the output size.
      cur_output_size++;
    }

    // Make output shape
    for (int in_idx = 0, out_idx = 0; out_idx < output_size; ++out_idx) {
      if (output_shape[out_idx] == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      }
    }

    return phi::make_ddim(output_shape);
  }
};

template <typename DeviceContext, typename T>
class UnsqueezeGradKernel : public framework::OpKernel<T> {
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
class Unsqueeze2GradKernel : public framework::OpKernel<T> {
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
