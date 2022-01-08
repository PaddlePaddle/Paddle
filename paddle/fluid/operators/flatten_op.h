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
#include "paddle/fluid/framework/pten_utils.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/pooling.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/kernels/empty_kernel.h"
#include "paddle/pten/kernels/flatten_grad_kernel.h"
#include "paddle/pten/kernels/flatten_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FlattenKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::Tensor>("X");
    auto *out = context.Output<framework::Tensor>("Out");

    auto &axes = context.Attr<int>("axis");
    auto x_dims = in->dims();
    auto out_dims = framework::make_ddim(GetOutputShape(axes, x_dims));

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }

  static std::vector<int32_t> GetOutputShape(const int axis,
                                             const framework::DDim &in_dims) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < in_dims.size(); ++i) {
      if (i < axis) {
        outer *= in_dims[i];
      } else {
        inner *= in_dims[i];
      }
    }
    std::vector<int32_t> out_shape(2);
    out_shape[0] = outer;
    out_shape[1] = inner;
    return out_shape;
  }
};

template <typename DeviceContext, typename T>
class FlattenGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto in_dims = ctx.Input<framework::Tensor>("X")->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(in_dims);
  }
};

template <typename DeviceContext, typename T>
class Flatten2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &axes = context.Attr<int>("axis");

    auto *in = context.Input<framework::Tensor>("X");
    auto x_dims = in->dims();

    auto *out = context.Output<framework::Tensor>("Out");

    auto out_dims = framework::make_ddim(
        FlattenKernel<DeviceContext, T>::GetOutputShape(axes, x_dims));

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class Flatten2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto xshape_dims = ctx.Input<framework::Tensor>("XShape")->dims();
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(x_dims);
  }
};

template <typename DeviceContext, typename T>
class FlattenContiguousRangeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::Tensor>("X");
    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data(context.GetPlace(), in->type());
    auto &start_axis = context.Attr<int>("start_axis");
    auto &stop_axis = context.Attr<int>("stop_axis");
    auto &dev_ctx = context.device_context<DeviceContext>();

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*in);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*out);

    // call new kernel
    pten::FlattenKernel<T, DeviceContext>(dev_ctx, *pt_x.get(), start_axis,
                                          stop_axis, pt_out.get());
  }
};

template <typename DeviceContext, typename T>
class FlattenContiguousRangeGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *xshape = ctx.Input<framework::Tensor>("XShape");

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    auto &dev_ctx = ctx.device_context<DeviceContext>();

    auto pt_d_x = paddle::experimental::MakePtenDenseTensor(*d_x);
    auto pt_d_out = paddle::experimental::MakePtenDenseTensor(*d_out);

    // Because the holder of xshape may be nullptr, we can't use
    // MakePtenDenseTensor.
    // So, we create a new DenseTensor to save the dims of xshape.
    pten::DenseTensorMeta xshape_meta{pten::TransToPtenDataType(d_x->type()),
                                      xshape->dims(), d_x->layout()};
    auto pt_xshape =
        pten::Empty<T, DeviceContext>(dev_ctx, std::move(xshape_meta));

    // call new kernel
    pten::FlattenGradKernel<T, DeviceContext>(dev_ctx, *pt_d_out.get(),
                                              pt_xshape, pt_d_x.get());
  }
};

}  // namespace operators
}  // namespace paddle
