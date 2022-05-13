// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/flatten_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FlattenMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");

    auto &axes = context.Attr<int>("axis");
    auto x_dims = in->dims();
    auto out_dims = phi::make_ddim(GetOutputShape(axes, x_dims));
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
class FlattenGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto in_dims = ctx.Input<framework::LoDTensor>("X")->dims();

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::MLUDeviceContext>(), d_x);
    d_x->Resize(in_dims);
  }
};

template <typename DeviceContext, typename T>
class Flatten2MLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &axes = context.Attr<int>("axis");

    auto *in = context.Input<framework::LoDTensor>("X");
    auto x_dims = in->dims();

    auto *out = context.Output<framework::LoDTensor>("Out");

    auto out_dims = phi::make_ddim(
        FlattenMLUKernel<DeviceContext, T>::GetOutputShape(axes, x_dims));

    out->mutable_data(context.GetPlace(), in->type());
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
};

template <typename DeviceContext, typename T>
class Flatten2GradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<platform::DeviceContext>(), d_x);
    d_x->Resize(x_dims);
  }
};

template <typename DeviceContext, typename T>
class FlattenContiguousRangeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");
    out->mutable_data(context.GetPlace(), in->type());
    auto &start_axis = context.Attr<int>("start_axis");
    auto &stop_axis = context.Attr<int>("stop_axis");

    // make out dims
    auto in_dims = in->dims();
    auto out_dims =
        phi::make_ddim(GetOutputShape(start_axis, stop_axis, in_dims));
    framework::TensorCopy(
        *in, context.GetPlace(),
        context.template device_context<platform::DeviceContext>(), out);
    out->Resize(out_dims);
  }
  static std::vector<int32_t> GetOutputShape(const int start_axis,
                                             const int stop_axis,
                                             const framework::DDim &in_dims) {
    int64_t outer = 1;
    std::vector<int32_t> out_shape;
    int in_dims_size = in_dims.size();
    out_shape.reserve(in_dims_size - stop_axis + start_axis);
    int real_start_axis = start_axis, real_stop_axis = stop_axis;
    if (start_axis < 0) {
      real_start_axis = start_axis + in_dims_size;
    }
    if (stop_axis < 0) {
      real_stop_axis = stop_axis + in_dims_size;
    }

    for (int i = 0; i < real_start_axis; ++i) {
      out_shape.push_back(in_dims[i]);
    }
    for (int i = real_start_axis; i <= real_stop_axis; i++) {
      if (in_dims[i] == -1 || outer == -1) {
        outer = -1;
      } else {
        outer *= in_dims[i];
      }
    }
    out_shape.push_back(outer);
    for (int i = real_stop_axis + 1; i < in_dims_size; i++) {
      out_shape.push_back(in_dims[i]);
    }

    return out_shape;
  }
};

template <typename DeviceContext, typename T>
class FlattenContiguousRangeGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out, ctx.GetPlace(),
        ctx.template device_context<paddle::platform::MLUDeviceContext>(), d_x);
    d_x->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(
    flatten, ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, int8_t>,
    ops::FlattenMLUKernel<paddle::platform::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    flatten_grad,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, int8_t>,
    ops::FlattenGradMLUKernel<paddle::platform::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    flatten2, ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, int8_t>,
    ops::Flatten2MLUKernel<paddle::platform::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    flatten2_grad,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, float>,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, double>,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, uint8_t>,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, int>,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, int8_t>,
    ops::Flatten2GradMLUKernel<paddle::platform::MLUDeviceContext, int64_t>);
REGISTER_OP_MLU_KERNEL(
    flatten_contiguous_range,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         float>,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         double>,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         uint8_t>,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         int>,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         int8_t>,
    ops::FlattenContiguousRangeMLUKernel<paddle::platform::MLUDeviceContext,
                                         int64_t>);
REGISTER_OP_MLU_KERNEL(
    flatten_contiguous_range_grad,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             float>,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             double>,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             uint8_t>,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             int>,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             int8_t>,
    ops::FlattenContiguousRangeGradMLUKernel<paddle::platform::MLUDeviceContext,
                                             int64_t>);
