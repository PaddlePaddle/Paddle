// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename T>
class Flatten2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<framework::LoDTensor>("X");
    auto *out = context.Output<framework::LoDTensor>("Out");
    auto &axis = context.Attr<int>("axis");
    out->mutable_data(context.GetPlace(), in->type());
    framework::NPUAttributeMap attr_input = {{"axis", axis}};

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto &runner = NpuOpRunner("FlattenV2", {*in}, {*out}, attr_input);
    runner.Run(stream);
  }
};

template <typename T>
class Flatten2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out,
        ctx.GetPlace(),
        ctx.template device_context<paddle::platform::NPUDeviceContext>(),
        d_x);
    d_x->Resize(x_dims);
  }
};

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class FlattenContiguousRangeNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<phi::DenseTensor>("X");
    auto *Out = ctx.Output<phi::DenseTensor>("Out");
    int start_axis = ctx.Attr<int>("start_axis");
    int stop_axis = ctx.Attr<int>("stop_axis");

    Out->mutable_data<T>(ctx.GetPlace());

    const auto &runner =
        NpuOpRunner("FlattenV2",
                    {*X},
                    {*Out},
                    {{"axis", static_cast<int32_t>(start_axis)},
                     {"end_axis", static_cast<int32_t>(stop_axis)}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class FlattenContiguousRangeGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *d_x = ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *d_out =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));

    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    auto x_dims = phi::slice_ddim(xshape_dims, 1, xshape_dims.size());

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    framework::TensorCopy(
        *d_out,
        ctx.GetPlace(),
        ctx.template device_context<paddle::platform::NPUDeviceContext>(),
        d_x);
    d_x->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(flatten2,
                       ops::Flatten2NPUKernel<float>,
                       ops::Flatten2NPUKernel<double>,
                       ops::Flatten2NPUKernel<uint8_t>,
                       ops::Flatten2NPUKernel<int>,
                       ops::Flatten2NPUKernel<int8_t>,
                       ops::Flatten2NPUKernel<int64_t>);
REGISTER_OP_NPU_KERNEL(flatten2_grad,
                       ops::Flatten2GradNPUKernel<float>,
                       ops::Flatten2GradNPUKernel<double>,
                       ops::Flatten2GradNPUKernel<uint8_t>,
                       ops::Flatten2GradNPUKernel<int>,
                       ops::Flatten2GradNPUKernel<int8_t>,
                       ops::Flatten2GradNPUKernel<int64_t>);

REGISTER_OP_NPU_KERNEL(
    flatten_contiguous_range,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         float>,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         double>,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         uint8_t>,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         int>,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         int8_t>,
    ops::FlattenContiguousRangeNPUKernel<paddle::platform::NPUDeviceContext,
                                         int64_t>);
REGISTER_OP_NPU_KERNEL(
    flatten_contiguous_range_grad,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             float>,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             double>,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             uint8_t>,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             int>,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             int8_t>,
    ops::FlattenContiguousRangeGradNPUKernel<paddle::platform::NPUDeviceContext,
                                             int64_t>);
