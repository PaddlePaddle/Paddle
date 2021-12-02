/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/norm_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using DDim = framework::DDim;
using Tensor = framework::Tensor;

void CheckAxis(int axis, int rank) {
  // check the axis is in [-rank, rank-1]
  if (axis <= rank - 1 && axis >= -rank) return;
  PADDLE_THROW(platform::errors::InvalidArgument(
      "axis in norm operator must between (%d) and (%d)"
      "but got (%d).",
      -rank, rank - 1, axis));
}

template <typename DeviceContext, typename T>
class NormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(4) << "Launch Norm Op Kernel on NPU." << std::endl;
    auto *in_x = ctx.Input<framework::Tensor>("X");
    auto *out_y = ctx.Output<framework::Tensor>("Out");
    auto *out_norm = ctx.Output<framework::Tensor>("Norm");
    out_y->mutable_data<T>(ctx.GetPlace());
    out_norm->mutable_data<T>(ctx.GetPlace());
    auto xdim = in_x->dims();
    float eps = ctx.Attr<float>("epsilon");
    int axis = ctx.Attr<int>("axis");
    CheckAxis(axis, xdim.size());
    if (axis < 0) axis = xdim.size() + axis;

    framework::NPUAttributeMap attr_input_norm;
    attr_input_norm["axes"] = std::vector<int>({axis});
    attr_input_norm["p"] = 2;
    attr_input_norm["keepdim"] = true;
    attr_input_norm["epsilon"] = eps;
    const auto &runner =
        NpuOpRunner("LpNorm", {*in_x}, {*out_norm}, attr_input_norm);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
    NpuOpRunner("Div", {*in_x, *out_norm}, {*out_y}, {}).Run(stream);
  }
};

template <typename DeviceContext, typename T>
class NormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    float epsilon = ctx.Attr<float>("epsilon");
    int axis = ctx.Attr<int>("axis");

    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Input<framework::Tensor>("Out");
    auto *dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto xdim = x->dims();
    CheckAxis(axis, xdim.size());

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    framework::NPUAttributeMap attr_input_norm;
    attr_input_norm["dim"] = std::vector<int>({axis});
    attr_input_norm["eps"] = epsilon;
    const auto &runner =
        NpuOpRunner("L2NormalizeGrad", {*x, *y, *dy}, {*dx}, attr_input_norm);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    norm, ops::NormNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::NormNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>)

REGISTER_OP_NPU_KERNEL(
    norm_grad, ops::NormGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::NormGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
