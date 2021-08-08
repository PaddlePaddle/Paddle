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

#include "paddle/fluid/operators/p_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PnormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_norm = ctx.Output<framework::Tensor>("Out");
    out_norm->mutable_data<T>(ctx.GetPlace());

    float porder = ctx.Attr<float>("porder");
    int axis = ctx.Attr<int>("axis");
    bool keepdim = ctx.Attr<bool>("keepdim");

    auto xdim = in_x->dims();
    if (axis < 0) axis = xdim.size() + axis;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    int p = 0;
    bool combine_op =
        !(porder == 0 || porder == INFINITY || porder == -INFINITY);
    if (porder == INFINITY) {
      p = INT_MAX;
    } else if (porder == -INFINITY) {
      p = INT_MIN;
    } else {
      p = static_cast<int>(porder);
    }

    if (!combine_op) {
      const auto& runner = NpuOpRunner("LpNorm", {*in_x}, {*out_norm},
                                       {{"p", p},
                                        {"axes", std::vector<int32_t>({axis})},
                                        {"keep_dims", keepdim}});
      runner.Run(stream);
    } else {
      Tensor tmp_x;
      tmp_x.mutable_data<T>(xdim, ctx.GetPlace());

      const auto& power_runner1 =
          NpuOpRunner("Power", {*in_x}, {tmp_x},
                      {{"power", porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner1.Run(stream);

      const auto& reduce_runner = NpuOpRunner(
          "ReduceSumD", {tmp_x}, {*out_norm},
          {{"axes", std::vector<int32_t>({axis})}, {"keep_dims", keepdim}});
      reduce_runner.Run(stream);

      const auto& power_runner2 = NpuOpRunner(
          "Power", {*out_norm}, {*out_norm},
          {{"power", 1 / porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner2.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class PnormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* in_norm = ctx.Input<framework::Tensor>("Out");
    auto* in_norm_dy =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out_dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    out_dx->mutable_data<T>(ctx.GetPlace());

    float eps = ctx.Attr<float>("epsilon");
    auto xdim = in_x->dims();
    float porder = ctx.Attr<float>("porder");

    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;

    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    if (porder == 0) {
      const auto& runner = NpuOpRunner(
          "FillV2D", {}, {*out_dx},
          {{"value", 0.0f}, {"dims", framework::vectorize<int32_t>(xdim)}});
      runner.Run(stream);
    } else {
      auto xdim_vec = framework::vectorize<int32_t>(xdim);
      xdim_vec[axis] = 1;
      Tensor tmp_in_norm, tmp_in_norm_dy;
      tmp_in_norm.ShareDataWith(*in_norm);
      tmp_in_norm_dy.ShareDataWith(*in_norm_dy);
      tmp_in_norm.Resize(framework::make_ddim(xdim_vec));
      tmp_in_norm_dy.Resize(framework::make_ddim(xdim_vec));

      Tensor b_in_norm, b_in_norm_dy, x_sign, x_abs;
      b_in_norm.mutable_data<T>(xdim, ctx.GetPlace());
      b_in_norm_dy.mutable_data<T>(xdim, ctx.GetPlace());
      x_sign.mutable_data<T>(xdim, ctx.GetPlace());
      x_abs.mutable_data<T>(xdim, ctx.GetPlace());

      const auto& tile_runner1 = NpuOpRunner(
          "TileWithAxis", {tmp_in_norm}, {b_in_norm},
          {{"axis", axis}, {"tiles", static_cast<int32_t>(xdim[axis])}});
      tile_runner1.Run(stream);
      const auto& tile_runner2 = NpuOpRunner(
          "TileWithAxis", {tmp_in_norm_dy}, {b_in_norm_dy},
          {{"axis", axis}, {"tiles", static_cast<int32_t>(xdim[axis])}});
      tile_runner2.Run(stream);
      const auto& sign_runner = NpuOpRunner("Sign", {*in_x}, {x_sign}, {});
      sign_runner.Run(stream);
      const auto& abs_runner = NpuOpRunner("Abs", {*in_x}, {x_abs}, {});
      abs_runner.Run(stream);

      if (porder == INFINITY || porder == -INFINITY) {
        Tensor mask;
        mask.mutable_data<uint8_t>(xdim, ctx.GetPlace());

        const auto& equal_runner =
            NpuOpRunner("Equal", {x_abs, b_in_norm}, {mask}, {});
        equal_runner.Run(stream);
        const auto& cast_runner =
            NpuOpRunner("Cast", {mask}, {x_abs},
                        {{"dst_type", ConvertToNpuDtype(x_abs.type())}});
        cast_runner.Run(stream);
        const auto& mul_runner1 =
            NpuOpRunner("Mul", {x_abs, b_in_norm_dy}, {x_abs}, {});
        mul_runner1.Run(stream);
        const auto& mul_runner2 =
            NpuOpRunner("Mul", {x_abs, x_sign}, {*out_dx}, {});
        mul_runner2.Run(stream);
      } else {
        const auto& power_runner1 = NpuOpRunner(
            "Power", {x_abs}, {x_abs},
            {{"power", porder - 1.0f}, {"scale", 1.0f}, {"shift", 0.0f}});
        power_runner1.Run(stream);
        const auto& power_runner2 = NpuOpRunner(
            "Power", {b_in_norm}, {b_in_norm},
            {{"power", porder - 1.0f}, {"scale", 1.0f}, {"shift", 0.0f}});
        power_runner2.Run(stream);
        const auto& adds_runner =
            NpuOpRunner("Adds", {b_in_norm}, {b_in_norm}, {{"value", eps}});
        adds_runner.Run(stream);
        const auto& div_runner =
            NpuOpRunner("Div", {x_abs, b_in_norm}, {x_abs}, {});
        div_runner.Run(stream);
        const auto& mul_runner1 =
            NpuOpRunner("Mul", {x_abs, b_in_norm_dy}, {x_abs}, {});
        mul_runner1.Run(stream);
        const auto& mul_runner2 =
            NpuOpRunner("Mul", {x_abs, x_sign}, {*out_dx}, {});
        mul_runner2.Run(stream);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    p_norm, ops::PnormNPUKernel<plat::NPUDeviceContext, float>,
    ops::PnormNPUKernel<plat::NPUDeviceContext, plat::float16>);
REGISTER_OP_NPU_KERNEL(
    p_norm_grad, ops::PnormGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::PnormGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
