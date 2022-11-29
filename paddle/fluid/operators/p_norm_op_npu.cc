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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class PnormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<phi::DenseTensor>("X");
    auto* out_norm = ctx.Output<phi::DenseTensor>("Out");
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
      float t = 0;
      float diff = abs(std::modf(porder, &t));
      if (diff < 1e-5) {
        combine_op = false;
      }
    }

    if (!combine_op) {
      const auto& runner = NpuOpRunner("LpNorm",
                                       {*in_x},
                                       {*out_norm},
                                       {{"p", p},
                                        {"axes", std::vector<int32_t>({axis})},
                                        {"keep_dims", keepdim}});
      runner.Run(stream);
    } else {
      Tensor tmp_x;
      tmp_x.mutable_data<T>(xdim, ctx.GetPlace());

      const auto& power_runner1 =
          NpuOpRunner("Power",
                      {*in_x},
                      {tmp_x},
                      {{"power", porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner1.Run(stream);

      const auto& reduce_runner = NpuOpRunner(
          "ReduceSumD",
          {tmp_x},
          {*out_norm},
          {{"axes", std::vector<int32_t>({axis})}, {"keep_dims", keepdim}});
      reduce_runner.Run(stream);

      const auto& power_runner2 = NpuOpRunner(
          "Power",
          {*out_norm},
          {*out_norm},
          {{"power", 1 / porder}, {"scale", 1.0f}, {"shift", 0.0f}});
      power_runner2.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class PnormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using Tensor = phi::DenseTensor;
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* y = ctx.Input<phi::DenseTensor>("Out");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();
    dx->mutable_data<T>(place);

    auto xdim = x->dims();
    float porder = ctx.Attr<float>("porder");
    bool keepdim = ctx.Attr<bool>("keepdim");

    int axis = ctx.Attr<int>("axis");
    axis = axis < 0 ? xdim.size() + axis : axis;

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor y_share(y->type());
    Tensor dy_share(dy->type());
    y_share.ShareDataWith(*y);
    dy_share.ShareDataWith(*dy);
    auto ydim = xdim;
    if (!keepdim) {
      ydim[axis] = 1;
    } else {
      ydim = y->dims();
    }
    y_share.Resize(ydim);
    dy_share.Resize(ydim);

    if (porder == 0) {
      FillNpuTensorWithConstant(dx, static_cast<T>(0));
      dx->Resize(xdim);
    } else if (porder == INFINITY || porder == -INFINITY) {
      Tensor x_abs;
      x_abs.mutable_data<T>(xdim, place);
      const auto& r_abs = NpuOpRunner("Abs", {*x}, {x_abs}, {});
      r_abs.Run(stream);

      Tensor t_cond;
      t_cond.mutable_data<bool>(xdim, place);
      const auto& r_equal =
          NpuOpRunner("Equal", {x_abs, y_share}, {t_cond}, {});
      r_equal.Run(stream);

      Tensor t_zero;
      t_zero.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant(&t_zero, static_cast<T>(0));

      Tensor x_sign;
      x_sign.mutable_data<T>(xdim, place);
      const auto& r_sign = NpuOpRunner("Sign", {*x}, {x_sign}, {});
      r_sign.Run(stream);

      const auto& r_mul = NpuOpRunner("Mul", {x_sign, dy_share}, {*dx}, {});
      r_mul.Run(stream);

      const auto& r_sel =
          NpuOpRunner("SelectV2", {t_cond, *dx, t_zero}, {*dx}, {});
      r_sel.Run(stream);
    } else {
      Tensor x_abs;
      x_abs.mutable_data<T>(xdim, place);
      const auto& r_abs = NpuOpRunner("Abs", {*x}, {x_abs}, {});
      r_abs.Run(stream);

      Tensor x_sign;
      x_sign.mutable_data<T>(xdim, place);
      const auto& r_sign = NpuOpRunner("Sign", {*x}, {x_sign}, {});
      r_sign.Run(stream);

      Tensor y_pow;
      y_pow.mutable_data<T>(ydim, place);
      if (porder >= 1) {
        const auto& r_pow1 = NpuOpRunner(
            "Power",
            {x_abs},
            {x_abs},
            {{"power", (porder - 1)}, {"scale", 1.0f}, {"shift", 0.0f}});
        r_pow1.Run(stream);

        const auto& r_pow2 = NpuOpRunner(
            "Power",
            {y_share},
            {y_pow},
            {{"power", (porder - 1)}, {"scale", 1.0f}, {"shift", 0.0f}});
        r_pow2.Run(stream);

        const auto& r_div = NpuOpRunner("DivNoNan", {x_abs, y_pow}, {*dx}, {});
        r_div.Run(stream);
      } else {
        const auto& r_pow1 = NpuOpRunner(
            "Power",
            {x_abs},
            {x_abs},
            {{"power", (1 - porder)}, {"scale", 1.0f}, {"shift", 0.0f}});
        r_pow1.Run(stream);

        const auto& r_pow2 = NpuOpRunner(
            "Power",
            {y_share},
            {y_pow},
            {{"power", (1 - porder)}, {"scale", 1.0f}, {"shift", 0.0f}});
        r_pow2.Run(stream);

        const auto& r_div = NpuOpRunner("DivNoNan", {y_pow, x_abs}, {*dx}, {});
        r_div.Run(stream);
      }

      const auto& r_mul1 = NpuOpRunner("Mul", {*dx, x_sign}, {*dx}, {});
      r_mul1.Run(stream);

      const auto& r_mul2 = NpuOpRunner("Mul", {*dx, dy_share}, {*dx}, {});
      r_mul2.Run(stream);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    p_norm,
    ops::PnormNPUKernel<plat::NPUDeviceContext, float>,
    ops::PnormNPUKernel<plat::NPUDeviceContext, plat::float16>);

REGISTER_OP_NPU_KERNEL(
    p_norm_grad,
    ops::PnormGradNPUKernel<plat::NPUDeviceContext, float>,
    ops::PnormGradNPUKernel<plat::NPUDeviceContext, plat::float16>);
