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

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ElementwiseDivNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto* out = ctx.Output<Tensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("Div", {*x, *y}, {*out}, {});
    runner.Run(stream);
  }
};

template <typename DeviceContext, typename T>
class ElementwiseDivGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto place = ctx.GetPlace();

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    if (dx) {
      dx->mutable_data<T>(place);

      Tensor tensor_one(y->type());
      tensor_one.mutable_data<float>({1}, place);
      FillNpuTensorWithConstant<float>(&tensor_one, static_cast<float>(1.0));

      // Use `Div` CANN OP to achieve `1/y` instead of `Power` CANN OP.
      // Because `Power` will cause precision overflow, that is, `float_status`
      // will be set to 1.
      Tensor y_div(y->type());
      y_div.mutable_data<T>(y->dims(), place);
      const auto& runner_one_div_y =
          NpuOpRunner("Div", {tensor_one, *y}, {y_div}, {});
      runner_one_div_y.Run(stream);

      Tensor tensor_zeros(x->type());
      tensor_zeros.mutable_data<T>(x->dims(), place);
      const auto& runner_tensor_zeros =
          NpuOpRunner("ZerosLike", {*x}, {tensor_zeros}, {});
      runner_tensor_zeros.Run(stream);

      Tensor x_zero(paddle::framework::proto::VarType::BOOL);
      x_zero.mutable_data<bool>(x->dims(), place);
      const auto& runner_x_zero =
          NpuOpRunner("Equal", {*x, tensor_zeros}, {x_zero}, {});
      runner_x_zero.Run(stream);

      Tensor x_nozero(paddle::framework::proto::VarType::BOOL);
      x_nozero.mutable_data<bool>(x->dims(), place);
      const auto& runner_x_nonzero =
          NpuOpRunner("LogicalNot", {x_zero}, {x_nozero}, {});
      runner_x_nonzero.Run(stream);

      Tensor x_nozero_f(x->type());
      x_nozero_f.mutable_data<T>(x->dims(), place);
      const auto& runner_x_nonzero_f =
          NpuOpRunner("Cast", {x_nozero}, {x_nozero_f},
                      {{"dst_type", static_cast<int32_t>(0)}});
      runner_x_nonzero_f.Run(stream);

      Tensor x_grad_w(x->type());
      x_grad_w.mutable_data<T>(x->dims(), place);
      const auto& runner_x_grad_w =
          NpuOpRunner("Mul", {x_nozero_f, y_div}, {x_grad_w}, {});
      runner_x_grad_w.Run(stream);

      const auto& runner_x_grad =
          NpuOpRunner("Mul", {x_grad_w, *dout}, {*dx}, {});
      runner_x_grad.Run(stream);
    }

    if (dy) {
      dy->mutable_data<T>(place);

      Tensor neg_out(y->type());
      neg_out.mutable_data<T>(y->dims(), place);
      const auto& runner_neg_out = NpuOpRunner("Neg", {*out}, {neg_out}, {});
      runner_neg_out.Run(stream);

      Tensor y_grad_w(y->type());
      y_grad_w.mutable_data<T>(y->dims(), place);
      const auto& runner_y_grad_w =
          NpuOpRunner("Div", {neg_out, *y}, {y_grad_w}, {});
      runner_y_grad_w.Run(stream);

      const auto& runner_y_grad =
          NpuOpRunner("Mul", {y_grad_w, *dout}, {*dy}, {});
      runner_y_grad.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseDivNPUKernel<paddle::platform::NPUDeviceContext,
                                 paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::ElementwiseDivGradNPUKernel<paddle::platform::NPUDeviceContext,
                                     paddle::platform::float16>);
