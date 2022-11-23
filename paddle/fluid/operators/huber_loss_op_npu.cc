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

using Tensor = phi::DenseTensor;

template <typename T>
void HuberLossSub(const platform::Place& place,
                  const aclrtStream& stream,
                  const phi::DenseTensor* x,
                  const phi::DenseTensor* y,
                  phi::DenseTensor* z) {
  //  Calculate z = x - y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossMuls(const platform::Place& place,
                   const aclrtStream& stream,
                   const phi::DenseTensor* x,
                   float scalar,
                   phi::DenseTensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
  runner.Run(stream);
}

template <typename T>
void HuberLossZerosLike(const platform::Place& place,
                        const aclrtStream& stream,
                        const phi::DenseTensor* x,
                        phi::DenseTensor* y) {
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("ZerosLike", {*x}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSmoothL1Loss(const platform::Place& place,
                           const aclrtStream& stream,
                           const phi::DenseTensor* x,
                           const phi::DenseTensor* y,
                           float delta,
                           phi::DenseTensor* z) {
  z->mutable_data<T>(x->dims(), place);
  const auto& runner =
      NpuOpRunner("SmoothL1Loss", {*x, *y}, {*z}, {{"sigma", delta}});
  runner.Run(stream);
}

template <typename T>
void HuberLossSmoothL1LossGrad(const platform::Place& place,
                               const aclrtStream& stream,
                               const phi::DenseTensor* pred,
                               const phi::DenseTensor* lab,
                               const phi::DenseTensor* dout,
                               float sigma,
                               phi::DenseTensor* grad) {
  grad->mutable_data<T>(pred->dims(), place);
  const auto& runner = NpuOpRunner(
      "SmoothL1LossGrad", {*pred, *lab, *dout}, {*grad}, {{"sigma", sigma}});
  runner.Run(stream);
}

template <typename T>
class HuberLossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in0 = ctx.Input<phi::DenseTensor>("X");
    auto* in1 = ctx.Input<phi::DenseTensor>("Y");
    auto* residual = ctx.Output<phi::DenseTensor>("Residual");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    auto delta = ctx.Attr<float>("delta");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();
    HuberLossSub<T>(place, stream, in1, in0, residual);

    HuberLossSmoothL1Loss<T>(place, stream, in0, in1, delta, out);
    HuberLossMuls<T>(place, stream, out, delta, out);
  }
};

template <typename T>
class HuberLossGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* residual = ctx.Input<phi::DenseTensor>("Residual");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<phi::DenseTensor>(framework::GradVarName("Y"));
    auto delta = ctx.Attr<float>("delta");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();

    Tensor t_grad_rd;
    if (dx || dy) {
      Tensor t_zero;
      HuberLossZerosLike<T>(place, stream, residual, &t_zero);
      HuberLossSmoothL1LossGrad<T>(
          place, stream, residual, &t_zero, dout, delta, &t_grad_rd);
    }
    if (dx) {
      HuberLossMuls<T>(place, stream, &t_grad_rd, -delta, dx);
    }
    if (dy) {
      HuberLossMuls<T>(place, stream, &t_grad_rd, delta, dy);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(huber_loss,
                       ops::HuberLossNPUKernel<float>,
                       ops::HuberLossNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(huber_loss_grad,
                       ops::HuberLossGradNPUKernel<float>,
                       ops::HuberLossGradNPUKernel<plat::float16>);
