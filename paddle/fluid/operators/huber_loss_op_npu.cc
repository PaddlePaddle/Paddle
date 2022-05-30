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

using Tensor = framework::Tensor;

template <typename T>
void HuberLossSub(const platform::Place& place, const aclrtStream& stream,
                  const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x - y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossMuls(const platform::Place& place, const aclrtStream& stream,
                   const Tensor* x, float scalar, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scalar}});
  runner.Run(stream);
}

template <typename T>
void HuberLossZerosLike(const platform::Place& place, const aclrtStream& stream,
                        const Tensor* x, Tensor* y) {
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("ZerosLike", {*x}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSmoothL1Loss(const platform::Place& place,
                           const aclrtStream& stream, const Tensor* x,
                           const Tensor* y, float delta, Tensor* z) {
  z->mutable_data<T>(x->dims(), place);
  const auto& runner =
      NpuOpRunner("SmoothL1Loss", {*x, *y}, {*z}, {{"sigma", delta}});
  runner.Run(stream);
}

template <typename T>
void HuberLossSmoothL1LossGrad(const platform::Place& place,
                               const aclrtStream& stream, const Tensor* pred,
                               const Tensor* lab, const Tensor* dout,
                               float sigma, Tensor* grad) {
  grad->mutable_data<T>(pred->dims(), place);
  const auto& runner = NpuOpRunner("SmoothL1LossGrad", {*pred, *lab, *dout},
                                   {*grad}, {{"sigma", sigma}});
  runner.Run(stream);
}

template <typename T>
class HuberLossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in0 = ctx.Input<Tensor>("X");
    auto* in1 = ctx.Input<Tensor>("Y");
    auto* residual = ctx.Output<Tensor>("Residual");
    auto* out = ctx.Output<Tensor>("Out");
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
    auto* residual = ctx.Input<Tensor>("Residual");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    auto delta = ctx.Attr<float>("delta");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();

    Tensor t_grad_rd;
    if (dx || dy) {
      Tensor t_zero;
      HuberLossZerosLike<T>(place, stream, residual, &t_zero);
      HuberLossSmoothL1LossGrad<T>(place, stream, residual, &t_zero, dout,
                                   delta, &t_grad_rd);
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

REGISTER_OP_NPU_KERNEL(huber_loss, ops::HuberLossNPUKernel<float>,
                       ops::HuberLossNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(huber_loss_grad, ops::HuberLossGradNPUKernel<float>,
                       ops::HuberLossGradNPUKernel<plat::float16>);
