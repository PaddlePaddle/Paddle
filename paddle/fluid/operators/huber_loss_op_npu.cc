/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/huber_loss_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
void DiffCheck(const platform::Place& place, const aclrtStream& stream,
               const Tensor* x, T delta, Tensor* y) {
  //  Calculate y = abs(x) <= delta
  y->mutable_data<bool>(x->dims(), place);
  Tensor x_abs;
  x_abs.mutable_data<bool>(x->dims(), place);
  const auto& runner_abs = NpuOpRunner("Abs", {*x}, {x_abs}, {});
  runner_abs.Run(stream);

  Tensor delta_t(framework::proto::VarType::FP32);
  // delta_t.mutable_data<T>({1}, place);
  delta_t.mutable_data<T>(x->dims(), place);
  FillNpuTensorWithConstant<T>(&delta_t, delta);

  const auto& runner_le = NpuOpRunner("LessEqual", {x_abs, delta_t}, {*y}, {});
  runner_le.Run(stream);
}

template <typename T>
void SubFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x - y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void AddsFun(const platform::Place& place, const aclrtStream& stream,
             const Tensor* x, float scale, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T>
void MulsFun(const platform::Place& place, const aclrtStream& stream,
             const Tensor* x, float scale, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Muls", {*x}, {*y}, {{"value", scale}});
  runner.Run(stream);
}

template <typename T>
void MulFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x * y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void SelFun(const platform::Place& place, const aclrtStream& stream,
            const Tensor* cond, const Tensor* x1, const Tensor* x2, Tensor* y) {
  //  Calculate y = cond ? x1 : x2;
  y->mutable_data<T>(x1->dims(), place);
  const auto& runner = NpuOpRunner("Select", {*cond, *x1, *x2}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void SquareFun(const platform::Place& place, const aclrtStream& stream,
               const Tensor* x, Tensor* y) {
  //  Calculate y = x ^ 2
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Square", {*x}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void SignFun(const platform::Place& place, const aclrtStream& stream,
             const Tensor* x, Tensor* y) {
  //  Calculate y = sign of x
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sign", {*x}, {*y}, {});
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
    auto delta = static_cast<T>(ctx.Attr<float>("delta"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();
    SubFun<T>(place, stream, in1, in0, residual);

    Tensor t_cond;
    DiffCheck<T>(place, stream, residual, delta, &t_cond);
    Tensor t_b0;
    Tensor t_mul_delta_residual;
    MulsFun<T>(place, stream, residual, delta, &t_mul_delta_residual);
    AddsFun<T>(place, stream, &t_mul_delta_residual, -0.5 * delta * delta,
               &t_b0);
    Tensor t_b1;
    SquareFun<T>(place, stream, residual, &t_b1);
    SelFun<T>(place, stream, &t_cond, &t_b1, &t_b0, out);
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
    auto delta = static_cast<T>(ctx.Attr<float>("delta"));

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();

    Tensor t_grad;
    if (dx || dy) {
      Tensor t_cond;
      DiffCheck<T>(place, stream, residual, delta, &t_cond);
      Tensor t_b0;
      Tensor t_sign_residual;
      SignFun<T>(place, stream, residual, &t_sign_residual);
      MulsFun<T>(place, stream, &t_sign_residual, delta, &t_b0);
      SelFun<T>(place, stream, &t_cond, &t_sign_residual, residual, &t_grad);
    }
    if (dx) {
      Tensor t_grad_x;
      MulsFun<T>(place, stream, &t_grad, -1, &t_grad_x);
      MulFun<T>(place, stream, &t_grad_x, dout, dx);
    }
    if (dy) {
      Tensor t_grad_y;
      MulsFun<T>(place, stream, &t_grad, 1, &t_grad_y);
      MulFun<T>(place, stream, &t_grad_y, dout, dy);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(huber_loss, ops::HuberLossNPUKernel<float>);
REGISTER_OP_NPU_KERNEL(huber_loss_grad, ops::HuberLossGradNPUKernel<float>);
