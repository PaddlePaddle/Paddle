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
void HuberLossAbs(const platform::Place& place, const aclrtStream& stream,
                  const Tensor* x, Tensor* y) {
  //  Calculate y = abs(x)
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Abs", {*x}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
framework::proto::VarType::Type HuberLossGetType();

template <>
framework::proto::VarType::Type HuberLossGetType<float>() {
  return framework::proto::VarType::FP32;
}

template <>
framework::proto::VarType::Type HuberLossGetType<double>() {
  return framework::proto::VarType::FP64;
}

template <>
framework::proto::VarType::Type HuberLossGetType<platform::float16>() {
  return framework::proto::VarType::FP16;
}

template <typename T>
Tensor HuberLossVal2Tsr(const platform::Place& place, T value) {
  auto vartype = HuberLossGetType<T>();
  Tensor val_t(vartype);
  val_t.mutable_data<T>({1}, place);
  FillNpuTensorWithConstant<T>(&val_t, value);
  return val_t;
}

template <typename T>
void HuberLossLessEqualValue(const platform::Place& place,
                             const aclrtStream& stream, const Tensor* x,
                             float val, Tensor* y) {
  //  Calculate y = x <= val, where y, x are tensors and val is scalar
  Tensor val_t = HuberLossVal2Tsr(place, static_cast<T>(val));

  y->mutable_data<bool>(x->dims(), place);
  const auto& runner = NpuOpRunner("LessEqual", {*x, val_t}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSub(const platform::Place& place, const aclrtStream& stream,
                  const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x - y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Sub", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossAdds(const platform::Place& place, const aclrtStream& stream,
                   const Tensor* x, float scalar, Tensor* y) {
  //  Calculate y = x + scale
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Adds", {*x}, {*y}, {{"value", scalar}});
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
void HuberLossMul(const platform::Place& place, const aclrtStream& stream,
                  const Tensor* x, const Tensor* y, Tensor* z) {
  //  Calculate z = x * y
  z->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Mul", {*x, *y}, {*z}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSel(const platform::Place& place, const aclrtStream& stream,
                  const Tensor* cond, const Tensor* x1, const Tensor* x2,
                  Tensor* y) {
  //  Calculate y = cond ? x1 : x2;
  y->mutable_data<T>(x1->dims(), place);
  const auto& runner = NpuOpRunner("Select", {*cond, *x1, *x2}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSquare(const platform::Place& place, const aclrtStream& stream,
                     const Tensor* x, Tensor* y) {
  //  Calculate y = x ^ 2
  y->mutable_data<T>(x->dims(), place);
  const auto& runner = NpuOpRunner("Square", {*x}, {*y}, {});
  runner.Run(stream);
}

template <typename T>
void HuberLossSign(const platform::Place& place, const aclrtStream& stream,
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
    auto delta = ctx.Attr<float>("delta");

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    auto place = ctx.GetPlace();
    HuberLossSub<T>(place, stream, in1, in0, residual);

    Tensor t_cond;
    Tensor t_abs_rd;
    HuberLossAbs<T>(place, stream, residual, &t_abs_rd);
    HuberLossLessEqualValue<T>(place, stream, &t_abs_rd, delta, &t_cond);
    Tensor t_b0;
    Tensor t_mul_delta_residual;
    HuberLossMuls<T>(place, stream, &t_abs_rd, delta, &t_mul_delta_residual);
    HuberLossAdds<T>(place, stream, &t_mul_delta_residual, -0.5 * delta * delta,
                     &t_b0);
    Tensor t_sqr;
    Tensor t_b1;
    HuberLossSquare<T>(place, stream, residual, &t_sqr);
    HuberLossMuls<T>(place, stream, &t_sqr, 0.5, &t_b1);
    HuberLossSel<T>(place, stream, &t_cond, &t_b1, &t_b0, out);
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
      Tensor t_cond;
      Tensor t_abs_rd;
      HuberLossAbs<T>(place, stream, residual, &t_abs_rd);
      HuberLossLessEqualValue<T>(place, stream, &t_abs_rd, delta, &t_cond);
      Tensor t_b0;
      Tensor t_sign_residual;
      HuberLossSign<T>(place, stream, residual, &t_sign_residual);
      HuberLossMuls<T>(place, stream, &t_sign_residual, delta, &t_b0);
      HuberLossSel<T>(place, stream, &t_cond, residual, &t_b0, &t_grad_rd);
    }
    if (dx) {
      Tensor t_grad_x;
      HuberLossMuls<T>(place, stream, &t_grad_rd, -1, &t_grad_x);
      HuberLossMul<T>(place, stream, &t_grad_x, dout, dx);
    }
    if (dy) {
      Tensor t_grad_y;
      HuberLossMuls<T>(place, stream, &t_grad_rd, 1, &t_grad_y);
      HuberLossMul<T>(place, stream, &t_grad_y, dout, dy);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(huber_loss, ops::HuberLossNPUKernel<float>,
                       ops::HuberLossNPUKernel<double>,
                       ops::HuberLossNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(huber_loss_grad, ops::HuberLossGradNPUKernel<float>,
                       ops::HuberLossGradNPUKernel<double>,
                       ops::HuberLossGradNPUKernel<plat::float16>);
