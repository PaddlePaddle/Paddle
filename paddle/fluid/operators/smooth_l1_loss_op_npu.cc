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

#include "paddle/fluid/operators/smooth_l1_loss_op.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SmoothL1LossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Y = context.Input<Tensor>("Y");
    auto* InsideWeight = context.Input<Tensor>("InsideWeight");
    auto* OutsideWeight = context.Input<Tensor>("OutsideWeight");
    auto* out_diff = context.Output<Tensor>("Diff");
    auto* out_loss = context.Output<Tensor>("Out");
    out_diff->mutable_data<T>(context.GetPlace());
    out_loss->mutable_data<T>(context.GetPlace());

    auto sigma = context.Attr<float>("sigma");
    T sigma2 = 1.0 / (sigma * sigma);
    bool has_weight = (InsideWeight != nullptr) && (OutsideWeight != nullptr);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner1 = NpuOpRunner("Sub", {*X, *Y}, {*out_diff}, {});
    runner1.Run(stream);

    Tensor no_reduce_loss(X->type());
    no_reduce_loss.Resize(X->dims());
    no_reduce_loss.mutable_data<T>(context.GetPlace());
    // multiply inside weight
    if (has_weight) {
      Tensor tmp_diff(out_diff->type());
      tmp_diff.Resize(out_diff->dims());
      tmp_diff.mutable_data<T>(context.GetPlace());
      const auto& runner2 =
          NpuOpRunner("Mul", {*out_diff, *InsideWeight}, {tmp_diff}, {});
      runner2.Run(stream);
      framework::TensorCopy(
          tmp_diff, context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          out_diff);

      Tensor tmp_x(X->type());
      tmp_x.Resize(X->dims());
      tmp_x.mutable_data<T>(context.GetPlace());

      Tensor tmp_y(Y->type());
      tmp_y.Resize(Y->dims());
      tmp_y.mutable_data<T>(context.GetPlace());

      const auto& runner_x =
          NpuOpRunner("Mul", {*X, *InsideWeight}, {tmp_x}, {});
      runner_x.Run(stream);
      const auto& runner_y =
          NpuOpRunner("Mul", {*Y, *InsideWeight}, {tmp_y}, {});
      runner_y.Run(stream);
      const auto& runner3 = NpuOpRunner("SmoothL1Loss", {tmp_x, tmp_y},
                                        {no_reduce_loss}, {{"sigma", sigma2}});
      runner3.Run(stream);
    } else {
      const auto& runner3 = NpuOpRunner("SmoothL1Loss", {*X, *Y},
                                        {no_reduce_loss}, {{"sigma", sigma2}});
      runner3.Run(stream);
    }

    // multiply outside weight
    if (has_weight) {
      Tensor tmp_loss(no_reduce_loss.type());
      tmp_loss.Resize(no_reduce_loss.dims());
      tmp_loss.mutable_data<T>(context.GetPlace());
      const auto& runner4 =
          NpuOpRunner("Mul", {no_reduce_loss, *OutsideWeight}, {tmp_loss}, {});
      runner4.Run(stream);
      const auto& runner5 =
          NpuOpRunner("ReduceSumD", {tmp_loss}, {*out_loss},
                      {{"axes", std::vector<int>{1}}, {"keep_dims", true}});
      runner5.Run(stream);
    } else {
      const auto& runner5 =
          NpuOpRunner("ReduceSumD", {no_reduce_loss}, {*out_loss},
                      {{"axes", std::vector<int>{1}}, {"keep_dims", true}});
      runner5.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class SmoothL1LossGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* InsideWeight = context.Input<Tensor>("InsideWeight");
    auto* OutsideWeight = context.Input<Tensor>("OutsideWeight");
    auto* diff = context.Input<Tensor>("Diff");
    auto* og = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    auto* out1 = context.Output<Tensor>(framework::GradVarName("Y"));
    auto sigma = context.Attr<T>("sigma");
    T sigma2 = 1.0 / (sigma * sigma);
    bool has_weight = (InsideWeight != nullptr) && (OutsideWeight != nullptr);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor tmp_zero(diff->type());
    tmp_zero.Resize(diff->dims());
    tmp_zero.mutable_data<T>(context.GetPlace());
    const auto& runner_zero = NpuOpRunner("ZerosLike", {*diff}, {tmp_zero}, {});
    runner_zero.Run(stream);

    Tensor grad(diff->type());
    grad.Resize(diff->dims());
    grad.mutable_data<T>(context.GetPlace());
    const auto& runner_broad =
        NpuOpRunner("BroadcastToD", {*og}, {grad},
                    {{"shape", framework::vectorize(diff->dims())}});
    runner_broad.Run(stream);

    Tensor gradient(diff->type());
    gradient.Resize(diff->dims());
    gradient.mutable_data<T>(context.GetPlace());
    const auto& runner_grad =
        NpuOpRunner("SmoothL1LossGrad", {*diff, tmp_zero, grad}, {gradient},
                    {{"sigma", sigma2}});
    runner_grad.Run(stream);

    if (has_weight) {
      Tensor weight(InsideWeight->type());
      weight.Resize(InsideWeight->dims());
      weight.mutable_data<T>(context.GetPlace());
      const auto& runner_weight =
          NpuOpRunner("Mul", {*InsideWeight, *OutsideWeight}, {weight}, {});
      runner_weight.Run(stream);

      Tensor tmp_grad(gradient.type());
      tmp_grad.Resize(gradient.dims());
      tmp_grad.mutable_data<T>(context.GetPlace());
      const auto& runner_weight_grad =
          NpuOpRunner("Mul", {gradient, weight}, {tmp_grad}, {});
      runner_weight_grad.Run(stream);

      framework::TensorCopy(
          tmp_grad, context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          &gradient);
    }
    if (out0) {
      out0->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(
          gradient, context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          out0);
    }

    if (out1) {
      out1->mutable_data<T>(context.GetPlace());
      Tensor coeff(framework::proto::VarType::FP32);
      coeff.mutable_data<float>({1}, context.GetPlace());
      FillNpuTensorWithConstant<float>(&coeff, -1);
      const auto& runner_y_grad =
          NpuOpRunner("Mul", {coeff, gradient}, {*out1}, {});
      runner_y_grad.Run(stream);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(
    smooth_l1_loss,
    ops::SmoothL1LossNPUKernel<paddle::platform::NPUDeviceContext, float>);

REGISTER_OP_NPU_KERNEL(
    smooth_l1_loss_grad,
    ops::SmoothL1LossGradNPUKernel<paddle::platform::NPUDeviceContext, float>);
