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

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/smooth_l1_loss_op.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SmoothL1LossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
<<<<<<< HEAD
    auto* in_x = context.Input<phi::DenseTensor>("X");
    auto* in_y = context.Input<phi::DenseTensor>("Y");
    auto* inside_weight = context.Input<phi::DenseTensor>("InsideWeight");
    auto* outside_weight = context.Input<phi::DenseTensor>("OutsideWeight");
    auto* out_diff = context.Output<phi::DenseTensor>("Diff");
    auto* out_loss = context.Output<phi::DenseTensor>("Out");
=======
    auto* in_x = context.Input<Tensor>("X");
    auto* in_y = context.Input<Tensor>("Y");
    auto* inside_weight = context.Input<Tensor>("InsideWeight");
    auto* outside_weight = context.Input<Tensor>("OutsideWeight");
    auto* out_diff = context.Output<Tensor>("Diff");
    auto* out_loss = context.Output<Tensor>("Out");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    out_diff->mutable_data<T>(context.GetPlace());
    out_loss->mutable_data<T>(context.GetPlace());

    auto sigma = context.Attr<float>("sigma");
    T sigma2 = 1.0 / (sigma * sigma);
    bool has_weight = (inside_weight != nullptr) && (outside_weight != nullptr);
    // out_diff = in_x - in_y
    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner1 = NpuOpRunner("Sub", {*in_x, *in_y}, {*out_diff}, {});
    runner1.Run(stream);

<<<<<<< HEAD
    phi::DenseTensor no_reduce_loss(in_x->dtype());
=======
    Tensor no_reduce_loss(in_x->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    no_reduce_loss.Resize(in_x->dims());
    no_reduce_loss.mutable_data<T>(context.GetPlace());
    // multiply inside weight before get the loss
    if (has_weight) {
<<<<<<< HEAD
      phi::DenseTensor tmp_diff(out_diff->dtype());
=======
      Tensor tmp_diff(out_diff->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      tmp_diff.Resize(out_diff->dims());
      tmp_diff.mutable_data<T>(context.GetPlace());
      const auto& runner2 =
          NpuOpRunner("Mul", {*out_diff, *inside_weight}, {tmp_diff}, {});
      runner2.Run(stream);
      framework::TensorCopy(
          tmp_diff,
          context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          out_diff);

<<<<<<< HEAD
      phi::DenseTensor tmp_x(in_x->dtype());
      tmp_x.Resize(in_x->dims());
      tmp_x.mutable_data<T>(context.GetPlace());

      phi::DenseTensor tmp_y(in_y->dtype());
=======
      Tensor tmp_x(in_x->dtype());
      tmp_x.Resize(in_x->dims());
      tmp_x.mutable_data<T>(context.GetPlace());

      Tensor tmp_y(in_y->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      tmp_y.Resize(in_y->dims());
      tmp_y.mutable_data<T>(context.GetPlace());

      // mul input and inside_weight
      const auto& runner_x =
          NpuOpRunner("Mul", {*in_x, *inside_weight}, {tmp_x}, {});
      runner_x.Run(stream);
      const auto& runner_y =
          NpuOpRunner("Mul", {*in_y, *inside_weight}, {tmp_y}, {});
      runner_y.Run(stream);
      const auto& runner3 = NpuOpRunner("SmoothL1Loss",
                                        {tmp_x, tmp_y},
                                        {no_reduce_loss},
                                        {{"sigma", sigma2}});
      runner3.Run(stream);
    } else {
      const auto& runner3 = NpuOpRunner("SmoothL1Loss",
                                        {*in_x, *in_y},
                                        {no_reduce_loss},
                                        {{"sigma", sigma2}});
      runner3.Run(stream);
    }

    // multiply outside weight and loss
    // reduceSum because the output'shape must be [B,1]
    if (has_weight) {
<<<<<<< HEAD
      phi::DenseTensor tmp_loss(no_reduce_loss.dtype());
=======
      Tensor tmp_loss(no_reduce_loss.dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      tmp_loss.Resize(no_reduce_loss.dims());
      tmp_loss.mutable_data<T>(context.GetPlace());
      const auto& runner4 =
          NpuOpRunner("Mul", {no_reduce_loss, *outside_weight}, {tmp_loss}, {});
      runner4.Run(stream);
      const auto& runner5 =
          NpuOpRunner("ReduceSumD",
                      {tmp_loss},
                      {*out_loss},
                      {{"axes", std::vector<int>{1}}, {"keep_dims", true}});
      runner5.Run(stream);
    } else {
      const auto& runner5 =
          NpuOpRunner("ReduceSumD",
                      {no_reduce_loss},
                      {*out_loss},
                      {{"axes", std::vector<int>{1}}, {"keep_dims", true}});
      runner5.Run(stream);
    }
  }
};

template <typename DeviceContext, typename T>
class SmoothL1LossGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
<<<<<<< HEAD
    auto* inside_weight = context.Input<phi::DenseTensor>("InsideWeight");
    auto* outside_weight = context.Input<phi::DenseTensor>("OutsideWeight");
    auto* diff = context.Input<phi::DenseTensor>("Diff");
    auto* og = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* outx_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* outy_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("Y"));
=======
    auto* inside_weight = context.Input<Tensor>("InsideWeight");
    auto* outside_weight = context.Input<Tensor>("OutsideWeight");
    auto* diff = context.Input<Tensor>("Diff");
    auto* og = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* outx_grad = context.Output<Tensor>(framework::GradVarName("X"));
    auto* outy_grad = context.Output<Tensor>(framework::GradVarName("Y"));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto sigma = context.Attr<T>("sigma");
    T sigma2 = 1.0 / (sigma * sigma);
    bool has_weight = (inside_weight != nullptr) && (outside_weight != nullptr);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // diff == in_x - in_y == diff - 0
<<<<<<< HEAD
    phi::DenseTensor tmp_zero(diff->dtype());
=======
    Tensor tmp_zero(diff->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    tmp_zero.Resize(diff->dims());
    tmp_zero.mutable_data<T>(context.GetPlace());
    const auto& runner_zero = NpuOpRunner("ZerosLike", {*diff}, {tmp_zero}, {});
    runner_zero.Run(stream);

<<<<<<< HEAD
    phi::DenseTensor grad(diff->dtype());
=======
    Tensor grad(diff->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    grad.Resize(diff->dims());
    grad.mutable_data<T>(context.GetPlace());
    // broadcast og(output_grad) to adapt to the npu interface
    const auto& runner_broad =
        NpuOpRunner("BroadcastToD",
                    {*og},
                    {grad},
                    {{"shape", phi::vectorize(diff->dims())}});
    runner_broad.Run(stream);

<<<<<<< HEAD
    phi::DenseTensor gradient(diff->dtype());
=======
    Tensor gradient(diff->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    gradient.Resize(diff->dims());
    gradient.mutable_data<T>(context.GetPlace());
    // diff == diff - 0 == in_x - in_y
    const auto& runner_grad = NpuOpRunner("SmoothL1LossGrad",
                                          {*diff, tmp_zero, grad},
                                          {gradient},
                                          {{"sigma", sigma2}});
    runner_grad.Run(stream);

    // mul weight and gradient
    if (has_weight) {
<<<<<<< HEAD
      phi::DenseTensor weight(inside_weight->dtype());
=======
      Tensor weight(inside_weight->dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      weight.Resize(inside_weight->dims());
      weight.mutable_data<T>(context.GetPlace());
      const auto& runner_weight =
          NpuOpRunner("Mul", {*inside_weight, *outside_weight}, {weight}, {});
      runner_weight.Run(stream);

<<<<<<< HEAD
      phi::DenseTensor tmp_grad(gradient.dtype());
=======
      Tensor tmp_grad(gradient.dtype());
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      tmp_grad.Resize(gradient.dims());
      tmp_grad.mutable_data<T>(context.GetPlace());
      const auto& runner_weight_grad =
          NpuOpRunner("Mul", {gradient, weight}, {tmp_grad}, {});
      runner_weight_grad.Run(stream);

      framework::TensorCopy(
          tmp_grad,
          context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          &gradient);
    }
    // outx_grad = gradient
    if (outx_grad) {
      outx_grad->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(
          gradient,
          context.GetPlace(),
          context.template device_context<paddle::platform::NPUDeviceContext>(),
          outx_grad);
    }

    // outy_grad = - gradient
    if (outy_grad) {
      outy_grad->mutable_data<T>(context.GetPlace());
<<<<<<< HEAD
      phi::DenseTensor coeff(experimental::DataType::FLOAT32);
=======
      Tensor coeff(experimental::DataType::FLOAT32);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      coeff.mutable_data<float>({1}, context.GetPlace());
      FillNpuTensorWithConstant<float>(&coeff, -1);
      const auto& runner_y_grad =
          NpuOpRunner("Mul", {coeff, gradient}, {*outy_grad}, {});
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
