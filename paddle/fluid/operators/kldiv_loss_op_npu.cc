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

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class KLDivLossNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* target = ctx.Input<phi::DenseTensor>("Target");
    auto* loss = ctx.Output<phi::DenseTensor>("Loss");
    auto reduction = ctx.Attr<std::string>("reduction");
    loss->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    if ("none" == reduction) {
      // log(label)
      auto ones_tensor = ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>(
          target->dims(), dev_ctx);
      const auto& ones_runner =
          NpuOpRunner("OnesLike", {*target}, {ones_tensor}, {});
      ones_runner.Run(stream);

      auto sub_tensor = ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>(
          target->dims(), dev_ctx);
      const auto& sub_runner =
          NpuOpRunner("Sub", {*target, ones_tensor}, {sub_tensor}, {});
      sub_runner.Run(stream);

      auto log_target = ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>(
          target->dims(), dev_ctx);
      const auto& log_runner =
          NpuOpRunner("Log1p", {sub_tensor}, {log_target}, {});
      log_runner.Run(stream);

      // log(label) - input
      const auto& sub_runner2 =
          NpuOpRunner("Sub", {log_target, *input}, {*loss}, {});
      sub_runner2.Run(stream);

      // label * (log(label) - input)
      auto min_value =
          ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({1}, dev_ctx);
      auto max_value =
          ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({1}, dev_ctx);
      FillNpuTensorWithConstant(&min_value, static_cast<T>(0));
      FillNpuTensorWithConstant(&max_value, std::numeric_limits<T>::max());

      auto cliped_target = ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>(
          target->dims(), dev_ctx);
      const auto& clip_runner = NpuOpRunner(
          "ClipByValue", {*target, min_value, max_value}, {cliped_target}, {});
      clip_runner.Run(stream);

      const auto& mul_runner =
          NpuOpRunner("Mul", {*loss, cliped_target}, {*loss}, {});
      mul_runner.Run(stream);
    } else if ("batchmean" == reduction || "sum" == reduction) {
      const auto& runner = NpuOpRunner(
          "KLDiv", {*input, *target}, {*loss}, {{"reduction", reduction}});
      runner.Run(stream);
    } else if ("mean" == reduction) {
      const auto& runner = NpuOpRunner("KLDiv",
                                       {*input, *target},
                                       {*loss},
                                       {{"reduction", std::string("sum")}});
      runner.Run(stream);

      const int numel = input->numel();
      const auto& muls_runner =
          NpuOpRunner("Muls",
                      {*loss},
                      {*loss},
                      {{"value", static_cast<float>(1.0 / numel)}});
      muls_runner.Run(stream);
    }
  }
};

template <typename T>
class KLDivLossGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* target = ctx.Input<phi::DenseTensor>("Target");
    auto* loss_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    auto* input_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto reduction = ctx.Attr<std::string>("reduction");
    input_grad->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::NPUDeviceContext>();
    auto stream = dev_ctx.stream();

    Tensor loss_grad_transformed;
    if ("none" == reduction) {
      loss_grad_transformed.ShareDataWith(*loss_grad);
    } else {
      loss_grad_transformed.mutable_data<T>(input_grad->dims(), ctx.GetPlace());

      NpuOpRunner broadcast_runner;
      broadcast_runner.SetType("BroadcastTo");
      broadcast_runner.AddInput(*loss_grad);
      broadcast_runner.AddInput(phi::vectorize<int>(input_grad->dims()));
      broadcast_runner.AddOutput(loss_grad_transformed);
      broadcast_runner.Run(stream);
    }
    auto min_value =
        ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({1}, dev_ctx);
    auto max_value =
        ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>({1}, dev_ctx);
    FillNpuTensorWithConstant(&min_value, static_cast<T>(0));
    FillNpuTensorWithConstant(&max_value, std::numeric_limits<T>::max());

    auto cliped_target = ctx.AllocateTmpTensor<T, platform::NPUDeviceContext>(
        target->dims(), dev_ctx);
    const auto& clip_runner = NpuOpRunner(
        "ClipByValue", {*target, min_value, max_value}, {cliped_target}, {});
    clip_runner.Run(stream);

    const auto& mul_runner = NpuOpRunner(
        "Mul", {cliped_target, loss_grad_transformed}, {*input_grad}, {});
    mul_runner.Run(stream);

    float k = -1.0f;

    if ("mean" == reduction) {
      k = static_cast<float>(-1.0 / input_grad->numel());
    } else if ("batchmean" == reduction) {
      k = static_cast<float>(-1.0 / input_grad->dims()[0]);
    }

    const auto& muls_runner =
        NpuOpRunner("Muls", {*input_grad}, {*input_grad}, {{"value", k}});
    muls_runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(kldiv_loss,
                       ops::KLDivLossNPUKernel<float>,
                       ops::KLDivLossNPUKernel<plat::float16>);

REGISTER_OP_NPU_KERNEL(kldiv_loss_grad,
                       ops::KLDivLossGradNPUKernel<float>,
                       ops::KLDivLossGradNPUKernel<plat::float16>);
