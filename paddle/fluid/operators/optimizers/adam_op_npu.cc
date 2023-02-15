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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AdamNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<phi::DenseTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be phi::DenseTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));
    auto* param = ctx.Input<phi::DenseTensor>("Param");
    auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<phi::DenseTensor>(),
                      true,
                      platform::errors::InvalidArgument(
                          "The Grad(%s)'s type should be phi::DenseTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(param_var->Type())));
    auto* grad = ctx.Input<phi::DenseTensor>("Grad");
    auto* mom1 = ctx.Input<phi::DenseTensor>("Moment1");
    auto* mom2 = ctx.Input<phi::DenseTensor>("Moment2");
    auto* lr = ctx.Input<phi::DenseTensor>("LearningRate");

    auto* beta1_pow = ctx.Input<phi::DenseTensor>("Beta1Pow");
    auto* beta2_pow = ctx.Input<phi::DenseTensor>("Beta2Pow");

    auto* param_out = ctx.Output<phi::DenseTensor>("ParamOut");
    auto* mom1_out = ctx.Output<phi::DenseTensor>("Moment1Out");
    auto* mom2_out = ctx.Output<phi::DenseTensor>("Moment2Out");
    auto* beta1_pow_out = ctx.Output<phi::DenseTensor>("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output<phi::DenseTensor>("Beta2PowOut");

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input<phi::DenseTensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      VLOG(4) << "Adam skip update";
      framework::TensorCopy(
          *param,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          param_out);
      framework::TensorCopy(
          *mom1,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          mom1_out);
      framework::TensorCopy(
          *mom2,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          mom2_out);
      framework::TensorCopy(
          *beta1_pow,
          beta1_pow->place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta1_pow_out);
      framework::TensorCopy(
          *beta2_pow,
          beta2_pow->place(),
          ctx.template device_context<platform::DeviceContext>(),
          beta2_pow_out);
      return;
    }

    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

    param_out->mutable_data<T>(ctx.GetPlace());
    mom1_out->mutable_data<T>(ctx.GetPlace());
    mom2_out->mutable_data<T>(ctx.GetPlace());

    // NOTE(zhiqiu): beta1_pow and beta2_pow may on CPU and not transform
    // place.
    phi::DenseTensor beta1_pow_tmp;
    phi::DenseTensor beta2_pow_tmp;
    if (beta1_pow->place() == platform::CPUPlace()) {
      T beta1 = *beta1_pow->data<T>();
      beta1_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&beta1_pow_tmp, beta1);
      beta1_pow = &beta1_pow_tmp;
    }
    if (beta2_pow->place() == platform::CPUPlace()) {
      T beta2 = *beta2_pow->data<T>();
      beta2_pow_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&beta2_pow_tmp, beta2);
      beta2_pow = &beta2_pow_tmp;
    }

    const phi::DenseTensor* beta1_tensor = nullptr;
    const phi::DenseTensor* beta2_tensor = nullptr;
    const phi::DenseTensor* epsilon_tensor = nullptr;

    phi::DenseTensor beta1_tmp(experimental::DataType::FLOAT32);
    phi::DenseTensor beta2_tmp(experimental::DataType::FLOAT32);
    phi::DenseTensor epsilon_tmp(experimental::DataType::FLOAT32);

    if (ctx.HasInput("Beta1Tensor")) {
      beta1_tensor = ctx.Input<phi::DenseTensor>("Beta1Tensor");
      PADDLE_ENFORCE_EQ(beta1_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(Beta1Tensor) size must be 1, but get %d",
                            beta1_tensor->numel()));
    } else {
      T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
      beta1_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&beta1_tmp, beta1);
      beta1_tensor = &beta1_tmp;
    }

    if (ctx.HasInput("Beta2Tensor")) {
      beta2_tensor = ctx.Input<phi::DenseTensor>("Beta2Tensor");
      PADDLE_ENFORCE_EQ(beta2_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(Beta2Tensor) size must be 1, but get %d",
                            beta2_tensor->numel()));
    } else {
      T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
      beta2_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&beta2_tmp, beta2);
      beta2_tensor = &beta2_tmp;
    }

    if (ctx.HasInput("EpsilonTensor")) {
      epsilon_tensor = ctx.Input<phi::DenseTensor>("EpsilonTensor");
      PADDLE_ENFORCE_EQ(epsilon_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(EpsilonTensor) size must be 1, but get %d",
                            epsilon_tensor->numel()));
    } else {
      T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
      epsilon_tmp.mutable_data<T>({1}, ctx.GetPlace());
      FillNpuTensorWithConstant<T>(&epsilon_tmp, epsilon);
      epsilon_tensor = &epsilon_tmp;
    }

    VLOG(3) << "beta1_pow.numel() : " << beta1_pow->numel()
            << "beta2_pow.numel() : " << beta2_pow->numel();
    VLOG(3) << "param.numel(): " << param->numel();

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(),
                      1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(),
                      1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    const auto& runner = NpuOpRunner("ApplyAdamD",
                                     {
                                         *param,
                                         *mom1,
                                         *mom2,
                                         *beta1_pow,
                                         *beta2_pow,
                                         *lr,
                                         *beta1_tensor,
                                         *beta2_tensor,
                                         *epsilon_tensor,
                                         *grad,
                                     },
                                     {
                                         *param_out,
                                         *mom1_out,
                                         *mom2_out,
                                     },
                                     {});
    runner.Run(stream);

    // NOTE(zhiqiu): ApplyAdamD updates params inplace, so
    // if param and param_out is not same, we need to do copy.
    if (param_out->data<T>() != param->data<T>()) {
      framework::TensorCopy(
          *param,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          param_out);
    }
    if (mom1_out->data<T>() != mom1->data<T>()) {
      framework::TensorCopy(
          *mom1,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          mom1_out);
    }
    if (mom2_out->data<T>() != mom2->data<T>()) {
      framework::TensorCopy(
          *mom2,
          ctx.GetPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          mom2_out);
    }
    if (!use_global_beta_pow) {
      beta1_pow_out->mutable_data<T>(ctx.GetPlace());
      beta2_pow_out->mutable_data<T>(ctx.GetPlace());
      const auto& runner_m1 =
          NpuOpRunner("Mul", {*beta1_pow, *beta1_tensor}, {*beta1_pow_out}, {});
      runner_m1.Run(stream);
      const auto& runner_m2 =
          NpuOpRunner("Mul", {*beta2_pow, *beta2_tensor}, {*beta2_pow_out}, {});
      runner_m2.Run(stream);
    }
  }
};

template <typename T>
class AdamWNPUKernel : public AdamNPUKernel<platform::NPUDeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    VLOG(3) << "NPU AdamW Kernel";
    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      VLOG(3) << "Has SkipUpdate";
      auto* skip_update_tensor = ctx.Input<phi::DenseTensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(),
                        1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      paddle::framework::TensorToVector(
          *skip_update_tensor, ctx.device_context(), &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    VLOG(3) << "Skip update" << skip_update;
    bool with_decay = ctx.Attr<bool>("with_decay");
    if (!skip_update && with_decay) {
      float coeff = ctx.Attr<float>("coeff");
      auto* lr = ctx.Input<phi::DenseTensor>("LearningRate");

      auto place = ctx.GetPlace();

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();

      phi::DenseTensor one(experimental::DataType::FLOAT32);
      phi::DenseTensor decay(experimental::DataType::FLOAT32);
      phi::DenseTensor tmp(experimental::DataType::FLOAT32);

      tmp.mutable_data<float>({1}, place);
      one.mutable_data<float>({1}, place);
      decay.mutable_data<float>({1}, place);

      FillNpuTensorWithConstant<float>(&one, 1.0f);
      framework::NPUAttributeMap attr_input = {{"value", coeff}};

      const auto& runner1 = NpuOpRunner("Muls", {*lr}, {tmp}, attr_input);
      runner1.Run(stream);

      const auto& runner2 = NpuOpRunner("Sub", {one, tmp}, {decay}, {});
      runner2.Run(stream);

      if (ctx.HasInput("MasterParam")) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Master Parma is not supported on npu"));
      } else {
        auto* param_out = ctx.Output<phi::DenseTensor>("ParamOut");
        param_out->mutable_data<T>(ctx.GetPlace());

        const auto* param_var = ctx.InputVar("Param");
        PADDLE_ENFORCE_EQ(param_var->IsType<phi::DenseTensor>(),
                          true,
                          platform::errors::InvalidArgument(
                              "The Var(%s)'s type should be phi::DenseTensor, "
                              "but the received is %s",
                              ctx.InputNames("Param").front(),
                              framework::ToTypeName(param_var->Type())));
        auto* param = ctx.Input<phi::DenseTensor>("Param");

        const auto& runner =
            NpuOpRunner("Mul",
                        {*param, decay},
                        {*const_cast<phi::DenseTensor*>(param)},
                        {});
        runner.Run(stream);
      }
    }
    AdamNPUKernel<platform::NPUDeviceContext, T>::Compute(ctx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    adam,
    ops::AdamNPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::AdamNPUKernel<paddle::platform::NPUDeviceContext,
                       paddle::platform::float16>);

REGISTER_OP_NPU_KERNEL(adamw,
                       ops::AdamWNPUKernel<float>,
                       ops::AdamWNPUKernel<paddle::platform::float16>);
