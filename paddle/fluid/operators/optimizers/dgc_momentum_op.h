// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>

#include "paddle/fluid/operators/optimizers/momentum_op.h"
#include "paddle/phi/kernels/momentum_kernel.h"
#include "paddle/phi/kernels/sgd_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCMomentumKernel : public framework::OpKernel<T> {
 public:
  DGCMomentumKernel() {}

  void Compute(const framework::ExecutionContext& context) const override {
    auto rampup_begin_step = context.Attr<float>("rampup_begin_step");
    if (static_cast<int>(rampup_begin_step) < 0) {
      return;
    }

    auto current_step_tensor = context.Input<phi::DenseTensor>("current_step");
    auto* current_step = current_step_tensor->data<T>();

    // nranks
    auto nranks_tensor = context.Input<phi::DenseTensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(
        nranks,
        1,
        platform::errors::InvalidArgument(
            "DGC is not useful when num_trainers <= 1, but now nranks=%d",
            nranks));

    const phi::DenseTensor* g = context.Input<phi::DenseTensor>("Grad");
    phi::DenseTensor* g_out = context.Output<phi::DenseTensor>("Grad_out");
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto g_out_e = framework::EigenVector<T>::Flatten(*g_out);

    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();

    // NOTE. In dgc_op we multi grad with nranks, so we need /nranks here.
    g_out_e.device(eigen_ctx) = (1.0 / nranks) * g_e;

    VLOG(10) << "current_step:" << *current_step
             << ", rampup_begin_step:" << rampup_begin_step;

    const auto* grad_var = context.InputVar("Grad");
    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << " so use momentum optimizer";
      auto* learning_rate = context.Input<phi::DenseTensor>("LearningRate");
      bool multi_precision = context.Attr<bool>("multi_precision");

      auto* param = context.Input<phi::DenseTensor>("Param");
      auto* velocity = context.Input<phi::DenseTensor>("Velocity");
      auto* param_out = context.Output<phi::DenseTensor>("ParamOut");
      auto* velocity_out = context.Output<phi::DenseTensor>("VelocityOut");
      auto* master_param_out =
          context.Output<phi::DenseTensor>("MasterParamOut");
      paddle::optional<phi::DenseTensor> master_param_opt(paddle::none);
      float mu = context.Attr<float>("mu");
      bool use_nesterov = context.Attr<bool>("use_nesterov");
      std::string regularization_method =
          context.Attr<std::string>("regularization_method");
      float regularization_coeff = context.Attr<float>("regularization_coeff");
      float rescale_grad = context.Attr<float>("rescale_grad");

      if (grad_var->IsType<phi::DenseTensor>()) {
        // sgd_dense
        auto* grad = context.Input<phi::DenseTensor>("Grad");
        phi::MomentumDenseKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param,
            *grad,
            *velocity,
            *learning_rate,
            master_param_opt,
            mu,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out);
      } else {
        // sgd dense param sparse grad
        auto* grad = context.Input<phi::SelectedRows>("Grad");
        phi::MomentumSparseKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param,
            *grad,
            *velocity,
            *learning_rate,
            master_param_opt,
            mu,
            use_nesterov,
            regularization_method,
            regularization_coeff,
            multi_precision,
            rescale_grad,
            param_out,
            velocity_out,
            master_param_out);
      }

      return;
    }

    VLOG(10) << " so use sgd optimizer";

    const auto* param_var = context.InputVar("Param");

    auto* learning_rate = context.Input<phi::DenseTensor>("LearningRate");
    bool multi_precision = context.Attr<bool>("multi_precision");
    if (param_var->IsType<framework::LoDTensor>()) {
      auto* param = context.Input<phi::DenseTensor>("Param");
      auto* param_out = context.Output<phi::DenseTensor>("ParamOut");
      auto* master_param_out =
          context.Output<phi::DenseTensor>("MasterParamOut");
      paddle::optional<phi::DenseTensor> master_param_opt(paddle::none);
      if (multi_precision) {
        auto* master_param = context.Input<phi::DenseTensor>("MasterParam");
        master_param_opt = *master_param;
      }

      if (grad_var->IsType<phi::DenseTensor>()) {
        // sgd_dense
        auto* grad = context.Input<phi::DenseTensor>("Grad");
        phi::SGDDenseKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param,
            *learning_rate,
            *grad,
            master_param_opt,
            multi_precision,
            param_out,
            master_param_out);
      } else {
        // sgd dense param sparse grad
        auto* grad = context.Input<phi::SelectedRows>("Grad");
        phi::SGDDenseParamSparseGradKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param,
            *learning_rate,
            *grad,
            master_param_opt,
            multi_precision,
            param_out,
            master_param_out);
      }
    } else if (param_var->IsType<phi::SelectedRows>() &&
               grad_var->IsType<phi::SelectedRows>() &&
               platform::is_cpu_place(context.GetPlace())) {
      // sgd sparse param sparse grad
      auto* param = context.Input<phi::SelectedRows>("Param");
      auto* param_out = context.Output<phi::SelectedRows>("ParamOut");
      auto* master_param_out =
          context.Output<phi::SelectedRows>("MasterParamOut");
      paddle::optional<phi::SelectedRows> master_param_opt(paddle::none);
      if (multi_precision) {
        auto* master_param = context.Input<phi::SelectedRows>("MasterParam");
        master_param_opt = *master_param;
      }
      auto* grad = context.Input<phi::SelectedRows>("Grad");
      phi::SGDSparseParamSparseGradKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *param,
          *learning_rate,
          *grad,
          master_param_opt,
          multi_precision,
          param_out,
          master_param_out);

    } else {
      PADDLE_THROW("gdc not support yet");
    }
  }
};

}  // namespace operators
}  // namespace paddle
