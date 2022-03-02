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
#include "paddle/phi/kernels/sgd_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DGCMomentumKernel : public framework::OpKernel<T> {
 public:
  DGCMomentumKernel()
      : _momentum_op_kernel(new MomentumOpKernel<DeviceContext, T>()) {}

  void Compute(const framework::ExecutionContext& context) const override {
    auto rampup_begin_step = context.Attr<float>("rampup_begin_step");
    if (static_cast<int>(rampup_begin_step) < 0) {
      return;
    }

    auto current_step_tensor = context.Input<framework::Tensor>("current_step");
    auto* current_step = current_step_tensor->data<T>();

    // nranks
    auto nranks_tensor = context.Input<framework::Tensor>("nranks");
    const int nranks = static_cast<const int>(*nranks_tensor->data<float>());
    PADDLE_ENFORCE_GT(
        nranks, 1,
        platform::errors::InvalidArgument(
            "DGC is not useful when num_trainers <= 1, but now nranks=%d",
            nranks));

    const framework::Tensor* g = context.Input<framework::Tensor>("Grad");
    framework::Tensor* g_out = context.Output<framework::Tensor>("Grad_out");
    auto g_e = framework::EigenVector<T>::Flatten(*g);
    auto g_out_e = framework::EigenVector<T>::Flatten(*g_out);

    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& eigen_ctx = *dev_ctx.eigen_device();

    // NOTE. In dgc_op we multi grad with nranks, so we need /nranks here.
    g_out_e.device(eigen_ctx) = (1.0 / nranks) * g_e;

    VLOG(10) << "current_step:" << *current_step
             << ", rampup_begin_step:" << rampup_begin_step;

    if (static_cast<int>(*current_step) < static_cast<int>(rampup_begin_step)) {
      VLOG(10) << " so use momentum optimizer";
      return _momentum_op_kernel->Compute(context);
    }

    VLOG(10) << " so use sgd optimizer";

    const auto* param_var = context.InputVar("Param");
    const auto* grad_var = context.InputVar("Grad");
    auto* learning_rate = context.Input<framework::Tensor>("LearningRate");
    bool multi_precision = context.Attr<bool>("multi_precision");
    if (param_var->IsType<framework::LoDTensor>()) {
      auto* param = context.Input<framework::Tensor>("Param");
      auto* param_out = context.Output<framework::Tensor>("ParamOut");
      auto* master_param_out =
          context.Output<framework::Tensor>("MasterParamOut");
      paddle::optional<const framework::Tensor&> master_param_opt =
          paddle::none;
      if (multi_precision) {
        auto* master_param = context.Input<framework::Tensor>("MasterParam");
        master_param_opt = *master_param;
      }

      if (grad_var->IsType<framework::Tensor>()) {
        // sgd_dense
        auto* grad = context.Input<framework::Tensor>("Grad");
        phi::SGDDenseKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param, *learning_rate, *grad, master_param_opt, multi_precision,
            param_out, master_param_out);
      } else {
        // sgd dense param sparse grad
        auto* grad = context.Input<phi::SelectedRows>("Grad");
        phi::SGDDenseParamSparseGradKernel<T>(
            static_cast<const typename framework::ConvertToPhiContext<
                DeviceContext>::TYPE&>(dev_ctx),
            *param, *learning_rate, *grad, master_param_opt, multi_precision,
            param_out, master_param_out);
      }
    } else if (param_var->IsType<phi::SelectedRows>() &&
               grad_var->IsType<phi::SelectedRows>() &&
               platform::is_cpu_place(context.GetPlace())) {
      // sgd sparse param sparse grad
      auto* param = context.Input<phi::SelectedRows>("Param");
      auto* param_out = context.Output<phi::SelectedRows>("ParamOut");
      auto* master_param_out =
          context.Output<phi::SelectedRows>("MasterParamOut");
      paddle::optional<const phi::SelectedRows&> master_param_opt =
          paddle::none;
      if (multi_precision) {
        auto* master_param = context.Input<phi::SelectedRows>("MasterParam");
        master_param_opt = *master_param;
      }
      auto* grad = context.Input<phi::SelectedRows>("Grad");
      phi::SGDSparseParamSparseGradKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *param, *learning_rate, *grad, master_param_opt, multi_precision,
          param_out, master_param_out);

    } else {
      PADDLE_THROW("gdc not support yet");
    }
  }

 private:
  std::unique_ptr<MomentumOpKernel<DeviceContext, T>> _momentum_op_kernel;
};

}  // namespace operators
}  // namespace paddle
