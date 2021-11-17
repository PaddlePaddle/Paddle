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

#pragma once
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/optimizers/momentum_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MTMomentumOpKernel : public framework::OpKernel<T> {
  using MPDType = MultiPrecisionType<T>;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      InnerCompute<MPDType>(ctx, multi_precision);
    } else {
      InnerCompute<T>(ctx, multi_precision);
    }
  }

 private:
  template <typename MT>
  void InnerCompute(const framework::ExecutionContext& ctx,
                    const bool multi_precision) const {
    // attr ctx
    std::vector<std::string> regularization_methods =
        ctx.Attr<std::vector<std::string>>("regularization_methods");
    std::vector<float> regularization_coeffs =
        ctx.Attr<std::vector<float>>("regularization_coeffs");
    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    VLOG(10) << "get all attr";

    // ipt/opt ctx
    auto params = ctx.MultiInput<framework::Tensor>("Params");
    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOuts");
    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocitys");
    auto velocity_outs = ctx.MultiOutput<framework::Tensor>("VelocityOuts");
    auto learning_rates = ctx.MultiInput<framework::Tensor>("LearningRates");
    std::vector<const framework::Tensor*> master_params(params.size(), nullptr);
    std::vector<framework::Tensor*> master_param_outs(params.size(), nullptr);
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParams") && ctx.HasOutput("MasterParamOuts");
      PADDLE_ENFORCE_EQ(
          has_master, true,
          platform::errors::InvalidArgument(
              "The Input(MasterParams) and Output(MasterParamOuts) "
              "should not be null when "
              "the attr `multi_precision` is true"));
      master_params = ctx.MultiInput<framework::Tensor>("MasterParams");
      master_param_outs = ctx.MultiOutput<framework::Tensor>("MasterParamOuts");
    }
    auto grad_vars = ctx.MultiInputVar("Grads");
    VLOG(10) << "get input / output";

    if (grad_vars[0]->IsType<framework::LoDTensor>()) {
      VLOG(10) << "compute LodTensor";
      auto grads = ctx.MultiInput<framework::Tensor>("Grads");
      // multi_tensor size
      size_t multi_tensor_size = params.size();
      for (size_t idx = 0; idx < multi_tensor_size; idx++) {
        std::string regularization_method = regularization_methods[idx];
        MT regularization_coeff = static_cast<MT>(regularization_coeffs[idx]);
        RegularizationType regularization_flag{
            RegularizationType::kNONE};  // disable regularization
        if (regularization_method == "l2_decay") {
          regularization_flag = RegularizationType::kL2DECAY;
        }
        param_outs[idx]->mutable_data<T>(ctx.GetPlace());
        velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace());
        const MT* master_in_data =
            multi_precision ? master_params[idx]->data<MT>() : nullptr;
        MT* master_out_data =
            multi_precision
                ? master_param_outs[idx]->mutable_data<MT>(ctx.GetPlace())
                : nullptr;
        if (platform::is_cpu_place(ctx.GetPlace())) {
          CPUDenseMomentumFunctor<MT> functor;
          functor(params[idx], grads[idx], velocitys[idx], learning_rates[idx],
                  mu, use_nesterov, regularization_flag, regularization_coeff,
                  param_outs[idx], velocity_outs[idx]);
        } else if (platform::is_gpu_place(ctx.GetPlace())) {
          platform::ForRange<DeviceContext> for_range(
              static_cast<const DeviceContext&>(ctx.device_context()),
              params[idx]->numel());
#define PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(__nesterov, __reg_type)         \
  DenseMomentumFunctor<T, MT, __reg_type, __nesterov> functor(                \
      params[idx]->data<T>(), grads[idx]->data<T>(),                          \
      velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),       \
      master_in_data, mu, rescale_grad, params[idx]->numel(),                 \
      regularization_coeff, param_outs[idx]->mutable_data<T>(ctx.GetPlace()), \
      velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()), master_out_data); \
  for_range(functor);
          if (use_nesterov) {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  UseNesterov, RegularizationType::kL2DECAY);
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(UseNesterov,
                                                    RegularizationType::kNONE);
            }
          } else {
            if (regularization_flag == RegularizationType::kL2DECAY) {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(
                  NoNesterov, RegularizationType::kL2DECAY);
            } else {
              PADDLE_LAUNCH_DENSE_MTMOMENTUM_KERNEL(NoNesterov,
                                                    RegularizationType::kNONE);
            }
          }
        }
      }
    } else if (grad_vars[0]->IsType<framework::SelectedRows>()) {
      VLOG(10) << "compute SelectedRows";
      auto grads = ctx.MultiInput<framework::SelectedRows>("Grads");
      // multi_tensor size
      size_t multi_tensor_size = params.size();
      for (size_t idx = 0; idx < multi_tensor_size; idx++) {
        std::string regularization_method = regularization_methods[idx];
        MT regularization_coeff = static_cast<MT>(regularization_coeffs[idx]);
        RegularizationType regularization_flag{
            RegularizationType::kNONE};  // disable regularization
        if (regularization_method == "l2_decay") {
          regularization_flag = RegularizationType::kL2DECAY;
        }
        param_outs[idx]->mutable_data<T>(ctx.GetPlace());
        velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace());
        const MT* master_in_data =
            multi_precision ? master_params[idx]->data<MT>() : nullptr;
        MT* master_out_data =
            multi_precision
                ? master_param_outs[idx]->mutable_data<MT>(ctx.GetPlace())
                : nullptr;
        // sparse update maybe empty.
        if (grads[idx]->rows().size() == 0) {
          VLOG(3) << "Grad SelectedRows contains no data!";
          return;
        }
        framework::SelectedRows tmp_merged_grad;
        framework::SelectedRows* merged_grad = &tmp_merged_grad;
        math::scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(ctx.template device_context<DeviceContext>(), *grads[idx],
                   merged_grad);

        const int64_t* rows = merged_grad->rows().Data(ctx.GetPlace());
        int64_t row_numel =
            merged_grad->value().numel() / merged_grad->rows().size();
        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            params[idx]->numel());
        if (use_nesterov) {
          SparseMomentumFunctor<T, MT, UseNesterov> functor(
              params[idx]->data<T>(), merged_grad->value().data<T>(),
              velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),
              master_in_data, mu, rescale_grad, rows, row_numel,
              static_cast<int64_t>(merged_grad->rows().size()),
              regularization_flag, regularization_coeff,
              param_outs[idx]->mutable_data<T>(ctx.GetPlace()),
              velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()),
              master_out_data);
          for_range(functor);
        } else {
          SparseMomentumFunctor<T, MT, NoNesterov> functor(
              params[idx]->data<T>(), merged_grad->value().data<T>(),
              velocitys[idx]->data<MT>(), learning_rates[idx]->data<MPDType>(),
              master_in_data, mu, rescale_grad, rows, row_numel,
              static_cast<int64_t>(merged_grad->rows().size()),
              regularization_flag, regularization_coeff,
              param_outs[idx]->mutable_data<T>(ctx.GetPlace()),
              velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace()),
              master_out_data);
          for_range(functor);
        }
      }
    } else {
      PADDLE_ENFORCE_EQ(
          false, true,
          platform::errors::PermissionDenied(
              "Unsupported Variable Type of Grads "
              "in MTMomentumOp. Excepted LodTensor "
              "or SelectedRows, But received [%s]",
              paddle::framework::ToTypeName(grad_vars[0]->Type())));
    }
  }
};

}  // namespace operators
}  // namespace paddle
