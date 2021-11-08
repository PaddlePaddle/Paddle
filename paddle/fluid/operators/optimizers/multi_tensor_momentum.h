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
    std::string regularization_method =
        ctx.Attr<std::string>("regularization_method");
    MT regularization_coeff =
        static_cast<MT>(ctx.Attr<float>("regularization_coeff"));
    RegularizationType regularization_flag{
        RegularizationType::kNONE};  // disable regularization
    if (regularization_method == "l2_decay") {
      regularization_flag = RegularizationType::kL2DECAY;
    }
    MT mu = static_cast<MT>(ctx.Attr<float>("mu"));
    MT rescale_grad = static_cast<MT>(ctx.Attr<float>("rescale_grad"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");
    // ipt/opt ctx
    auto params = ctx.MultiInput<framework::Tensor>("Params");
    auto param_outs = ctx.MultiOutput<framework::Tensor>("ParamOuts");
    auto velocitys = ctx.MultiInput<framework::Tensor>("Velocitys");
    auto velocity_outs = ctx.MultiOutput<framework::Tensor>("VelocityOuts");
    auto learning_rates = ctx.MultiInput<framework::Tensor>("LearningRates");
    if (multi_precision) {
      bool has_master =
          ctx.HasInput("MasterParams") && ctx.HasOutput("MasterParamOuts");
      PADDLE_ENFORCE_EQ(
          has_master, true,
          platform::errors::InvalidArgument(
              "The Input(MasterParams) and Output(MasterParamOuts) "
              "should not be null when "
              "the attr `multi_precision` is true"));
      auto master_params = ctx.MultiInput<framework::Tensor>("MasterParams");
      auto master_param_outs =
          ctx.MultiOutput<framework::Tensor>("MasterParamOuts");
    }
    auto grad_vars = ctx.MultiInputVar("Grads");
    if (grad_vars[0]->IsType<framework::LoDTensor>()) {
      auto grads = ctx.MultiInput<framework::Tensor>("Grads");
    } else if (grad_vars[0]->IsType<framework::SelectedRows>()) {
      auto grads = ctx.MultiInput<framework::SelectedRows>("Grads");
    } else {
      PADDLE_ENFORCE_EQ(
          false, true,
          platform::errors::PermissionDenied(
              "Unsupported Variable Type of Grads "
              "in MTMomentumOp. Excepted LodTensor "
              "or SelectedRows, But received [%s]",
              paddle::framework::ToTypeName(grad_vars[0]->Type())));
    }
    // multi_tensor size
    size_t multi_tensor_size = params.size();
    for (auto idx = 0; idx < multi_tensor_size; idx++) {
      param_outs[idx]->mutable_data<T>(ctx.GetPlace());
      velocity_outs[idx]->mutable_data<MT>(ctx.GetPlace());
      const MT* master_in_data =
          multi_precision ? master_params[idx]->data<MT>() : nullptr;
      MT* master_out_data =
          multi_precision
              ? master_param_outs[idx]->mutable_data<MT>(ctx.GetPlace())
              : nullptr;
      if (grad_vars[idx]->IsType<framework::LoDTensor>()) {
        if (platform::is_cpu_place(ctx.GetPlace())) {
          CPUDenseMomentumFunctor<MT> functor;
          functor(params[idx], grads[idx], velocitys[idx], learning_rate, mu,
                  use_nesterov, regularization_flag, regularization_coeff,
                  param_out, velocity_out);

        } else {  // grad is SelectedRows
          functor(params[idx], grads[idx], velocitys[idx], learning_rate, mu,
                  use_nesterov, regularization_flag, regularization_coeff,
                  param_out, velocity_out);
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
