/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <paddle/fluid/operators/optimizers/adam_op.h>

namespace paddle {
namespace operators {

class AdamWOp : public AdamOp {
  using AdamOp::AdamOp;
};

struct CPUAdamW;

template <typename T, typename Flavour>
class AdamWFunctor;

template <typename T>
class AdamWFunctor<T, CPUAdamW> {
 private:
  const float coeff_;
  const float learning_rate_;
  T* param_;

 public:
  AdamWFunctor(const float& coeff, const float& learning_rate, T* param)
      : coeff_(coeff), learning_rate_(learning_rate), param_(param) {}

  inline HOSTDEVICE void operator()(size_t numel) const {
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};
    // Calculation
    param = param * (1.0f - learning_rate_ * coeff_);
  }
};

template <typename DeviceContext, typename T>
class AdamWOpKernel : public AdamOpKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    using paddle::framework::LoDTensor;
    bool skip_update = false;
    // TODO(liupeng):
    if (ctx.HasInput("SkipUpdate")) {
      VLOG(3) << "Has SkipUpdate";
      auto* skip_update_tensor = ctx.Input<framework::Tensor>("SkipUpdate");
      PADDLE_ENFORCE_EQ(skip_update_tensor->numel(), 1,
                        platform::errors::InvalidArgument(
                            "Input(SkipUpdate) size must be 1, but get %d",
                            skip_update_tensor->numel()));
      std::vector<bool> skip_update_vec;
      TensorToVector(*skip_update_tensor, ctx.device_context(),
                     &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    VLOG(3) << "Skip update" << skip_update;
    float coeff = ctx.Attr<float>("weight_decay_coeff");

    if (skip_update || coeff == 0.0f) {
      AdamOpKernel<DeviceContext, T>::Compute(ctx);
      return;
    }

    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    LoDTensor* param;

    if (ctx.HasInput("MasterParam")) {
      // TODO(liupeng): master
      param = const_cast<LoDTensor*>(ctx.Input<LoDTensor>("MasterParam"));
    } else {
      param = const_cast<LoDTensor*>(ctx.Input<LoDTensor>("Param"));
    }

    AdamWFunctor<T, CPUAdamW> functor(coeff, *lr->data<float>(),
                                      param->data<T>());
    functor(param->numel());

    AdamOpKernel<DeviceContext, T>::Compute(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
