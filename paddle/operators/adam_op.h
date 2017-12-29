/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include <math.h>  // for sqrt in CPU and CUDA
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/safe_ref.h"
#include "paddle/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct AdamFunctor {
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;

  AdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
              const T* beta2_pow, const T* mom1, T* mom1_out, const T* mom2,
              T* mom2_out, const T* lr, const T* grad, const T* param,
              T* param_out)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        lr_(lr),
        grad_(grad),
        param_(param),
        param_out_(param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    // Merge all memory access together.
    T g = grad_[i];
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);
    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
    p -= lr * (mom1 / (sqrt(mom2) + epsilon_));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = p;
  }
};

template <typename DeviceContext, typename T>
class AdamOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using paddle::framework::LoDTensor;
    using paddle::operators::detail::Ref;

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto& param = Ref(ctx.Input<LoDTensor>("Param"), "Must set Param");
    auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");
    auto& mom1 = Ref(ctx.Input<LoDTensor>("Moment1"), "Must set Moment1");
    auto& mom2 = Ref(ctx.Input<LoDTensor>("Moment2"), "Must set Moment2");
    auto& lr =
        Ref(ctx.Input<LoDTensor>("LearningRate"), "Must set LearningRate");

    auto& beta1_pow =
        Ref(ctx.Input<LoDTensor>("Beta1Pow"), "Must set Beta1Pow");
    auto& beta2_pow =
        Ref(ctx.Input<LoDTensor>("Beta2Pow"), "Must set Beta2Pow");

    auto& param_out =
        Ref(ctx.Output<LoDTensor>("ParamOut"), "Must set ParamOut");
    auto& mom1_out =
        Ref(ctx.Output<LoDTensor>("Moment1Out"), "Must set Moment1Out");
    auto& mom2_out =
        Ref(ctx.Output<LoDTensor>("Moment2Out"), "Must set Moment1Out");

    AdamFunctor<T> functor(beta1, beta2, epsilon, beta1_pow.template data<T>(),
                           beta2_pow.template data<T>(),
                           mom1.template data<T>(),
                           mom1_out.template mutable_data<T>(ctx.GetPlace()),
                           mom2.template data<T>(),
                           mom2_out.template mutable_data<T>(ctx.GetPlace()),
                           lr.template data<T>(), grad.template data<T>(),
                           param.template data<T>(),
                           param_out.template mutable_data<T>(ctx.GetPlace()));
    platform::ForRange<DeviceContext> for_range(
        static_cast<const DeviceContext&>(ctx.device_context()), param.numel());
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
