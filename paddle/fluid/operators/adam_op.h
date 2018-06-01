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
#include <math.h>  // for sqrt in CPU and CUDA
#include <Eigen/Dense>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

struct GPUAdam;
struct CPUAdam;

template <typename T, typename Flavour>
struct AdamFunctor;

template <typename T>
struct AdamFunctor<T, GPUAdam> {
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

template <typename T>
struct AdamFunctor<T, CPUAdam> {
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

  void operator()(size_t numel) const {
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> g{
        grad_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom1{
        moment1_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> mom2{
        moment2_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param_out{
        param_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment1_out{
        moment1_out_, static_cast<Eigen::Index>(numel)};
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> moment2_out{
        moment2_out_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;

    // Calculation
    lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

    moment1_out = beta1_ * mom1 + (1 - beta1_) * g;
    moment2_out = beta2_ * mom2 + (1 - beta2_) * g * g;
    param_out = param - lr * (moment1_out / (moment2_out.sqrt() + epsilon_));
  }
};

template <typename T>
struct SparseAdamFunctor {
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

  const int64_t* rows_;
  int64_t row_numel_;

  SparseAdamFunctor(T beta1, T beta2, T epsilon, const T* beta1_pow,
                    const T* beta2_pow, const T* mom1, T* mom1_out,
                    const T* mom2, T* mom2_out, const T* lr, const T* grad,
                    const T* param, T* param_out, const int64_t* rows,
                    int64_t row_numel)
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
        param_out_(param_out),
        rows_(rows),
        row_numel_(row_numel) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    for (int64_t j = 0; j < row_numel_; ++j) {
      T g = grad_[i * row_numel_ + j];
      T mom1 = moment1_[rows_[i] * row_numel_ + j];
      T mom2 = moment2_[rows_[i] * row_numel_ + j];
      T lr = *lr_;
      T p = param_[rows_[i] * row_numel_ + j];

      lr *= sqrt(1 - beta2_pow) / (1 - beta1_pow);

      mom1 = beta1_ * mom1 + (1 - beta1_) * g;
      mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;
      p -= lr * (mom1 / (sqrt(mom2) + epsilon_));

      moment1_out_[rows_[i] * row_numel_ + j] = mom1;
      moment2_out_[rows_[i] * row_numel_ + j] = mom2;
      param_out_[rows_[i] * row_numel_ + j] = p;
    }  // for col id
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
    // auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");
    auto* grad_var = ctx.InputVar("Grad");
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

    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad");

      if (platform::is_cpu_place(ctx.GetPlace())) {
        AdamFunctor<T, CPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad.template data<T>(),
            param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()));
        functor(param.numel());
      } else if (platform::is_gpu_place(ctx.GetPlace())) {
        AdamFunctor<T, GPUAdam> functor(
            beta1, beta2, epsilon, beta1_pow.template data<T>(),
            beta2_pow.template data<T>(), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()),
            lr.template data<T>(), grad.template data<T>(),
            param.template data<T>(),
            param_out.template mutable_data<T>(ctx.GetPlace()));

        platform::ForRange<DeviceContext> for_range(
            static_cast<const DeviceContext&>(ctx.device_context()),
            param.numel());
        for_range(functor);
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto& grad =
          Ref(ctx.Input<framework::SelectedRows>("Grad"), "Must set Grad");
      // merge duplicated rows if any.
      scatter::MergeAdd<DeviceContext, T> merge_func;
      auto grad_merge =
          merge_func(ctx.template device_context<DeviceContext>(), grad);
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      int64_t* rows = nullptr;
      if (platform::is_gpu_place(ctx.GetPlace())) {
        rows = grad_merge.mutable_rows()->CUDAMutableData(ctx.GetPlace());
      } else {
        rows = grad_merge.mutable_rows()->data();
      }
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      SparseAdamFunctor<T> functor(
          beta1, beta2, epsilon, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(), mom1.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2.template data<T>(),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          lr.template data<T>(), grad_data, param.template data<T>(),
          param_out.template mutable_data<T>(ctx.GetPlace()), rows, row_numel);
      platform::ForRange<DeviceContext> for_range(
          static_cast<const DeviceContext&>(ctx.device_context()),
          grad_merge.rows().size());
      for_range(functor);
    } else {
      PADDLE_THROW("Variable type not supported by adam_op");
    }
  }
};

}  // namespace operators
}  // namespace paddle
