/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

template <typename T>
struct LambMomentUpdateFunctor {
  T weight_decay_;
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* grad_;
  const T* param_;
  T* trust_ratio_div_;

  LambMomentUpdateFunctor(T weight_decay, T beta1, T beta2, T epsilon,
                          const T* beta1_pow, const T* beta2_pow, const T* mom1,
                          T* mom1_out, const T* mom2, T* mom2_out,
                          const T* grad, const T* param, T* trust_ratio_div)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    T g = grad_[i];
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T p = param_[i];

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    trust_ratio_div_[i] = mom1 / (sqrt(mom2) + epsilon_) + weight_decay_ * p;
  }
};

template <typename T>
struct SparseLambMomentUpdateFunctor {
  T weight_decay_;
  T beta1_;
  T beta2_;
  T epsilon_;

  const T* beta1_pow_;
  const T* beta2_pow_;
  const T* moment1_;
  T* moment1_out_;
  const T* moment2_;
  T* moment2_out_;
  const T* grad_;
  const T* param_;
  T* trust_ratio_div_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;

  SparseLambMomentUpdateFunctor(T weight_decay, T beta1, T beta2, T epsilon,
                                const T* beta1_pow, const T* beta2_pow,
                                const T* mom1, T* mom1_out, const T* mom2,
                                T* mom2_out, const T* grad, const T* param,
                                T* trust_ratio_div, const int64_t* rows,
                                int64_t row_numel, int64_t row_count)
      : weight_decay_(weight_decay),
        beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        beta1_pow_(beta1_pow),
        beta2_pow_(beta2_pow),
        moment1_(mom1),
        moment1_out_(mom1_out),
        moment2_(mom2),
        moment2_out_(mom2_out),
        grad_(grad),
        param_(param),
        trust_ratio_div_(trust_ratio_div),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  inline HOSTDEVICE void update(size_t i, T g) const {
    // The following code is same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T p = param_[i];

    mom1 = beta1_ * mom1 + (1 - beta1_) * g;
    mom2 = beta2_ * mom2 + (1 - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    trust_ratio_div_[i] = mom1 / (sqrt(mom2) + epsilon_) + weight_decay_ * p;
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    T g = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_] : 0;
    update(i, g);
  }
};

template <typename T>
struct LambParamUpateFunctor {
  const T* lr_;
  const T* param_;
  const T* param_norm_;
  const T* trust_ratio_div_;
  const T* trust_ratio_div_norm_;
  T* param_out_;

  LambParamUpateFunctor(const T* lr, const T* param, const T* param_norm,
                        const T* trust_ratio_div, const T* trust_ratio_div_norm,
                        T* param_out)
      : lr_(lr),
        param_(param),
        param_norm_(param_norm),
        trust_ratio_div_(trust_ratio_div),
        trust_ratio_div_norm_(trust_ratio_div_norm),
        param_out_(param_out) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    T lr = *lr_;
    T p = *param_norm_;
    T t = *trust_ratio_div_norm_;

    T r = (p > 0 && t > 0) ? p / t : 1.0;
    lr *= r;
    param_out_[i] = param_[i] - lr * trust_ratio_div_[i];
  }
};

template <typename DeviceContext, typename T>
class LambOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.InputNames("Param").front(),
                   framework::ToTypeName(param_var->Type()));

    using paddle::framework::LoDTensor;
    using paddle::operators::detail::Ref;

    T weight_decay = static_cast<T>(ctx.Attr<float>("weight_decay"));
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    auto& param = Ref(ctx.Input<LoDTensor>("Param"), "Must set Param.");
    auto* grad_var = ctx.InputVar("Grad");
    auto& mom1 = Ref(ctx.Input<LoDTensor>("Moment1"), "Must set Moment1.");
    auto& mom2 = Ref(ctx.Input<LoDTensor>("Moment2"), "Must set Moment2.");
    auto& lr =
        Ref(ctx.Input<LoDTensor>("LearningRate"), "Must set LearningRate.");

    auto& beta1_pow =
        Ref(ctx.Input<LoDTensor>("Beta1Pow"), "Must set Beta1Pow.");
    auto& beta2_pow =
        Ref(ctx.Input<LoDTensor>("Beta2Pow"), "Must set Beta2Pow.");

    auto& param_out =
        Ref(ctx.Output<LoDTensor>("ParamOut"), "Must set ParamOut.");
    auto& mom1_out =
        Ref(ctx.Output<LoDTensor>("Moment1Out"), "Must set Moment1Out.");
    auto& mom2_out =
        Ref(ctx.Output<LoDTensor>("Moment2Out"), "Must set Moment1Out.");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, param.numel());
    framework::Tensor trust_ratio_div =
        ctx.AllocateTmpTensor<T, DeviceContext>(param.dims(), dev_ctx);

    // Update moments
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = Ref(ctx.Input<LoDTensor>("Grad"), "Must set Grad.");

      LambMomentUpdateFunctor<T> moment_update_functor(
          weight_decay, beta1, beta2, epsilon, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(), mom1.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2.template data<T>(),
          mom2_out.template mutable_data<T>(ctx.GetPlace()),
          grad.template data<T>(), param.template data<T>(),
          trust_ratio_div.template data<T>());
      for_range(moment_update_functor);
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto& grad =
          Ref(ctx.Input<framework::SelectedRows>("Grad"), "Must set Grad.");
      if (grad.rows().size() == 0) {
        VLOG(3) << "grad row size is 0!!";
        return;
      }

      std::vector<int64_t> cpu_rows(grad.rows().begin(), grad.rows().end());
      bool is_strict_sorted = true;
      for (size_t i = 1; i < cpu_rows.size(); ++i) {
        if (cpu_rows[i - 1] >= cpu_rows[i]) {
          is_strict_sorted = false;
          break;
        }
      }

      framework::SelectedRows tmp_grad_merge;
      const framework::SelectedRows* grad_merge_ptr;
      if (is_strict_sorted) {
        grad_merge_ptr = &grad;
      } else {
        // merge duplicated rows if any.
        // The rows of grad_merge have been sorted inside MergeAdd functor
        scatter::MergeAdd<DeviceContext, T> merge_func;
        merge_func(dev_ctx, grad, &tmp_grad_merge, true);
        grad_merge_ptr = &tmp_grad_merge;
      }

      auto& grad_merge = *grad_merge_ptr;
      auto& grad_tensor = grad_merge.value();
      const T* grad_data = grad_tensor.template data<T>();
      const int64_t* rows = grad_merge.rows().Data(ctx.GetPlace());
      auto row_numel = grad_tensor.numel() / grad_merge.rows().size();

      SparseLambMomentUpdateFunctor<T> moment_update_functor(
          weight_decay, beta1, beta2, epsilon, beta1_pow.template data<T>(),
          beta2_pow.template data<T>(), mom1.template data<T>(),
          mom1_out.template mutable_data<T>(ctx.GetPlace()),
          mom2.template data<T>(),
          mom2_out.template mutable_data<T>(ctx.GetPlace()), grad_data,
          param.template data<T>(), trust_ratio_div.template data<T>(), rows,
          row_numel, grad_merge.rows().size());
      for_range(moment_update_functor);
    } else {
      PADDLE_THROW("Variable type not supported by lamb_op.");
    }

    // Update parameter
    framework::Tensor p_norm_t =
        ctx.AllocateTmpTensor<T, DeviceContext>({1}, dev_ctx);
    framework::Tensor trust_ratio_div_norm_t =
        ctx.AllocateTmpTensor<T, DeviceContext>({1}, dev_ctx);
    auto p_norm = framework::EigenScalar<T>::From(p_norm_t);
    auto trust_ratio_div_norm =
        framework::EigenScalar<T>::From(trust_ratio_div_norm_t);

    auto p = framework::EigenVector<T>::Flatten(param);
    auto t = framework::EigenVector<T>::Flatten(trust_ratio_div);

    auto* place = dev_ctx.eigen_device();
    p_norm.device(*place) = p.square().sum().sqrt();
    trust_ratio_div_norm.device(*place) = t.square().sum().sqrt();

    LambParamUpateFunctor<T> param_update_functor(
        lr.template data<T>(), param.template data<T>(),
        p_norm_t.template data<T>(), trust_ratio_div.template data<T>(),
        trust_ratio_div_norm_t.template data<T>(),
        param_out.template mutable_data<T>(ctx.GetPlace()));
    for_range(param_update_functor);
  }
};

}  // namespace operators
}  // namespace paddle
