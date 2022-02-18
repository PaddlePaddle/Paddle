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
#include "paddle/fluid/memory/buffer.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/operators/math/squared_l2_norm.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/algorithm.h"
#include "paddle/pten/kernels/funcs/eigen/extensions.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

template <typename T, bool IsMultiPrecision>
struct LambMomentREGUpdateFunctor {
  using MT = typename std::conditional<
      IsMultiPrecision, typename details::MPTypeTrait<T>::Type, T>::type;

  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  MT beta1_pow_;
  MT* beta1_pow_out_;
  MT beta2_pow_;
  MT* beta2_pow_out_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const MT* param_;
  MT* trust_ratio_div_;
  const bool* skip_update_;

  LambMomentREGUpdateFunctor(MT weight_decay, MT beta1, MT beta2, MT epsilon,
                             MT beta1_pow, MT beta2_pow, const MT* mom1,
                             MT* mom1_out, const MT* mom2, MT* mom2_out,
                             const T* grad, const MT* param,
                             MT* trust_ratio_div, const bool* skip_update)
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
        skip_update_(skip_update) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    if (skip_update_ && *skip_update_) return;

    MT g = static_cast<MT>(grad_[i]);
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = beta1_pow_;
    MT beta2_pow = beta2_pow_;
    MT p = param_[i];

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (Eigen::numext::sqrt(mom2_unbiased) + epsilon_) +
        weight_decay_ * p;
  }
};

template <typename T, bool IsMultiPrecision>
struct LambMomentMENUpdateFunctor {
  using MT = typename std::conditional<
      IsMultiPrecision, typename details::MPTypeTrait<T>::Type, T>::type;

  MT weight_decay_;
  MT beta1_;
  MT beta2_;
  MT epsilon_;

  const MT* beta1_pow_;
  const MT* beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const T* grad_;
  const MT* param_;
  MT* trust_ratio_div_;
  const bool* skip_update_;

  LambMomentMENUpdateFunctor(MT weight_decay, MT beta1, MT beta2, MT epsilon,
                             const MT* beta1_pow, const MT* beta2_pow,
                             const MT* mom1, MT* mom1_out, const MT* mom2,
                             MT* mom2_out, const T* grad, const MT* param,
                             MT* trust_ratio_div, const bool* skip_update)
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
        skip_update_(skip_update) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    if (skip_update_ && *skip_update_) return;
    MT g = static_cast<MT>(grad_[i]);
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = param_[i];

    mom1 = beta1_ * mom1 + (static_cast<MT>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    MT mom1_unbiased = mom1 / (static_cast<MT>(1) - beta1_pow);
    MT mom2_unbiased = mom2 / (static_cast<MT>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (Eigen::numext::sqrt(mom2_unbiased) + epsilon_) +
        weight_decay_ * p;
  }
};

template <typename T>
struct SparseLambMomentREGUpdateFunctor {
  T weight_decay_;
  T beta1_;
  T beta2_;
  T epsilon_;

  T beta1_pow_;
  T beta2_pow_;
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

  const bool* skip_update_;

  SparseLambMomentREGUpdateFunctor(T weight_decay, T beta1, T beta2, T epsilon,
                                   T beta1_pow, T beta2_pow, const T* mom1,
                                   T* mom1_out, const T* mom2, T* mom2_out,
                                   const T* grad, const T* param,
                                   T* trust_ratio_div, const int64_t* rows,
                                   int64_t row_numel, int64_t row_count,
                                   const bool* skip_update)
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
        row_count_(row_count),
        skip_update_(skip_update) {}

  inline HOSTDEVICE void update(size_t i, T g) const {
    // The following code is same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T beta1_pow = beta1_pow_;
    T beta2_pow = beta2_pow_;
    T p = param_[i];

    mom1 = beta1_ * mom1 + (static_cast<T>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<T>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    T mom1_unbiased = mom1 / (static_cast<T>(1) - beta1_pow);
    T mom2_unbiased = mom2 / (static_cast<T>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (Eigen::numext::sqrt(mom2_unbiased) + epsilon_) +
        weight_decay_ * p;
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    if (skip_update_ && *skip_update_) return;
    auto row_idx =
        pten::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    T g = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_]
                       : static_cast<T>(0);
    update(i, g);
  }
};

template <typename T>
struct SparseLambMomentMENUpdateFunctor {
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

  const bool* skip_update_;

  SparseLambMomentMENUpdateFunctor(T weight_decay, T beta1, T beta2, T epsilon,
                                   const T* beta1_pow, const T* beta2_pow,
                                   const T* mom1, T* mom1_out, const T* mom2,
                                   T* mom2_out, const T* grad, const T* param,
                                   T* trust_ratio_div, const int64_t* rows,
                                   int64_t row_numel, int64_t row_count,
                                   const bool* skip_update)
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
        row_count_(row_count),
        skip_update_(skip_update) {}

  inline HOSTDEVICE void update(size_t i, T g) const {
    // The following code is same as dense
    T mom1 = moment1_[i];
    T mom2 = moment2_[i];
    T beta1_pow = *beta1_pow_;
    T beta2_pow = *beta2_pow_;
    T p = param_[i];

    mom1 = beta1_ * mom1 + (static_cast<T>(1) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<T>(1) - beta2_) * g * g;

    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;

    T mom1_unbiased = mom1 / (static_cast<T>(1) - beta1_pow);
    T mom2_unbiased = mom2 / (static_cast<T>(1) - beta2_pow);
    trust_ratio_div_[i] =
        mom1_unbiased / (Eigen::numext::sqrt(mom2_unbiased) + epsilon_) +
        weight_decay_ * p;
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    if (skip_update_ && *skip_update_) return;
    auto row_idx =
        pten::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    T g = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_]
                       : static_cast<T>(0);
    update(i, g);
  }
};

template <typename MT, bool NeedUpdateBetaPow /*=true*/>
struct LambBetaPowUpdateFunctor {
  void SetBetaPows(const MT* beta1pow, const MT* beta2pow, MT* beta1pow_out,
                   MT* beta2pow_out, MT beta1, MT beta2) {
    beta1pow_ = beta1pow;
    beta2pow_ = beta2pow;
    beta1pow_out_ = beta1pow_out;
    beta2pow_out_ = beta2pow_out;
    beta1_ = beta1;
    beta2_ = beta2;
  }

  HOSTDEVICE void UpdateBetaPow(size_t i) const {
    if (i == 0) {
      beta1pow_out_[0] = beta1pow_[0] * beta1_;
      beta2pow_out_[0] = beta2pow_[0] * beta2_;
    }
  }

 private:
  const MT* beta1pow_;
  const MT* beta2pow_;
  MT* beta1pow_out_;
  MT* beta2pow_out_;
  MT beta1_;
  MT beta2_;
};

template <typename MT>
struct LambBetaPowUpdateFunctor<MT, /*NeedUpdateBetaPow=*/false> {
  void SetBetaPows(const MT* beta1pow, const MT* beta2pow, MT* beta1pow_out,
                   MT* beta2pow_out, MT beta1, MT beta2) {}
  HOSTDEVICE void UpdateBetaPow(size_t) const {}
};

template <typename T, typename MT, bool IsMultiPrecision, bool UpdateBetaPow>
struct LambParamUpateFunctor
    : public LambBetaPowUpdateFunctor<MT, UpdateBetaPow> {
  const MT* lr_;
  const T* param_;
  const MT* master_param_;
  const MT* param_norm_;
  const MT* trust_ratio_div_;
  const MT* trust_ratio_div_norm_;
  T* param_out_;
  MT* master_param_out_;

  const bool* skip_update_;

  LambParamUpateFunctor(const MT* lr, const T* param, const MT* master_param,
                        const MT* param_norm, const MT* trust_ratio_div,
                        const MT* trust_ratio_div_norm, T* param_out,
                        MT* master_param_out, const bool* skip_update)
      : lr_(lr),
        param_(param),
        master_param_(master_param),
        param_norm_(param_norm),
        trust_ratio_div_(trust_ratio_div),
        trust_ratio_div_norm_(trust_ratio_div_norm),
        param_out_(param_out),
        master_param_out_(master_param_out),
        skip_update_(skip_update) {}

  inline HOSTDEVICE void operator()(size_t i) const {
    if (skip_update_ && *skip_update_) return;
    MT lr = *lr_;
    MT pn = Eigen::numext::sqrt(*param_norm_);
    MT tn = Eigen::numext::sqrt(*trust_ratio_div_norm_);

    MT r = (pn > static_cast<MT>(0) && tn > static_cast<MT>(0))
               ? pn / tn
               : static_cast<MT>(1);
    lr *= r;
    MT p = IsMultiPrecision ? master_param_[i] : static_cast<MT>(param_[i]);
    MT param_out = p - lr * trust_ratio_div_[i];
    param_out_[i] = static_cast<T>(param_out);
    if (IsMultiPrecision) {
      master_param_out_[i] = param_out;
    }
    this->UpdateBetaPow(i);
  }
};

template <typename DeviceContext, typename T>
class LambOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using MT = typename details::MPTypeTrait<T>::Type;
    bool multi_precision = ctx.Attr<bool>("multi_precision");
    if (multi_precision) {
      ComputeImpl<MT, true>(ctx);
    } else {
      ComputeImpl<T, false>(ctx);
    }
  }

 private:
  template <typename MT, bool IsMultiPrecision>
  void ComputeImpl(const framework::ExecutionContext& ctx) const {
    if (!IsMultiPrecision) {
      constexpr auto kIsSameType = std::is_same<T, MT>::value;
      PADDLE_ENFORCE_EQ(
          kIsSameType, true,
          platform::errors::InvalidArgument(
              "When multi_precision=False, T and MT must be the same type."));
    }
    const auto* skip_update = ctx.Input<framework::LoDTensor>("SkipUpdate");
    const bool* skip_update_flag = skip_update && skip_update->IsInitialized()
                                       ? skip_update->data<bool>()
                                       : nullptr;
    if (skip_update_flag && platform::is_cpu_place(skip_update->place()) &&
        (*skip_update_flag)) {
      return;
    }

    auto weight_decay = static_cast<MT>(ctx.Attr<float>("weight_decay"));
    auto beta1 = static_cast<MT>(ctx.Attr<float>("beta1"));
    auto beta2 = static_cast<MT>(ctx.Attr<float>("beta2"));
    auto epsilon = static_cast<MT>(ctx.Attr<float>("epsilon"));
    const auto& param = GET_DATA_SAFELY(
        ctx.Input<framework::LoDTensor>("Param"), "Input", "Param", "Lamb");
    const auto* grad_var = ctx.InputVar("Grad");
    const auto& mom1 = GET_DATA_SAFELY(
        ctx.Input<framework::LoDTensor>("Moment1"), "Input", "Moment1", "Lamb");
    const auto& mom2 = GET_DATA_SAFELY(
        ctx.Input<framework::LoDTensor>("Moment2"), "Input", "Moment2", "Lamb");
    const auto& lr =
        GET_DATA_SAFELY(ctx.Input<framework::LoDTensor>("LearningRate"),
                        "Input", "LearningRate", "Lamb");

    const auto& beta1_pow =
        GET_DATA_SAFELY(ctx.Input<framework::LoDTensor>("Beta1Pow"), "Input",
                        "Beta1Pow", "Lamb");
    const auto& beta2_pow =
        GET_DATA_SAFELY(ctx.Input<framework::LoDTensor>("Beta2Pow"), "Input",
                        "Beta2Pow", "Lamb");

    auto& param_out =
        GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("ParamOut"), "Output",
                        "ParamOut", "Lamb");
    auto& mom1_out =
        GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Moment1Out"),
                        "Output", "Moment1Out", "Lamb");
    auto& mom2_out =
        GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Moment2Out"),
                        "Output", "Moment2Out", "Lamb");
    auto& beta1_pow_out =
        GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Beta1PowOut"),
                        "Output", "Beta1PowOut", "Lamb");
    auto& beta2_pow_out =
        GET_DATA_SAFELY(ctx.Output<framework::LoDTensor>("Beta2PowOut"),
                        "Output", "Beta2PowOut", "Lamb");
    const auto* master_param =
        IsMultiPrecision ? ctx.Input<framework::LoDTensor>("MasterParam")
                         : nullptr;
    auto* master_param_out =
        IsMultiPrecision ? ctx.Output<framework::LoDTensor>("MasterParamOut")
                         : nullptr;

    if (IsMultiPrecision) {
      PADDLE_ENFORCE_NOT_NULL(master_param,
                              platform::errors::InvalidArgument(
                                  "Input(MasterParam) must be provided when "
                                  "multi_precision=True."));
      PADDLE_ENFORCE_NOT_NULL(master_param_out,
                              platform::errors::InvalidArgument(
                                  "Output(MasterParamOut) must be provided "
                                  "when multi_precision=True."));
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto numel = param.numel();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    auto trust_ratio_div =
        ctx.AllocateTmpTensor<MT, DeviceContext>(param.dims(), dev_ctx);
    auto* trust_ratio_div_ptr = trust_ratio_div.template data<MT>();

    const void* param_ptr = param.data();
    const void* master_param_ptr =
        master_param ? master_param->data() : nullptr;
    void* param_out_ptr = param_out.template mutable_data<T>(ctx.GetPlace());
    void* master_param_out_ptr =
        master_param_out
            ? master_param_out->template mutable_data<MT>(ctx.GetPlace())
            : nullptr;

    // Update moments
    bool should_update_beta_pow_later = false;
    const MT *beta1_pow_ptr = nullptr, *beta2_pow_ptr = nullptr;
    MT *beta1_pow_out_ptr = nullptr, *beta2_pow_out_ptr = nullptr;
    VLOG(10) << "Beta1Pow place: " << beta1_pow.place()
             << " , Beta2Pow place: " << beta2_pow.place();
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto& grad = grad_var->Get<framework::LoDTensor>();
      if (platform::is_gpu_place(ctx.GetPlace()) &&
          beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        LambMomentREGUpdateFunctor<T, IsMultiPrecision> moment_update_functor(
            weight_decay, beta1, beta2, epsilon, *beta1_pow.template data<MT>(),
            *beta2_pow.template data<MT>(), mom1.template data<MT>(),
            mom1_out.template mutable_data<MT>(ctx.GetPlace()),
            mom2.template data<MT>(),
            mom2_out.template mutable_data<MT>(ctx.GetPlace()),
            grad.template data<T>(),
            static_cast<const MT*>(IsMultiPrecision ? master_param_ptr
                                                    : param_ptr),
            trust_ratio_div_ptr, skip_update_flag);
        for_range(moment_update_functor);
        beta1_pow_out.template mutable_data<MT>(platform::CPUPlace())[0] =
            beta1 * beta1_pow.template data<MT>()[0];
        beta2_pow_out.template mutable_data<MT>(platform::CPUPlace())[0] =
            beta2 * beta2_pow.template data<MT>()[0];
      } else {
        beta1_pow_ptr = beta1_pow.template data<MT>();
        beta2_pow_ptr = beta2_pow.template data<MT>();
        beta1_pow_out_ptr =
            beta1_pow_out.template mutable_data<MT>(ctx.GetPlace());
        beta2_pow_out_ptr =
            beta2_pow_out.template mutable_data<MT>(ctx.GetPlace());
        should_update_beta_pow_later = true;
        LambMomentMENUpdateFunctor<T, IsMultiPrecision> moment_update_functor(
            weight_decay, beta1, beta2, epsilon,
            static_cast<const MT*>(beta1_pow_ptr),
            static_cast<const MT*>(beta2_pow_ptr), mom1.template data<MT>(),
            mom1_out.template mutable_data<MT>(ctx.GetPlace()),
            mom2.template data<MT>(),
            mom2_out.template mutable_data<MT>(ctx.GetPlace()),
            grad.template data<T>(),
            static_cast<const MT*>(IsMultiPrecision ? master_param_ptr
                                                    : param_ptr),
            trust_ratio_div_ptr, skip_update_flag);
        for_range(moment_update_functor);
      }
    } else if (grad_var->IsType<pten::SelectedRows>()) {
      PADDLE_ENFORCE_EQ(IsMultiPrecision, false,
                        platform::errors::Unimplemented(
                            "SelectedRows gradient is not supported when "
                            "multi_precision=True."));
      constexpr bool kIsSameType = std::is_same<T, MT>::value;
      PADDLE_ENFORCE_EQ(kIsSameType, true,
                        platform::errors::Unimplemented(
                            "SelectedRows gradient is not supported when "
                            "multi_precision=True."));
      auto& grad = GET_DATA_SAFELY(ctx.Input<pten::SelectedRows>("Grad"),
                                   "Input", "Grad", "Lamb");
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

      pten::SelectedRows tmp_grad_merge;
      const pten::SelectedRows* grad_merge_ptr;
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
      if (platform::is_gpu_place(ctx.GetPlace()) &&
          beta1_pow.place() == platform::CPUPlace() &&
          beta2_pow.place() == platform::CPUPlace()) {
        SparseLambMomentREGUpdateFunctor<T> moment_update_functor(
            static_cast<T>(weight_decay), static_cast<T>(beta1),
            static_cast<T>(beta2), static_cast<T>(epsilon),
            *beta1_pow.template data<T>(), *beta2_pow.template data<T>(),
            mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()), grad_data,
            param.template data<T>(), trust_ratio_div.template data<T>(), rows,
            row_numel, grad_merge.rows().size(), skip_update_flag);
        for_range(moment_update_functor);
        beta1_pow_out.template mutable_data<T>(platform::CPUPlace())[0] =
            static_cast<T>(beta1) * beta1_pow.template data<T>()[0];
        beta2_pow_out.template mutable_data<T>(platform::CPUPlace())[0] =
            static_cast<T>(beta2) * beta2_pow.template data<T>()[0];
      } else {
        beta1_pow_ptr = beta1_pow.template data<MT>();
        beta2_pow_ptr = beta2_pow.template data<MT>();
        beta1_pow_out_ptr =
            beta1_pow_out.template mutable_data<MT>(ctx.GetPlace());
        beta2_pow_out_ptr =
            beta2_pow_out.template mutable_data<MT>(ctx.GetPlace());
        should_update_beta_pow_later = true;
        SparseLambMomentMENUpdateFunctor<T> moment_update_functor(
            static_cast<T>(weight_decay), static_cast<T>(beta1),
            static_cast<T>(beta2), static_cast<T>(epsilon),
            reinterpret_cast<const T*>(beta1_pow_ptr),
            reinterpret_cast<const T*>(beta2_pow_ptr), mom1.template data<T>(),
            mom1_out.template mutable_data<T>(ctx.GetPlace()),
            mom2.template data<T>(),
            mom2_out.template mutable_data<T>(ctx.GetPlace()), grad_data,
            param.template data<T>(), trust_ratio_div.template data<T>(), rows,
            row_numel, grad_merge.rows().size(), skip_update_flag);
        for_range(moment_update_functor);
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Variable type not supported by lamb_op. Expect LoDTensor or "
          "SelectedRows, but got %s",
          framework::ToTypeName(grad_var->Type())));
    }

    // Update parameter
    auto p_norm_t = ctx.AllocateTmpTensor<MT, DeviceContext>({1}, dev_ctx);
    auto* p_norm_ptr = p_norm_t.template data<MT>();

    auto trust_ratio_div_norm_t =
        ctx.AllocateTmpTensor<MT, DeviceContext>({1}, dev_ctx);
    auto* trust_ratio_div_norm_ptr = trust_ratio_div_norm_t.template data<MT>();

    // TODO(zengjinle): remove the following Eigen operations when
    // *skip_update == true.
    memory::Buffer buffer(dev_ctx.GetPlace());
    math::SquaredL2Norm(
        dev_ctx, reinterpret_cast<const MT*>(IsMultiPrecision ? master_param_ptr
                                                              : param_ptr),
        p_norm_ptr, numel, &buffer);
    math::SquaredL2Norm(dev_ctx, trust_ratio_div_ptr, trust_ratio_div_norm_ptr,
                        numel, &buffer);

#define CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(__should_update_beta_pow)         \
  do {                                                                       \
    LambParamUpateFunctor<T, MT, IsMultiPrecision, __should_update_beta_pow> \
    param_update_functor(                                                    \
        lr.template data<MT>(), static_cast<const T*>(param_ptr),            \
        static_cast<const MT*>(master_param_ptr), p_norm_ptr,                \
        trust_ratio_div_ptr, trust_ratio_div_norm_ptr,                       \
        static_cast<T*>(param_out_ptr),                                      \
        static_cast<MT*>(master_param_out_ptr), skip_update_flag);           \
    if (__should_update_beta_pow) {                                          \
      param_update_functor.SetBetaPows(beta1_pow_ptr, beta2_pow_ptr,         \
                                       beta1_pow_out_ptr, beta2_pow_out_ptr, \
                                       beta1, beta2);                        \
    }                                                                        \
    for_range(param_update_functor);                                         \
  } while (0)

    if (should_update_beta_pow_later) {
      CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(true);
    } else {
      CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC(false);
    }

#undef CALL_PADDLE_UPDATE_LAMB_PARAM_FUNC
  }
};

}  // namespace operators
}  // namespace paddle
