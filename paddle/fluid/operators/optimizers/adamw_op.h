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
#include <paddle/fluid/operators/optimizers/adam_op.h>

namespace paddle {
namespace operators {

class AdamWOp : public AdamOp {
  using AdamOp::AdamOp;
};

struct GPUAdamW;
struct CPUAdamW;

template <typename T, typename Flavour>
class AdamWFunctor;

template <typename T>
class AdamWFunctor<T, CPUAdamW> {
 private:
  const T coeff_;
  const T lr_ratio_;
  const T* lr_;
  T* param_;

 public:
  AdamWFunctor(const T coeff, const T lr_ratio, const T* lr, T* param)
      : coeff_(coeff), lr_ratio_(lr_ratio), lr_(lr), param_(param) {}

  inline HOSTDEVICE void operator()(size_t numel) const {
    Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> param{
        param_, static_cast<Eigen::Index>(numel)};

    T lr = *lr_;

    // Calculation
    param -= lr * lr_ratio_ * coeff_ * param;
  }
};

template <typename T, typename Flavour, typename MT = T>
class SparseAdamWFunctor;

template <typename T, typename MT>
class SparseAdamWFunctor<T, GPUAdamW, MT> {
 private:
  MT beta1_;
  MT beta2_;
  MT epsilon_;
  MT coeff_;
  MT lr_ratio_;

  const MT* beta1_pow_;
  const MT* beta2_pow_;
  const MT* moment1_;
  MT* moment1_out_;
  const MT* moment2_;
  MT* moment2_out_;
  const MT* lr_;
  const T* grad_;
  const T* param_;
  T* param_out_;
  const MT* master_param_;
  MT* master_param_out_;

  const int64_t* rows_;
  int64_t row_numel_;
  int64_t row_count_;
  bool lazy_mode_;

 public:
  SparseAdamWFunctor(MT beta1, MT beta2, MT epsilon, MT coeff, MT lr_ratio,
                     const MT* beta1_pow, const MT* beta2_pow, const MT* mom1,
                     MT* mom1_out, const MT* mom2, MT* mom2_out, const MT* lr,
                     const T* grad, const T* param, T* param_out,
                     const MT* master_param, MT* master_param_out,
                     const int64_t* rows, int64_t row_numel, int64_t row_count,
                     bool lazy_mode)
      : beta1_(beta1),
        beta2_(beta2),
        epsilon_(epsilon),
        coeff_(coeff),
        lr_ratio_(lr_ratio),
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
        master_param_(master_param),
        master_param_out_(master_param_out),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count),
        lazy_mode_(lazy_mode) {}

  inline HOSTDEVICE void adamw_update(size_t i, MT g) const {
    // The following code is the same as dense
    MT mom1 = moment1_[i];
    MT mom2 = moment2_[i];
    MT lr = *lr_ * lr_ratio_;
    MT lr_orig = lr;
    MT beta1_pow = *beta1_pow_;
    MT beta2_pow = *beta2_pow_;
    MT p = master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);

    // Calculation
    lr *= sqrt(static_cast<MT>(1.0) - beta2_pow) /
          (static_cast<MT>(1.0) - beta1_pow);

    mom1 = beta1_ * mom1 + (static_cast<MT>(1.0) - beta1_) * g;
    mom2 = beta2_ * mom2 + (static_cast<MT>(1.0) - beta2_) * g * g;
    p -= lr_orig * coeff_ * p;
    p -= lr * (mom1 / (sqrt(mom2) +
                       epsilon_ * sqrt(static_cast<MT>(1.0) - beta2_pow)));

    // Write back to global memory
    moment1_out_[i] = mom1;
    moment2_out_[i] = mom2;
    param_out_[i] = static_cast<T>(p);
    if (master_param_out_) {
      master_param_out_[i] = p;
    }
  }

  inline HOSTDEVICE void operator()(size_t i) const {
    auto row_idx =
        math::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    if (lazy_mode_ && row_idx < 0) {
      return;
    } else {
      MT g = row_idx >= 0
                 ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_])
                 : static_cast<MT>(0);
      adamw_update(i, g);
    }
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
      paddle::framework::TensorToVector(*skip_update_tensor,
                                        ctx.device_context(), &skip_update_vec);
      skip_update = skip_update_vec[0];
    }
    VLOG(3) << "Skip update" << skip_update;
    bool with_decay = ctx.Attr<bool>("with_decay");

    if (skip_update || !with_decay) {
      AdamOpKernel<DeviceContext, T>::Compute(ctx);
      return;
    }

    T coeff = static_cast<T>(ctx.Attr<float>("coeff"));
    T lr_ratio = static_cast<T>(ctx.Attr<float>("lr_ratio"));
    auto* lr = ctx.Input<LoDTensor>("LearningRate");

    LoDTensor* param;

    if (ctx.HasInput("MasterParam")) {
      // TODO(liupeng): master
      param = const_cast<LoDTensor*>(ctx.Input<LoDTensor>("MasterParam"));
    } else {
      param = const_cast<LoDTensor*>(ctx.Input<LoDTensor>("Param"));
    }

    AdamWFunctor<T, CPUAdamW> functor(coeff, lr_ratio, lr->data<T>(),
                                      param->data<T>());
    functor(param->numel());

    AdamOpKernel<DeviceContext, T>::Compute(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
