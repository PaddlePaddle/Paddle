/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "paddle/phi/kernels/funcs/squared_l2_norm.h"
#include "paddle/phi/kernels/funcs/tensor_to_string.h"

namespace phi {

namespace scatter = phi::funcs::scatter;

template <typename T, bool IsMultiPrecision>
struct LambMomentREGUpdateFunctor {
  using MT =
      typename std::conditional<IsMultiPrecision,
                                typename phi::dtype::MPTypeTrait<T>::Type,
                                T>::type;

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

  LambMomentREGUpdateFunctor(MT weight_decay,
                             MT beta1,
                             MT beta2,
                             MT epsilon,
                             MT beta1_pow,
                             MT beta2_pow,
                             const MT* mom1,
                             MT* mom1_out,
                             const MT* mom2,
                             MT* mom2_out,
                             const T* grad,
                             const MT* param,
                             MT* trust_ratio_div,
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
  using MT =
      typename std::conditional<IsMultiPrecision,
                                typename phi::dtype::MPTypeTrait<T>::Type,
                                T>::type;

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

  LambMomentMENUpdateFunctor(MT weight_decay,
                             MT beta1,
                             MT beta2,
                             MT epsilon,
                             const MT* beta1_pow,
                             const MT* beta2_pow,
                             const MT* mom1,
                             MT* mom1_out,
                             const MT* mom2,
                             MT* mom2_out,
                             const T* grad,
                             const MT* param,
                             MT* trust_ratio_div,
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

  SparseLambMomentREGUpdateFunctor(T weight_decay,
                                   T beta1,
                                   T beta2,
                                   T epsilon,
                                   T beta1_pow,
                                   T beta2_pow,
                                   const T* mom1,
                                   T* mom1_out,
                                   const T* mom2,
                                   T* mom2_out,
                                   const T* grad,
                                   const T* param,
                                   T* trust_ratio_div,
                                   const int64_t* rows,
                                   int64_t row_numel,
                                   int64_t row_count,
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
        phi::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
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

  SparseLambMomentMENUpdateFunctor(T weight_decay,
                                   T beta1,
                                   T beta2,
                                   T epsilon,
                                   const T* beta1_pow,
                                   const T* beta2_pow,
                                   const T* mom1,
                                   T* mom1_out,
                                   const T* mom2,
                                   T* mom2_out,
                                   const T* grad,
                                   const T* param,
                                   T* trust_ratio_div,
                                   const int64_t* rows,
                                   int64_t row_numel,
                                   int64_t row_count,
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
        phi::funcs::BinarySearch<int64_t>(rows_, row_count_, i / row_numel_);
    T g = row_idx >= 0 ? grad_[row_idx * row_numel_ + i % row_numel_]
                       : static_cast<T>(0);
    update(i, g);
  }
};

template <typename MT, bool NeedUpdateBetaPow /*=true*/>
struct LambBetaPowUpdateFunctor {
  void SetBetaPows(const MT* beta1pow,
                   const MT* beta2pow,
                   MT* beta1pow_out,
                   MT* beta2pow_out,
                   MT beta1,
                   MT beta2) {
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
  void SetBetaPows(const MT* beta1pow UNUSED,
                   const MT* beta2pow UNUSED,
                   MT* beta1pow_out UNUSED,
                   MT* beta2pow_out UNUSED,
                   MT beta1 UNUSED,
                   MT beta2 UNUSED) {}
  HOSTDEVICE void UpdateBetaPow(size_t) const {}
};

template <typename T, typename MT, bool IsMultiPrecision, bool UpdateBetaPow>
struct LambParamUpdateFunctor
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

  LambParamUpdateFunctor(const MT* lr,
                         const T* param,
                         const MT* master_param,
                         const MT* param_norm,
                         const MT* trust_ratio_div,
                         const MT* trust_ratio_div_norm,
                         T* param_out,
                         MT* master_param_out,
                         const bool* skip_update)
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

}  // namespace phi
