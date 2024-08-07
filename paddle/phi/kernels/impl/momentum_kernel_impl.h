// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "paddle/phi/kernels/momentum_kernel.h"

namespace phi {

template <typename T>
using MultiPrecisionType = typename phi::dtype::MPTypeTrait<T>::Type;

template <typename T>
struct CPUDenseUpdater {
  template <typename G>
  void operator()(const DenseTensor& param,
                  const DenseTensor& velocity,
                  const T& mu,
                  const T& lr,
                  const bool use_nesterov,
                  G&& grad,
                  DenseTensor* param_out,
                  DenseTensor* velocity_out) const {
    auto param_out_vec = EigenVector<T>::Flatten(*param_out);
    auto velocity_out_vec = EigenVector<T>::Flatten(*velocity_out);

    auto param_vec = EigenVector<T>::Flatten(param);
    auto velocity_vec = EigenVector<T>::Flatten(velocity);
    velocity_out_vec = velocity_vec * mu + grad;
    if (use_nesterov) {
      param_out_vec = param_vec - (grad + velocity_out_vec * mu) * lr;
    } else {
      param_out_vec = param_vec - lr * velocity_out_vec;
    }
  }
};

struct NoNesterov;
struct UseNesterov;

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

template <typename T>
class CPUDenseMomentumFunctor {
 public:
  void operator()(const DenseTensor* param,
                  const DenseTensor* grad,
                  const DenseTensor* velocity,
                  const DenseTensor* learning_rate,
                  const T mu,
                  const bool use_nesterov,
                  const RegularizationType regularization_flag,
                  const T regularization_coeff,
                  DenseTensor* param_out,
                  DenseTensor* velocity_out) {
    auto grad_vec = EigenVector<T>::Flatten(*grad);
    auto* lr = learning_rate->data<MultiPrecisionType<T>>();

    CPUDenseUpdater<T> updater;
    if (regularization_flag == RegularizationType::kL2DECAY) {
      auto param_vec = EigenVector<T>::Flatten(*param);
      updater(*param,
              *velocity,
              mu,
              static_cast<T>(lr[0]),
              use_nesterov,
              param_vec * regularization_coeff + grad_vec,
              param_out,
              velocity_out);
    } else {
      updater(*param,
              *velocity,
              mu,
              static_cast<T>(lr[0]),
              use_nesterov,
              grad_vec,
              param_out,
              velocity_out);
    }
  }
};

template <typename T,
          typename TG,
          typename MT,
          RegularizationType kRegType,
          typename UpdateMethod>
class DenseMomentumFunctor;

// NOTE(dzh) for performance.
// avoid if/else in inside kernel, implement GPU UseNesterov/NoNesterov as two
// functor.
template <typename T, typename TG, typename MT, RegularizationType kRegType>
class DenseMomentumFunctor<T, TG, MT, kRegType, UseNesterov> {
 private:
  const T* param_;
  const TG* grad_;
  const MT* velocity_;
  const MultiPrecisionType<MT>* lr_;
  const MT* master_param_;
  const MT mu_;
  const MT rescale_grad_;
  const int64_t num_;
  T* param_out_;
  MT* velocity_out_;
  MT* master_param_out_;
  const MT regularization_coeff_;

 public:
  DenseMomentumFunctor(const T* param,
                       const TG* grad,
                       const MT* velocity,
                       const MultiPrecisionType<MT>* learning_rate,
                       const MT* master_param,
                       const MT mu,
                       const MT rescale_grad,
                       const int64_t num,
                       const MT regularization_coeff,
                       T* param_out,
                       MT* velocity_out,
                       MT* master_param_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(learning_rate),
        master_param_(master_param),
        mu_(mu),
        rescale_grad_(rescale_grad),
        num_(num),
        param_out_(param_out),
        velocity_out_(velocity_out),
        master_param_out_(master_param_out),
        regularization_coeff_(regularization_coeff) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const MT param =
        master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    MT grad = static_cast<MT>(grad_[i]) * rescale_grad_;
    const MT lr = static_cast<MT>(lr_[0]);
    const MT velocity = velocity_[i];

    if (kRegType == RegularizationType::kL2DECAY) {
      grad += regularization_coeff_ * param;
    }

    MT velocity_out = velocity * mu_ + grad;
    MT param_out = param - (grad + velocity_out * mu_) * lr;
    // write register to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T, typename TG, typename MT, RegularizationType kRegType>
class DenseMomentumFunctor<T, TG, MT, kRegType, NoNesterov> {
 private:
  const T* param_;
  const TG* grad_;
  const MT* velocity_;
  const MultiPrecisionType<MT>* lr_;
  const MT* master_param_;
  const MT mu_;
  const MT rescale_grad_;
  const int64_t num_;
  T* param_out_;
  MT* velocity_out_;
  MT* master_param_out_;
  const MT regularization_coeff_;

 public:
  DenseMomentumFunctor(const T* param,
                       const TG* grad,
                       const MT* velocity,
                       const MultiPrecisionType<MT>* learning_rate,
                       const MT* master_param,
                       const MT mu,
                       const MT rescale_grad,
                       const int64_t num,
                       const MT regularization_coeff,
                       T* param_out,
                       MT* velocity_out,
                       MT* master_param_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(learning_rate),
        master_param_(master_param),
        mu_(mu),
        rescale_grad_(rescale_grad),
        num_(num),
        param_out_(param_out),
        velocity_out_(velocity_out),
        master_param_out_(master_param_out),
        regularization_coeff_(regularization_coeff) {}
  inline HOSTDEVICE void operator()(size_t i) const {
    // put memory access in register
    const MT param =
        master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    MT grad = static_cast<MT>(grad_[i]) * rescale_grad_;
    const MT lr = static_cast<MT>(lr_[0]);
    const MT velocity = velocity_[i];

    if (kRegType == RegularizationType::kL2DECAY) {
      grad += regularization_coeff_ * param;
    }

    MT velocity_out = velocity * mu_ + grad;
    MT param_out = param - lr * velocity_out;
    // write register to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T, typename MT, typename UpdateMethod>
class SparseMomentumFunctor;

template <typename T, typename MT>
class SparseMomentumFunctor<T, MT, UseNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const MT* velocity_;
  const MultiPrecisionType<MT>* lr_;
  const MT* master_param_;
  const MT mu_;
  const MT rescale_grad_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* param_out_;
  MT* velocity_out_;
  MT* master_param_out_;
  const RegularizationType regularization_flag_;
  const MT regularization_coeff_;

 public:
  SparseMomentumFunctor(const T* param,
                        const T* grad,
                        const MT* velocity,
                        const MultiPrecisionType<MT>* lr,
                        const MT* master_param,
                        const MT mu,
                        const MT rescale_grad,
                        const int64_t* rows,
                        int64_t row_numel,
                        int64_t row_height,
                        const RegularizationType regularization_flag,
                        const MT regularization_coeff,
                        T* param_out,
                        MT* velocity_out,
                        MT* master_param_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(lr),
        master_param_(master_param),
        mu_(mu),
        rescale_grad_(rescale_grad),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        param_out_(param_out),
        velocity_out_(velocity_out),
        master_param_out_(master_param_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        phi::funcs::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    MT grad =
        row_idx >= 0
            ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_]) *
                  rescale_grad_
            : static_cast<MT>(0);
    // put memory access in register
    const MT param =
        master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    MT velocity_out = velocity * mu_ + grad;
    MT param_out = param - (grad + velocity_out * mu_) * lr;
    // write register to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T, typename MT>
class SparseMomentumFunctor<T, MT, NoNesterov> {
 private:
  const T* param_;
  const T* grad_;
  const MT* velocity_;
  const MultiPrecisionType<MT>* lr_;
  const MT* master_param_;
  const MT mu_;
  const MT rescale_grad_;
  const int64_t* rows_;
  const int64_t row_numel_;
  const int64_t row_height_;
  T* param_out_;
  MT* velocity_out_;
  MT* master_param_out_;
  const RegularizationType regularization_flag_;
  const MT regularization_coeff_;

 public:
  SparseMomentumFunctor(const T* param,
                        const T* grad,
                        const MT* velocity,
                        const MultiPrecisionType<MT>* lr,
                        const MT* master_param,
                        const MT mu,
                        const MT rescale_grad,
                        const int64_t* rows,
                        int64_t row_numel,
                        int64_t row_height,
                        const RegularizationType regularization_flag,
                        const MT regularization_coeff,
                        T* param_out,
                        MT* velocity_out,
                        MT* master_param_out)
      : param_(param),
        grad_(grad),
        velocity_(velocity),
        lr_(lr),
        master_param_(master_param),
        mu_(mu),
        rescale_grad_(rescale_grad),
        rows_(rows),
        row_numel_(row_numel),
        row_height_(row_height),
        param_out_(param_out),
        velocity_out_(velocity_out),
        master_param_out_(master_param_out),
        regularization_flag_(regularization_flag),
        regularization_coeff_(regularization_coeff) {}

  inline HOSTDEVICE void operator()(size_t i) {
    auto row_idx =
        phi::funcs::BinarySearch<int64_t>(rows_, row_height_, i / row_numel_);
    MT grad =
        row_idx >= 0
            ? static_cast<MT>(grad_[row_idx * row_numel_ + i % row_numel_]) *
                  rescale_grad_
            : static_cast<MT>(0);
    // put memory access in register
    const MT param =
        master_param_ ? master_param_[i] : static_cast<MT>(param_[i]);
    const MT lr = static_cast<MT>(lr_[0]);
    const MT velocity = velocity_[i];

    grad = regularization_flag_ == RegularizationType::kL2DECAY
               ? grad + regularization_coeff_ * param
               : grad;

    MT velocity_out = velocity * mu_ + grad;
    MT param_out = param - velocity_out * lr;
    // write register to memory
    velocity_out_[i] = velocity_out;
    param_out_[i] = static_cast<T>(param_out);
    if (master_param_out_) {
      master_param_out_[i] = param_out;
    }
  }
};

template <typename T, typename MT, typename Context>
void MomentumDenseImpl(const Context& ctx,
                       const DenseTensor& param,
                       const DenseTensor& grad,
                       const DenseTensor& velocity,
                       const DenseTensor& learning_rate,
                       const paddle::optional<DenseTensor>& master_param_opt,
                       float mu_t,
                       bool use_nesterov,
                       const std::string& regularization_method,
                       float regularization_coeff_t,
                       bool multi_precision,
                       float rescale_grad_t,
                       DenseTensor* param_out,
                       DenseTensor* velocity_out,
                       DenseTensor* master_param_out) {
  MT regularization_coeff = static_cast<MT>(regularization_coeff_t);
  RegularizationType regularization_flag{
      RegularizationType::kNONE};  // disable regularization
  if (regularization_method == "l2_decay") {
    regularization_flag = RegularizationType::kL2DECAY;
  }
  MT mu = static_cast<MT>(mu_t);
  MT rescale_grad = static_cast<MT>(rescale_grad_t);
  auto master_param = master_param_opt.get_ptr();
  if (multi_precision) {
    bool has_master = ((master_param_opt.get_ptr() != nullptr) &&
                       (master_param_out != nullptr));
    PADDLE_ENFORCE_EQ(has_master,
                      true,
                      common::errors::InvalidArgument(
                          "The Input(MasterParam) and Output(MasterParamOut) "
                          "should not be null when "
                          "the attr `multi_precision` is true"));
  }

  ctx.template Alloc<T>(param_out);
  ctx.template Alloc<MT>(velocity_out);
  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? ctx.template Alloc<MT>(master_param_out) : nullptr;
  if (ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    CPUDenseMomentumFunctor<MT> functor;
    functor(&param,
            &grad,
            &velocity,
            &learning_rate,
            mu,
            use_nesterov,
            regularization_flag,
            regularization_coeff,
            param_out,
            velocity_out);
  } else if (ctx.GetPlace().GetType() == phi::AllocationType::GPU) {
    funcs::ForRange<Context> for_range(ctx, param.numel());
    const auto grad_type = grad.dtype();
#define PADDLE_LAUNCH_DENSE_MOMENTUM_KERNEL(__nesterov, __reg_type)     \
  if (grad_type == phi::DataType::FLOAT32) {                            \
    DenseMomentumFunctor<T, float, MT, __reg_type, __nesterov> functor( \
        param.data<T>(),                                                \
        grad.data<float>(),                                             \
        velocity.data<MT>(),                                            \
        learning_rate.data<MultiPrecisionType<T>>(),                    \
        master_in_data,                                                 \
        mu,                                                             \
        rescale_grad,                                                   \
        param.numel(),                                                  \
        regularization_coeff,                                           \
        ctx.template Alloc<T>(param_out),                               \
        ctx.template Alloc<MT>(velocity_out),                           \
        master_out_data);                                               \
    for_range(functor);                                                 \
  } else {                                                              \
    DenseMomentumFunctor<T, T, MT, __reg_type, __nesterov> functor(     \
        param.data<T>(),                                                \
        grad.data<T>(),                                                 \
        velocity.data<MT>(),                                            \
        learning_rate.data<MultiPrecisionType<T>>(),                    \
        master_in_data,                                                 \
        mu,                                                             \
        rescale_grad,                                                   \
        param.numel(),                                                  \
        regularization_coeff,                                           \
        ctx.template Alloc<T>(param_out),                               \
        ctx.template Alloc<MT>(velocity_out),                           \
        master_out_data);                                               \
    for_range(functor);                                                 \
  }

    if (use_nesterov) {
      if (regularization_flag == RegularizationType::kL2DECAY) {
        PADDLE_LAUNCH_DENSE_MOMENTUM_KERNEL(UseNesterov,
                                            RegularizationType::kL2DECAY);
      } else {
        PADDLE_LAUNCH_DENSE_MOMENTUM_KERNEL(UseNesterov,
                                            RegularizationType::kNONE);
      }
    } else {
      if (regularization_flag == RegularizationType::kL2DECAY) {
        PADDLE_LAUNCH_DENSE_MOMENTUM_KERNEL(NoNesterov,
                                            RegularizationType::kL2DECAY);
      } else {
        PADDLE_LAUNCH_DENSE_MOMENTUM_KERNEL(NoNesterov,
                                            RegularizationType::kNONE);
      }
    }
  }
}

template <typename T, typename MT, typename Context>
void MomentumSparseImpl(const Context& ctx,
                        const DenseTensor& param,
                        const SelectedRows& grad,
                        const DenseTensor& velocity,
                        const DenseTensor& learning_rate,
                        const paddle::optional<DenseTensor>& master_param_opt,
                        float mu_t,
                        bool use_nesterov,
                        const std::string& regularization_method,
                        float regularization_coeff_t,
                        bool multi_precision,
                        float rescale_grad_t,
                        DenseTensor* param_out,
                        DenseTensor* velocity_out,
                        DenseTensor* master_param_out) {
  MT regularization_coeff = static_cast<MT>(regularization_coeff_t);
  RegularizationType regularization_flag{
      RegularizationType::kNONE};  // disable regularization
  if (regularization_method == "l2_decay") {
    regularization_flag = RegularizationType::kL2DECAY;
  }

  MT mu = static_cast<MT>(mu_t);
  MT rescale_grad = static_cast<MT>(rescale_grad_t);

  auto master_param = master_param_opt.get_ptr();
  if (multi_precision) {
    bool has_master = ((master_param_opt.get_ptr() != nullptr) &&
                       (master_param_out != nullptr));
    PADDLE_ENFORCE_EQ(has_master,
                      true,
                      common::errors::InvalidArgument(
                          "The Input(MasterParam) and Output(MasterParamOut) "
                          "should not be null when "
                          "the attr `multi_precision` is true"));
  }

  ctx.template Alloc<T>(param_out);
  ctx.template Alloc<MT>(velocity_out);

  const MT* master_in_data =
      multi_precision ? master_param->data<MT>() : nullptr;
  MT* master_out_data =
      multi_precision ? ctx.template Alloc<MT>(master_param_out) : nullptr;

  // sparse update maybe empty.
  if (grad.rows().size() == 0) {
    VLOG(3) << "Grad SelectedRows contains no data!";
    return;
  }

  phi::SelectedRows tmp_merged_grad;
  phi::SelectedRows* merged_grad = &tmp_merged_grad;
  phi::funcs::scatter::MergeAdd<Context, T> merge_func;
  merge_func(ctx, grad, merged_grad);

  auto* grad_merge_rows = merged_grad->mutable_rows();
  phi::MixVector<int64_t> mixv_grad_merge_rows(grad_merge_rows);
  const int64_t* rows = mixv_grad_merge_rows.Data(ctx.GetPlace());
  int64_t row_numel = merged_grad->value().numel() / merged_grad->rows().size();
  funcs::ForRange<Context> for_range(ctx, param.numel());
  if (use_nesterov) {
    SparseMomentumFunctor<T, MT, UseNesterov> functor(
        param.data<T>(),
        merged_grad->value().data<T>(),
        velocity.data<MT>(),
        learning_rate.data<MultiPrecisionType<MT>>(),
        master_in_data,
        mu,
        rescale_grad,
        rows,
        row_numel,
        static_cast<int64_t>(merged_grad->rows().size()),
        regularization_flag,
        regularization_coeff,
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<MT>(velocity_out),
        master_out_data);
    for_range(functor);

  } else {
    SparseMomentumFunctor<T, MT, NoNesterov> functor(
        param.data<T>(),
        merged_grad->value().data<T>(),
        velocity.data<MT>(),
        learning_rate.data<MultiPrecisionType<MT>>(),
        master_in_data,
        mu,
        rescale_grad,
        rows,
        row_numel,
        static_cast<int64_t>(merged_grad->rows().size()),
        regularization_flag,
        regularization_coeff,
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<MT>(velocity_out),
        master_out_data);
    for_range(functor);
  }
}

template <typename T, typename Context>
void MomentumDenseKernel(const Context& dev_ctx,
                         const DenseTensor& param,
                         const DenseTensor& grad,
                         const DenseTensor& velocity,
                         const DenseTensor& learning_rate,
                         const paddle::optional<DenseTensor>& master_param,
                         float mu,
                         bool use_nesterov,
                         const std::string& regularization_method,
                         float regularization_coeff,
                         bool multi_precision,
                         float rescale_grad,
                         DenseTensor* param_out,
                         DenseTensor* velocity_out,
                         DenseTensor* master_param_out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  if (multi_precision) {
    MomentumDenseImpl<T, MT>(dev_ctx,
                             param,
                             grad,
                             velocity,
                             learning_rate,
                             master_param,
                             mu,
                             use_nesterov,
                             regularization_method,
                             regularization_coeff,
                             multi_precision,
                             rescale_grad,
                             param_out,
                             velocity_out,
                             master_param_out);
  } else {
    MomentumDenseImpl<T, T>(dev_ctx,
                            param,
                            grad,
                            velocity,
                            learning_rate,
                            master_param,
                            mu,
                            use_nesterov,
                            regularization_method,
                            regularization_coeff,
                            multi_precision,
                            rescale_grad,
                            param_out,
                            velocity_out,
                            master_param_out);
  }
}

template <typename T, typename Context>
void MomentumSparseKernel(const Context& dev_ctx,
                          const DenseTensor& param,
                          const SelectedRows& grad,
                          const DenseTensor& velocity,
                          const DenseTensor& learning_rate,
                          const paddle::optional<DenseTensor>& master_param,
                          float mu,
                          bool use_nesterov,
                          const std::string& regularization_method,
                          float regularization_coeff,
                          bool multi_precision,
                          float rescale_grad,
                          DenseTensor* param_out,
                          DenseTensor* velocity_out,
                          DenseTensor* master_param_out) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  if (multi_precision) {
    MomentumSparseImpl<T, MT>(dev_ctx,
                              param,
                              grad,
                              velocity,
                              learning_rate,
                              master_param,
                              mu,
                              use_nesterov,
                              regularization_method,
                              regularization_coeff,
                              multi_precision,
                              rescale_grad,
                              param_out,
                              velocity_out,
                              master_param_out);
  } else {
    MomentumSparseImpl<T, T>(dev_ctx,
                             param,
                             grad,
                             velocity,
                             learning_rate,
                             master_param,
                             mu,
                             use_nesterov,
                             regularization_method,
                             regularization_coeff,
                             multi_precision,
                             rescale_grad,
                             param_out,
                             velocity_out,
                             master_param_out);
  }
}

}  // namespace  phi
