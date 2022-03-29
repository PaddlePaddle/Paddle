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

#include <math.h>

#include "paddle/phi/kernels/rmsprop_kernel.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct DenseRmspropGradFunctor {
  inline explicit DenseRmspropGradFunctor(const T *grad) : grad_(grad) {}

  HOSTDEVICE inline T operator()(int64_t idx) const { return grad_[idx]; }

  const T *grad_;
};

template <typename T>
struct SparseRmspropGradFunctor {
  inline SparseRmspropGradFunctor(const T *grad,
                                  const int64_t *rows,
                                  int64_t row_numel,
                                  int64_t row_count)
      : grad_(grad),
        rows_(rows),
        row_numel_(row_numel),
        row_count_(row_count) {}

  HOSTDEVICE inline T operator()(int64_t idx) const {
    auto row_idx =
        phi::funcs::BinarySearch(rows_, row_count_, idx / row_numel_);
    return row_idx >= 0 ? grad_[row_idx * row_numel_ + idx % row_numel_] : 0;
  }

  const T *grad_;
  const int64_t *rows_;
  int64_t row_numel_;
  int64_t row_count_;
};

template <typename T, typename GradFunctor>
struct UncenteredRmspropFunctor {
  UncenteredRmspropFunctor(T *param,
                           T *ms,
                           T *mom,
                           const T *lr,
                           T rho,
                           T epsilon,
                           T momentum,
                           const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    T g = grad_functor_(idx);
    T ms_out = rho_ * ms_[idx] + (1 - rho_) * g * g;
    T mom_out = momentum_ * mom_[idx] + lr_[0] * g / sqrt(ms_out + epsilon_);
    param_[idx] -= mom_out;
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
  }

  T *param_;
  T *ms_;
  T *mom_;
  const T *lr_;
  T rho_;
  T epsilon_;
  T momentum_;
  GradFunctor grad_functor_;
};

template <typename T, typename GradFunctor>
struct CenteredRmspropFunctor {
  CenteredRmspropFunctor(T *param,
                         T *ms,
                         T *mom,
                         T *mean_grad,
                         const T *lr,
                         T rho,
                         T epsilon,
                         T momentum,
                         const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        mean_grad_(mean_grad),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    T g = grad_functor_(idx);
    T ms_out = rho_ * ms_[idx] + (1 - rho_) * g * g;
    T mg_out = rho_ * mean_grad_[idx] + (1 - rho_) * g;
    T mom_out = momentum_ * mom_[idx] +
                lr_[0] * g / sqrt(ms_out - mg_out * mg_out + epsilon_);
    param_[idx] -= mom_out;
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
    mean_grad_[idx] = mg_out;
  }

  T *param_;
  T *ms_;
  T *mom_;
  T *mean_grad_;
  const T *lr_;
  T rho_;
  T epsilon_;
  T momentum_;
  GradFunctor grad_functor_;
};

template <typename T, typename Context>
void RmspropDenseKernel(const Context &ctx,
                        const DenseTensor &param,
                        const DenseTensor &mean_square,
                        const DenseTensor &grad,
                        const DenseTensor &moment,
                        const DenseTensor &learning_rate,
                        paddle::optional<const DenseTensor &> mean_grad_opt,
                        float epsilon_t,
                        float decay_t,
                        float momentum_t,
                        bool centered,
                        DenseTensor *param_out,
                        DenseTensor *moment_out,
                        DenseTensor *mean_square_out,
                        DenseTensor *mean_grad_out) {
  auto epsilon = static_cast<T>(epsilon_t);
  auto rho = static_cast<T>(decay_t);
  auto momentum = static_cast<T>(momentum_t);

  auto &p_tensor = param;
  auto &ms_tensor = mean_square;
  auto &lr_tensor = learning_rate;
  auto &mom_tensor = moment;

  PADDLE_ENFORCE_EQ(p_tensor.IsSharedBufferWith(*param_out),
                    true,
                    phi::errors::InvalidArgument(
                        "Param and ParamOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(mom_tensor.IsSharedBufferWith(*moment_out),
                    true,
                    phi::errors::InvalidArgument(
                        "Moment and MomentOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(
      ms_tensor.IsSharedBufferWith(*mean_square_out),
      true,
      phi::errors::InvalidArgument(
          "MeanSquare and MeanSquareOut must be the same Tensor"));
  size_t limit = static_cast<size_t>(ms_tensor.numel());
  auto &grad_tensor = grad;
  if (paddle::platform::is_cpu_place(ctx.GetPlace())) {
    auto &place = *ctx.eigen_device();
    auto lr_value = lr_tensor.data<T>()[0];

    auto p = EigenVector<T>::Flatten(p_tensor);
    auto ms = EigenVector<T>::Flatten(ms_tensor);
    auto g = EigenVector<T>::Flatten(grad_tensor);
    auto mom = EigenVector<T>::Flatten(mom_tensor);

    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto mom_out = EigenVector<T>::Flatten(*moment_out);
    auto ms_out = EigenVector<T>::Flatten(*mean_square_out);

    ms_out.device(place) = rho * ms + (1 - rho) * g * g;
    if (centered) {
      auto mg_tensor = mean_grad_opt.get_ptr();
      auto mg = EigenVector<T>::Flatten(*mg_tensor);
      PADDLE_ENFORCE_EQ(
          mg_tensor,
          mean_grad_out,
          phi::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
      auto mg_out = EigenVector<T>::Flatten(*mean_grad_out);

      mg_out.device(place) = rho * mg + (1 - rho) * g;
      mom_out.device(place) =
          momentum * mom +
          lr_value * g / (ms_out - mg_out.square() + epsilon).sqrt();
    } else {
      mom_out.device(place) =
          momentum * mom + lr_value * g / (ms_out + epsilon).sqrt();
    }
    p_out.device(place) = p - mom_out;
  } else {
    DenseRmspropGradFunctor<T> grad_func(grad_tensor.data<T>());
    funcs::ForRange<Context> for_range(ctx, limit);
    if (centered) {
      auto mg_tensor = mean_grad_opt.get_ptr();

      PADDLE_ENFORCE_EQ(
          mg_tensor,
          mean_grad_out,
          phi::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
      for_range(CenteredRmspropFunctor<T, DenseRmspropGradFunctor<T>>(
          ctx.template Alloc<T>(param_out),
          ctx.template Alloc<T>(mean_square_out),
          ctx.template Alloc<T>(moment_out),
          ctx.template Alloc<T>(mean_grad_out),
          lr_tensor.data<T>(),
          rho,
          epsilon,
          momentum,
          grad_func));
    } else {
      for_range(UncenteredRmspropFunctor<T, DenseRmspropGradFunctor<T>>(
          ctx.template Alloc<T>(param_out),
          ctx.template Alloc<T>(mean_square_out),
          ctx.template Alloc<T>(moment_out),
          lr_tensor.data<T>(),
          rho,
          epsilon,
          momentum,
          grad_func));
    }
  }
}

template <typename T, typename Context>
void RmspropSparseKernel(const Context &ctx,
                         const DenseTensor &param,
                         const DenseTensor &mean_square,
                         const SelectedRows &grad,
                         const DenseTensor &moment,
                         const DenseTensor &learning_rate,
                         paddle::optional<const DenseTensor &> mean_grad_opt,
                         float epsilon_t,
                         float decay_t,
                         float momentum_t,
                         bool centered,
                         DenseTensor *param_out,
                         DenseTensor *moment_out,
                         DenseTensor *mean_square_out,
                         DenseTensor *mean_grad_out) {
  auto epsilon = static_cast<T>(epsilon_t);
  auto rho = static_cast<T>(decay_t);
  auto momentum = static_cast<T>(momentum_t);

  auto &p_tensor = param;
  auto &ms_tensor = mean_square;
  auto &lr_tensor = learning_rate;
  auto &mom_tensor = moment;

  PADDLE_ENFORCE_EQ(p_tensor.IsSharedBufferWith(*param_out),
                    true,
                    phi::errors::InvalidArgument(
                        "Param and ParamOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(mom_tensor.IsSharedBufferWith(*moment_out),
                    true,
                    phi::errors::InvalidArgument(
                        "Moment and MomentOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(
      ms_tensor.IsSharedBufferWith(*mean_square_out),
      true,
      phi::errors::InvalidArgument(
          "MeanSquare and MeanSquareOut must be the same Tensor"));
  size_t limit = static_cast<size_t>(ms_tensor.numel());

  phi::SelectedRows tmp_merged_grad;
  phi::SelectedRows *merged_grad = &tmp_merged_grad;
  paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
  merge_func(ctx, grad, merged_grad);

  funcs::ForRange<Context> for_range(ctx, limit);
  auto &grad_merge_rows = merged_grad->rows();
  paddle::framework::MixVector<int64_t> mixv_grad_merge_rows(&grad_merge_rows);
  const int64_t *rows = mixv_grad_merge_rows.Data(ctx.GetPlace());

  auto &merged_tensor = merged_grad->value();
  int64_t row_count = merged_grad->rows().size();
  int64_t row_numel = merged_tensor.numel() / row_count;
  SparseRmspropGradFunctor<T> grad_func(
      merged_tensor.data<T>(), rows, row_numel, row_count);

  if (centered) {
    auto mg_tensor = mean_grad_opt.get_ptr();

    PADDLE_ENFORCE_EQ(mg_tensor,
                      mean_grad_out,
                      phi::errors::InvalidArgument(
                          "MeanGrad and MeanGradOut must be the same Tensor"));
    for_range(CenteredRmspropFunctor<T, SparseRmspropGradFunctor<T>>(
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<T>(mean_square_out),
        ctx.template Alloc<T>(moment_out),
        ctx.template Alloc<T>(mean_grad_out),
        lr_tensor.data<T>(),
        rho,
        epsilon,
        momentum,
        grad_func));
  } else {
    for_range(UncenteredRmspropFunctor<T, SparseRmspropGradFunctor<T>>(
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<T>(mean_square_out),
        ctx.template Alloc<T>(moment_out),
        lr_tensor.data<T>(),
        rho,
        epsilon,
        momentum,
        grad_func));
  }
}

}  // namespace phi
