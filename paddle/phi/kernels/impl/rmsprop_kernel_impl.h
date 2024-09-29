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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/funcs/algorithm.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "paddle/phi/kernels/rmsprop_kernel.h"
namespace phi {

template <typename T, typename Context>
struct RmsFunctor {
  RmsFunctor(const Context &ctx,
             const DenseTensor &param,
             const DenseTensor &mean_square,
             const DenseTensor &grad,
             const DenseTensor &moment,
             const DenseTensor &learning_rate,
             const paddle::optional<DenseTensor> &mean_grad_opt,
             const paddle::optional<DenseTensor> &master_param,
             float epsilon_t,
             float decay_t,
             float momentum_t,
             bool centered,
             bool multi_precision,
             DenseTensor *param_out,
             DenseTensor *moment_out,
             DenseTensor *mean_square_out,
             DenseTensor *mean_grad_out,
             DenseTensor *master_param_outs);
};

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
    return row_idx >= 0 ? grad_[row_idx * row_numel_ + idx % row_numel_]
                        : static_cast<T>(0);
  }

  const T *grad_;
  const int64_t *rows_;
  int64_t row_numel_;
  int64_t row_count_;
};

template <typename T, typename MT, typename GradFunctor>
struct UncenteredRmspropFunctor {
  UncenteredRmspropFunctor(T *param,
                           MT *ms,
                           MT *mom,
                           const MT *lr,
                           MT *master_p,
                           MT rho,
                           MT epsilon,
                           MT momentum,
                           const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        master_p_(master_p),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    MT g = static_cast<MT>(grad_functor_(idx));
    MT l_rho = static_cast<MT>(1) - rho_;
    MT ms_out = rho_ * ms_[idx] + l_rho * g * g;
    MT mom_out = momentum_ * mom_[idx] +
                 static_cast<MT>(lr_[0]) * g / sqrt(ms_out + epsilon_);
    MT p = master_p_ ? master_p_[idx] : static_cast<MT>(param_[idx]);
    MT p_m = p - mom_out;
    param_[idx] = static_cast<T>(p_m);
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
    if (master_p_) master_p_[idx] = p_m;
  }

  T *param_;
  MT *ms_;
  MT *mom_;
  MT *master_p_;
  const MT *lr_;
  MT rho_;
  MT epsilon_;
  MT momentum_;
  GradFunctor grad_functor_;
};

template <typename T, typename MT, typename GradFunctor>
struct CenteredRmspropFunctor {
  CenteredRmspropFunctor(T *param,
                         MT *ms,
                         MT *mom,
                         MT *mean_grad,
                         const MT *lr,
                         MT *master_param,
                         MT rho,
                         MT epsilon,
                         MT momentum,
                         const GradFunctor &grad_functor)
      : param_(param),
        ms_(ms),
        mom_(mom),
        master_p_(master_param),
        mean_grad_(mean_grad),
        lr_(lr),
        rho_(rho),
        epsilon_(epsilon),
        momentum_(momentum),
        grad_functor_(grad_functor) {}

  HOSTDEVICE inline void operator()(int64_t idx) const {
    MT g = static_cast<MT>(grad_functor_(idx));
    MT l_rho = static_cast<MT>(1) - rho_;
    MT ms_out = rho_ * ms_[idx] + l_rho * g * g;
    MT mg_out = rho_ * mean_grad_[idx] + l_rho * g;
    MT mom_out =
        momentum_ * mom_[idx] +
        static_cast<MT>(lr_[0]) * g / sqrt(ms_out - mg_out * mg_out + epsilon_);

    MT p = master_p_ ? master_p_[idx] : static_cast<MT>(param_[idx]);
    MT p_m = p - mom_out;
    param_[idx] = static_cast<T>(p_m);
    ms_[idx] = ms_out;
    mom_[idx] = mom_out;
    mean_grad_[idx] = mg_out;
    if (master_p_) master_p_[idx] = p_m;
  }

  T *param_;
  MT *ms_;
  MT *mom_;
  MT *master_p_;
  MT *mean_grad_;
  const MT *lr_;
  MT rho_;
  MT epsilon_;
  MT momentum_;
  GradFunctor grad_functor_;
};

template <typename T, typename Context>
void RmspropDenseKernel(const Context &ctx,
                        const DenseTensor &param,
                        const DenseTensor &mean_square,
                        const DenseTensor &grad,
                        const DenseTensor &moment,
                        const DenseTensor &learning_rate,
                        const paddle::optional<DenseTensor> &mean_grad_opt,
                        const paddle::optional<DenseTensor> &master_param,
                        float epsilon_t,
                        float decay_t,
                        float momentum_t,
                        bool centered,
                        bool multi_precision,
                        DenseTensor *param_out,
                        DenseTensor *moment_out,
                        DenseTensor *mean_square_out,
                        DenseTensor *mean_grad_out,
                        DenseTensor *master_param_outs) {
  RmsFunctor<T, Context> functor(ctx,
                                 param,
                                 mean_square,
                                 grad,
                                 moment,
                                 learning_rate,
                                 mean_grad_opt,
                                 master_param,
                                 epsilon_t,
                                 decay_t,
                                 momentum_t,
                                 centered,
                                 multi_precision,
                                 param_out,
                                 moment_out,
                                 mean_square_out,
                                 mean_grad_out,
                                 master_param_outs);
}

template <typename T, typename Context>
void RmspropSparseKernel(const Context &ctx,
                         const DenseTensor &param,
                         const DenseTensor &mean_square,
                         const SelectedRows &grad,
                         const DenseTensor &moment,
                         const DenseTensor &learning_rate,
                         const paddle::optional<DenseTensor> &mean_grad_opt,
                         const paddle::optional<DenseTensor> &master_param
                             UNUSED,
                         float epsilon_t,
                         float decay_t,
                         float momentum_t,
                         bool centered,
                         bool multi_precision,
                         DenseTensor *param_out,
                         DenseTensor *moment_out,
                         DenseTensor *mean_square_out,
                         DenseTensor *mean_grad_out,
                         DenseTensor *master_param_outs) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  auto epsilon = static_cast<MPDType>(epsilon_t);
  auto rho = static_cast<MPDType>(decay_t);
  auto momentum = static_cast<MPDType>(momentum_t);

  auto &p_tensor = param;
  auto &ms_tensor = mean_square;
  auto &lr_tensor = learning_rate;
  auto &mom_tensor = moment;

  PADDLE_ENFORCE_EQ(p_tensor.IsSharedBufferWith(*param_out),
                    true,
                    common::errors::InvalidArgument(
                        "Param and ParamOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(mom_tensor.IsSharedBufferWith(*moment_out),
                    true,
                    common::errors::InvalidArgument(
                        "Moment and MomentOut must be the same Tensor"));
  PADDLE_ENFORCE_EQ(
      ms_tensor.IsSharedBufferWith(*mean_square_out),
      true,
      common::errors::InvalidArgument(
          "MeanSquare and MeanSquareOut must be the same Tensor"));
  size_t limit = static_cast<size_t>(ms_tensor.numel());

  phi::SelectedRows tmp_merged_grad;
  phi::SelectedRows *merged_grad = &tmp_merged_grad;
  phi::funcs::scatter::MergeAdd<Context, T> merge_func;
  merge_func(ctx, grad, merged_grad);

  funcs::ForRange<Context> for_range(ctx, limit);
  auto &grad_merge_rows = merged_grad->rows();
  phi::MixVector<int64_t> mixv_grad_merge_rows(&grad_merge_rows);
  const int64_t *rows = mixv_grad_merge_rows.Data(ctx.GetPlace());

  auto &merged_tensor = merged_grad->value();
  int64_t row_count = merged_grad->rows().size();
  int64_t row_numel = merged_tensor.numel() / row_count;
  SparseRmspropGradFunctor<T> grad_func(
      merged_tensor.data<T>(), rows, row_numel, row_count);

  MPDType *master_out_data =
      multi_precision ? ctx.template Alloc<MPDType>(master_param_outs)
                      : nullptr;

  if (centered) {
    auto mg_tensor = mean_grad_opt.get_ptr();
    if (mg_tensor) {
      PADDLE_ENFORCE_EQ(
          mg_tensor->Holder(),
          mean_grad_out->Holder(),
          common::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
    } else {
      PADDLE_ENFORCE_EQ(
          mg_tensor,
          mean_grad_out,
          common::errors::InvalidArgument(
              "MeanGrad and MeanGradOut must be the same Tensor"));
    }

    for_range(CenteredRmspropFunctor<T, MPDType, SparseRmspropGradFunctor<T>>(
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<MPDType>(mean_square_out),
        ctx.template Alloc<MPDType>(moment_out),
        ctx.template Alloc<MPDType>(mean_grad_out),
        lr_tensor.data<MPDType>(),
        master_out_data,
        rho,
        epsilon,
        momentum,
        grad_func));
  } else {
    for_range(UncenteredRmspropFunctor<T, MPDType, SparseRmspropGradFunctor<T>>(
        ctx.template Alloc<T>(param_out),
        ctx.template Alloc<MPDType>(mean_square_out),
        ctx.template Alloc<MPDType>(moment_out),
        lr_tensor.data<MPDType>(),
        master_out_data,
        rho,
        epsilon,
        momentum,
        grad_func));
  }
}

}  // namespace phi
