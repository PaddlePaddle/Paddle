// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T, typename AttrT>
struct Pow2DecayWithLinearWarmupFunctor {
  template <typename U>
  using RestrictPtr = U* PADDLE_RESTRICT;

 public:
  HOSTDEVICE Pow2DecayWithLinearWarmupFunctor(RestrictPtr<T> lr,
                                              RestrictPtr<int64_t> step,
                                              size_t warmup_steps,
                                              size_t total_steps,
                                              AttrT base_lr,
                                              AttrT end_lr)
      : lr_(lr),
        step_(step),
        warmup_steps_(warmup_steps),
        total_steps_(total_steps),
        base_lr_(base_lr),
        end_lr_(end_lr) {}

  HOSTDEVICE void operator()(size_t) const {
    size_t step = static_cast<size_t>(*step_) + 1;
    *step_ = static_cast<int64_t>(step);
    if (step <= warmup_steps_) {
      auto new_lr = static_cast<double>(step) / warmup_steps_ * base_lr_;
      *lr_ = static_cast<T>(new_lr);
    } else if (step < total_steps_) {
      auto factor = 1 - static_cast<double>(step - warmup_steps_) /
                            (total_steps_ - warmup_steps_);
      auto new_lr =
          static_cast<double>(base_lr_ - end_lr_) * (factor * factor) + end_lr_;
      *lr_ = static_cast<T>(new_lr);
    } else {
      *lr_ = static_cast<T>(end_lr_);
    }
  }

 private:
  RestrictPtr<T> lr_;
  RestrictPtr<int64_t> step_;
  size_t warmup_steps_;
  size_t total_steps_;
  AttrT base_lr_;
  AttrT end_lr_;
};

template <typename T, typename Context>
void Pow2DecayWithLinearWarmupKernel(const Context& dev_ctx,
                                     const DenseTensor& lr,
                                     const DenseTensor& step,
                                     int64_t warmup_steps,
                                     int64_t total_steps,
                                     float base_lr,
                                     float end_lr,
                                     DenseTensor* lr_out,
                                     DenseTensor* step_out) {
  PADDLE_ENFORCE_EQ(&lr,
                    lr_out,
                    phi::errors::InvalidArgument("Input(LearningRate) and "
                                                 "Output(LearningRateOut) "
                                                 "must be the same."));
  PADDLE_ENFORCE_EQ(&step,
                    step_out,
                    phi::errors::InvalidArgument(
                        "Input(Step) and Output(StepOut) must be the same."));
  PADDLE_ENFORCE_EQ(
      step.IsInitialized(),
      true,
      phi::errors::InvalidArgument("Input(Step) must be initialized."));

  PADDLE_ENFORCE_LE(warmup_steps,
                    total_steps,
                    phi::errors::InvalidArgument(
                        "warmup_steps must not be larger than total_steps."));

  auto* lr_data = lr_out->data<T>();
  auto* step_data = step_out->data<int64_t>();
  phi::funcs::ForRange<Context> for_range(dev_ctx, 1);
  using AttrT = double;
  Pow2DecayWithLinearWarmupFunctor<T, AttrT> functor(
      lr_data,
      step_data,
      static_cast<size_t>(warmup_steps),
      static_cast<size_t>(total_steps),
      static_cast<AttrT>(base_lr),
      static_cast<AttrT>(end_lr));
  for_range(functor);
}
}  // namespace phi
