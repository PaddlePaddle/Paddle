// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {

template <typename T, typename AttrT>
struct Pow2WarmupFunctor {
 public:
  HOSTDEVICE Pow2WarmupFunctor(T *PADDLE_RESTRICT lr,
                               int64_t *PADDLE_RESTRICT step,
                               size_t warmup_steps, size_t total_steps,
                               AttrT start_lr, AttrT base_lr, AttrT end_lr)
      : lr_(lr),
        step_(step),
        warmup_steps_(warmup_steps),
        total_steps_(total_steps),
        start_lr_(start_lr),
        base_lr_(base_lr),
        end_lr_(end_lr) {}

  HOSTDEVICE void operator()(size_t) const {
    size_t step = static_cast<size_t>(*step_);
    *step_ = static_cast<int64_t>(step + 1);
    if (step < warmup_steps_) {
      auto new_lr =
          static_cast<double>(base_lr_ - start_lr_) * step / warmup_steps_ +
          start_lr_;
      *lr_ = static_cast<T>(new_lr);
    } else if (step < total_steps_) {
      auto factor = 1 -
                    static_cast<double>(step - warmup_steps_) /
                        (total_steps_ - warmup_steps_);
      auto new_lr =
          static_cast<double>(base_lr_ - end_lr_) * factor * factor + end_lr_;
      *lr_ = static_cast<T>(new_lr);
    } else {
      *lr_ = static_cast<T>(end_lr_);
    }
  }

 private:
  T *PADDLE_RESTRICT lr_;
  int64_t *PADDLE_RESTRICT step_;
  size_t warmup_steps_;
  size_t total_steps_;
  AttrT start_lr_;
  AttrT base_lr_;
  AttrT end_lr_;
};

template <typename DeviceContext, typename T>
class Pow2WarmupOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const {
    const auto *lr = ctx.Input<framework::Tensor>("LearningRate");
    const auto *step = ctx.Input<framework::Tensor>("Step");
    auto *lr_out = ctx.Output<framework::Tensor>("LearningRateOut");
    auto *step_out = ctx.Output<framework::Tensor>("StepOut");
    PADDLE_ENFORCE_EQ(
        lr, lr_out, platform::errors::InvalidArgument("Input(LearningRate) and "
                                                      "Output(LearningRateOut) "
                                                      "must be the same."));
    PADDLE_ENFORCE_EQ(step, step_out,
                      platform::errors::InvalidArgument(
                          "Input(Step) and Output(StepOut) must be the same."));
    PADDLE_ENFORCE_EQ(
        step->IsInitialized(), true,
        platform::errors::InvalidArgument("Input(Step) must be initialized."));

    auto warmup_steps = static_cast<size_t>(ctx.Attr<int64_t>("warmup_steps"));
    auto total_steps = static_cast<size_t>(ctx.Attr<int64_t>("total_steps"));
    PADDLE_ENFORCE_LE(warmup_steps, total_steps,
                      platform::errors::InvalidArgument(
                          "warmup_steps must not be larger than total_steps."));
    auto start_lr = ctx.Attr<float>("start_lr");
    auto base_lr = ctx.Attr<float>("base_lr");
    auto end_lr = ctx.Attr<float>("end_lr");

    auto *lr_data = lr_out->data<T>();
    auto *step_data = step_out->data<int64_t>();
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, 1);
    using AttrT = float;
    Pow2WarmupFunctor<T, AttrT> functor(
        lr_data, step_data, warmup_steps, total_steps,
        static_cast<AttrT>(start_lr), static_cast<AttrT>(base_lr),
        static_cast<AttrT>(end_lr));
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
