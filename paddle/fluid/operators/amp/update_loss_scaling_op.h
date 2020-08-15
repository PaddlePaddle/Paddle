// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <cmath>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
HOSTDEVICE void Update(const bool* found_inf_v, const T* pre_loss_scaling_v,
                       const int* good_in_v, const int* bad_in_v,
                       const int incr_every_n_steps,
                       const int decr_every_n_nan_or_inf,
                       const float incr_ratio, const float decr_ratio,
                       T* updated_loss_scaling_v, int* good_out_v,
                       int* bad_out_v) {
  if (*found_inf_v) {
    *good_out_v = 0;
    *bad_out_v = *bad_in_v + 1;
    if (*bad_out_v == decr_every_n_nan_or_inf) {
      T new_loss_scaling = *pre_loss_scaling_v * decr_ratio;
      *updated_loss_scaling_v = new_loss_scaling < static_cast<T>(1)
                                    ? static_cast<T>(1)
                                    : new_loss_scaling;
      *bad_out_v = 0;
    }
  } else {
    *bad_out_v = 0;
    *good_out_v = *good_in_v + 1;
    if (*good_out_v == incr_every_n_steps) {
      T new_loss_scaling = *pre_loss_scaling_v * incr_ratio;
      *updated_loss_scaling_v = std::isfinite(new_loss_scaling)
                                    ? new_loss_scaling
                                    : *pre_loss_scaling_v;
      *good_out_v = 0;
    }
  }
}

template <typename DeviceContext, typename T>
class UpdateLossScalingFunctor {
 public:
  void operator()(const DeviceContext& dev_ctx, const bool* found_inf_v,
                  const T* pre_loss_scaling_v, const int* good_in_v,
                  const int* bad_in_v, const int incr_every_n_steps,
                  const int decr_every_n_nan_or_inf, const float incr_ratio,
                  const float decr_ratio, T* updated_loss_scaling_v,
                  int* good_out_v, int* bad_out_v);
};

template <typename DeviceContext, typename T>
class UpdateLossScalingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* found_inf = ctx.Input<Tensor>("FoundInfinite");
    auto* pre_loss_scaling = ctx.Input<Tensor>("PrevLossScaling");
    auto* good_in = ctx.Input<Tensor>("InGoodSteps");
    auto* bad_in = ctx.Input<Tensor>("InBadSteps");
    auto* updated_loss_scaling = ctx.Output<Tensor>("LossScaling");
    auto* good_out = ctx.Output<Tensor>("OutGoodSteps");
    auto* bad_out = ctx.Output<Tensor>("OutBadSteps");

    const bool* found_inf_v = found_inf->data<bool>();
    const T* pre_loss_scaling_v = pre_loss_scaling->data<T>();
    const int* good_in_v = good_in->data<int>();
    const int* bad_in_v = bad_in->data<int>();

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    T* updated_loss_scaling_v =
        updated_loss_scaling->mutable_data<T>(dev_ctx.GetPlace());
    int* good_out_v = good_out->mutable_data<int>(dev_ctx.GetPlace());
    int* bad_out_v = bad_out->mutable_data<int>(dev_ctx.GetPlace());

    const int incr_every_n_steps = ctx.Attr<int>("incr_every_n_steps");
    const int decr_every_n_nan_or_inf =
        ctx.Attr<int>("decr_every_n_nan_or_inf");
    const float incr_ratio = ctx.Attr<float>("incr_ratio");
    const float decr_ratio = ctx.Attr<float>("decr_ratio");
    UpdateLossScalingFunctor<DeviceContext, T>{}(
        dev_ctx, found_inf_v, pre_loss_scaling_v, good_in_v, bad_in_v,
        incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio,
        updated_loss_scaling_v, good_out_v, bad_out_v);
  }
};

}  // namespace operators
}  // namespace paddle
