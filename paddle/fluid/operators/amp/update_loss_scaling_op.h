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

#if defined(PADDLE_WITH_CUDA) && defined(__NVCC__)
#include <cuda.h>
#endif  // PADDLE_WITH_CUDA && __NVCC__
#include <cmath>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
inline HOSTDEVICE bool check_finite(T value) {
#if defined(PADDLE_WITH_CUDA) && defined(__NVCC__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

template <typename T>
inline HOSTDEVICE void Update(const bool* found_inf_data,
                              const T* pre_loss_scaling_data,
                              const int* good_in_data, const int* bad_in_data,
                              const int incr_every_n_steps,
                              const int decr_every_n_nan_or_inf,
                              const float incr_ratio, const float decr_ratio,
                              T* updated_loss_scaling_data, int* good_out_data,
                              int* bad_out_data) {
  if (*found_inf_data) {
    *good_out_data = 0;
    *bad_out_data = *bad_in_data + 1;
    if (*bad_out_data == decr_every_n_nan_or_inf) {
      T new_loss_scaling = *pre_loss_scaling_data * decr_ratio;
      *updated_loss_scaling_data = new_loss_scaling < static_cast<T>(1)
                                       ? static_cast<T>(1)
                                       : new_loss_scaling;
      *bad_out_data = 0;
    }
  } else {
    *bad_out_data = 0;
    *good_out_data = *good_in_data + 1;
    if (*good_out_data == incr_every_n_steps) {
      T new_loss_scaling = *pre_loss_scaling_data * incr_ratio;
      *updated_loss_scaling_data = check_finite(new_loss_scaling)
                                       ? new_loss_scaling
                                       : *pre_loss_scaling_data;
      *good_out_data = 0;
    }
  }
}

template <typename DeviceContext, typename T>
class UpdateLossScalingFunctor {
 public:
  void operator()(const DeviceContext& dev_ctx, const bool* found_inf_data,
                  const T* pre_loss_scaling_data, const int* good_in_data,
                  const int* bad_in_data, const int incr_every_n_steps,
                  const int decr_every_n_nan_or_inf, const float incr_ratio,
                  const float decr_ratio, T* updated_loss_scaling_data,
                  int* good_out_data, int* bad_out_data) const;
};

template <typename DeviceContext, typename T>
class LazyZeros {
 public:
  void operator()(const DeviceContext& dev_ctx, const bool* found_inf_data,
                  const std::vector<const framework::Tensor*>& xs,
                  const std::vector<framework::Tensor*>& outs) const;
};

template <typename DeviceContext, typename T>
class UpdateLossScalingKernel : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    const auto* found_inf = ctx.Input<Tensor>("FoundInfinite");
    PADDLE_ENFORCE_EQ(found_inf->numel(), 1,
                      platform::errors::InvalidArgument(
                          "FoundInfinite must has only one element."));
    const bool* found_inf_data = found_inf->data<bool>();

    LazyZeros<DeviceContext, T>{}(dev_ctx, found_inf_data, xs, outs);
    const bool stop_update = ctx.Attr<bool>("stop_update");
    if (stop_update) {
      return;
    }

    const auto* pre_loss_scaling = ctx.Input<Tensor>("PrevLossScaling");
    const auto* good_in = ctx.Input<Tensor>("InGoodSteps");
    const auto* bad_in = ctx.Input<Tensor>("InBadSteps");
    auto* updated_loss_scaling = ctx.Output<Tensor>("LossScaling");
    auto* good_out = ctx.Output<Tensor>("OutGoodSteps");
    auto* bad_out = ctx.Output<Tensor>("OutBadSteps");
    const MPDType* pre_loss_scaling_data = pre_loss_scaling->data<MPDType>();
    const int* good_in_data = good_in->data<int>();
    const int* bad_in_data = bad_in->data<int>();

    MPDType* updated_loss_scaling_data =
        updated_loss_scaling->mutable_data<MPDType>(dev_ctx.GetPlace());
    int* good_out_data = good_out->mutable_data<int>(dev_ctx.GetPlace());
    int* bad_out_data = bad_out->mutable_data<int>(dev_ctx.GetPlace());

    const int incr_every_n_steps = ctx.Attr<int>("incr_every_n_steps");
    const int decr_every_n_nan_or_inf =
        ctx.Attr<int>("decr_every_n_nan_or_inf");
    const float incr_ratio = ctx.Attr<float>("incr_ratio");
    const float decr_ratio = ctx.Attr<float>("decr_ratio");
    UpdateLossScalingFunctor<DeviceContext, MPDType>{}(
        dev_ctx, found_inf_data, pre_loss_scaling_data, good_in_data,
        bad_in_data, incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio,
        decr_ratio, updated_loss_scaling_data, good_out_data, bad_out_data);
  }
};

}  // namespace operators
}  // namespace paddle
