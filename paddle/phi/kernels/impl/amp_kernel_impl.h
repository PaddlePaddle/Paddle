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

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/kernels/amp_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

template <typename T>
inline HOSTDEVICE bool CheckFinite(T value) {
#if defined(PADDLE_WITH_CUDA) && defined(__NVCC__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

inline HOSTDEVICE bool IsFoundNanInf(const bool found_nan_inf_data) {
  return found_nan_inf_data;
}

inline HOSTDEVICE bool IsFoundNanInf(const bool* found_nan_inf_data) {
  return *found_nan_inf_data;
}

template <typename T, typename FoundInfFlagT>
inline HOSTDEVICE void Update(const FoundInfFlagT found_inf_data,
                              const T* pre_loss_scaling_data,
                              const int* good_in_data,
                              const int* bad_in_data,
                              const int incr_every_n_steps,
                              const int decr_every_n_nan_or_inf,
                              const float incr_ratio,
                              const float decr_ratio,
                              T* updated_loss_scaling_data,
                              int* good_out_data,
                              int* bad_out_data) {
  if (IsFoundNanInf(found_inf_data)) {
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
      *updated_loss_scaling_data = CheckFinite(new_loss_scaling)
                                       ? new_loss_scaling
                                       : *pre_loss_scaling_data;
      *good_out_data = 0;
    }
  }
}

template <typename Context, typename T>
class LazyZeros {
 public:
  void operator()(const DeviceContext& dev_ctx UNUSED,
                  const bool* found_inf_data UNUSED,
                  const std::vector<const DenseTensor*>& xs UNUSED,
                  const std::vector<DenseTensor*>& outs UNUSED) const {}
};

template <typename Context, typename T, bool IsFoundInfOnCPU>
class UpdateLossScalingFunctor {
 public:
  void operator()(const DeviceContext& dev_ctx,
                  const bool* found_inf_data,
                  const T* pre_loss_scaling_data,
                  const int* good_in_data,
                  const int* bad_in_data,
                  const int incr_every_n_steps,
                  const int decr_every_n_nan_or_inf,
                  const float incr_ratio,
                  const float decr_ratio,
                  T* updated_loss_scaling_data,
                  int* good_out_data,
                  int* bad_out_data) const;
};

template <typename T, typename Context>
void UpdateLossScalingKernel(const Context& dev_ctx,
                             const std::vector<const DenseTensor*>& xs,
                             const DenseTensor& found_infinite,
                             const DenseTensor& prev_loss_scaling,
                             const DenseTensor& in_good_steps,
                             const DenseTensor& in_bad_steps,
                             int incr_every_n_steps,
                             int decr_every_n_nan_or_inf,
                             float incr_ratio,
                             float decr_ratio,
                             const Scalar& stop_update,
                             std::vector<DenseTensor*> outs,
                             DenseTensor* loss_scaling,
                             DenseTensor* out_good_steps,
                             DenseTensor* out_bad_steps) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(
      found_infinite.numel(),
      1,
      phi::errors::InvalidArgument("FoundInfinite must has only one element."));
  const bool* found_inf_data = found_infinite.data<bool>();
  bool is_found_inf_on_cpu =
      found_infinite.place().GetType() == AllocationType::CPU;

  if (is_found_inf_on_cpu) {
    if (*found_inf_data) {
      for (auto* out : outs) {
        Full<T>(dev_ctx, vectorize(out->dims()), static_cast<T>(0), out);
      }
    }
  } else {
    LazyZeros<Context, T>{}(dev_ctx, found_inf_data, xs, outs);
  }

  auto stop_update_val = stop_update.to<bool>();
  if (stop_update_val) {
    return;
  }

  const MPDType* pre_loss_scaling_data = prev_loss_scaling.data<MPDType>();
  const int* good_in_data = in_good_steps.data<int>();
  const int* bad_in_data = in_bad_steps.data<int>();

  MPDType* updated_loss_scaling_data =
      dev_ctx.template Alloc<MPDType>(loss_scaling);
  int* good_out_data = dev_ctx.template Alloc<int>(out_good_steps);
  int* bad_out_data = dev_ctx.template Alloc<int>(out_bad_steps);

  if (is_found_inf_on_cpu) {
    UpdateLossScalingFunctor<Context, MPDType, true>{}(
        dev_ctx,
        found_inf_data,
        pre_loss_scaling_data,
        good_in_data,
        bad_in_data,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        updated_loss_scaling_data,
        good_out_data,
        bad_out_data);
  } else {
    UpdateLossScalingFunctor<Context, MPDType, false>{}(
        dev_ctx,
        found_inf_data,
        pre_loss_scaling_data,
        good_in_data,
        bad_in_data,
        incr_every_n_steps,
        decr_every_n_nan_or_inf,
        incr_ratio,
        decr_ratio,
        updated_loss_scaling_data,
        good_out_data,
        bad_out_data);
  }
}

}  // namespace phi
