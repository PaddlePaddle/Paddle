/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/amp_kernel.h"

#include <cstring>
#include <string>
#include <vector>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

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
  using XPUTyp = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(
      found_infinite.numel(),
      1,
      phi::errors::InvalidArgument("FoundInfinite must has only one element."));
  const bool* found_inf_data = found_infinite.data<bool>();
  bool cpu_found_inf_data = false;
  if (found_infinite.place().GetType() == phi::AllocationType::XPU) {
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_found_inf_data),
                         found_infinite.place(),
                         static_cast<const void*>(found_inf_data),
                         sizeof(bool));
  } else {
    cpu_found_inf_data = (*found_inf_data);
  }

  for (size_t i = 0; i < xs.size(); ++i) {
    auto* out = outs[i];
    T* out_data = dev_ctx.template Alloc<T>(out);
    int num = out->numel();
    if (cpu_found_inf_data) {
      VLOG(1) << "-- UpdateLossScaling: Find infinite grads. --";
      int r = 0;
      r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUTyp*>(out_data),
                        num,
                        XPUTyp(0.0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    }
  }
  if (stop_update.to<bool>()) {
    return;
  }

  const MPDType* pre_loss_scaling_data = prev_loss_scaling.data<MPDType>();
  const int* good_in_data = in_good_steps.data<int>();
  const int* bad_in_data = in_bad_steps.data<int>();
  MPDType* updated_loss_scaling_data =
      dev_ctx.template Alloc<MPDType>(loss_scaling);

  int* good_out_data = dev_ctx.template Alloc<int>(out_good_steps);
  int* bad_out_data = dev_ctx.template Alloc<int>(out_bad_steps);

  int cpu_bad_in_data;
  int cpu_good_in_data;
  MPDType cpu_pre_loss_scaling_data;
  if (in_bad_steps.place().GetType() == phi::AllocationType::XPU) {
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_bad_in_data),
                         in_bad_steps.place(),
                         static_cast<const void*>(bad_in_data),
                         sizeof(int));
  } else {
    cpu_bad_in_data = (*bad_in_data);
  }

  if (in_good_steps.place().GetType() == phi::AllocationType::XPU) {
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_good_in_data),
                         in_good_steps.place(),
                         static_cast<const void*>(good_in_data),
                         sizeof(int));
  } else {
    cpu_good_in_data = (*good_in_data);
  }

  if (prev_loss_scaling.place().GetType() == phi::AllocationType::XPU) {
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_pre_loss_scaling_data),
                         prev_loss_scaling.place(),
                         static_cast<const void*>(pre_loss_scaling_data),
                         sizeof(MPDType));
  } else {
    cpu_pre_loss_scaling_data = (*pre_loss_scaling_data);
  }
  int cpu_good_out_data = 0;
  int cpu_bad_out_data = 0;
  MPDType cpu_updated_loss_scaling_data = cpu_pre_loss_scaling_data;

  if (cpu_found_inf_data) {
    cpu_good_out_data = 0;
    cpu_bad_out_data = cpu_bad_in_data + 1;
    if (cpu_bad_out_data == decr_every_n_nan_or_inf) {
      MPDType new_loss_scaling = cpu_pre_loss_scaling_data * decr_ratio;
      cpu_updated_loss_scaling_data =
          (new_loss_scaling < static_cast<MPDType>(1))
              ? (static_cast<MPDType>(1))
              : (new_loss_scaling);
      cpu_bad_out_data = 0;
    }
  } else {
    cpu_bad_out_data = 0;
    cpu_good_out_data = cpu_good_in_data + 1;
    if (cpu_good_out_data == incr_every_n_steps) {
      MPDType new_loss_scaling = cpu_pre_loss_scaling_data * incr_ratio;
      cpu_updated_loss_scaling_data = (std::isfinite(new_loss_scaling))
                                          ? new_loss_scaling
                                          : cpu_pre_loss_scaling_data;
      cpu_good_out_data = 0;
    }
  }
  // copy to device
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       bad_out_data,
                       phi::CPUPlace(),
                       &cpu_bad_out_data,
                       sizeof(int));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       good_out_data,
                       phi::CPUPlace(),
                       &cpu_good_out_data,
                       sizeof(int));
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       updated_loss_scaling_data,
                       phi::CPUPlace(),
                       &cpu_updated_loss_scaling_data,
                       sizeof(MPDType));
}

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUType = typename XPUTypeTrait<T>::Type;
  using float16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

  const MPDType* scale_data = scale.data<MPDType>();
  bool* found_inf_data = dev_ctx.template Alloc<bool>(found_infinite);

  // cpy to cpu
  bool cpu_found_inf_data = false;

  // number of inf and nans
  int nums_inf_nans = 0;
  MPDType cpu_scale_data;
  if (scale.place().GetType() == phi::AllocationType::XPU) {
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void*>(&cpu_scale_data),
                         scale.place(),
                         static_cast<const void*>(scale_data),
                         sizeof(MPDType));

  } else {
    cpu_scale_data = (*scale_data);
  }
  MPDType inverse_scale = 1.0 / cpu_scale_data;
  for (size_t i = 0; i < xs.size(); ++i) {
    const auto* x = xs[i];
    auto* out = outs[i];
    dev_ctx.template Alloc<T>(out);

    DenseTensor inf_nan_count;
    inf_nan_count.Resize(found_infinite->dims());
    dev_ctx.template Alloc<int>(&inf_nan_count);

    if (nums_inf_nans == 0) {
      int r =
          xpu::count_nan_or_inf(dev_ctx.x_context(),
                                reinterpret_cast<const XPUType*>(x->data<T>()),
                                inf_nan_count.data<int>(),
                                x->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "count_nan_or_inf");
      paddle::memory::Copy(phi::CPUPlace(),
                           &nums_inf_nans,
                           dev_ctx.GetPlace(),
                           inf_nan_count.data<int>(),
                           sizeof(int));
    }

    if (nums_inf_nans > 0) {
      cpu_found_inf_data = true;
      inverse_scale = 0.0;
    }

    auto version =
        phi::backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
    DenseTensor float_x;
    DenseTensor float_out;
    if (std::is_same<T, phi::dtype::float16>::value &&
        (version == phi::backends::xpu::XPUVersion::XPU1)) {
      dev_ctx.template Alloc<MPDType>(&float_x, x->numel() * sizeof(MPDType));
      dev_ctx.template Alloc<MPDType>(&float_out,
                                      out->numel() * sizeof(MPDType));

      int r = xpu::cast_v2(dev_ctx.x_context(),
                           reinterpret_cast<const float16*>(x->data<T>()),
                           float_x.data<MPDType>(),
                           x->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");

      r = xpu::scale(dev_ctx.x_context(),
                     float_x.data<MPDType>(),
                     float_out.data<MPDType>(),
                     x->numel(),
                     false,
                     inverse_scale,
                     0.0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

      r = xpu::cast_v2(dev_ctx.x_context(),
                       float_out.data<MPDType>(),
                       reinterpret_cast<float16*>(out->data<T>()),
                       out->numel());
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");
    } else {
      int r = xpu::scale(dev_ctx.x_context(),
                         reinterpret_cast<const XPUType*>(x->data<T>()),
                         reinterpret_cast<XPUType*>(out->data<T>()),
                         x->numel(),
                         false,
                         inverse_scale,
                         0.0);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    }
  }
  paddle::memory::Copy(dev_ctx.GetPlace(),
                       found_inf_data,
                       phi::CPUPlace(),
                       &cpu_found_inf_data,
                       sizeof(bool));
}

}  // namespace phi

PD_REGISTER_KERNEL(update_loss_scaling,
                   XPU,
                   ALL_LAYOUT,
                   phi::UpdateLossScalingKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(check_finite_and_unscale,
                   XPU,
                   ALL_LAYOUT,
                   phi::CheckFiniteAndUnscaleKernel,
                   float,
                   phi::dtype::float16) {}
