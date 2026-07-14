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

#include "glog/logging.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/memory_utils.h"
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
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUType = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(found_infinite.numel(),
                    1,
                    common::errors::InvalidArgument(
                        "FoundInfinite must has only one element."));
  const bool* found_inf_data = found_infinite.data<bool>();
  bool cpu_found_inf_data = false;
  if (found_infinite.place().GetType() == AllocationType::XPU) {
    memory_utils::Copy(CPUPlace(),
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
    int64_t num = out->numel();
    if (cpu_found_inf_data) {
      VLOG(1) << "-- UpdateLossScaling: Find infinite grads. --";
      int r = 0;
      r = xpu::constant(dev_ctx.x_context(),
                        reinterpret_cast<XPUType*>(out_data),
                        num,
                        XPUType(0.0));
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    }
  }
  if (stop_update.to<bool>()) {
    return;
  }

  const MT* pre_loss_scaling_data = prev_loss_scaling.data<MT>();
  const int* good_in_data = in_good_steps.data<int>();
  const int* bad_in_data = in_bad_steps.data<int>();
  MT* updated_loss_scaling_data = dev_ctx.template Alloc<MT>(loss_scaling);

  int* good_out_data = dev_ctx.template Alloc<int>(out_good_steps);
  int* bad_out_data = dev_ctx.template Alloc<int>(out_bad_steps);

  int cpu_bad_in_data;
  int cpu_good_in_data;
  MT cpu_pre_loss_scaling_data;
  if (in_bad_steps.place().GetType() == AllocationType::XPU) {
    memory_utils::Copy(CPUPlace(),
                       static_cast<void*>(&cpu_bad_in_data),
                       in_bad_steps.place(),
                       static_cast<const void*>(bad_in_data),
                       sizeof(int));
  } else {
    cpu_bad_in_data = (*bad_in_data);
  }

  if (in_good_steps.place().GetType() == AllocationType::XPU) {
    memory_utils::Copy(CPUPlace(),
                       static_cast<void*>(&cpu_good_in_data),
                       in_good_steps.place(),
                       static_cast<const void*>(good_in_data),
                       sizeof(int));
  } else {
    cpu_good_in_data = (*good_in_data);
  }

  if (prev_loss_scaling.place().GetType() == AllocationType::XPU) {
    memory_utils::Copy(CPUPlace(),
                       static_cast<void*>(&cpu_pre_loss_scaling_data),
                       prev_loss_scaling.place(),
                       static_cast<const void*>(pre_loss_scaling_data),
                       sizeof(MT));
  } else {
    cpu_pre_loss_scaling_data = (*pre_loss_scaling_data);
  }
  int cpu_good_out_data = 0;
  int cpu_bad_out_data = 0;
  MT cpu_updated_loss_scaling_data = cpu_pre_loss_scaling_data;

  if (cpu_found_inf_data) {
    cpu_good_out_data = 0;
    cpu_bad_out_data = cpu_bad_in_data + 1;
    if (cpu_bad_out_data == decr_every_n_nan_or_inf) {
      MT new_loss_scaling = cpu_pre_loss_scaling_data * decr_ratio;
      cpu_updated_loss_scaling_data = (new_loss_scaling < static_cast<MT>(1))
                                          ? (static_cast<MT>(1))
                                          : (new_loss_scaling);
      cpu_bad_out_data = 0;
    }
  } else {
    cpu_bad_out_data = 0;
    cpu_good_out_data = cpu_good_in_data + 1;
    if (cpu_good_out_data == incr_every_n_steps) {
      MT new_loss_scaling = cpu_pre_loss_scaling_data * incr_ratio;
      cpu_updated_loss_scaling_data = (std::isfinite(new_loss_scaling))
                                          ? new_loss_scaling
                                          : cpu_pre_loss_scaling_data;
      cpu_good_out_data = 0;
    }
  }
  // copy to device
  memory_utils::Copy(dev_ctx.GetPlace(),
                     bad_out_data,
                     CPUPlace(),
                     &cpu_bad_out_data,
                     sizeof(int));
  memory_utils::Copy(dev_ctx.GetPlace(),
                     good_out_data,
                     CPUPlace(),
                     &cpu_good_out_data,
                     sizeof(int));
  memory_utils::Copy(dev_ctx.GetPlace(),
                     updated_loss_scaling_data,
                     CPUPlace(),
                     &cpu_updated_loss_scaling_data,
                     sizeof(MT));
}

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite) {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeFP16 = typename XPUTypeTrait<phi::float16>::Type;

  const MT* scale_data = scale.data<MT>();
  bool* found_inf_data = dev_ctx.template Alloc<bool>(found_infinite);

  // cpy to cpu
  bool cpu_found_inf_data = false;

  // has nans or infs
  bool has_inf_nans = false;
  MT cpu_scale_data;
  if (scale.place().GetType() == AllocationType::XPU) {
    memory_utils::Copy(CPUPlace(),
                       static_cast<void*>(&cpu_scale_data),
                       scale.place(),
                       static_cast<const void*>(scale_data),
                       sizeof(MT));

  } else {
    cpu_scale_data = (*scale_data);
  }
  MT inverse_scale = 1.0 / cpu_scale_data;
  auto version =
      backends::xpu::get_xpu_version(dev_ctx.GetPlace().GetDeviceId());
  if (version == backends::xpu::XPUVersion::XPU3) {
    int64_t num_grads = xs.size();
    DenseTensor cpu_found_tensor;
    cpu_found_tensor.Resize({num_grads});
    dev_ctx.template HostAlloc<bool>(&cpu_found_tensor);
    DenseTensor inf_nan_check;
    inf_nan_check.Resize({num_grads});
    dev_ctx.template Alloc<bool>(&inf_nan_check);
    bool* inf_nan_check_ptr = inf_nan_check.data<bool>();
    for (int64_t i = 0; i < num_grads; ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      dev_ctx.template Alloc<T>(out);

      int r = xpu::check_finite_unscale(
          dev_ctx.x_context(),
          reinterpret_cast<const XPUType*>(x->data<T>()),
          reinterpret_cast<XPUType*>(out->data<T>()),
          x->numel(),
          inverse_scale,
          inf_nan_check_ptr + i);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "check_finite_unscale");
    }
    memory_utils::Copy(CPUPlace(),
                       cpu_found_tensor.data<bool>(),
                       dev_ctx.GetPlace(),
                       inf_nan_check.data<bool>(),
                       num_grads * sizeof(bool));
    for (int64_t i = 0; i < num_grads; ++i) {
      if (cpu_found_tensor.data<bool>()[i]) {
        cpu_found_inf_data = true;
        break;
      }
    }
  } else {
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      dev_ctx.template Alloc<T>(out);

      DenseTensor inf_nan_check;
      inf_nan_check.Resize({1});
      dev_ctx.template Alloc<bool>(&inf_nan_check);

      if (!has_inf_nans) {
        int r = xpu::check_nan_or_inf(
            dev_ctx.x_context(),
            reinterpret_cast<const XPUType*>(x->data<T>()),
            inf_nan_check.data<bool>(),
            x->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "check_nan_or_inf");
        memory_utils::Copy(CPUPlace(),
                           &has_inf_nans,
                           dev_ctx.GetPlace(),
                           inf_nan_check.data<bool>(),
                           sizeof(bool));
      }

      if (has_inf_nans) {
        cpu_found_inf_data = true;
        break;
      }

      DenseTensor float_x;
      DenseTensor float_out;
      if (std::is_same<T, phi::float16>::value &&
          (version == backends::xpu::XPUVersion::XPU1)) {
        dev_ctx.template Alloc<MT>(&float_x, x->numel() * sizeof(MT));
        dev_ctx.template Alloc<MT>(&float_out, out->numel() * sizeof(MT));

        int r = xpu::cast(dev_ctx.x_context(),
                          reinterpret_cast<const XPUTypeFP16*>(x->data<T>()),
                          float_x.data<MT>(),
                          x->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");

        r = xpu::scale(dev_ctx.x_context(),
                       float_x.data<MT>(),
                       float_out.data<MT>(),
                       x->numel(),
                       false,
                       inverse_scale,
                       0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

        r = xpu::cast(dev_ctx.x_context(),
                      float_out.data<MT>(),
                      reinterpret_cast<XPUTypeFP16*>(out->data<T>()),
                      out->numel());
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
      } else {
        int r = xpu::scale(dev_ctx.x_context(),
                           reinterpret_cast<const XPUType*>(x->data<T>()),
                           reinterpret_cast<XPUType*>(out->data<T>()),
                           x->numel(),
                           false,
                           inverse_scale,
                           0.0f);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
      }
    }
  }
  memory_utils::Copy(dev_ctx.GetPlace(),
                     found_inf_data,
                     CPUPlace(),
                     &cpu_found_inf_data,
                     sizeof(bool));
}

}  // namespace phi

PD_REGISTER_KERNEL(update_loss_scaling,
                   XPU,
                   ALL_LAYOUT,
                   phi::UpdateLossScalingKernel,
                   float,
                   phi::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  }
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::INT32);
}

PD_REGISTER_KERNEL(check_finite_and_unscale,
                   XPU,
                   ALL_LAYOUT,
                   phi::CheckFiniteAndUnscaleKernel,
                   float,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::BOOL);
}
