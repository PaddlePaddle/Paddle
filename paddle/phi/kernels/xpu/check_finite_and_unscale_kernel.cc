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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/amp_type_traits.h"
// #include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CheckFiniteAndUnscaleKernel(const Context& dev_ctx,
                                 const std::vector<const DenseTensor*>& xs,
                                 const DenseTensor& scale,
                                 std::vector<DenseTensor*> outs,
                                 DenseTensor* found_infinite) {
  using MPDType = typename phi::dtype::MPTypeTrait<T>::Type;
  using XPUTyp = typename XPUTypeTrait<T>::Type;
  using float16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

  const MPDType* scale_data = scale.data<MPDType>();
  bool* found_inf_data = dev_ctx.template Alloc<bool>(found_infinite)

                         // cpy to cpu
                         bool cpu_found_inf_data = false;

  // number of inf and nans
  int nums_inf_nans = 0;
  MPDType cpu_scale_data;
  if (paddle::platform::is_xpu_place(scale.place())) {
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
    DenseTensor inf_nan_count =
        ctx.AllocateTmpTensor<int, platform::XPUDeviceContext>(  // ?
            found_infinite->dims(),
            dev_ctx);

    if (nums_inf_nans == 0) {
      int r =
          xpu::count_nan_or_inf(dev_ctx.x_context(),  // ?
                                reinterpret_cast<const XPUTyp*>(x->data<T>()),
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
      dev_ctx.template Alloc<MPDType>(float_x, x->numel() * sizeof(MPDType));
      dev_ctx.template Alloc<MPDType>(float_out,
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
                         reinterpret_cast<const XPUTyp*>(x->data<T>()),
                         reinterpret_cast<XPUTyp*>(out->data<T>()),
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

PD_REGISTER_KERNEL(check_finite_and_unscale,
                   XPU,
                   ALL_LAYOUT,
                   phi::CheckFiniteAndUnscaleKernel,
                   float,
                   phi::dtype::float16) {}
