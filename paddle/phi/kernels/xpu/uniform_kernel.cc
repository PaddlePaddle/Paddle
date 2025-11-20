/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/uniform_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"

namespace phi {

template <typename T, typename Context>
void UniformKernel(const Context &dev_ctx,
                   const IntArray &shape,
                   DataType dtype,
                   const Scalar &min,
                   const Scalar &max,
                   int seed,
                   DenseTensor *out) {
  out->Resize(common::make_ddim(shape.GetData()));
  T *data = dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  int64_t real_seed = seed != 0 ? seed : dev_ctx.GetGenerator()->Random64();
  // algo:
  //       0: philox4x32_10_pytorch
  //       1: mt
  //       2: philox4x32_10_curand
  int algo = 0;

  // Handle complex types separately
  if constexpr (std::is_same_v<T, phi::dtype::complex<float>> ||
                std::is_same_v<T, phi::dtype::complex<double>>) {
    using RealType = phi::dtype::Real<T>;  // float or double
    using XPUType = typename XPUTypeTrait<RealType>::Type;
    RealType min_val = min.to<RealType>();
    RealType max_val = max.to<RealType>();

    // Generate random values for real and imaginary parts separately
    DenseTensor real_part, imag_part;
    real_part.Resize(out->dims());
    imag_part.Resize(out->dims());
    RealType *real_data = dev_ctx.template Alloc<RealType>(&real_part);
    RealType *imag_data = dev_ctx.template Alloc<RealType>(&imag_part);

    // Generate real part
    int r = xpu::uniform<XPUType>(dev_ctx.x_context(),
                                  reinterpret_cast<XPUType *>(real_data),
                                  real_part.numel(),
                                  static_cast<XPUType>(min_val),
                                  static_cast<XPUType>(max_val),
                                  real_seed,
                                  algo);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "uniform");

    // Generate imaginary part with different seed
    r = xpu::uniform<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<XPUType *>(imag_data),
                              imag_part.numel(),
                              static_cast<XPUType>(min_val),
                              static_cast<XPUType>(max_val),
                              real_seed + 1,
                              algo);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "uniform");

    // Combine real and imaginary parts using ComplexKernel
    ComplexKernel<RealType, Context>(dev_ctx, real_part, imag_part, out);
  } else {
    // Original implementation for non-complex types
    using XPUType = typename XPUTypeTrait<T>::Type;
    int r = xpu::uniform<XPUType>(dev_ctx.x_context(),
                                  reinterpret_cast<XPUType *>(data),
                                  out->numel(),
                                  static_cast<XPUType>(min.to<float>()),
                                  static_cast<XPUType>(max.to<float>()),
                                  real_seed,
                                  algo);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "uniform");
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(uniform,
                   XPU,
                   ALL_LAYOUT,
                   phi::UniformKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64) {}
