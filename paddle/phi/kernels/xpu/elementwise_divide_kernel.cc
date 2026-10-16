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

#include "paddle/phi/kernels/elementwise_divide_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

#ifdef PADDLE_WITH_XPU_FFT
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#endif

namespace phi {

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_div<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
}

#ifdef PADDLE_WITH_XPU_FFT
// Complex64 division specialization: XPU does not have a native complex64
// divide kernel, so we implement complex division using real/imag
// decomposition. Formula: (a + bi) / (c + di) = ((ac + bd) / (c^2 + d^2)) +
// ((bc - ad) / (c^2 + d^2))i
template <>
void DivideKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                              const DenseTensor& x,
                                              const DenseTensor& y,
                                              DenseTensor* out) {
  using T = phi::complex64;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  const DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
  const DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
  const DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);

  // Denominator: y_real^2 + y_imag^2
  DenseTensor denom = Add<float, XPUContext>(
      dev_ctx,
      Multiply<float, XPUContext>(dev_ctx, y_real, y_real),
      Multiply<float, XPUContext>(dev_ctx, y_imag, y_imag));

  // Real part: (x_real * y_real + x_imag * y_imag) / denom
  DenseTensor real_out = Divide<float, XPUContext>(
      dev_ctx,
      Add<float, XPUContext>(
          dev_ctx,
          Multiply<float, XPUContext>(dev_ctx, x_real, y_real),
          Multiply<float, XPUContext>(dev_ctx, x_imag, y_imag)),
      denom);

  // Imaginary part: (x_imag * y_real - x_real * y_imag) / denom
  DenseTensor imag_out = Divide<float, XPUContext>(
      dev_ctx,
      Subtract<float, XPUContext>(
          dev_ctx,
          Multiply<float, XPUContext>(dev_ctx, x_imag, y_real),
          Multiply<float, XPUContext>(dev_ctx, x_real, y_imag)),
      denom);

  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(divide,
                   XPU,
                   ALL_LAYOUT,
                   phi::DivideKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
#endif
                   int,
                   int64_t) {
}
