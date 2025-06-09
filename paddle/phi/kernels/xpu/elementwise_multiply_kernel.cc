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

#include "paddle/phi/kernels/elementwise_multiply_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_mul<XPUType>(ctx, x, y, z, xshape, yshape);
  };

  XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void MultiplyKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& x,
    const DenseTensor& y,
    DenseTensor* out) {
  using T = phi::dtype::complex<float>;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  const DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
  const DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
  const DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);
  DenseTensor real_out = Subtract<float, XPUContext>(
      dev_ctx,
      Multiply<float, XPUContext>(dev_ctx, x_real, y_real),
      Multiply<float, XPUContext>(dev_ctx, x_imag, y_imag));
  DenseTensor imag_out = Add<float, XPUContext>(
      dev_ctx,
      Multiply<float, XPUContext>(dev_ctx, x_real, y_imag),
      Multiply<float, XPUContext>(dev_ctx, x_imag, y_real));
  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(multiply,
                   XPU,
                   ALL_LAYOUT,
                   phi::MultiplyKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   float,
                   int,
                   int64_t) {
}
