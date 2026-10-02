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

#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/xpu/elementwise.h"
namespace phi {

template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto f = [](xpu::Context* xpu_ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_sub<XPUType>(xpu_ctx, x, y, z, xshape, yshape);
  };

  phi::XPUElementwise<T, XPUType>(dev_ctx, x, y, -1, out, f);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void SubtractKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                                const DenseTensor& x,
                                                const DenseTensor& y,
                                                DenseTensor* out) {
  using T = phi::complex64;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto f = [](xpu::Context* xpu_ctx,
              const float* x,
              const float* y,
              float* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_sub<float>(xpu_ctx, x, y, z, xshape, yshape);
  };
  const DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
  const DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
  const DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);
  DenseTensor real_out, imag_out;
  real_out.Resize(out->dims());
  imag_out.Resize(out->dims());
  dev_ctx.template Alloc<float>(&real_out);
  dev_ctx.template Alloc<float>(&imag_out);

  XPUElementwise<float, float>(dev_ctx, x_real, y_real, -1, &real_out, f);
  XPUElementwise<float, float>(dev_ctx, x_imag, y_imag, -1, &imag_out, f);
  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void SubtractKernel<phi::complex128, XPUContext>(const XPUContext& dev_ctx,
                                                 const DenseTensor& x,
                                                 const DenseTensor& y,
                                                 DenseTensor* out) {
  using T = phi::complex128;
  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  // XPU xdnn does not support broadcast_sub<double>, so we decompose
  // complex subtraction into real/imag parts and use the float path
  // with type casting: cast real/imag double parts to float, subtract
  // in float, then cast result back to double and recombine as complex128.
  const DenseTensor x_real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor x_imag = Imag<T, XPUContext>(dev_ctx, x);
  const DenseTensor y_real = Real<T, XPUContext>(dev_ctx, y);
  const DenseTensor y_imag = Imag<T, XPUContext>(dev_ctx, y);

  DenseTensor x_real_f = Cast<float>(dev_ctx, x_real, DataType::FLOAT32);
  DenseTensor x_imag_f = Cast<float>(dev_ctx, x_imag, DataType::FLOAT32);
  DenseTensor y_real_f = Cast<float>(dev_ctx, y_real, DataType::FLOAT32);
  DenseTensor y_imag_f = Cast<float>(dev_ctx, y_imag, DataType::FLOAT32);

  DenseTensor real_out_f, imag_out_f;
  real_out_f.Resize(out->dims());
  imag_out_f.Resize(out->dims());
  dev_ctx.template Alloc<float>(&real_out_f);
  dev_ctx.template Alloc<float>(&imag_out_f);

  auto f = [](xpu::Context* xpu_ctx,
              const float* x,
              const float* y,
              float* z,
              const std::vector<int64_t>& xshape,
              const std::vector<int64_t>& yshape) {
    return xpu::broadcast_sub<float>(xpu_ctx, x, y, z, xshape, yshape);
  };
  XPUElementwise<float, float>(dev_ctx, x_real_f, y_real_f, -1, &real_out_f, f);
  XPUElementwise<float, float>(dev_ctx, x_imag_f, y_imag_f, -1, &imag_out_f, f);

  DenseTensor real_out = Cast<double>(dev_ctx, real_out_f, DataType::FLOAT64);
  DenseTensor imag_out = Cast<double>(dev_ctx, imag_out_f, DataType::FLOAT64);

  phi::ComplexKernel<double>(dev_ctx, real_out, imag_out, out);
}
#endif
#endif

}  // namespace phi
PD_REGISTER_KERNEL(subtract,
                   XPU,
                   ALL_LAYOUT,
                   phi::SubtractKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::complex64,
                   phi::complex128,
#endif
                   int,
                   int64_t) {
}
