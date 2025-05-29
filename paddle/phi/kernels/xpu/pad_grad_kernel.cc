// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/pad_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"

namespace phi {
template <typename T, typename Context>
void PadGradKernel(const Context& dev_ctx,
                   const DenseTensor& d_out,
                   const std::vector<int>& paddings,
                   const Scalar& pad_value,
                   DenseTensor* d_x) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int64_t> pad_left, pad_right;
  std::vector<int64_t> out_shape = common::vectorize<int64_t>(d_out.dims());
  dev_ctx.template Alloc<T>(d_x);

  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    pad_left.push_back(-paddings[i * 2]);
    pad_right.push_back(-paddings[i * 2 + 1]);
  }

  XPUType value = static_cast<XPUType>(pad_value.to<T>());
  int r = xpu::pad<XPUType>(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(d_out.data<T>()),
                            reinterpret_cast<XPUType*>(d_x->data<T>()),
                            out_shape,
                            pad_left,
                            pad_right,
                            value);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void PadGradKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& d_out,
    const std::vector<int>& paddings,
    const Scalar& pad_value,
    DenseTensor* d_x) {
  using T = phi::dtype::complex<float>;
  std::vector<int64_t> pad_left, pad_right;
  std::vector<int64_t> out_shape = common::vectorize<int64_t>(d_out.dims());
  dev_ctx.template Alloc<T>(d_x);

  for (size_t i = 0; i < paddings.size() / 2; ++i) {
    pad_left.push_back(-paddings[i * 2]);
    pad_right.push_back(-paddings[i * 2 + 1]);
  }

  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  DenseTensor real_out, imag_out;
  real_out.Resize(d_x->dims());
  imag_out.Resize(d_x->dims());
  dev_ctx.template Alloc<float>(&real_out);
  dev_ctx.template Alloc<float>(&imag_out);
  const DenseTensor real = Real<T, XPUContext>(dev_ctx, d_out);
  const DenseTensor imag = Imag<T, XPUContext>(dev_ctx, d_out);
  T complex_val = pad_value.to<T>();
  float real_part = complex_val.real;
  float imag_part = complex_val.imag;
  int r = xpu::pad<float>(dev_ctx.x_context(),
                          real.data<float>(),
                          real_out.data<float>(),
                          out_shape,
                          pad_left,
                          pad_right,
                          real_part);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
  r = xpu::pad<float>(dev_ctx.x_context(),
                      imag.data<float>(),
                      imag_out.data<float>(),
                      out_shape,
                      pad_left,
                      pad_right,
                      imag_part);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, d_x);
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(pad_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PadGradKernel,
                   float,
                   int,
                   int16_t,
                   int64_t,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
}
