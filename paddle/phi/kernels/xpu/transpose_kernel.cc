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

#include "paddle/phi/kernels/transpose_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"

namespace phi {

template <typename T, typename Context>
void TransposeKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     DenseTensor* out) {
  size_t x_rank = x.dims().size();
  std::vector<int64_t> formatted_axis(axis.begin(), axis.end());
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
  }

  using XPUType = typename XPUTypeTrait<T>::Type;

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formatted_axis.size() == 0) {
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  std::vector<int64_t> x_dim_vec = common::vectorize<int64_t>(x.dims());
  int r = xpu::transpose<XPUType>(dev_ctx.x_context(),
                                  reinterpret_cast<const XPUType*>(x.data<T>()),
                                  reinterpret_cast<XPUType*>(out->data<T>()),
                                  x_dim_vec,
                                  formatted_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void TransposeKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx,
    const DenseTensor& x,
    const std::vector<int>& axis,
    DenseTensor* out) {
  using T = phi::dtype::complex<float>;
  size_t x_rank = x.dims().size();
  std::vector<int64_t> formatted_axis(axis.begin(), axis.end());
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = axis[i] + x_rank;
    }
  }

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formatted_axis.size() == 0) {
    phi::Copy<XPUContext>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  DenseTensor real_out, imag_out;
  real_out.Resize(out->dims());
  imag_out.Resize(out->dims());
  dev_ctx.template Alloc<float>(&real_out);
  dev_ctx.template Alloc<float>(&imag_out);
  const DenseTensor real = Real<T, XPUContext>(dev_ctx, x);
  const DenseTensor imag = Imag<T, XPUContext>(dev_ctx, x);
  std::vector<int64_t> x_dim_vec = common::vectorize<int64_t>(x.dims());
  int r = xpu::transpose<float>(dev_ctx.x_context(),
                                real.data<float>(),
                                real_out.data<float>(),
                                x_dim_vec,
                                formatted_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  r = xpu::transpose<float>(dev_ctx.x_context(),
                            imag.data<float>(),
                            imag_out.data<float>(),
                            x_dim_vec,
                            formatted_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "transpose");
  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
}
#endif
}  // namespace phi

PD_REGISTER_KERNEL(transpose,
                   XPU,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   int64_t,
                   int,
                   bool) {
}
