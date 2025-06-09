/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/contiguous_kernel.h"

#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  // use XPUCopyTypeTrait to deal with double and int16_t copy instead of
  // XPUTypeTrait
  using XPUType = typename XPUCopyTypeTrait<T>::Type;

  int r = 0;
  auto input_data = reinterpret_cast<const XPUType*>(input.data<T>());
  auto output_data = reinterpret_cast<XPUType*>(dev_ctx.template Alloc<T>(out));
  if (input.numel() == 1) {
    r = xpu::copy<XPUType>(dev_ctx.x_context(), input_data, output_data, 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = xpu::as_strided<XPUType>(dev_ctx.x_context(),
                                 input_data,
                                 output_data,
                                 common::vectorize<int64_t>(input.dims()),
                                 common::vectorize<int64_t>(input.strides()),
                                 0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "as_strided");
  }
}

#ifdef PADDLE_WITH_XPU_FFT
template <>
void ContiguousKernel<phi::dtype::complex<float>, XPUContext>(
    const XPUContext& dev_ctx, const DenseTensor& input, DenseTensor* out) {
  using T = phi::dtype::complex<float>;

  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  // The current complex number implementation uses separate real/imaginary
  // parts,resulting in redundant operations and performance
  // penalties.Optimization should address this in future iterations.
  dev_ctx.template Alloc<T>(out);
  const DenseTensor real = Real<T, XPUContext>(dev_ctx, input);
  const DenseTensor imag = Imag<T, XPUContext>(dev_ctx, input);
  DenseTensor real_out, imag_out;
  real_out.Resize(out->dims());
  imag_out.Resize(out->dims());
  dev_ctx.template Alloc<float>(&real_out);
  dev_ctx.template Alloc<float>(&imag_out);

  int r = 0;
  if (input.numel() == 1) {
    r = xpu::copy<float>(
        dev_ctx.x_context(), real.data<float>(), real_out.data<float>(), 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
    r = xpu::copy<float>(
        dev_ctx.x_context(), imag.data<float>(), imag_out.data<float>(), 1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = xpu::as_strided<float>(dev_ctx.x_context(),
                               real.data<float>(),
                               real_out.data<float>(),
                               common::vectorize<int64_t>(input.dims()),
                               common::vectorize<int64_t>(input.strides()),
                               0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "as_strided");
    r = xpu::as_strided<float>(dev_ctx.x_context(),
                               imag.data<float>(),
                               imag_out.data<float>(),
                               common::vectorize<int64_t>(input.dims()),
                               common::vectorize<int64_t>(input.strides()),
                               0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "as_strided");
  }
  phi::ComplexKernel<float>(dev_ctx, real_out, imag_out, out);
}
#endif

}  // namespace phi

PD_REGISTER_KERNEL(contiguous,
                   XPU,
                   ALL_LAYOUT,
                   phi::ContiguousKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
#ifdef PADDLE_WITH_XPU_FFT
                   phi::dtype::complex<float>,
#endif
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16) {
}
