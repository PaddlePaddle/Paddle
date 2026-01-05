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

#include <cstdint>
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
  DenseTensorMeta meta = input.meta();
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
template <typename T>
typename std::enable_if<std::is_same<T, phi::complex64>::value ||
                        std::is_same<T, phi::complex128>::value>::type
ComplexContiguousKernelImpl(const XPUContext& dev_ctx,
                            const DenseTensor& input,
                            DenseTensor* out) {
  DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  if (out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  // For strided complex tensors, avoid using Real/Imag kernels that assume
  // contiguous complex layout. Instead, materialize bytes with
  // as_strided<int8_t> to preserve both real/imag parts and handle large
  // strides safely.
  dev_ctx.template Alloc<T>(out);
  auto bytes_shape = common::vectorize<int64_t>(input.dims());
  auto bytes_strides = common::vectorize<int64_t>(input.strides());
  const int64_t bytes_per_elem = static_cast<int64_t>(sizeof(T));
  for (auto& s : bytes_strides) {
    s *= bytes_per_elem;
  }
  bytes_shape.push_back(bytes_per_elem);
  bytes_strides.push_back(1);

  const auto* input_bytes = reinterpret_cast<const int8_t*>(input.data<T>());
  auto* output_bytes = reinterpret_cast<int8_t*>(out->data<T>());

  int r = 0;
  if (input.numel() == 1) {
    r = xpu::copy<int8_t>(
        dev_ctx.x_context(), input_bytes, output_bytes, bytes_per_elem);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
  } else {
    r = xpu::as_strided<int8_t>(dev_ctx.x_context(),
                                input_bytes,
                                output_bytes,
                                bytes_shape,
                                bytes_strides,
                                0);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "as_strided");
  }
}
template <>
void ContiguousKernel<phi::complex64, XPUContext>(const XPUContext& dev_ctx,
                                                  const DenseTensor& input,
                                                  DenseTensor* out) {
  ComplexContiguousKernelImpl<phi::complex64>(dev_ctx, input, out);
}

template <>
void ContiguousKernel<phi::complex128, XPUContext>(const XPUContext& dev_ctx,
                                                   const DenseTensor& input,
                                                   DenseTensor* out) {
  ComplexContiguousKernelImpl<phi::complex128>(dev_ctx, input, out);
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
                   phi::complex64,
                   phi::complex128,
#endif
                   phi::float16,
                   phi::bfloat16) {
}
