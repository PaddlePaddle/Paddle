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
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  using XPUT = typename XPUTypeTrait<T>::Type;
  phi::DenseTensorMeta meta = input.meta();
  meta.stride = meta.calc_stride(meta.dims, meta.layout);
  out->set_meta(meta);

  auto input_data = reinterpret_cast<const XPUT*>(input.data<T>());
  auto output_data = reinterpret_cast<XPUT*>(dev_ctx.template Alloc<T>(out));

  int r = xpu::as_strided<XPUT>(dev_ctx.x_context(),
                                input_data,
                                output_data,
                                phi::vectorize<int64_t>(input.dims()),
                                phi::vectorize<int64_t>(input.stride()),
                                0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "as_strided");
}
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
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16,
                   ::phi::dtype::complex<float>,
                   ::phi::dtype::complex<double>) {}
