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
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  int r = 0;

  if (std::is_same<T, float>::value) {
    auto input_data = reinterpret_cast<const float*>(input.data<T>());
    auto output_data = reinterpret_cast<float*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<float>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<float>(dev_ctx.x_context(),
                                 input_data,
                                 output_data,
                                 common::vectorize<int64_t>(input.dims()),
                                 common::vectorize<int64_t>(input.strides()),
                                 0);
    }
  } else if (std::is_same<T, double>::value) {
    auto input_data = reinterpret_cast<const int64_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int64_t*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<int64_t>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<int64_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   common::vectorize<int64_t>(input.dims()),
                                   common::vectorize<int64_t>(input.strides()),
                                   0);
    }
  } else if (std::is_same<T, ::phi::dtype::float16>::value) {
    using XPUFLOAT16 = typename XPUTypeTrait<float16>::Type;
    auto input_data = reinterpret_cast<const XPUFLOAT16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<XPUFLOAT16*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<XPUFLOAT16>(
          dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<XPUFLOAT16>(
          dev_ctx.x_context(),
          input_data,
          output_data,
          common::vectorize<int64_t>(input.dims()),
          common::vectorize<int64_t>(input.strides()),
          0);
    }
  } else if (std::is_same<T, ::phi::dtype::bfloat16>::value) {
    using XPUFLOAT16 = typename XPUTypeTrait<float16>::Type;
    auto input_data = reinterpret_cast<const XPUFLOAT16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<XPUFLOAT16*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<XPUFLOAT16>(
          dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<XPUFLOAT16>(
          dev_ctx.x_context(),
          input_data,
          output_data,
          common::vectorize<int64_t>(input.dims()),
          common::vectorize<int64_t>(input.strides()),
          0);
    }
  } else if (std::is_same<T, int16_t>::value) {
    using XPUFLOAT16 = typename XPUTypeTrait<float16>::Type;
    auto input_data = reinterpret_cast<const XPUFLOAT16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<XPUFLOAT16*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<XPUFLOAT16>(
          dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<XPUFLOAT16>(
          dev_ctx.x_context(),
          input_data,
          output_data,
          common::vectorize<int64_t>(input.dims()),
          common::vectorize<int64_t>(input.strides()),
          0);
    }
  } else if (std::is_same<T, uint8_t>::value) {
    auto input_data = reinterpret_cast<const int8_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int8_t*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<int8_t>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<int8_t>(dev_ctx.x_context(),
                                  input_data,
                                  output_data,
                                  common::vectorize<int64_t>(input.dims()),
                                  common::vectorize<int64_t>(input.strides()),
                                  0);
    }
  } else if (std::is_same<T, int8_t>::value) {
    auto input_data = reinterpret_cast<const int8_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int8_t*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<int8_t>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<int8_t>(dev_ctx.x_context(),
                                  input_data,
                                  output_data,
                                  common::vectorize<int64_t>(input.dims()),
                                  common::vectorize<int64_t>(input.strides()),
                                  0);
    }
  } else if (std::is_same<T, int32_t>::value) {
    auto input_data = reinterpret_cast<const int32_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int32_t*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<int32_t>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<int32_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   common::vectorize<int64_t>(input.dims()),
                                   common::vectorize<int64_t>(input.strides()),
                                   0);
    }
  } else if (std::is_same<T, int64_t>::value) {
    auto input_data = reinterpret_cast<const int64_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int64_t*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<int64_t>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<int64_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   common::vectorize<int64_t>(input.dims()),
                                   common::vectorize<int64_t>(input.strides()),
                                   0);
    }
  } else if (std::is_same<T, bool>::value) {
    auto input_data = reinterpret_cast<const bool*>(input.data<T>());
    auto output_data = reinterpret_cast<bool*>(dev_ctx.template Alloc<T>(out));
    if (input.numel() == 1) {
      r = xpu::copy<bool>(dev_ctx.x_context(), input_data, output_data, 1);
    } else {
      r = xpu::as_strided<bool>(dev_ctx.x_context(),
                                input_data,
                                output_data,
                                common::vectorize<int64_t>(input.dims()),
                                common::vectorize<int64_t>(input.strides()),
                                0);
    }
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Received unsupported dtype : %s.", input.dtype()));
  }

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "contiguous");
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
