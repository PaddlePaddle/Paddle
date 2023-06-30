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

#include "paddle/phi/kernels/strided_copy_kernel.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace phi {

template <typename T, typename Context>
void StridedCopyKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& out_stride,
                       int64_t offset,
                       DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = phi::make_ddim(out_stride);
  meta.dims = phi::make_ddim(dims);
  meta.offset = offset;
  out->set_meta(meta);

  PADDLE_ENFORCE_EQ(input.dims(),
                    out->dims(),
                    phi::errors::InvalidArgument(
                        "Input shape(%s) must be equal with out shape(%s).",
                        input.dims(),
                        out->dims()));

  PADDLE_ENFORCE_EQ(input.numel(),
                    out->numel(),
                    phi::errors::InvalidArgument(
                        "Input numel(%d) must be equal with out numel(%d).",
                        input.numel(),
                        out->numel()));

  int r = 0;
  if (std::is_same<T, float>::value) {
    auto input_data = reinterpret_cast<const float*>(input.data<T>());
    auto output_data = reinterpret_cast<float*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<float>(dev_ctx.x_context(),
                                 input_data,
                                 output_data,
                                 phi::vectorize<int64_t>(input.dims()),
                                 phi::vectorize<int64_t>(out->dims()),
                                 phi::vectorize<int64_t>(input.strides()),
                                 phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, double>::value) {
    auto input_data = reinterpret_cast<const int64_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int64_t*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<int64_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   phi::vectorize<int64_t>(input.dims()),
                                   phi::vectorize<int64_t>(out->dims()),
                                   phi::vectorize<int64_t>(input.strides()),
                                   phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, ::phi::dtype::float16>::value) {
    bfloat16 a = 0;
    auto input_data = reinterpret_cast<const cpp_float16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<cpp_float16*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<cpp_float16>(dev_ctx.x_context(),
                                       input_data,
                                       output_data,
                                       phi::vectorize<int64_t>(input.dims()),
                                       phi::vectorize<int64_t>(out->dims()),
                                       phi::vectorize<int64_t>(input.strides()),
                                       phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, ::phi::dtype::bfloat16>::value) {
    auto input_data = reinterpret_cast<const cpp_float16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<cpp_float16*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<cpp_float16>(dev_ctx.x_context(),
                                       input_data,
                                       output_data,
                                       phi::vectorize<int64_t>(input.dims()),
                                       phi::vectorize<int64_t>(out->dims()),
                                       phi::vectorize<int64_t>(input.strides()),
                                       phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, int16_t>::value) {
    auto input_data = reinterpret_cast<const cpp_float16*>(input.data<T>());
    auto output_data =
        reinterpret_cast<cpp_float16*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<cpp_float16>(dev_ctx.x_context(),
                                       input_data,
                                       output_data,
                                       phi::vectorize<int64_t>(input.dims()),
                                       phi::vectorize<int64_t>(out->dims()),
                                       phi::vectorize<int64_t>(input.strides()),
                                       phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, uint8_t>::value) {
    auto input_data = reinterpret_cast<const int8_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int8_t*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<int8_t>(dev_ctx.x_context(),
                                  input_data,
                                  output_data,
                                  phi::vectorize<int64_t>(input.dims()),
                                  phi::vectorize<int64_t>(out->dims()),
                                  phi::vectorize<int64_t>(input.strides()),
                                  phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, int8_t>::value) {
    auto input_data = reinterpret_cast<const int8_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int8_t*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<int8_t>(dev_ctx.x_context(),
                                  input_data,
                                  output_data,
                                  phi::vectorize<int64_t>(input.dims()),
                                  phi::vectorize<int64_t>(out->dims()),
                                  phi::vectorize<int64_t>(input.strides()),
                                  phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, int32_t>::value) {
    auto input_data = reinterpret_cast<const int32_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int32_t*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<int32_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   phi::vectorize<int64_t>(input.dims()),
                                   phi::vectorize<int64_t>(out->dims()),
                                   phi::vectorize<int64_t>(input.strides()),
                                   phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, int64_t>::value) {
    auto input_data = reinterpret_cast<const int64_t*>(input.data<T>());
    auto output_data =
        reinterpret_cast<int64_t*>(dev_ctx.template Alloc<T>(out));
    r = xpu::strided_copy<int64_t>(dev_ctx.x_context(),
                                   input_data,
                                   output_data,
                                   phi::vectorize<int64_t>(input.dims()),
                                   phi::vectorize<int64_t>(out->dims()),
                                   phi::vectorize<int64_t>(input.strides()),
                                   phi::vectorize<int64_t>(out->strides()));
  } else if (std::is_same<T, bool>::value) {
    auto input_data = reinterpret_cast<const bool*>(input.data<T>());
    auto output_data = reinterpret_cast<bool*>(dev_ctx.template Alloc<T>(out));

    r = xpu::strided_copy<bool>(dev_ctx.x_context(),
                                input_data,
                                output_data,
                                phi::vectorize<int64_t>(input.dims()),
                                phi::vectorize<int64_t>(out->dims()),
                                phi::vectorize<int64_t>(input.strides()),
                                phi::vectorize<int64_t>(out->strides()));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Received unsupported dtype : %s.", input.dtype()));
  }
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "strided_copy");
}

}  // namespace phi

PD_REGISTER_KERNEL(strided_copy,
                   XPU,
                   ALL_LAYOUT,
                   phi::StridedCopyKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16) {}
