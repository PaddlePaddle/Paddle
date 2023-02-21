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

#include <vector>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/memcpy_with_strides_kernel.h"

namespace phi {

template <typename T>
void hostCopyWithStrides(int ndims,
                         const int64_t* dims,
                         const T* src,
                         const int64_t* srcStrides,
                         T* dst,
                         const int64_t* dstStrides) {
  if (ndims > 0) {
    for (int64_t k = 0; k < *dims; ++k) {
      hostCopyWithStrides(
          ndims - 1, dims + 1, src, srcStrides + 1, dst, dstStrides + 1);
      src += *srcStrides;
      dst += *dstStrides;
    }
  } else {
    *dst = *src;
  }
}

template <typename T, typename Context>
void memcpyWithStridesKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             DenseTensor* out) {
  const T* src = input.data<T>();
  T* dst = out->data<T>();
  auto input_meta = input.meta();
  int ndims = input_meta.dims.size();
  const int64_t* dims = input_meta.dims.Get();
  const int64_t* srcStrides = input_meta.strides.Get();
  const int64_t* dstStrides = out->meta().strides.Get();

  hostCopyWithStrides<T>(ndims, dims, src, srcStrides, dst, dstStrides);
}
}  // namespace phi

PD_REGISTER_KERNEL(copyWithStrides,
                   CPU,
                   ALL_LAYOUT,
                   phi::memcpyWithStridesKernel,
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
