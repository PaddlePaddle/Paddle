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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/transpose_grad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void ContiguousKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      DenseTensor* out) {
  phi::DenseTensorMeta meta = input.meta();
  meta.strides = meta.calc_strides(meta.dims);
  meta.offset = 0;
  out->set_meta(meta);

  const T* input_data = input.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  int rank = input.dims().size();
  auto dims = input.dims();
  auto input_stride = input.strides();
  auto numel = input.numel();

  for (int64_t i = 0; i < numel; i++) {
    int64_t input_offset = 0;
    int64_t index_tmp = i;
    for (int dim = rank - 1; dim >= 0; --dim) {
      int64_t mod = index_tmp % dims[dim];
      index_tmp = index_tmp / dims[dim];
      input_offset += mod * input_stride[dim];
    }

    output_data[i] = input_data[input_offset];
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(contiguous,
                   CPU,
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
                   ::phi::dtype::complex<double>,
                   ::phi::dtype::float8_e4m3fn,
                   ::phi::dtype::float8_e5m2) {}
