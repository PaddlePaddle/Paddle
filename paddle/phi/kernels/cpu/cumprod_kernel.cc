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

#include "paddle/phi/kernels/cumprod_kernel.h"

#include <cstdint>
#include <type_traits>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"

namespace phi {
template <typename T, typename Context>
void CumprodKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int dim,
                   DenseTensor* out) {
  const DenseTensor* x = &input;
  auto* x_data = x->data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  DDim shape = x->dims();

  size_t outer_dim = 1;
  size_t mid_dim = 1;
  size_t inner_dim = 1;
  GetCumprodDimInfo(shape, dim, &outer_dim, &mid_dim, &inner_dim);

  for (size_t i = 0; i < outer_dim; i++) {
    for (size_t j = 0; j < mid_dim; j++) {
      for (size_t k = 0; k < inner_dim; k++) {
        size_t pos = i * mid_dim * inner_dim + j * inner_dim + k;
        if (j == 0) {
          out_data[pos] = x_data[pos];
        } else {
          out_data[pos] = out_data[pos - inner_dim] * x_data[pos];
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cumprod,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumprodKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
