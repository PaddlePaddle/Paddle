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

#include "paddle/phi/kernels/cummax_kernel.h"

#include <cstdint>
#include <type_traits>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cummax.h"

namespace phi {

template <typename T, typename Context>
void CummaxKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  DenseTensor* out DenseTensor* indices) {
  const auto& dims = input->dims();

  PADDLE_ENFORCE_EQ(
      axis < dims.size() && axis >= (0 - dims.size()),
      true,
      phi::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          dims.size(),
          dims.size() - 1,
          axis));
  if (axis < 0) {
    axis += dims.size();
  }

  auto* x_data = x->data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  size_t outer_dim = 1, inner_dim = 1, mid_dim = dims[axis];
  for (size_t i = 0; i < axis; ++i) outer_dim *= dims[i];
  for (size_t i = axis + 1; i < dims.size(); ++i) inner_dim *= dims[i];

  for (size_t i = 0; i < outer_dim; i++) {
    for (size_t j = 0; j < mid_dim; j++) {
      for (size_t k = 0; k < inner_dim; k++) {
        size_t pos = i * mid_dim * inner_dim + j * inner_dim + k;
        if (j == 0) {
          out_data[pos] = x_data[pos];
          indices_data[pos] = 0;
        } else if (x_data[pos] > out_data[pos - inner_dim]) {
          out_data[pos] = x_data[pos];
          indices_data[pos] = k;
        } else {
          out_data[pos] = out_data[pos - inner_dim];
          indices_data[pos] = indices_data[pos - inner_dim];
        }
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(cummax,
                   CPU,
                   ALL_LAYOUT,
                   phi::CummaxKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
