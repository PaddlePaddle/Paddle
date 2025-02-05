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

#include "paddle/phi/kernels/fill_diagonal_tensor_grad_kernel.h"
#include <array>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FillDiagonalTensorGradKernel(const Context& ctx,
                                  const DenseTensor& out_grad,
                                  int64_t offset,
                                  int dim1,
                                  int dim2,
                                  DenseTensor* x_grad) {
  int matrows = 1;

  if (x_grad) {
    auto* data = ctx.template Alloc<T>(x_grad);

    auto dx_dims = x_grad->dims();
    for (int i = 0; i < dx_dims.size(); i++) {
      if (i != dim1 && i != dim2) {
        matrows *= static_cast<int>(dx_dims[i]);
      }
    }

    std::array<int64_t, 2> new_dims = {};
    std::array<int64_t, 2> strides = {};
    std::vector<int64_t> matdim;
    matdim.resize(matrows);
    CalMatDims(dx_dims,
               dim1,
               dim2,
               &offset,
               new_dims.data(),
               strides.data(),
               matdim.data());

    auto size = x_grad->numel();
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);

    for (int64_t i = 0; i < new_dims[0]; i += 1) {
      auto sumoff = matdim[i] + offset;
      for (int64_t j = 0; j < new_dims[1]; j += 1) {
        auto fill_index = j * (strides[1] + strides[0]) + sumoff;
        if (fill_index < size) {
          data[fill_index] = 0;
        }
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>,
                   bool) {}
