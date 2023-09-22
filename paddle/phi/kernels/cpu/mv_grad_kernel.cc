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

#include "paddle/phi/kernels/mv_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T, typename Context>
void MvGradKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& vec,
                  const DenseTensor& out_grad,
                  DenseTensor* x_grad,
                  DenseTensor* vec_grad) {
  auto dout = out_grad;
  auto dx = x_grad;
  auto dvec = vec_grad;

  const auto& dim_x = x.dims();
  int m = static_cast<int>(dim_x[0]);
  int n = static_cast<int>(dim_x[1]);

  // get data ptr
  const T* x_data = x.data<T>();
  const T* vec_data = vec.data<T>();
  const T* dout_data = dout.data<T>();

  if (dx) {
    T* dx_data = dev_ctx.template Alloc<T>(dx);

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        dx_data[i * n + j] = dout_data[i] * vec_data[j];
      }
    }
  }

  if (dvec) {
    T* dvec_data = dev_ctx.template Alloc<T>(dvec);

    auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

    blas.GEMV(true,
              dim_x[0],
              dim_x[1],
              static_cast<T>(1),
              x_data,
              dout_data,
              static_cast<T>(0),
              dvec_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(mv_grad, CPU, ALL_LAYOUT, phi::MvGradKernel, float, double) {
}
