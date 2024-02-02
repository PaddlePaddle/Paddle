// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/swiglu_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace phi {

template <typename T, typename Context>
void SwiGLUGradKernelImpl(const Context &ctx,
                          const T *x,
                          const T *y,
                          const T *dz,
                          T *dx,
                          T *dy,
                          int64_t m,
                          int64_t n) {
  funcs::SwiGLUGradFunctor<T> functor;
  if (y) {
    int64_t numel = m * n;
    for (int64_t i = 0; i < numel; ++i) {
      T dx_tmp, dy_tmp;
      functor(x[i], y[i], dz[i], &dx_tmp, &dy_tmp);
      if (dx) {
        dx[i] = dx_tmp;
      }
      if (dy) {
        dy[i] = dy_tmp;
      }
    }
  } else {
    int64_t stride = 2 * n;
    y = x + n;
    dy = dx + n;
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        functor(x[i * stride + j],
                y[i * stride + j],
                dz[i * n + j],
                &dx[i * stride + j],
                &dy[i * stride + j]);
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    swiglu_grad, CPU, ALL_LAYOUT, phi::SwiGLUGradKernel, float, double) {}
