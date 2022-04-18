/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

namespace phi {
template <typename T, typename Context>
void LinearGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      const DenseTensor& b,
                      const DenseTensor& out,
                      const DenseTensor& mm,
                      const DenseTensor& out_grad,
                      bool transpose_x,
                      bool transpose_y,
                      DenseTensor* dx,
                      DenseTensor* dy,
                      DenseTensor* db) {
  DenseTensor dmm;
  EmptyLikeKernel<T, Context>(dev_ctx, mm, mm.type(), &dmm);
  AddGradKernel<T, Context>(dev_ctx, mm, b, out_grad, -1, &dmm, db);
  MatmulGradKernel<T, Context>(
      dev_ctx, x, y, dmm, transpose_x, transpose_y, dx, dy);
}

}  // namespace phi
