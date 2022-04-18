/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/matmul_kernel.h"

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void LinearKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const DenseTensor& b,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out,
                  DenseTensor* mm) {
  PADDLE_ENFORCE_NE(
      phi::product(x.dims()),
      0,
      phi::errors::InvalidArgument("The Input(X) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  PADDLE_ENFORCE_NE(
      phi::product(y.dims()),
      0,
      phi::errors::InvalidArgument("The Input(Y) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  MatmulKernel<T, Context>(dev_ctx, x, y, transpose_x, transpose_y, mm);
  AddKernel<T, Context>(dev_ctx, *mm, b, out);
}

}  // namespace phi
