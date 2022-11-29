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

#pragma once

#include "paddle/phi/kernels/inverse_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace phi {

template <typename T, typename Context>
void InverseKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  phi::funcs::MatrixInverseFunctor<Context, T> mat_inv;
  mat_inv(dev_ctx, x, out);
}

}  // namespace phi
