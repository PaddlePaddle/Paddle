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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/elementwise.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MatrixRankKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      bool hermitian,
                      bool use_default_tol,
                      float tol,
                      DenseTensor* out) {
  DenseTensor atol_tensor;
  paddle::framework::TensorFromVector<T>(
      std::vector<T>{tol}, dev_ctx, &atol_tensor);
  MatrixRankTolKernel(dev_ctx, x, atol_tensor, hermite, use_default_tol, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(matrix_rank,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
