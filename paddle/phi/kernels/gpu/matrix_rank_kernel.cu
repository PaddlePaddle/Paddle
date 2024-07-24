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

#include "paddle/phi/kernels/matrix_rank_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/matrix_rank_tol_kernel.h"

namespace phi {

template <typename T, typename Context>
void MatrixRankKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      float tol,
                      bool use_default_tol,
                      bool hermitian,
                      DenseTensor* out) {
  DenseTensor atol_tensor;
  if (use_default_tol) {
    atol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(0));
  } else {
    atol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(tol));
  }
  MatrixRankTolKernel<T, Context>(
      dev_ctx, x, atol_tensor, use_default_tol, hermitian, out);
}

template <typename T, typename Context>
void MatrixRank2Kernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       float atol,
                       float rtol,
                       bool use_default_tol,
                       bool hermitian,
                       DenseTensor* out) {
  DenseTensor atol_tensor, rtol_tensor;
  atol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(atol));
  rtol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(rtol));
  MatrixRankAtolRtolKernel<T, Context>(
      dev_ctx, x, atol_tensor, rtol_tensor, use_default_tol, hermitian, out);
}

template <typename T, typename Context>
void MatrixRankAtolKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& atol_tensor,
                          float rtol,
                          bool hermitian,
                          DenseTensor* out) {
  DenseTensor rtol_tensor;
  rtol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(rtol));
  MatrixRankAtolRtolKernel<T, Context>(
      dev_ctx, x, atol_tensor, rtol_tensor, false, hermitian, out);
}

template <typename T, typename Context>
void MatrixRankRtolKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& rtol_tensor,
                          float atol,
                          bool hermitian,
                          DenseTensor* out) {
  DenseTensor atol_tensor;
  atol_tensor = phi::Full<T, Context>(dev_ctx, {1}, static_cast<T>(atol));
  MatrixRankAtolRtolKernel<T, Context>(
      dev_ctx, x, atol_tensor, rtol_tensor, false, hermitian, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(matrix_rank,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(matrix_rank2,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRank2Kernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(matrix_rank_atol,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankAtolKernel,
                   float,
                   double) {}

PD_REGISTER_KERNEL(matrix_rank_rtol,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankRtolKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
