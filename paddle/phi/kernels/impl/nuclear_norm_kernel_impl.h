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

#include "paddle/phi/kernels/nuclear_norm_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/values_vectors_functor.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/svd_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void NuclearNormKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const std::vector<int>& axis,
                       bool keepdim,
                       bool reduce_all UNUSED,
                       DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto in_dims = x.dims();
  int x_rank = x.dims().size();

  int m = axis[0] >= 0 ? axis[0] : static_cast<int>(axis[0] + x_rank);
  int n = axis[1] >= 0 ? axis[1] : static_cast<int>(axis[1] + x_rank);

  // axis put back
  std::vector<int> formated_axis(x_rank);
  int cur = 0;
  for (int i = 0; i < x_rank; i++) {
    if (i != m && i != n) formated_axis[cur++] = static_cast<int>(i);
  }
  formated_axis[x_rank - 2] = m;
  formated_axis[x_rank - 1] = n;

  // transpose dims
  phi::DDim trans_dims(x.dims());
  for (size_t i = 0; i < formated_axis.size(); i++) {
    trans_dims[static_cast<int>(i)] = in_dims[formated_axis[i]];
  }

  // x_input: A
  DenseTensor x_input;
  x_input.Resize(trans_dims);
  dev_ctx.template Alloc<T>(&x_input);
  TransposeKernel<T, Context>(dev_ctx, x, formated_axis, &x_input);

  int M = trans_dims[x_rank - 2];
  int N = trans_dims[x_rank - 1];
  int K = std::min(M, N);

  // singular
  DenseTensor singular_tensor;
  singular_tensor.Resize(detail::GetEigenvalueDim(x_input.dims(), K));
  dev_ctx.template Alloc<T>(&singular_tensor);

  // U
  DenseTensor u_tensor;
  u_tensor.Resize(detail::GetUDDim(x_input.dims(), K));
  dev_ctx.template Alloc<T>(&u_tensor);

  // VH
  DenseTensor vh_tensor;
  vh_tensor.Resize(detail::GetVHDDim(x_input.dims(), K));
  dev_ctx.template Alloc<T>(&vh_tensor);

  SvdKernel<T, Context>(
      dev_ctx, x_input, false, &u_tensor, &singular_tensor, &vh_tensor);

  if (keepdim) {
    singular_tensor.Resize(detail::NewAxisDim(singular_tensor.dims(), 1));
    phi::SumKernel<T, Context>(
        dev_ctx, singular_tensor, {-2}, singular_tensor.dtype(), keepdim, out);
  } else {
    phi::SumKernel<T, Context>(
        dev_ctx, singular_tensor, {-1}, singular_tensor.dtype(), keepdim, out);
  }

  std::cout << "sum_grad.dims():" << singular_tensor.dims() << std::endl;
}
}  // namespace phi
