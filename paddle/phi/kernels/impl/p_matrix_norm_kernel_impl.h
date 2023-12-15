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

#include "paddle/phi/kernels/p_matrix_norm_kernel.h"

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
#include "paddle/phi/kernels/reduce_min_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/svd_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void PMatrixNormKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       float porder,
                       const std::vector<int>& axis,
                       float epsilon UNUSED,
                       bool keepdim UNUSED,
                       bool asvector,
                       DenseTensor* out) {
  std::cout << "C++++++++\n";
  auto in_dims = x.dims();
  int x_rank = x.dims().size();

  int m = axis[0] >= 0 ? axis[0] : static_cast<int>(axis[0] + x_rank);
  int n = axis[1] >= 0 ? axis[1] : static_cast<int>(axis[1] + x_rank);
  if (m > n) {
    std::swap(m, n);
  }
  // axis put back
  std::vector<int> formated_axis(x_rank);
  int cur = 0;
  for (int i = 0; i < x_rank; i++) {
    if (i != m && i != n) formated_axis[cur++] = static_cast<int>(i);
  }
  formated_axis[x_rank - 2] = m;
  formated_axis[x_rank - 1] = n;

  std::cout << "m,n:" << m << " " << n << std::endl;

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

  if (porder == INFINITY || porder == -INFINITY || porder == 1 ||
      porder == -1) {
    phi::AbsKernel<T, Context>(dev_ctx, x_input, &x_input);

    DenseTensor x_sum;
    if (porder == INFINITY || porder == -INFINITY) {
      x_sum.Resize(
          detail::GetEigenvalueDim(x_input.dims(), trans_dims[x_rank - 2]));
      dev_ctx.template Alloc<T>(&x_sum);
      phi::SumKernel<T, Context>(
          dev_ctx, x_input, {-1}, x_input.dtype(), false, &x_sum);
    } else if (porder == 1 || porder == -1) {
      x_sum.Resize(
          detail::GetEigenvalueDim(x_input.dims(), trans_dims[x_rank - 1]));
      dev_ctx.template Alloc<T>(&x_sum);
      phi::SumKernel<T, Context>(
          dev_ctx, x_input, {-2}, x_input.dtype(), false, &x_sum);
    }
    if (porder == INFINITY || porder == 1) {
      phi::MaxKernel<T, Context>(
          dev_ctx, x_sum, phi::IntArray({-1}), false, out);
    } else {
      phi::MinKernel<T, Context>(
          dev_ctx, x_sum, phi::IntArray({-1}), false, out);
    }

  } else {
    formated_axis[x_rank - 2] = n;
    formated_axis[x_rank - 1] = m;
    for (size_t i = 0; i < formated_axis.size(); i++) {
      trans_dims[static_cast<int>(i)] = in_dims[formated_axis[i]];
    }

    // x_transposed: A.T
    DenseTensor x_transposed;
    x_transposed.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&x_transposed);
    TransposeKernel<T, Context>(dev_ctx, x, formated_axis, &x_transposed);
    std::cout << "x_transpose's numel:" << x_transposed.numel() << std::endl;
    std::cout << "x_transpose's size:" << x_transposed.dims() << std::endl;

    std::cout << "transpose over\n";

    // A.T @ A 's dims
    formated_axis[x_rank - 1] = n;
    for (size_t i = 0; i < formated_axis.size(); i++) {
      trans_dims[static_cast<int>(i)] = in_dims[formated_axis[i]];
    }

    // x_2: A.T @ A
    DenseTensor x_2;
    x_2.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&x_2);
    MatmulKernel<T, Context>(
        dev_ctx, x_transposed, x_input, false, false, &x_2);

    // singular
    DenseTensor singular_tensor;
    singular_tensor.Resize(detail::RemoveLastDim(x_2.dims()));
    dev_ctx.template Alloc<T>(&singular_tensor);

    // U
    DenseTensor u_tensor;
    u_tensor.Resize(x_2.dims());
    dev_ctx.template Alloc<T>(&u_tensor);

    // VH
    DenseTensor vh_tensor;
    vh_tensor.Resize(x_2.dims());
    dev_ctx.template Alloc<T>(&vh_tensor);

    SvdKernel<T, Context>(
        dev_ctx, x_2, false, &u_tensor, &singular_tensor, &vh_tensor);

    // abs eigenvalue
    DenseTensor singular_tensor_abs;
    singular_tensor_abs.Resize(detail::RemoveLastDim(x_2.dims()));
    dev_ctx.template Alloc<T>(&singular_tensor_abs);
    phi::AbsKernel<T, Context>(dev_ctx, singular_tensor, &singular_tensor_abs);

    DenseTensor max_singular_tensor;
    max_singular_tensor.Resize(
        detail::RemoveLastDim(singular_tensor_abs.dims()));
    dev_ctx.template Alloc<T>(&max_singular_tensor);
    phi::MaxKernel<T, Context>(dev_ctx,
                               singular_tensor_abs,
                               phi::IntArray({-1}),
                               false,
                               &max_singular_tensor);

    dev_ctx.template Alloc<T>(out);
    SqrtKernel<T, Context>(dev_ctx, max_singular_tensor, out);
  }
}
}  // namespace phi
