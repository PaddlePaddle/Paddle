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

#include "paddle/phi/kernels/p_matrix_norm_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_grad_kernel.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/arg_min_max_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/values_vectors_functor.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"
#include "paddle/phi/kernels/matmul_grad_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/reduce_max_grad_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_min_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/svd_grad_kernel.h"
#include "paddle/phi/kernels/svd_kernel.h"
#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T>
struct DivSqrtFunctor {
  inline HOSTDEVICE T operator()(T in, float epsilon) { return 1.0 / 2 * in; }
};

template <typename T, typename Context>
void PMatrixNormGradKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& out,
                           const DenseTensor& out_grad,
                           float porder,
                           const std::vector<int>& axis,
                           float epsilon,
                           bool keepdim UNUSED,
                           bool asvector,
                           DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  // forward
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
    // forward
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

    // backward
    DenseTensor max_min_grad;
    max_min_grad.Resize(x_sum.dims());
    dev_ctx.template Alloc<T>(&max_min_grad);

    if (porder == INFINITY || porder == 1) {
      ReduceMaxGradKernel<T, Context>(
          dev_ctx, x_sum, out, out_grad, {-1}, false, false, &max_min_grad);

    } else {
      ReduceMinGradKernel<T, Context>(
          dev_ctx, x_sum, out, out_grad, {-1}, false, false, &max_min_grad);
    }

    DenseTensor sum_grad;
    sum_grad.Resize(x_input.dims());
    dev_ctx.template Alloc<T>(&sum_grad);
    if (porder == INFINITY || porder == -INFINITY) {
      ReduceSumGradKernel<T, Context>(
          dev_ctx, x_input, max_min_grad, {-1}, false, false, &sum_grad);
    } else {
      ReduceSumGradKernel<T, Context>(
          dev_ctx, x_input, max_min_grad, {-2}, false, false, &sum_grad);
    }

    DenseTensor abs_grad;
    abs_grad.Resize(x_input.dims());
    dev_ctx.template Alloc<T>(&abs_grad);
    phi::AbsGradKernel<T, Context>(dev_ctx, x_input, sum_grad, &abs_grad);
    TransposeGradKernel<T, Context>(dev_ctx, abs_grad, formated_axis, x_grad);

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

    /*
        backward
    */
    DenseTensor one;
    one.Resize(out_grad.dims());
    dev_ctx.template Alloc<float>(&one);

    DenseTensor two;
    two.Resize(out_grad.dims());
    dev_ctx.template Alloc<float>(&two);

    DenseTensor ep;
    ep.Resize(out_grad.dims());
    dev_ctx.template Alloc<float>(&ep);

    phi::funcs::set_constant(dev_ctx, &one, 1);
    phi::funcs::set_constant(dev_ctx, &two, 2);
    phi::funcs::set_constant(dev_ctx, &ep, epsilon);

    std::cout << "out_grad:" << out_grad.dims() << "\n";
    std::cout << "out:" << out.dims() << "\n";

    DenseTensor sqrt_grad;
    sqrt_grad.Resize(out_grad.dims());
    dev_ctx.template Alloc<T>(&sqrt_grad);
    MultiplyKernel<T, Context>(dev_ctx, two, out, &sqrt_grad);
    AddKernel<T, Context>(dev_ctx, sqrt_grad, ep, &sqrt_grad);
    DivideKernel<T, Context>(dev_ctx, one, sqrt_grad, &sqrt_grad);
    MultiplyKernel<T, Context>(dev_ctx, out_grad, sqrt_grad, &sqrt_grad);

    // std::cout<<"sqrt_grad:\n";
    // for(size_t i = 0; i < static_cast<size_t>(sqrt_grad.numel()); i++) {
    //   std::cout<<sqrt_grad.data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";

    DenseTensor max_grad;
    max_grad.Resize(singular_tensor.dims());
    dev_ctx.template Alloc<T>(&max_grad);
    ReduceMaxGradKernel<T, Context>(dev_ctx,
                                    singular_tensor_abs,
                                    max_singular_tensor,
                                    sqrt_grad,
                                    {-1},
                                    false,
                                    false,
                                    &max_grad);

    // std::cout<<"max_grad:\n";
    // for(size_t i = 0; i < static_cast<size_t>(max_grad.numel()); i++) {
    //   std::cout<<max_grad.data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";

    DenseTensor abs_grad;
    abs_grad.Resize(singular_tensor.dims());
    dev_ctx.template Alloc<T>(&abs_grad);
    AbsGradKernel<T, Context>(dev_ctx, singular_tensor, max_grad, &abs_grad);

    // std::cout<<"abs_grad:\n";
    // for(size_t i = 0; i < static_cast<size_t>(abs_grad.numel()); i++) {
    //   std::cout<<abs_grad.data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";

    // phi::funcs::set_constant(dev_ctx, &abs_grad, 1);

    DenseTensor singular_grad;
    singular_grad.Resize(x_2.dims());
    dev_ctx.template Alloc<T>(&singular_grad);

    DenseTensor u_grad;
    u_grad.Resize(x_2.dims());
    dev_ctx.template Alloc<T>(&u_grad);

    DenseTensor vh_grad;
    vh_grad.Resize(x_2.dims());
    dev_ctx.template Alloc<T>(&vh_grad);

    phi::funcs::set_constant(dev_ctx, &u_grad, 0);
    phi::funcs::set_constant(dev_ctx, &vh_grad, 0);

    SvdGradKernel<T, Context>(dev_ctx,
                              x_2,
                              u_tensor,
                              vh_tensor,
                              singular_tensor,
                              u_grad,
                              vh_grad,
                              abs_grad,
                              false,
                              &singular_grad);

    // std::cout<<"singular_grad:\n";
    // for(size_t i = 0; i < static_cast<size_t>(singular_grad.numel()); i++) {
    //   std::cout<<singular_grad.data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";

    DenseTensor x_input_grad;
    x_input_grad.Resize(x_input.dims());
    dev_ctx.template Alloc<T>(&x_input_grad);

    DenseTensor x_transposed_grad;
    x_transposed_grad.Resize(x_transposed.dims());
    dev_ctx.template Alloc<T>(&x_transposed_grad);

    MatmulGradKernel<T, Context>(dev_ctx,
                                 x_transposed,
                                 x_input,
                                 singular_grad,
                                 false,
                                 false,
                                 &x_transposed_grad,
                                 &x_input_grad);

    // std::cout<<"x_input_grad:\n";
    // for(size_t i = 0; i < static_cast<size_t>(x_input_grad.numel()); i++) {
    //   std::cout<<x_input_grad.data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";

    two.Resize(x_input_grad.dims());
    dev_ctx.template Alloc<float>(&two);
    phi::funcs::set_constant(dev_ctx, &two, 2);
    MultiplyKernel<T, Context>(dev_ctx, two, x_input_grad, &x_input_grad);

    formated_axis[x_rank - 2] = m;
    formated_axis[x_rank - 1] = n;
    TransposeGradKernel<T, Context>(
        dev_ctx, x_input_grad, formated_axis, x_grad);

    // std::cout<<"x_grad:\n";
    // // auto x_grad_data
    // for(size_t i = 0; i < static_cast<size_t>(x_grad->numel()); i++) {
    //   std::cout<<x_grad->data<T>()[static_cast<int>(i)]<<" ";
    // }
    // std::cout<<"\n";
  }
}
}  // namespace phi
