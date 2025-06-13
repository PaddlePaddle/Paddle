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

#include "glog/logging.h"

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/impl/determinant_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/isfinite_kernel_impl.h"
#include "paddle/phi/kernels/slogdeterminant_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
namespace phi {

template <typename T, typename Context>
void SlogDeterminantGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& sign,
                               const DenseTensor& logdet,
                               const DenseTensor& sign_grad,
                               const DenseTensor& logdet_grad,
                               DenseTensor* x_grad) {
  using RealT = typename phi::dtype::Real<T>;
  const auto& x_dims = x.dims();
  const auto& grad_dims = logdet_grad.dims();
  int x_rank = x_dims.size();
  int grad_rank = grad_dims.size();

  PADDLE_ENFORCE_GE(
      x_rank,
      2,
      phi::errors::InvalidArgument(
          "Input tensor X's rank must be at least 2, but received %d.",
          x_rank));

  if (x_rank == 2)
    PADDLE_ENFORCE_EQ(
        grad_rank,
        0,
        phi::errors::InvalidArgument(
            "For a 2D input tensor X, the gradient tensor (logdet_grad) "
            "should be a 0D tensor (scalar), but received rank %d.",
            grad_rank));
  else if (x_rank > 2)
    PADDLE_ENFORCE_EQ(
        grad_rank + 2,
        x_rank,
        phi::errors::InvalidArgument(
            "The rank of gradient tensor (logdet_grad) should be 2 less than "
            "the input tensor X's rank, but received grad rank %d and X rank "
            "%d.",
            grad_rank,
            x_rank));

  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad->numel() == 0) {
    return;
  }

  // Check Whether the matrix is invertible
  // (matrix A not invertible) == (absslogdet(A)=0)
  if (!detail::CheckMatrixInvertible<RealT, Context>(dev_ctx, &logdet)) {
    // The matrix is not invertible
    VLOG(3) << "The input matrix not invertible!";
    phi::Full<T>(dev_ctx,
                 common::vectorize(x.dims()),
                 std::numeric_limits<T>::quiet_NaN(),
                 x_grad);
    return;
  }

  // The matrix is invertible
  // let sl|A| = SlogDeterminant(A)
  // Ref to https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  // we set dsl|A| = unsqueeze(dslA, [-1, -2]) *
  // inverse(A).conj().transpose(-2, -1)

  // First: inverse(A)
  DenseTensor inverse_A;
  // A must be square matrices!
  inverse_A.Resize(x_dims);
  dev_ctx.template Alloc<T>(&inverse_A);

  phi::funcs::MatrixInverseFunctor<Context, T> mat_inv;
  mat_inv(dev_ctx, x, &inverse_A);

  VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

  // Second: inverse(A).conj() for complex
  DenseTensor conj_inverse_A;
  if constexpr (is_complex64_or_complex128<T>::value) {
    conj_inverse_A = phi::Conj<T>(dev_ctx, inverse_A);
    VLOG(3) << "Performed complex conjugate.";
  } else {
    conj_inverse_A.ShareDataWith(inverse_A);
    VLOG(3) << "Skipped complex conjugate for real type.";
  }

  VLOG(3) << "inverse(A).conj() dims: " << conj_inverse_A.dims();

  // Third: inverse(A).conj().transpose(-2, -1)
  DenseTensor transpose_inverse_A =
      phi::TransposeLast2Dim<T>(dev_ctx, conj_inverse_A);
  VLOG(3) << "inverse(A).conj().transpose(-2, -1) dims: "
          << transpose_inverse_A.dims();

  DenseTensor logdet_grad_term = logdet_grad;
  if constexpr (is_complex64_or_complex128<T>::value) {
    // change logdet_grad datatype from <RealT> to <ComplexT>
    DenseTensor logdet_grad_complex =
        Empty<T>(dev_ctx, common::vectorize(grad_dims));

    int64_t logdet_numel = logdet_grad.numel();
    phi::funcs::ForRange<Context> for_range(dev_ctx, logdet_numel);
    phi::funcs::RealToComplexFunctor<T> functor(
        logdet_grad.data<RealT>(), logdet_grad_complex.data<T>(), logdet_numel);

    for_range(functor);
    logdet_grad_term = logdet_grad_complex;
  }
  DenseTensor unsqueezed_combined_grad =
      phi::funcs::Unsqueeze(logdet_grad_term, -1);
  unsqueezed_combined_grad =
      phi::funcs::Unsqueeze(unsqueezed_combined_grad, -2);
  VLOG(3) << "unsqueezed_combined_grad dims: "
          << unsqueezed_combined_grad.dims();

  phi::Multiply<T, Context>(
      dev_ctx, unsqueezed_combined_grad, transpose_inverse_A, x_grad);
  VLOG(3) << x_grad->dims();
}

}  // namespace phi
