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
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/impl/determinant_grad_kernel_impl.h"
#include "paddle/phi/kernels/slogdeterminant_grad_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <typename T, typename Context>
void SlogDeterminantGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& out,
                               const DenseTensor& out_grad,
                               DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(
      out_grad.dims()[0],
      2,
      errors::InvalidArgument("The grad tensor of SlogDet should contain two"
                              " grad: sign and absslogdet, but here %ld.",
                              out_grad.dims()[0]));
  if (x.dims().size() > 2) {
    PADDLE_ENFORCE_EQ(
        out_grad.dims().size() + 1,
        x.dims().size(),
        errors::InvalidArgument(
            "The grad tensor of slogdet dims size should 1 less than"
            " input tensor's, but here differ %d",
            x.dims().size() - out_grad.dims().size()));
  }

  // Check Whether the matrix is invertible
  // (matrix A not invertible) == (absslogdet(A)=0)
  auto slogdet_vec = out.Split(1, 0);
  auto absslogdet_val = slogdet_vec[0];
  if (!detail::CheckMatrixInvertible<T, Context>(dev_ctx, &absslogdet_val)) {
    // The matrix is not invertible
    VLOG(3) << "The input matrix not invertible!";
    x_grad->Resize(x.dims());
    phi::Full<T>(dev_ctx,
                 phi::vectorize(x.dims()),
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
  inverse_A.Resize(x.dims());
  dev_ctx.template Alloc<T>(&inverse_A);

  phi::funcs::MatrixInverseFunctor<Context, T> mat_inv;
  mat_inv(dev_ctx, x, &inverse_A);

  VLOG(3) << "inverse(A) dims: " << inverse_A.dims();

  // Second: inverse(A).conj()
  auto conj_inverse_A = phi::Conj<T>(dev_ctx, inverse_A);

  VLOG(3) << "inverse(A).conj() dims: " << conj_inverse_A.dims();

  // Third: inverse(A).conj().transpose(-2, -1)
  DenseTensor transpose_inverse_A =
      phi::TransposeLast2Dim<T>(dev_ctx, conj_inverse_A);
  VLOG(3) << "inverse(A).conj().transpose(-2, -1) dims: "
          << transpose_inverse_A.dims();

  // Fourth: split grad value to [sign_grad, absslogdet_grad]
  auto grad_vec = out_grad.Split(1, 0);
  auto det_grad = grad_vec[1];

  // remmove useless first dimension
  int det_grad_size = det_grad.dims().size();
  std::vector<int> det_grad_vec;
  for (int i = 1; i < det_grad_size; ++i) {
    det_grad_vec.emplace_back(det_grad.dims()[i]);
  }
  det_grad.Resize(det_grad.dims().reshape(det_grad_vec));

  // Fifth: unsqueeze(dslA, [-1, -2])
  auto unsqueeze1 = phi::funcs::Unsqueeze(det_grad, -1);
  auto unsqueeze2 = phi::funcs::Unsqueeze(unsqueeze1, -2);
  VLOG(3) << "unsqueezed(dslA, [-1, -2]) dims: " << unsqueeze2.dims();

  // Finally: unsqueeze(dslA) * inverse(A)
  auto res = phi::Multiply<T>(dev_ctx, unsqueeze2, transpose_inverse_A);
  VLOG(3) << "unsqueeze(dslA) * inverse(A) dims: " << res.dims();

  phi::Copy(dev_ctx, res, dev_ctx.GetPlace(), false, x_grad);
  x_grad->Resize(x.dims());
  VLOG(3) << "dsl|A| dims: " << x_grad->dims();
}

}  // namespace phi
