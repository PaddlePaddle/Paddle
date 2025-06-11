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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/diagonal_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/triangular_solve_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& dev_ctx,
                        std::vector<int> shape,
                        float fill_value) {
  DenseTensor ret;
  ret.Resize(common::make_ddim(shape));
  dev_ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(dev_ctx, &ret, T(fill_value));
  return ret;
}

template <typename T, typename Context>
void QrGradKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& q,
                  const DenseTensor& r,
                  const DenseTensor& q_grad,
                  const DenseTensor& r_grad,
                  const std::string& mode,
                  DenseTensor* x_grad) {
  // Using alias names
  const DenseTensor& A = x;
  const DenseTensor& Q = q;
  const DenseTensor& R = r;
  const DenseTensor& dQ = q_grad;
  const DenseTensor& dR = r_grad;
  DenseTensor& dA = *x_grad;

  dev_ctx.template Alloc<T>(&dA);
  phi::funcs::SetConstant<Context, T>()(dev_ctx, &dA, T(0));

  bool compute_q, reduced;
  std::tie(compute_q, reduced) = phi::funcs::ParseQrMode(mode);
  if (!compute_q) {
    PADDLE_THROW(errors::InvalidArgument(
        "The derivative of qr is not implemented when mode='%s'.", mode));
  }

  auto a_dims = A.dims();
  int a_rank = a_dims.size();
  int m = a_dims[a_rank - 2];
  int n = a_dims[a_rank - 1];

  if ((m > n) && (!reduced)) {
    PADDLE_THROW(errors::InvalidArgument(
        "The derivative of qr is not implemented when mode='complete' and "
        "%d > %d.",
        m,
        n));
  }

  // m >= n case
  auto m_gt_n_case = [](const Context& dev_ctx,
                        const DenseTensor& dQ,
                        const DenseTensor& dR,
                        const DenseTensor& A UNUSED,
                        const DenseTensor& Q,
                        const DenseTensor& R) -> DenseTensor {
    // Roberts, D., & Roberts, L. (2020). QR and LQ Decomposition Matrix
    // Backpropagation Algorithms for Square, Wide, and Deep Matrices and Their
    // Software Implementation. https://arxiv.org/abs/2009.10071v4

    // dR^H
    DenseTensor R_term;
    if (dR.initialized()) {
      R_term = Matmul<T, Context>(dev_ctx,
                                  R,
                                  TransposeLast2Dim<T, Context>(
                                      dev_ctx, Conj<T, Context>(dev_ctx, dR)));
    } else {
      R_term = Fill<T, Context>(dev_ctx, common::vectorize<int>(R.dims()), 0);
    }

    // dQ^H * Q
    DenseTensor Q_term;
    if (dQ.initialized()) {
      Q_term = Matmul<T, Context>(
          dev_ctx,
          TransposeLast2Dim<T, Context>(dev_ctx, Conj<T, Context>(dev_ctx, dQ)),
          Q);
    } else {
      Q_term = Fill<T, Context>(dev_ctx, common::vectorize<int>(R.dims()), 0);
    }

    DenseTensor M_tmp1 = Subtract<T, Context>(dev_ctx, R_term, Q_term);
    DenseTensor M;
#ifdef PADDLE_WITH_HIP
    // Compute M = (tril(M) + tril(M).mH()) * 0.5 Identity
    DenseTensor M_tril_0 = TrilTriu<T, Context>(dev_ctx, M_tmp1, 0, true);
    DenseTensor M_tril_1 = TrilTriu<T, Context>(dev_ctx, M_tmp1, -1, true);
    M = Add<T, Context>(
        dev_ctx, M_tril_0, TransposeLast2Dim<T, Context>(dev_ctx, M_tril_1));
#else
    if (std::is_same<T, phi::dtype::complex<float>>::value ||
        std::is_same<T, phi::dtype::complex<double>>::value) {
      DenseTensor M_tril_tmp = TrilTriu<T, Context>(dev_ctx, M_tmp1, -1, true);
      DenseTensor M_tril =
          Add<T, Context>(dev_ctx,
                          M_tril_tmp,
                          TransposeLast2Dim<T, Context>(
                              dev_ctx, Conj<T, Context>(dev_ctx, M_tril_tmp)));

      size_t rank = M_tmp1.dims().size();
      DenseTensor M_diag_tmp =
          Diagonal<T, Context>(dev_ctx, M_tmp1, 0, rank - 2, rank - 1);
      DenseTensor M_diag_real = Real<T, Context>(dev_ctx, M_diag_tmp);
      DenseTensor M_diag_imag = Fill<phi::dtype::Real<T>, Context>(
          dev_ctx, common::vectorize<int>(M_diag_real.dims()), 0);

      DenseTensor M_diag;
      M_diag.Resize(M_diag_real.dims());
      dev_ctx.template Alloc<T>(&M_diag);
      phi::ComplexKernel<phi::dtype::Real<T>>(
          dev_ctx, M_diag_real, M_diag_imag, &M_diag);

      M = FillDiagonalTensor<T, Context>(
          dev_ctx, M_tril, M_diag, 0, rank - 2, rank - 1);
    } else {
      // Compute M = (tril(M) + tril(M).mH()) * 0.5 Identity
      DenseTensor M_tril_0 = TrilTriu<T, Context>(dev_ctx, M_tmp1, 0, true);
      DenseTensor M_tril_1 = TrilTriu<T, Context>(dev_ctx, M_tmp1, -1, true);
      M = Add<T, Context>(
          dev_ctx, M_tril_0, TransposeLast2Dim<T, Context>(dev_ctx, M_tril_1));
    }
#endif

    DenseTensor rhs_term;
    if (dQ.initialized()) {
      rhs_term =
          Add<T, Context>(dev_ctx, dQ, Matmul<T, Context>(dev_ctx, Q, M));
    } else {
      rhs_term = Matmul<T, Context>(dev_ctx, Q, M);
    }

    // dA * R^H = rhs_term
    auto dA = TriangularSolve<T, Context>(
        dev_ctx,
        TransposeLast2Dim<T, Context>(
            dev_ctx,
            Conj<T, Context>(dev_ctx,
                             TransposeLast2Dim<T, Context>(dev_ctx, R))),
        TransposeLast2Dim<T, Context>(dev_ctx, rhs_term),
        /*upper=*/true,
        /*transpose=*/false,
        /*unitriangular=*/false);

    return TransposeLast2Dim<T, Context>(dev_ctx, dA);
  };

  if (m >= n) {
    auto dA_tmp = m_gt_n_case(dev_ctx, dQ, dR, A, Q, R);
    phi::Copy(dev_ctx, dA_tmp, dA.place(), false, &dA);
  } else {
    // If m < n for input matrices A, we partition A = [X|Y] and R = [U|V]
    // Calculate dX and dY individually and concatenate them to get dA
    dev_ctx.template Alloc<phi::dtype::Real<T>>(&dA);

    auto Y = Slice<T, Context>(dev_ctx, A, {A.dims().size() - 1}, {m}, {n});
    auto U = Slice<T, Context>(dev_ctx, R, {R.dims().size() - 1}, {0}, {m});
    DenseTensor dY, dX, dV, dU, dQ_prime;

    if (dR.initialized()) {
      dV = Slice<T, Context>(dev_ctx, dR, {dR.dims().size() - 1}, {m}, {n});
      dU = Slice<T, Context>(dev_ctx, dR, {dR.dims().size() - 1}, {0}, {m});
      // Y * dV^H
      dQ_prime =
          Matmul<T, Context>(dev_ctx,
                             Y,
                             TransposeLast2Dim<T, Context>(
                                 dev_ctx, Conj<T, Context>(dev_ctx, dV)));
    } else {
      dV = Fill<T, Context>(dev_ctx, common::vectorize<int>(Y.dims()), 0);
      dQ_prime = Fill<T, Context>(dev_ctx, common::vectorize<int>(Q.dims()), 0);
    }

    if (dQ.initialized()) {
      dQ_prime = Add<T, Context>(dev_ctx, dQ, dQ_prime);
    }
    dX = m_gt_n_case(dev_ctx, dQ_prime, dU, A, Q, U);
    dY = Matmul<T, Context>(dev_ctx, Q, dV);
    // Concatenate dX and dY to get dA.
    auto dA_tmp = Concat<T, Context>(dev_ctx, {&dX, &dY}, -1);
    phi::Copy(dev_ctx, dA_tmp, dA.place(), false, &dA);
  }
}

}  // namespace phi
