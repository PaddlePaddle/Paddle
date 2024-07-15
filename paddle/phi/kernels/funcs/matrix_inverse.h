/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
struct MapMatrixInverseFunctor {
  void operator()(
      const Context& dev_ctx, const T* a_ptr, T* a_inv_ptr, int offset, int n) {
    using Matrix =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using EigenMatrixMap = Eigen::Map<Matrix>;
    using ConstEigenMatrixMap = Eigen::Map<const Matrix>;

    ConstEigenMatrixMap mat(a_ptr + offset, n, n);
    EigenMatrixMap mat_inv(a_inv_ptr + offset, n, n);
    Eigen::PartialPivLU<Matrix> lu;
    lu.compute(mat);

    const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
    PADDLE_ENFORCE_GT(min_abs_pivot,
                      static_cast<T>(0),
                      errors::InvalidArgument("Input is not invertible."));
    mat_inv.noalias() = lu.inverse();
  }
};

template <typename Context, typename T>
struct MapMatrixInverseFunctor<Context, phi::dtype::complex<T>> {
  void operator()(const Context& dev_ctx,
                  const phi::dtype::complex<T>* a_ptr,
                  phi::dtype::complex<T>* a_inv_ptr,
                  int offset,
                  int n) {
    using Matrix = Eigen::Matrix<std::complex<T>,
                                 Eigen::Dynamic,
                                 Eigen::Dynamic,
                                 Eigen::RowMajor>;
    using EigenMatrixMap = Eigen::Map<Matrix>;
    using ConstEigenMatrixMap = Eigen::Map<const Matrix>;
    std::complex<T>* std_ptr = new std::complex<T>[n * n];
    std::complex<T>* std_inv_ptr = new std::complex<T>[n * n];
    for (int i = 0; i < n * n; i++) {
      *(std_ptr + i) = static_cast<std::complex<T>>(*(a_ptr + offset + i));
    }
    ConstEigenMatrixMap mat(std_ptr, n, n);
    EigenMatrixMap mat_inv(std_inv_ptr, n, n);
    Eigen::PartialPivLU<Matrix> lu;
    lu.compute(mat);

    const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
    PADDLE_ENFORCE_NE(min_abs_pivot,
                      static_cast<std::complex<T>>(0),
                      errors::InvalidArgument("Input is not invertible."));
    mat_inv.noalias() = lu.inverse();
    for (int i = 0; i < n * n; i++) {
      *(a_inv_ptr + offset + i) =
          static_cast<phi::dtype::complex<T>>(*(std_inv_ptr + i));
    }
    delete[] std_ptr;
    delete[] std_inv_ptr;
  }
};

template <typename Context, typename T>
void ComputeInverseEigen(const Context& dev_ctx,
                         const DenseTensor& a,
                         DenseTensor* a_inv) {
  const auto& mat_dims = a.dims();
  const int rank = mat_dims.size();
  int n = mat_dims[rank - 1];
  int batch_size = rank > 2 ? a.numel() / (n * n) : 1;

  const T* a_ptr = a.data<T>();
  T* a_inv_ptr = dev_ctx.template Alloc<T>(a_inv);

  // Putting phi::dtype::complex into eigen::matrix has a problem,
  // it's not going to get the right result,
  // so we're going to convert it to std::complex and
  // then we're going to put it into eigen::matrix.
  for (int i = 0; i < batch_size; ++i) {
    MapMatrixInverseFunctor<Context, T> functor;
    functor(dev_ctx, a_ptr, a_inv_ptr, i * n * n, n);
  }
}

template <typename Context, typename T>
class MatrixInverseFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor& a,
                  DenseTensor* a_inv);
};

}  // namespace funcs
}  // namespace phi
