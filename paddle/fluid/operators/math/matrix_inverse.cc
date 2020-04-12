/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/matrix_inverse.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class MatrixInverseFunctor<platform::CPUDeviceContext, T> {
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrixMap = Eigen::Map<Matrix>;
  using ConstEigenMatrixMap = Eigen::Map<const Matrix>;

 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& A, framework::Tensor* A_inv) {
    const auto& mat_dims = A.dims();
    const int rank = mat_dims.size();
    int N = mat_dims[rank - 1];
    int batch_size = rank > 2 ? A.numel() / (N * N) : 1;

    const T* A_ptr = A.data<T>();
    T* A_inv_ptr = A_inv->mutable_data<T>(context.GetPlace());

#ifdef PADDLE_WITH_MKLML
    framework::Tensor ipiv;
    int* ipiv_ptr = ipiv.mutable_data<int>({N}, context.GetPlace());

    if (A_ptr != A_inv_ptr) {
      framework::TensorCopy(A, context.GetPlace(), A_inv);
    }

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int i = 0; i < batch_size; ++i) {
      T* mat_ptr = A_inv_ptr + i * N * N;

      // Compute the LU Factorization of a general m-by-n matrix: A = P*L*U
      blas.GETRF(N, N, mat_ptr, ipiv_ptr);

      // Computes the inverse of an LU-factored general matrix.
      blas.GETRI(N, mat_ptr, ipiv_ptr);
    }
#else
    for (int i = 0; i < batch_size; ++i) {
      ConstEigenMatrixMap mat(A_ptr + i * N * N, N, N);
      EigenMatrixMap mat_inv(A_inv_ptr + i * N * N, N, N);
      Eigen::PartialPivLU<Matrix> lu;
      lu.compute(mat);

      const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
      PADDLE_ENFORCE_GT(
          min_abs_pivot, static_cast<T>(0),
          platform::errors::InvalidArgument("Input is not invertible."));
      mat_inv.noalias() = lu.inverse();
    }
#endif
  }
};

template class MatrixInverseFunctor<platform::CPUDeviceContext, float>;
template class MatrixInverseFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
