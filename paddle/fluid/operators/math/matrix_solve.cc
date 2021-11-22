/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/matrix_solve.h"
#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class MatrixSolveFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& dev_ctx,
                  const framework::Tensor& a, const framework::Tensor& b,
                  framework::Tensor* out) {
    compute_solve_eigen<platform::CPUDeviceContext, T>(dev_ctx, a, b, out);
  }
};

template class MatrixSolveFunctor<platform::CPUDeviceContext, float>;
template class MatrixSolveFunctor<platform::CPUDeviceContext, double>;

template <typename T>
class TriangularSolveFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor* a, framework::Tensor* b, bool left,
                  bool upper, bool transpose, bool unitriangular) {
    CBLAS_SIDE side = left ? CblasLeft : CblasRight;
    CBLAS_UPLO uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE transA = transpose ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG diag = unitriangular ? CblasUnit : CblasNonUnit;

    const T* a_data = a->data<T>();
    T* b_data = b->mutable_data<T>(context.GetPlace());

    int a_dim_size = a->dims().size();
    int b_dim_size = b->dims().size();

    int M = static_cast<int>(b->dims()[b_dim_size - 2]);
    int N = static_cast<int>(b->dims()[b_dim_size - 1]);
    auto lda = left ? std::max(1, M) : std::max(1, N);
    auto ldb = std::max(1, N);

    int batch_size = 1;
    auto& a_dim = a->dims();
    for (int i = 0; i < a_dim_size - 2; i++) {
      batch_size *= a_dim[i];
    }

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int i = 0; i < batch_size; i++) {
      blas.TRSM(side, uplo, transA, diag, M, N, T(1), a_data + i * M * M, lda,
                b_data + i * N * M, ldb);
    }
  }
};

template class TriangularSolveFunctor<platform::CPUDeviceContext, float>;
template class TriangularSolveFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
