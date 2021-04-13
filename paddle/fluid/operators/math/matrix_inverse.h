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

#pragma once

#include <string>
#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
void compute_inverse_eigen(const DeviceContext& context,
                           const framework::Tensor& a,
                           framework::Tensor* a_inv) {
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrixMap = Eigen::Map<Matrix>;
  using ConstEigenMatrixMap = Eigen::Map<const Matrix>;
  const auto& mat_dims = a.dims();
  const int rank = mat_dims.size();
  int n = mat_dims[rank - 1];
  int batch_size = rank > 2 ? a.numel() / (n * n) : 1;

  const T* a_ptr = a.data<T>();
  T* a_inv_ptr = a_inv->mutable_data<T>(context.GetPlace());

  for (int i = 0; i < batch_size; ++i) {
    ConstEigenMatrixMap mat(a_ptr + i * n * n, n, n);
    EigenMatrixMap mat_inv(a_inv_ptr + i * n * n, n, n);
    Eigen::PartialPivLU<Matrix> lu;
    lu.compute(mat);

    const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
    PADDLE_ENFORCE_GT(
        min_abs_pivot, static_cast<T>(0),
        platform::errors::InvalidArgument("Input is not invertible."));
    mat_inv.noalias() = lu.inverse();
  }
}

template <typename DeviceContext, typename T>
class MatrixInverseFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& a,
                  framework::Tensor* a_inv);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
