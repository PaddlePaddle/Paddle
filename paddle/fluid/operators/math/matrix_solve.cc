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
#include "paddle/phi/kernels/funcs/blas/blas.h"

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

}  // namespace math
}  // namespace operators
}  // namespace paddle
