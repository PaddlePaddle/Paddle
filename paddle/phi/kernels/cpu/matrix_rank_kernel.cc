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

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
void BatchEigenvalues(const T* x_data,
                      T* eigenvalues_data,
                      int batches,
                      int rows,
                      int cols,
                      int k) {
  // Eigen::Matrix API need non-const pointer.
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, rows);
    Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eigen_solver(m);
    auto eigenvalues = eigen_solver.eigenvalues().cwiseAbs();
    for (int j = 0; j < k; j++) {
      *(eigenvalues_data + i * k + j) = eigenvalues[j];
    }
  }
}

template <typename T>
void BatchSVD(const T* x_data,
              T* eigenvalues_data,
              int batches,
              int rows,
              int cols,
              int k) {
  // Eigen::Matrix API need non-const pointer.
  T* input = const_cast<T*>(x_data);
  int stride = rows * cols;
  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd;
  for (int i = 0; i < batches; i++) {
    auto m = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        input + i * stride, rows, cols);
    svd.compute(m);
    auto res_s = svd.singularValues();
    for (int j = 0; j < k; j++) {
      eigenvalues_data[i * k + j] = res_s[j];
    }
  }
}

template <typename T, typename Context>
void MatrixRankKernel(const Context& ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  const Tensor* x = context.Input<Tensor>("X");
  auto* x_data = x->data<T>();
  auto* out = context.Output<Tensor>("Out");
  out->mutable_data<int64_t>(context.GetPlace());
  bool hermitian = context.Attr<bool>("hermitian");

  auto dim_x = x->dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  int k = std::min(rows, cols);
  auto numel = x->numel();
  int batches = numel / (rows * cols);

  bool use_default_tol = context.Attr<bool>("use_default_tol");
  const Tensor* atol_tensor = nullptr;
  Tensor temp_tensor;
  T rtol_T = 0;
  if (use_default_tol) {
    framework::TensorFromVector<T>(
        std::vector<T>{0}, context.device_context(), &temp_tensor);
    atol_tensor = &temp_tensor;
    rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
  } else if (context.HasInput("TolTensor")) {
    atol_tensor = context.Input<Tensor>("TolTensor");
  } else {
    framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                   context.device_context(),
                                   &temp_tensor);
    atol_tensor = &temp_tensor;
  }

  Tensor eigenvalue_tensor;
  auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
      detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
  if (hermitian) {
    BatchEigenvalues<T>(x_data, eigenvalue_data, batches, rows, cols, k);
  } else {
    BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols, k);
  }

  auto dito_T =
      math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext, T>(
          context);
  std::vector<int> max_eigenvalue_shape =
      phi::vectorize<int>(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  Tensor max_eigenvalue_tensor =
      dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);

  Tensor temp_rtol_tensor;
  framework::TensorFromVector<T>(std::vector<T>{rtol_T}, &temp_rtol_tensor);
  Tensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);
  Tensor tol_tensor;
  tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
  ElementwiseComputeEx<GreaterElementFunctor<T>,
                       platform::CPUDeviceContext,
                       T,
                       T>(context,
                          atol_tensor,
                          &rtol_tensor,
                          -1,
                          GreaterElementFunctor<T>(),
                          &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  Tensor compare_result;
  compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
                                       context.GetPlace());

  int axis = -1;
  if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
    ElementwiseComputeEx<GreaterThanFunctor<T, int64_t>,
                         platform::CPUDeviceContext,
                         T,
                         int>(context,
                              &eigenvalue_tensor,
                              &tol_tensor,
                              axis,
                              GreaterThanFunctor<T, int64_t>(),
                              &compare_result);
  } else {
    ElementwiseComputeEx<LessThanFunctor<T, int64_t>,
                         platform::CPUDeviceContext,
                         T,
                         int>(context,
                              &eigenvalue_tensor,
                              &tol_tensor,
                              axis,
                              LessThanFunctor<T, int64_t>(),
                              &compare_result);
  }
  auto dito_int =
      math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
                                               int64_t>(context);
  std::vector<int> result_shape = phi::vectorize<int>(dim_out);
  Tensor result = dito_int.ReduceSum(compare_result, result_shape);
  out->ShareDataWith(result);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    matrix_rank, CPU, ALL_LAYOUT, phi::MatrixRankKernel, float, double) {}
