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

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
// #include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"

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
void MatrixRankKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const Scalar& atol_tensor,
                      bool hermitian,
                      bool use_default_tol,
                      float tol,
                      DenseTensor* out) {
  // const Tensor* x = context.Input<Tensor>("X");
  auto* x_data = x.data<T>();
  // auto* out = context.Output<Tensor>("Out");
  dev_ctx.template Alloc<int64_t>(out);
  // out->mutable_data<int64_t>(context.GetPlace());
  // bool hermitian = context.Attr<bool>("hermitian");

  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  int k = std::min(rows, cols);
  auto numel = x.numel();
  int batches = numel / (rows * cols);

  // bool use_default_tol = context.Attr<bool>("use_default_tol");
  // const DenseTensor* atol_tensor = nullptr;
  T rtol_T = 0;
  DenseTensor atol_dense_tensor;
  if (use_default_tol) {
    // paddle::framework::TensorFromVector<T>(
    //     std::vector<T>{0}, dev_ctx, &temp_tensor);
    atol_dense_tensor = Full<int>(dev_ctx, {0}, atol_tensor);
    rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
  } else {
    // paddle::framework::TensorFromVector<T>(std::vector<T>{tol},
    //                                dev_ctx,
    //                                &temp_tensor);
    // const Scalar temp_tensor(tol);
    // atol_tensor = temp_tensor;
    atol_dense_tensor = Full<float>(dev_ctx, {tol}, atol_tensor);
  }

  DenseTensor eigenvalue_tensor;
  // auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
  //     detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<T>(&eigenvalue_tensor);
  if (hermitian) {
    BatchEigenvalues<T>(x_data, eigenvalue_data, batches, rows, cols, k);
  } else {
    BatchSVD<T>(x_data, eigenvalue_data, batches, rows, cols, k);
  }

  auto dito_T = math::DeviceIndependenceTensorOperations<
      paddle::platform::CPUDeviceContext,
      T>(context);
  std::vector<int> max_eigenvalue_shape =
      phi::vectorize<int>(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  DenseTensor max_eigenvalue_tensor =
      dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);

  DenseTensor temp_rtol_tensor;
  paddle::framework::TensorFromVector<T>(std::vector<T>{rtol_T},
                                         &temp_rtol_tensor);
  DenseTensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);

  DenseTensor tol_tensor;
  tol_tensor.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<T>(&tol_tensor);
  // tol_tensor.mutable_data<T>(dim_out, context.GetPlace());

  phi::ElementwiseCompute<GreaterElementFunctor<T>, T, T>(
      dev_ctx,
      atol_dense_tensor,
      rtol_tensor,
      -1,
      GreaterElementFunctor<T>(),
      &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<T>(&compare_result);
  // compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
  //                                      context.GetPlace());

  int axis = -1;
  if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
    phi::ElementwiseCompute<paddle::operators::GreaterThanFunctor<T, int64_t>,
                            T,
                            int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        axis,
        paddle::operators::GreaterThanFunctor<T, int64_t>(),
        &compare_result);
  } else {
    phi::ElementwiseCompute<paddle::operators::LessThanFunctor<T, int64_t>,
                            T,
                            int>(
        dev_ctx,
        eigenvalue_tensor,
        tol_tensor,
        axis,
        paddle::operators::LessThanFunctor<T, int64_t>(),
        &compare_result);
    auto dito_int = math::DeviceIndependenceTensorOperations<
        paddle::platform::CPUDeviceContext,
        int64_t>(context);
    std::vector<int> result_shape = phi::vectorize<int>(dim_out);
    DenseTensor result = dito_int.ReduceSum(compare_result, result_shape);
    out->ShareDataWith(result);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    matrix_rank, CPU, ALL_LAYOUT, phi::MatrixRankKernel, float, double) {}
