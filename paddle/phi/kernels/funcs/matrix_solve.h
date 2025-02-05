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

#pragma once

#include <string>

#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

// for TransposeNormal
static std::vector<int> getNewAxis(const int b_rank) {
  std::vector<int> axis_1 = {0};
  std::vector<int> axis_2 = {1, 0};
  std::vector<int> axis_3 = {0, 2, 1};
  std::vector<int> axis_4 = {0, 1, 3, 2};
  std::vector<int> axis_5 = {0, 1, 2, 4, 3};
  std::vector<int> axis_6 = {0, 1, 2, 3, 5, 4};
  std::vector<int> axis_7 = {0, 1, 2, 3, 4, 6, 5};
  std::vector<int> axis_8 = {0, 1, 2, 3, 4, 5, 7, 6};
  std::vector<int> axis_9 = {0, 1, 2, 3, 4, 5, 6, 8, 7};
  switch (b_rank) {
    case 1:
      return axis_1;
      break;
    case 2:
      return axis_2;
      break;
    case 3:
      return axis_3;
      break;
    case 4:
      return axis_4;
      break;
    case 5:
      return axis_5;
      break;
    case 6:
      return axis_6;
      break;
    case 7:
      return axis_7;
      break;
    case 8:
      return axis_8;
      break;
    default:
      return axis_9;
  }
}

// for Resize
static std::vector<int64_t> getNewDimsVec(const DDim& b_dims) {
  std::vector<int64_t> b_dims_vec = common::vectorize(b_dims);
  int size = b_dims_vec.size();
  if (size >= 2) {
    // swap the last 2 elements in b_dims_vec
    int64_t temp = b_dims_vec[size - 1];
    b_dims_vec[size - 1] = b_dims_vec[size - 2];
    b_dims_vec[size - 2] = temp;
    return b_dims_vec;
  }
  PADDLE_ENFORCE_NE(
      b_dims_vec.empty(),
      true,
      common::errors::PreconditionNotMet(
          "The size of tensor b must not be %d after getting new dims", 0));
  // if b_dims_vec.size() == 1, just return original vec
  return b_dims_vec;
}

template <typename Context, typename T>
void compute_solve_eigen(const Context& context,
                         const DenseTensor& a,
                         const DenseTensor& b,
                         DenseTensor* out) {
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrixMap = Eigen::Map<Matrix>;
  using ConstEigenMatrixMap = Eigen::Map<const Matrix>;
  // prepare for a
  const auto& a_mat_dims = a.dims();
  const int a_rank = a_mat_dims.size();
  int n = a_mat_dims[a_rank - 1];
  int a_batch_size = a_rank > 2 ? a.numel() / (n * n) : 1;

  // prepare for b
  const auto& b_mat_dims = b.dims();
  const int b_rank = b_mat_dims.size();
  int b_h = n;
  int b_w = b_mat_dims[b_rank - 1];
  int b_batch_size = b_rank > 2 ? b.numel() / (b_h * b_w) : 1;

  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  out->Resize(b_mat_dims);  // make sure the out dims is right

  T* out_ptr = context.template Alloc<T>(out);
  if (a_batch_size == b_batch_size) {
    for (int i = 0; i < a_batch_size; ++i) {
      ConstEigenMatrixMap a_mat(a_ptr + i * n * n, n, n);
      ConstEigenMatrixMap b_mat(b_ptr + i * b_h * b_w, b_h, b_w);
      EigenMatrixMap out_mat(out_ptr + i * b_h * b_w, b_h, b_w);
      Eigen::PartialPivLU<Matrix> lu;
      lu.compute(a_mat);
      const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
      PADDLE_ENFORCE_GT(
          min_abs_pivot,
          static_cast<T>(0),
          common::errors::InvalidArgument("Input is not invertible."));
      out_mat.noalias() = lu.solve(b_mat);
    }
  } else {
    PADDLE_ENFORCE_EQ(a_batch_size,
                      b_batch_size,
                      common::errors::InvalidArgument(
                          "All input tensors must have the same rank."));
  }
}

// only used for complex input
template <typename T>
void SolveLinearSystem(T* matrix_data,
                       T* rhs_data,
                       T* out_data,
                       int order,
                       int rhs_cols,
                       int batch) {
  using Treal = typename Eigen::NumTraits<T>::Real;

  // cast paddle::complex into std::complex
  std::complex<Treal>* matrix_data_ =
      reinterpret_cast<std::complex<Treal>*>(matrix_data);
  std::complex<Treal>* rhs_data_ =
      reinterpret_cast<std::complex<Treal>*>(rhs_data);
  std::complex<Treal>* out_data_ =
      reinterpret_cast<std::complex<Treal>*>(out_data);

  using Matrix = Eigen::Matrix<std::complex<Treal>,
                               Eigen::Dynamic,
                               Eigen::Dynamic,
                               Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<Matrix>;
  using OutputMatrixMap = Eigen::Map<Matrix>;

  for (int i = 0; i < batch; ++i) {
    auto input_matrix =
        InputMatrixMap(matrix_data_ + i * order * order, order, order);
    auto input_rhs =
        InputMatrixMap(rhs_data_ + i * order * rhs_cols, order, rhs_cols);
    auto output =
        OutputMatrixMap(out_data_ + i * order * rhs_cols, order, rhs_cols);

    Eigen::PartialPivLU<Matrix> lu_decomposition(order);
    lu_decomposition.compute(input_matrix);

    const Treal min_abs_piv =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    PADDLE_ENFORCE_GT(min_abs_piv,
                      Treal(0),
                      common::errors::InvalidArgument(
                          "Something's wrong with SolveLinearSystem. "));

    output = lu_decomposition.solve(input_rhs);
  }
}

template <typename Context, typename T>
class MatrixSolveFunctor {
 public:
  void operator()(const Context& context,
                  const DenseTensor& a,
                  const DenseTensor& b,
                  DenseTensor* out);
};

}  // namespace funcs
}  // namespace phi
