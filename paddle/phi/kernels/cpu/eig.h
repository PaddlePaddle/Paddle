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

#include <math.h>

#include <algorithm>
#include <complex>

#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#define EPSILON 1e-6

namespace phi {

inline int BatchCount(const DenseTensor& matrix) {
  int count = 1;
  int num_dims = matrix.dims().size();
  for (int i = 0; i < num_dims - 2; ++i) {
    count *= matrix.dims()[i];
  }
  return count;
}

inline int MatrixStride(const DenseTensor& matrix) {
  phi::DDim dims_list = matrix.dims();
  int num_dims = dims_list.size();
  return dims_list[num_dims - 1] * dims_list[num_dims - 2];
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
    PADDLE_ENFORCE_GT(
        min_abs_piv,
        Treal(0),
        errors::InvalidArgument("Something's wrong with SolveLinearSystem. "));

    output = lu_decomposition.solve(input_rhs);
  }
}

template <typename T, typename Context>
void TransposeTwoAxis(const DenseTensor& input,
                      DenseTensor* transposed_input,
                      const int axis1,
                      const int axis2,
                      const Context& dev_ctx) {
  std::vector<int> permute(input.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis1] = axis2;
  permute[axis2] = axis1;

  transposed_input->Resize(input.dims());
  dev_ctx.template Alloc<T>(transposed_input);

  funcs::TransCompute<Context, T>(
      input.dims().size(), dev_ctx, input, transposed_input, permute);
}

// Apply eig to a batch of matrices, values, vectors and (intermidiate
// DenseTensor) info are overwritten
template <typename T, typename Context>
void LapackEig(DenseTensor* input,
               DenseTensor* values,
               DenseTensor* vectors,
               int info,
               const Context& dev_ctx) {
  char jobvl = 'N';
  char jobvr = 'V';  // only right eigenvectors are computed
  int num_dims = input->dims().size();
  int order = input->dims()[num_dims - 1];

  T* input_data = input->data<T>();
  int lda = std::max<int>(1, order);

  T* values_data = dev_ctx.template Alloc<T>(values);
  T* lvector_data = nullptr;
  int ldvl = 1;
  T* rvector_data = dev_ctx.template Alloc<T>(vectors);
  int ldvr = lda;
  int lwork = -1;

  int batch_count = BatchCount(*input);
  int matrix_stride = MatrixStride(*input);
  int values_stride = values->dims()[values->dims().size() - 1];

  DenseTensor rwork;
  phi::dtype::Real<T>* rwork_data = nullptr;

  rwork.Resize(common::make_ddim({lda * 2}));
  rwork_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(&rwork);

  // call lapackEig once to compute the size of work;
  T computed_work_size;
  phi::funcs::lapackEig<T, phi::dtype::Real<T>>(jobvl,
                                                jobvr,
                                                order,
                                                input_data,
                                                lda,
                                                values_data,
                                                lvector_data,
                                                ldvl,
                                                rvector_data,
                                                ldvr,
                                                &computed_work_size,
                                                lwork,
                                                rwork_data,
                                                &info);

  lwork = std::max<int>(
      1, static_cast<int>(phi::dtype::Real<T>(computed_work_size)));
  DenseTensor work;
  work.Resize(common::make_ddim({lwork}));
  T* work_data = dev_ctx.template Alloc<T>(&work);

  for (auto i = 0; i < batch_count; ++i) {
    T* current_matrix = &input_data[i * matrix_stride];
    T* current_values = &values_data[i * values_stride];
    T* current_rvectors = &rvector_data[i * matrix_stride];

    phi::funcs::lapackEig<T, phi::dtype::Real<T>>(jobvl,
                                                  jobvr,
                                                  order,
                                                  current_matrix,
                                                  lda,
                                                  current_values,
                                                  lvector_data,
                                                  ldvl,
                                                  current_rvectors,
                                                  ldvr,
                                                  work_data,
                                                  lwork,
                                                  rwork_data,
                                                  &info);
    PADDLE_ENFORCE_EQ(
        info,
        0,
        errors::PreconditionNotMet(
            "current info is not 0, computation failed. "
            "= 0:  successful exit."
            "< 0:  if INFO = -i, the i-th argument had an illegal value."
            "> 0:  if INFO = i, the QR algorithm failed to compute all the "
            "eigenvalues, and no eigenvectors have been computed; "
            "elements i+1:N of WR and WI contain eigenvalues which "
            "have converged."));
  }
}

template <typename T, typename Context>
void ApplyEigKernel(const DenseTensor& input,
                    DenseTensor* values,
                    DenseTensor* vectors,
                    const Context& dev_ctx) {
  DenseTensor input_column_major;
  DenseTensor vectors_row_major;
  int num_dims = input.dims().size();

  // transfer to column-major memory layout i.e. common::make_ddim from
  // transposed_input: [batch,row,col]->[batch,col,row]
  TransposeTwoAxis<T, Context>(
      input, &input_column_major, num_dims - 1, num_dims - 2, dev_ctx);
  // make sure 'vectors_row_major' holds memory before passed to LapackEig()
  vectors_row_major.Resize(input.dims());
  int info = 0;
  LapackEig<T, Context>(
      &input_column_major, values, &vectors_row_major, info, dev_ctx);

  // transfer column-major layout back
  // vectors_row_major: column-major layout
  // vector: original layout
  TransposeTwoAxis<T, Context>(
      vectors_row_major, vectors, num_dims - 1, num_dims - 2, dev_ctx);
}

// template <typename T, typename Tout>
template <typename T, typename Tout, typename Context>
void ConstructComplexVectors(DenseTensor* c_vectors,
                             const DenseTensor& c_values,
                             const DenseTensor& r_vectors,
                             const Context& dev_ctx,
                             int batch_count,
                             int order) {
  int matrix_stride = MatrixStride(r_vectors);

  auto* c_vectors_data = dev_ctx.template Alloc<Tout>(c_vectors);
  auto* c_values_data = c_values.data<Tout>();
  auto* r_v_data = r_vectors.data<T>();

  for (int b = 0; b < batch_count; b++) {
    auto* vecs = &r_v_data[b * matrix_stride];
    auto* res = &c_vectors_data[b * matrix_stride];
    auto* vals = &c_values_data[b * order];

    for (int j = 0; j < order; j++) {
      if (vals[j].imag < EPSILON) {
        for (int i = 0; i < order; i++) {
          res[j * order + i] = dtype::complex<T>(vecs[j * order + i], 0);
        }
      } else {
        for (int i = 0; i < order; i++) {
          res[j * order + i] =
              dtype::complex<T>(vecs[j * order + i], vecs[(j + 1) * order + i]);
          res[(j + 1) * order + i] = dtype::complex<T>(
              vecs[j * order + i], -vecs[(j + 1) * order + i]);
        }
        j++;
      }
    }
  }
}

template <typename T, typename Context>
void ComputeBackwardForComplexInput(const DenseTensor& L,
                                    const DenseTensor& V,
                                    const DenseTensor& gL,
                                    const DenseTensor& gV,
                                    T* x_grad_data,
                                    int batch_count,
                                    int order,
                                    const Context& dev_ctx) {
  DenseTensor trans_v = phi::TransposeLast2Dim<T>(dev_ctx, V);
  DenseTensor Vh = phi::Conj<T>(dev_ctx, trans_v);
  DenseTensor Lconj = phi::Conj<T>(dev_ctx, L);
  DenseTensor Econj = phi::Subtract<T>(dev_ctx,
                                       phi::funcs::Unsqueeze(Lconj, -2),
                                       phi::funcs::Unsqueeze(Lconj, -1));
  DenseTensor VhgV = phi::Matmul<T>(dev_ctx, Vh, gV);
  DenseTensor diag_real = phi::Real<T>(dev_ctx, VhgV);
  DenseTensor diag_res =
      phi::funcs::BatchDiag<T>(dev_ctx, diag_real, batch_count);
  DenseTensor diag_unsqueezed = phi::funcs::Unsqueeze(diag_res, -2);

  // turn diag_unsqueezed into complex
  auto numel = diag_unsqueezed.numel();
  DenseTensor diag_unsqueezed_complex;
  auto* data_diag_un = diag_unsqueezed.data<phi::dtype::Real<T>>();
  diag_unsqueezed_complex.Resize(diag_unsqueezed.dims());
  auto* data_diag_un_com = dev_ctx.template Alloc<T>(
      &diag_unsqueezed_complex, static_cast<size_t>(numel * sizeof(T)));

  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  phi::funcs::RealToComplexFunctor<T> functor(
      data_diag_un, data_diag_un_com, numel);
  for_range(functor);
  // real tensor multiply complex tensor in broadcast manner
  DenseTensor res1 = phi::Multiply<T>(dev_ctx, V, diag_unsqueezed_complex);
  DenseTensor res2 = phi::Matmul<T>(dev_ctx, Vh, res1);
  DenseTensor result = phi::Subtract<T>(dev_ctx, VhgV, res2);

  result.Resize(V.dims());
  dev_ctx.template Alloc<T>(&result);
  result = phi::Divide<T>(dev_ctx, result, Econj);
  result =
      phi::funcs::DiagFill<T, T>(dev_ctx, order, order, order, 0, gL, result);
  DenseTensor rhs = phi::Matmul<T>(dev_ctx, result, Vh);

  // solve linear system
  // solve(Vh, rhs, out, m, k)
  // Vh: matrix with shape [m,m]
  // rhs: rhs with shape [m,k]
  // x_grad: out
  int m = Vh.dims()[Vh.dims().size() - 1];
  int k = rhs.dims()[rhs.dims().size() - 1];
  auto* matrix_data = Vh.data<T>();
  auto* rhs_data = rhs.data<T>();

  SolveLinearSystem<T>(matrix_data, rhs_data, x_grad_data, m, k, batch_count);
}

}  // namespace phi
