// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/operators/math/matrix_solve.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/pten/kernels/funcs/math_function.h"
#define EPSILON 1e-6

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

inline int BatchCount(const Tensor& matrix) {
  int count = 1;
  int num_dims = matrix.dims().size();
  for (int i = 0; i < num_dims - 2; ++i) {
    count *= matrix.dims()[i];
  }
  return count;
}

inline int MatrixStride(const Tensor& matrix) {
  framework::DDim dims_list = matrix.dims();
  int num_dims = dims_list.size();
  return dims_list[num_dims - 1] * dims_list[num_dims - 2];
}

// Transpose two axis of a Tensor
template <typename DeviceContext, typename T>
void TransposeTwoAxis(const Tensor& input, Tensor* transposed_input,
                      const int axis1, const int axis2,
                      const framework::ExecutionContext& context) {
  std::vector<int> permute(input.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis1] = axis2;
  permute[axis2] = axis1;

  transposed_input->mutable_data<T>(input.dims(), context.GetPlace());
  auto& dev_ctx = context.template device_context<platform::CPUDeviceContext>();

  TransCompute<DeviceContext, T>(input.dims().size(), dev_ctx, input,
                                 transposed_input, permute);
}

// Apply eig to a batch of matrices, values, vectors and (intermidiate
// tensor) info are overritten
template <typename T>
void LapackEig(Tensor* input, Tensor* values, Tensor* vectors, int info,
               const framework::ExecutionContext& context) {
  char jobvl = 'N';
  char jobvr = 'V';  // only right eigenvectors are computed
  int num_dims = input->dims().size();
  int order = input->dims()[num_dims - 1];

  T* input_data = input->data<T>();
  int lda = std::max<int>(1, order);
  T* values_data = values->mutable_data<T>(context.GetPlace());
  T* lvector_data = nullptr;
  int ldvl = 1;
  T* rvector_data = vectors->mutable_data<T>(context.GetPlace());
  int ldvr = lda;
  int lwork = -1;

  int batch_count = BatchCount(*input);
  int matrix_stride = MatrixStride(*input);
  int values_stride = values->dims()[values->dims().size() - 1];

  Tensor rwork;
  math::Real<T>* rwork_data = nullptr;

  rwork.Resize(framework::make_ddim({lda * 2}));
  rwork_data = rwork.mutable_data<math::Real<T>>(context.GetPlace());

  // call lapackEig once to compute the size of work;
  T computed_work_size;
  math::lapackEig<T, math::Real<T>>(
      jobvl, jobvr, order, input_data, lda, values_data, lvector_data, ldvl,
      rvector_data, ldvr, &computed_work_size, lwork, rwork_data, &info);

  lwork = std::max<int>(1, static_cast<int>(math::Real<T>(computed_work_size)));
  Tensor work;
  work.Resize(framework::make_ddim({lwork}));
  T* work_data = work.mutable_data<T>(context.GetPlace());

  for (auto i = 0; i < batch_count; ++i) {
    T* current_matrix = &input_data[i * matrix_stride];
    T* current_values = &values_data[i * values_stride];
    T* current_rvectors = &rvector_data[i * matrix_stride];

    math::lapackEig<T, math::Real<T>>(
        jobvl, jobvr, order, current_matrix, lda, current_values, lvector_data,
        ldvl, current_rvectors, ldvr, work_data, lwork, rwork_data, &info);
    PADDLE_ENFORCE_EQ(
        info, 0,
        platform::errors::PreconditionNotMet(
            "current info is not 0, computation failed. "
            "= 0:  successful exit."
            "< 0:  if INFO = -i, the i-th argument had an illegal value."
            "> 0:  if INFO = i, the QR algorithm failed to compute all the "
            "eigenvalues, and no eigenvectors have been computed; "
            "elements i+1:N of WR and WI contain eigenvalues which "
            "have converged."));
  }
}

template <typename DeviceContext, typename T>
void ApplyEigKernel(const Tensor& input, Tensor* values, Tensor* vectors,
                    const framework::ExecutionContext& context) {
  Tensor input_column_major;
  Tensor vectors_row_major;
  int num_dims = input.dims().size();

  // transfer to column-major memory layout i.e. make_ddim from tranposed_input:
  // [batch,row,col]->[batch,col,row]
  TransposeTwoAxis<DeviceContext, T>(input, &input_column_major, num_dims - 1,
                                     num_dims - 2, context);
  // make sure 'vectors_row_major' holds memory before passed to LapackEig()
  vectors_row_major.Resize(input.dims());
  int info = 0;
  LapackEig<T>(&input_column_major, values, &vectors_row_major, info, context);

  // transfer column-major layout back
  // vectors_row_major: column-major layout
  // vector: original layout
  TransposeTwoAxis<DeviceContext, T>(vectors_row_major, vectors, num_dims - 1,
                                     num_dims - 2, context);
}

template <typename T, typename Tout>
void ConstructComplexVectors(Tensor* c_vectors, const Tensor& c_values,
                             const Tensor& r_vectors,
                             const framework::ExecutionContext& ctx,
                             int batch_count, int order) {
  int matrix_stride = MatrixStride(r_vectors);

  auto* c_vectors_data = c_vectors->mutable_data<Tout>(ctx.GetPlace());
  auto* c_values_data = c_values.data<Tout>();
  auto* r_v_data = r_vectors.data<T>();

  for (int b = 0; b < batch_count; b++) {
    auto* vecs = &r_v_data[b * matrix_stride];
    auto* res = &c_vectors_data[b * matrix_stride];
    auto* vals = &c_values_data[b * order];

    for (int j = 0; j < order; j++) {
      if (vals[j].imag < EPSILON) {
        for (int i = 0; i < order; i++) {
          res[j * order + i] = platform::complex<T>(vecs[j * order + i], 0);
        }
      } else {
        for (int i = 0; i < order; i++) {
          res[j * order + i] = platform::complex<T>(vecs[j * order + i],
                                                    vecs[(j + 1) * order + i]);
          res[(j + 1) * order + i] = platform::complex<T>(
              vecs[j * order + i], -vecs[(j + 1) * order + i]);
        }
        j++;
      }
    }
  }
}

template <typename DeviceContext, typename T, typename Tout>
class EigKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out_values = context.Output<Tensor>("Eigenvalues");
    auto* out_vectors = context.Output<Tensor>("Eigenvectors");

    if (!framework::IsComplexType(framework::TransToProtoVarType(x->dtype()))) {
      out_values->mutable_data<Tout>(context.GetPlace());
      out_vectors->mutable_data<Tout>(context.GetPlace());

      int batch_count = BatchCount(*x);
      int order = x->dims()[x->dims().size() - 1];

      Tensor real_values;
      Tensor real_vectors;
      // double the size of real_values, the first half stores the real part,
      // the next half stores the imag part
      std::vector<int> origin_dim =
          framework::vectorize<int>(out_values->dims());
      int last_item = origin_dim.back();
      origin_dim.pop_back();
      origin_dim.push_back(last_item * 2);
      framework::DDim big_dim = framework::make_ddim(origin_dim);

      real_values.mutable_data<math::Real<T>>(big_dim, context.GetPlace());
      real_vectors.mutable_data<math::Real<T>>(x->dims(), context.GetPlace());

      ApplyEigKernel<DeviceContext, math::Real<T>>(*x, &real_values,
                                                   &real_vectors, context);
      auto dito =
          math::DeviceIndependenceTensorOperations<DeviceContext, math::Real<T>,
                                                   Tout>(context);

      // 1. extract real part & imag part from real_values
      Tensor real_part = dito.Slice(real_values, {-1}, {0}, {order});
      Tensor imag_part = dito.Slice(real_values, {-1}, {order}, {order * 2});

      // 2. construct complex values
      auto* real_part_data = real_part.data<math::Real<T>>();
      auto* imag_part_data = imag_part.data<math::Real<T>>();
      int out_values_numel = out_values->numel();
      platform::ForRange<DeviceContext> for_range(
          context.template device_context<DeviceContext>(), out_values_numel);
      math::RealImagToComplexFunctor<Tout> functor(
          real_part_data, imag_part_data,
          out_values->mutable_data<Tout>(context.GetPlace()), out_values_numel);
      for_range(functor);

      // 3. construct complex vectors
      Tensor real_vector_trans = dito.Transpose(real_vectors);
      Tensor out_vectors_trans;
      out_vectors_trans.mutable_data<Tout>(x->dims(), context.GetPlace());
      ConstructComplexVectors<math::Real<T>, Tout>(
          &out_vectors_trans, *out_values, real_vector_trans, context,
          batch_count, order);
      TransposeTwoAxis<DeviceContext, Tout>(out_vectors_trans, out_vectors,
                                            x->dims().size() - 1,
                                            x->dims().size() - 2, context);
    } else {
      out_values->mutable_data<T>(context.GetPlace());
      out_vectors->mutable_data<T>(context.GetPlace());

      ApplyEigKernel<DeviceContext, T>(*x, out_values, out_vectors, context);
    }
  }
};

template <typename DeviceContext, typename Tout>
void ComputeBackwardForComplexInput(
    const Tensor& V, const Tensor& L, const Tensor& gL, const Tensor& gV,
    Tout* x_grad_data, int batch_count, int order,
    const framework::ExecutionContext& context) {
  auto dito =
      math::DeviceIndependenceTensorOperations<DeviceContext, Tout, Tout>(
          context);

  Tensor trans_v = dito.Transpose(V);
  Tensor Vh = dito.Conj(trans_v);
  Tensor Lconj = dito.Conj(L);
  Tensor Econj = dito.Sub(dito.Unsqueeze(Lconj, -2), dito.Unsqueeze(Lconj, -1));
  Tensor VhgV = dito.Matmul(Vh, gV);
  Tensor diag_real = dito.Real(VhgV);
  Tensor diag_res = dito.BatchDiag(diag_real, batch_count);
  Tensor diag_unsqueezed = dito.Unsqueeze(diag_res, -2);

  // turn diag_unsqueezed into complex
  auto numel = diag_unsqueezed.numel();
  Tensor diag_unsqueezed_complex;
  auto* data_diag_un = diag_unsqueezed.data<math::Real<Tout>>();
  auto* data_diag_un_com = diag_unsqueezed_complex.mutable_data<Tout>(
      diag_unsqueezed.dims(), context.GetPlace(),
      static_cast<size_t>(numel * sizeof(Tout)));
  auto& dev_ctx = context.template device_context<DeviceContext>();
  platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
  math::RealToComplexFunctor<Tout> functor(data_diag_un, data_diag_un_com,
                                           numel);
  for_range(functor);
  // real tensor multiply complex tensor in broadcast manner
  Tensor res1 = dito.RealMulComplex(V, diag_unsqueezed_complex);
  Tensor res2 = dito.Matmul(Vh, res1);
  Tensor result = dito.Sub(VhgV, res2);

  result.mutable_data<Tout>(V.dims(), context.GetPlace());
  result = dito.Div(result, Econj);
  result = dito.DiagFill(order, order, order, 0, gL, result);
  Tensor rhs = dito.Matmul(result, Vh);

  // solve linear system
  // solve(Vh, rhs, out, m, k)
  // Vh: matrix with shape [m,m]
  // rhs: rhs with shape [m,k]
  // x_grad: out
  int m = Vh.dims()[Vh.dims().size() - 1];
  int k = rhs.dims()[rhs.dims().size() - 1];
  auto* matrix_data = Vh.data<Tout>();
  auto* rhs_data = rhs.data<Tout>();
  math::SolveLinearSystem<Tout>(matrix_data, rhs_data, x_grad_data, m, k,
                                batch_count);
}

template <typename DeviceContext, typename T, typename Tout>
class EigGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& L = *context.Input<Tensor>("Eigenvalues");
    auto& V = *context.Input<Tensor>("Eigenvectors");
    auto& gL = *context.Input<Tensor>(framework::GradVarName("Eigenvalues"));
    auto& gV = *context.Input<Tensor>(framework::GradVarName("Eigenvectors"));

    auto& x_grad = *context.Output<Tensor>(framework::GradVarName("X"));
    auto* x_grad_data = x_grad.mutable_data<Tout>(context.GetPlace());

    auto& dims = V.dims();
    framework::DDim dim_origin = dims;
    int num_dims = dim_origin.size();
    int batch_count = BatchCount(V);
    const int order = dim_origin[num_dims - 1];

    ComputeBackwardForComplexInput<DeviceContext, Tout>(
        V, L, gL, gV, x_grad_data, batch_count, order, context);
  }
};

}  // namespace operators
}  // namespace paddle
