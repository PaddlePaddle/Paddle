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
#define EPSILON 1e-6

//#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct TemplateInTemplate {
  using type = T;
};

template <typename T>
struct TemplateInTemplate<platform::complex<T>> {
  using type = T;
};

template <typename T, typename Tin = T>
constexpr Tin GetReal(T z) {
  return z;
}

template <>
constexpr float GetReal<platform::complex<float>, float>(
    platform::complex<float> z) {
  return z.real;
}

template <>
constexpr double GetReal<platform::complex<double>, double>(
    platform::complex<double> z) {
  return z.real;
}

using paddle::framework::Tensor;

inline int BatchCount(const Tensor& matrices) {
  int count = 1;
  int num_dims = matrices.dims().size();
  for (int i = 0; i < num_dims - 2; ++i) {
    count *= matrices.dims()[i];
  }
  return count;
}

inline int MatrixStride(const Tensor& matrices) {
  framework::DDim dims_list = matrices.dims();
  int num_dims = dims_list.size();
  return dims_list[num_dims - 1] * dims_list[num_dims - 2];
}

std::vector<int> ExtendDims(const framework::DDim& in_dims, int batch_size) {
  int cum = 1;
  std::vector<int> res;
  for (int i = 0; i < in_dims.size(); ++i) {
    cum *= in_dims[i];
    res.push_back(in_dims[i]);
    if (cum == batch_size) {
      break;
    }
  }
  return res;
}

// Transpose two axis of a Tensor
template <typename DeviceContext, typename T>
void TransposeTwoAxis(const Tensor& input, Tensor& transposed_input,
                      const int axis1, const int axis2,
                      const framework::ExecutionContext& context) {
  std::vector<int> permute(input.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis1] = axis2;
  permute[axis2] = axis1;

  transposed_input.mutable_data<T>(input.dims(), context.GetPlace());
  auto& dev_ctx = context.template device_context<platform::CPUDeviceContext>();

  TransCompute<DeviceContext, T>(input.dims().size(), dev_ctx, input,
                                 &transposed_input, permute);
}

// Apply eig to a batch of matrices, values, vectors and (intermidiate
// tensor)infos are overritten by
template <typename T, typename Tout>
void ApplyEig(Tensor& input, Tensor& values, Tensor& vectors, Tensor& infos,
              const framework::ExecutionContext& context) {
  using Tin = typename TemplateInTemplate<T>::type;

  char jobvl = 'N';
  char jobvr = 'V';  // only right eigenvectors are computed
  int num_dims = input.dims().size();
  int order = input.dims()[num_dims - 1];

  T* input_data = input.data<T>();
  int lda = std::max<int>(1, order);
  T* values_data = values.mutable_data<T>(context.GetPlace());
  T* lvector_data = nullptr;
  int ldvl = 1;
  T* rvector_data = vectors.mutable_data<T>(context.GetPlace());
  int ldvr = lda;
  int lwork = -1;

  int batch_count = BatchCount(input);
  int matrix_stride = MatrixStride(input);
  int values_stride = values.dims()[values.dims().size() - 1];
  infos.Resize(framework::make_ddim({batch_count}));
  int* info = infos.mutable_data<int>(context.GetPlace());

  Tensor rwork;
  Tin* rwork_data = nullptr;

  rwork.Resize(framework::make_ddim({lda * 2}));
  rwork_data = rwork.mutable_data<Tin>(context.GetPlace());

  // call lapackEig once to compute the size of work;
  T computed_work_size;
  math::lapackEig<T, Tin>(jobvl, jobvr, order, input_data, lda, values_data,
                          lvector_data, ldvl, rvector_data, ldvr,
                          &computed_work_size, lwork, rwork_data, &info[0]);

  lwork =
      std::max<int>(1, static_cast<int>(GetReal<T, Tin>(computed_work_size)));
  Tensor work;
  work.Resize(framework::make_ddim({lwork}));
  T* work_data = work.mutable_data<T>(context.GetPlace());

  for (auto i = 0; i < batch_count; ++i) {
    T* current_matrix = &input_data[i * matrix_stride];
    T* current_values = &values_data[i * values_stride];
    T* current_rvectors = &rvector_data[i * matrix_stride];
    int* current_info = &info[i];
    math::lapackEig<T, Tin>(jobvl, jobvr, order, current_matrix, lda,
                            current_values, lvector_data, ldvl,
                            current_rvectors, ldvr, work_data, lwork,
                            rwork_data, current_info);
    PADDLE_ENFORCE_EQ(
        *current_info, 0,
        platform::errors::PreconditionNotMet(
            "current_info is not 0, computation failed. "
            "= 0:  successful exit."
            "< 0:  if INFO = -i, the i-th argument had an illegal value."
            "> 0:  if INFO = i, the QR algorithm failed to compute all the "
            "eigenvalues, and no eigenvectors have been computed; "
            "elements i+1:N of WR and WI contain eigenvalues which "
            "have converged."));
  }
}

template <typename DeviceContext, typename T, typename Tout>
void ApplyEigKernel(const Tensor& input, Tensor& values, Tensor& vectors,
                    Tensor& infos, const framework::ExecutionContext& context) {
  Tensor input_fortran_mem_layout;
  Tensor vectors_fortran_mem_layout;
  int num_dims = input.dims().size();

  // transfer to Fortran memory layout i.e. make_ddim from tranposed_input:
  // [batch,row,col]->[batch,col,row]
  TransposeTwoAxis<DeviceContext, T>(input, input_fortran_mem_layout,
                                     num_dims - 1, num_dims - 2, context);
  // make sure 'vectors_fortran_mem_layout' holds memory before passed to
  // ApplyEig()
  vectors_fortran_mem_layout.Resize(input.dims());
  ApplyEig<T, Tout>(input_fortran_mem_layout, values,
                    vectors_fortran_mem_layout, infos, context);

  // transfer output memory layout back
  // vectors_fortran_mem_layout: fortran layout
  // vector: original layout
  TransposeTwoAxis<DeviceContext, T>(vectors_fortran_mem_layout, vectors,
                                     num_dims - 1, num_dims - 2, context);
}

template <typename T, typename Tout>
void ConstructComplexValues(Tensor& c_values, Tensor& real_part,
                            Tensor& imag_part,
                            const framework::ExecutionContext& ctx,
                            int batch_count, int order) {
  auto* c_values_data = c_values.mutable_data<Tout>(ctx.GetPlace());
  auto* real_data = real_part.data<T>();
  auto* imag_data = imag_part.data<T>();

  for (int b = 0; b < batch_count; b++) {
    for (int j = 0; j < order; j++) {
      c_values_data[b * order + j] = platform::complex<T>(
          real_data[b * order + j], imag_data[b * order + j]);
    }
  }
}

template <typename T, typename Tout>
void ConstructComplexVectors(Tensor& c_vectors, Tensor& c_values,
                             Tensor& r_vectors,
                             const framework::ExecutionContext& ctx,
                             int batch_count, int order) {
  int matrix_stride = MatrixStride(r_vectors);

  auto* c_vectors_data = c_vectors.mutable_data<Tout>(ctx.GetPlace());
  auto* c_values_data = c_values.mutable_data<Tout>(ctx.GetPlace());
  auto* r_v_data = r_vectors.data<T>();

  for (auto b = decltype(batch_count){0}; b < batch_count; b++) {
    auto* vecs = &r_v_data[b * matrix_stride];
    auto* res = &c_vectors_data[b * matrix_stride];
    auto* vals = &c_values_data[b * order];

    for (auto j = decltype(order){0}; j < order; j++) {
      if (vals[j].imag < EPSILON) {
        for (auto i = decltype(order){0}; i < order; i++) {
          res[j * order + i] = platform::complex<T>(vecs[j * order + i], 0);
        }
      } else {
        for (auto i = decltype(order){0}; i < order; i++) {
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

template <typename DeviceContext, typename T, typename Tout, typename Tbase>
class EigKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out_values = context.Output<Tensor>("Eigenvalues");
    auto* out_vectors = context.Output<Tensor>("Eigenvectors");

    if (!framework::IsComplexType(x->type())) {
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

      real_values.mutable_data<Tbase>(big_dim, context.GetPlace());
      real_vectors.mutable_data<Tbase>(x->dims(), context.GetPlace());
      Tensor infos;
      ApplyEigKernel<DeviceContext, Tbase, Tout>(*x, real_values, real_vectors,
                                                 infos, context);
      auto dito =
          math::DeviceIndependenceTensorOperations<DeviceContext, Tbase, Tout>(
              context);

      // 1. extract real part & imag part from real_values
      Tensor real_part = dito.Slice(real_values, {-1}, {0}, {order});
      Tensor imag_part = dito.Slice(real_values, {-1}, {order}, {order * 2});

      // 2. construct complex values
      ConstructComplexValues<Tbase, Tout>(*out_values, real_part, imag_part,
                                          context, batch_count, order);

      // 3. construct complex vectors
      Tensor real_vector_trans = dito.Transpose(real_vectors);
      Tensor out_vectors_trans;
      out_vectors_trans.mutable_data<Tout>(x->dims(), context.GetPlace());
      ConstructComplexVectors<Tbase, Tout>(out_vectors_trans, *out_values,
                                           real_vector_trans, context,
                                           batch_count, order);
      TransposeTwoAxis<DeviceContext, Tout>(out_vectors_trans, *out_vectors,
                                            x->dims().size() - 1,
                                            x->dims().size() - 2, context);
    } else {
      out_values->mutable_data<T>(context.GetPlace());
      out_vectors->mutable_data<T>(context.GetPlace());

      Tensor infos;
      ApplyEigKernel<DeviceContext, T, Tout>(*x, *out_values, *out_vectors,
                                             infos, context);
    }
  }
};

template <typename DeviceContext, typename Tout>
void ComputeBackwardForComplexInput(
    Tensor& V, Tensor& L, Tensor& gL, Tensor& gV, Tout* x_grad_data,
    int batch_count, int order, const framework::ExecutionContext& context) {
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
    auto& L = const_cast<Tensor&>(*context.Input<Tensor>("Eigenvalues"));
    auto& V = const_cast<Tensor&>(*context.Input<Tensor>("Eigenvectors"));
    auto& gL = const_cast<Tensor&>(
        *context.Input<Tensor>(framework::GradVarName("Eigenvalues")));
    auto& gV = const_cast<Tensor&>(
        *context.Input<Tensor>(framework::GradVarName("Eigenvectors")));

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
