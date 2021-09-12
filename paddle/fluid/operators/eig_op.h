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
#ifdef PADDLE_WITH_MKLML
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#else
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#include "Eigen/Core"
#include "Eigen/Eigenvalues"
#include "paddle/fluid/operators/eig_op_helper.h"  // must before lapack.h
#include "Eigen/src/misc/lapacke.h"                // LAPACK_dgeev
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/transpose_op.h"  // TransCompute()
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

// complex_util
template <typename T>
struct TemplateInTemplate {
  using type = T;
};

template <typename T>
struct TemplateInTemplate<platform::complex<T>> {
  using type = T;
};

// math complex_util
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

template <class T, class Tin = T>
void LapackEig(char jobvl, char jobvr, int n, T* a, int lda, T* w, T* vl,
               int ldvl, T* vr, int ldvr, T* work, int lwork, Tin* rwork,
               int* info);

template <>
void LapackEig<double>(char jobvl, char jobvr, int n, double* a, int lda,
                       double* w, double* vl, int ldvl, double* vr, int ldvr,
                       double* work, int lwork, double* rwork, int* info) {
  // lapack [sd]geev wants to separate output arrays: wr and wi for the real
  // and imaginary parts
  double* wr = w;
  double* wi = w + n;
  (void)rwork;
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
         &lwork, info);
}

template <>
void LapackEig<float>(char jobvl, char jobvr, int n, float* a, int lda,
                      float* w, float* vl, int ldvl, float* vr, int ldvr,
                      float* work, int lwork, float* rwork, int* info) {
  float* wr = w;
  float* wi = w + n;
  (void)rwork;
  sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
         &lwork, info);
}

template <>
void LapackEig<platform::complex<double>, double>(
    char jobvl, char jobvr, int n, platform::complex<double>* a, int lda,
    platform::complex<double>* w, platform::complex<double>* vl, int ldvl,
    platform::complex<double>* vr, int ldvr, platform::complex<double>* work,
    int lwork, double* rwork, int* info) {
  zgeev_(&jobvl, &jobvr, &n, reinterpret_cast<std::complex<double>*>(a), &lda,
         reinterpret_cast<std::complex<double>*>(w),
         reinterpret_cast<std::complex<double>*>(vl), &ldvl,
         reinterpret_cast<std::complex<double>*>(vr), &ldvr,
         reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, info);
}

template <>
void LapackEig<platform::complex<float>, float>(
    char jobvl, char jobvr, int n, platform::complex<float>* a, int lda,
    platform::complex<float>* w, platform::complex<float>* vl, int ldvl,
    platform::complex<float>* vr, int ldvr, platform::complex<float>* work,
    int lwork, float* rwork, int* info) {
  cgeev_(&jobvl, &jobvr, &n, reinterpret_cast<std::complex<float>*>(a), &lda,
         reinterpret_cast<std::complex<float>*>(w),
         reinterpret_cast<std::complex<float>*>(vl), &ldvl,
         reinterpret_cast<std::complex<float>*>(vr), &ldvr,
         reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, info);
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
  int order = input.dims()[num_dims - 1];  // the order"阶" of matrix input

  T* input_data = input.data<T>();  // input.data_ptr;
  int lda = std::max<int>(1, order);
  T* values_data = values.mutable_data<T>(context.GetPlace());
  T* lvector_data = nullptr;
  int ldvl = 1;
  T* rvector_data = vectors.mutable_data<T>(context.GetPlace());
  int ldvr = lda;
  int lwork = -1;

  int batch_count = BatchCount(input);
  int matrix_stride = MatrixStride(input);
  LOG(INFO) << "matrix_stride: " << MatrixStride;
  int values_stride = order;  // value stride is equal to the order of matrix
  LOG(INFO) << "values_stride" << values_stride;
  infos.Resize(framework::make_ddim({batch_count}));
  int* info = infos.mutable_data<int>(context.GetPlace());
  LOG(INFO) << "infos.type: " << infos.type();

  // if input is complex 会使用到 Tin
  Tensor rwork;
  Tin* rwork_data = nullptr;
  // if (framework::IsComplexType(input.type())) {  //
  // 说明input没有给转为complex，！！
  //   // 下面一句需要指定类型吗
  //   rwork.Resize(framework::make_ddim({lda*2}));
  //   rwork_data = rwork.mutable_data<Tin>(context.GetPlace());
  // }
  if (framework::IsComplexType(input.type())) {
    LOG(INFO) << "IsComplexType(input.type())";
  }

  rwork.Resize(framework::make_ddim({lda * 2}));
  rwork_data = rwork.mutable_data<Tin>(context.GetPlace());

  LOG(INFO) << "input dtype: " << input.type();
  LOG(INFO) << "order: " << order;
  LOG(INFO) << "input_data: " << input_data;
  LOG(INFO) << "values_data: " << values_data;
  LOG(INFO) << "rvector_data: " << rvector_data;
  LOG(INFO) << "rwork_data: " << rwork_data;
  LOG(INFO) << "info: " << info;

  // call LapackEig once to compute the size of work;
  T computed_work_size;
  LapackEig<T, Tin>(jobvl, jobvr, order, input_data, lda, values_data,
                    lvector_data, ldvl, rvector_data, ldvr, &computed_work_size,
                    lwork, rwork_data, &info[0]);

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
    LapackEig<T, Tin>(jobvl, jobvr, order, current_matrix, lda, current_values,
                      lvector_data, ldvl, current_rvectors, ldvr, work_data,
                      lwork, rwork_data, current_info);
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
  LOG(INFO) << "👍 ApplyEig Done";
}

template <typename DeviceContext, typename T, typename Tout>
void ApplyEigKernel(const Tensor& input, Tensor& values, Tensor& vectors,
                    Tensor& infos, const framework::ExecutionContext& context) {
  Tensor input_fortran_mem_layout;
  Tensor vectors_fortran_mem_layout;
  int num_dims = input.dims().size();

  LOG(INFO) << "input dims(): " << input.dims();

  LOG(INFO) << "💕  begin";
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
  LOG(INFO) << "💕  end";
}

// template <typename DeviceContext, typename Tin, typename Tout, typename
// Tbase>
template <typename DeviceContext, typename T, typename Tout, typename Tbase>
class EigKernel : public framework::OpKernel<T> {  // T 为注册的输入精度
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");  // T
    // convert input from real to complex
    auto* out_values = context.Output<Tensor>("OutValues");
    auto* out_vectors = context.Output<Tensor>("OutVectors");

    LOG(INFO) << "👀  Compute started";
    if (!framework::IsComplexType(
            x->type())) {  // 如果输入x是实数，则将x转为复数
      LOG(INFO) << "RealToComnplex";
      auto numel = x->numel();
      framework::Tensor x_c;
      auto* x_data = x->data<Tbase>();  // 保证编译时 Tbase 只为实数
      auto* x_c_data =
          x_c.mutable_data<Tout>(x->dims(),  // 未设置dims
                                 context.GetPlace(),
                                 static_cast<size_t>(numel * sizeof(Tout)));  //

      auto& dev_ctx = context.template device_context<DeviceContext>();
      platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
      math::RealToComplexFunctor<Tout> functor(x_data, x_c_data, numel);
      for_range(functor);

      LOG(INFO) << "x_c_data: " << x_c_data;  // 不应该为 nullptr

      out_values->mutable_data<Tout>(context.GetPlace());
      out_vectors->mutable_data<Tout>(context.GetPlace());

      LOG(INFO) << "Forward x type: " << x->type();
      LOG(INFO) << "Forward out_values type: " << out_values->type();
      LOG(INFO) << "Forward out_vectors type: " << out_vectors->type();
      LOG(INFO) << "input x: " << x->type();
      Tensor infos;
      ApplyEigKernel<DeviceContext, Tout, Tout>(x_c, *out_values, *out_vectors,
                                                infos, context);
    } else {
      LOG(INFO) << "Comnplex Original";
      out_values->mutable_data<T>(context.GetPlace());
      out_vectors->mutable_data<T>(context.GetPlace());

      LOG(INFO) << "Forward x type: " << x->type();
      LOG(INFO) << "Forward out_values type: " << out_values->type();
      LOG(INFO) << "Forward out_vectors type: " << out_vectors->type();
      LOG(INFO) << "input x: " << x->type();

      Tensor infos;
      ApplyEigKernel<DeviceContext, T, Tout>(*x, *out_values, *out_vectors,
                                             infos, context);
    }
    LOG(INFO) << "👀  Compute done";
  }
};

template <typename DeviceContext, typename T, typename Tout>
class EigGradKernel : public framework::OpKernel<T> {  //
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // T: 正向输入的类型
    // Tout: 正向输出的类型

    LOG(INFO) << "👉 backward starts";
    // using Tin = typename TemplateInTemplate<T>::type;

    auto& L = *context.Input<Tensor>("OutValues");
    auto& V = *context.Input<Tensor>("OutVectors");
    auto& gL = *context.Input<Tensor>(framework::GradVarName("OutValues"));
    auto& gV = *context.Input<Tensor>(framework::GradVarName("OutVectors"));

    auto& x_grad = *context.Output<Tensor>(framework::GradVarName("X"));
    auto* x_grad_data =
        x_grad.mutable_data<Tout>(context.GetPlace());  // 应是复数类型

    LOG(INFO) << "backward input L (values): " << L.dims();
    LOG(INFO) << "backward input V (vectors): " << V.dims();
    LOG(INFO) << "backward input gL (grad_val): " << gL.dims();
    LOG(INFO) << "backward input gV (grad_vec): " << gV.dims();
    LOG(INFO) << "backward output x_grad: " << x_grad.dims();

    // LOG(INFO) << "preparation DONE";

    auto& dims = V.dims();
    int num_dims = dims.size();
    int batch_count = BatchCount(V);
    const int order = dims[num_dims - 1];  // the order"阶" of matrix input
    // int tensor_size = batch_count * order * order;

    LOG(INFO) << "HERE 0";

    // outvalues: ValueType
    // outVectors: T
    auto dito = math::DeviceIndependenceTensorOperations<DeviceContext,
                                                         /*Tin*/ Tout, Tout>(
        context);  //谁是Tin，谁是Tout
    LOG(INFO) << "HERE 1";

    //(1.1) Vh = V.transpose(-2, -1).conj();
    Tensor trans_v = dito.Transpose(V);  // 每个方阵转置
    LOG(INFO) << "HERE 2";
    Tensor Vh = dito.Conj(trans_v);  // 每个元素Conj

    LOG(INFO) << "HERE 3";

    //(1.2) Lconj = L.conj();
    Tensor Lconj = dito.Conj(L);
    LOG(INFO) << "HERE 4";

    //(1.3) Econj = Lconj.unsqueeze(-2) - Lconj.unsqueeze(-1);
    Tensor Econj =
        dito.SubBroadcast(dito.Unsqueeze(Lconj, -2), dito.Unsqueeze(Lconj, -1),
                          batch_count, order);
    LOG(INFO) << "HERE 5";
    //(1.4) Econj.diagonal(0, -2, -1).fill_(1.);  //
    //(2.1) const auto VhgV = at::matmul(Vh, gV);
    Tensor VhgV = dito.Matmul(Vh, gV);  // matmul会批处理吗√
    LOG(INFO) << "HERE 6";

    //(2.2) const auto diag_re_VhgV = at::real(VhgV).diagonal(0, -2, -1);
    Tensor diag_real_VhgV;
    // if (framework::IsComplexType(V.type())) {  // vectors总是复数
    //   LOG(INFO) << "V is COMPLEX";
    //   diag_real_VhgV = dito.Diag(dito.Real(VhgV),batch_count);
    // } else {
    //   diag_real_VhgV = dito.Diag(VhgV,batch_count);
    // }
    LOG(INFO) << "🐽 VhgV dims: " << VhgV.dims();
    math::ShowTensor<Tout>(VhgV);
    diag_real_VhgV = dito.Diag(VhgV, batch_count);  // ⭕️
    math::ShowTensor<Tout>(diag_real_VhgV);
    LOG(INFO) << "🐽 VhgV after diag: " << diag_real_VhgV.dims();
    LOG(INFO) << "HERE 7";
    //(2.3) auto result = VhgV - at::matmul(Vh, V * diag_re_VhgV.unsqueeze(-2));
    //⭕️ * 是对应元素乘还是矩阵乘
    Tensor res1 = dito.Matmul(V, dito.Unsqueeze(diag_real_VhgV, -2));
    Tensor res2 = dito.Matmul(Vh, res1);
    Tensor result = dito.Sub(VhgV, res2);
    result.mutable_data<Tout>(dims, context.GetPlace());  // T
    LOG(INFO) << "HERE 8";

    //(3.1) result.div_(Econj);
    result = dito.Div(result, Econj);
    LOG(INFO) << "HERE 9";

    //(3.2) result.diagonal(0, -2, -1).copy_(gL);
    result = dito.DiagFill(order, order, order, 0, gL, result);
    LOG(INFO) << "HERE 10";

    //(4) result = at::linalg_solve(Vh, at::matmul(result, Vh));
    Tensor tmp = dito.Matmul(result, Vh);
    LOG(INFO) << "HERE 11";

    // solve linear system
    // solve(Vh, tmp, out, m, k)
    // Vh: matrix with shape [m,m]
    // tmp: rhs with shape [m,k]
    // x_grad: out
    int m = Vh.dims()[Vh.dims().size() - 1];
    int k = tmp.dims()[tmp.dims().size() - 1];
    auto* matrix_data = Vh.data<Tout>();
    auto* rhs_data = tmp.data<Tout>();
    LOG(INFO) << "HERE 12";
    math::SolveLinearSystem<DeviceContext, Tout>(
        matrix_data, rhs_data, x_grad_data, m, k,
        batch_count);  // 需要循环batch处理

    // x_grad.mutable_data<Tout>(context.GetPlace());

    //(5) return self.is_complex() ? result : at::real(result);
    // x_grad.ShareDataWith(result);
    LOG(INFO) << "👍 backward DONE";
  }
};

}  // namespace operators
}  // namespace paddle
