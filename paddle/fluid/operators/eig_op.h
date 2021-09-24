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

#include "paddle/fluid/operators/eig_op_helper.h"  
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/transpose_op.h"  
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/operators/math/lapack_function.h"

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
  int values_stride = order;  // value stride is equal to the order of matrix
  infos.Resize(framework::make_ddim({batch_count}));
  int* info = infos.mutable_data<int>(context.GetPlace());

  // if input is complex 会使用到 Tin
  Tensor rwork;
  Tin* rwork_data = nullptr;

  rwork.Resize(framework::make_ddim({lda * 2}));
  rwork_data = rwork.mutable_data<Tin>(context.GetPlace());

  // call LapackEig once to compute the size of work;
  T computed_work_size;
  math::lapackEig<T, Tin>(jobvl, jobvr, order, input_data, lda, values_data,
                    lvector_data, ldvl, rvector_data, ldvr, &computed_work_size,
                    lwork, rwork_data, &info[0]);

  lwork =
      std::max<int>(1, static_cast<int>(GetReal<T, Tin>(computed_work_size)));
  Tensor work;
  work.Resize(framework::make_ddim({lwork}));
  T* work_data = work.mutable_data<T>(context.GetPlace());

  for (auto i = 0; i < batch_count; ++i) {
    auto* current_matrix = &input_data[i * matrix_stride];
    auto* current_values = &values_data[i * values_stride];
    auto* current_rvectors = &rvector_data[i * matrix_stride];
    auto* current_info = &info[i];
    math::lapackEig<T, Tin>(jobvl, jobvr, order, current_matrix, lda, current_values,
                      lvector_data, ldvl, current_rvectors, ldvr, work_data,
                      lwork, rwork_data, current_info);
    PADDLE_ENFORCE_EQ(
          *current_info, 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              *current_info));
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

  TransposeTwoAxis<DeviceContext, T>(vectors_fortran_mem_layout, vectors,
                                     num_dims - 1, num_dims - 2, context);
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

    if (!framework::IsComplexType(
            x->type())) {  // 如果输入x是实数，则将x转为复数
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

      out_values->mutable_data<Tout>(context.GetPlace());
      out_vectors->mutable_data<Tout>(context.GetPlace());
      Tensor infos;
      ApplyEigKernel<DeviceContext, Tout, Tout>(x_c, *out_values, *out_vectors,
                                                infos, context);
    } else {
      out_values->mutable_data<T>(context.GetPlace());
      out_vectors->mutable_data<T>(context.GetPlace());
      Tensor infos;
      ApplyEigKernel<DeviceContext, T, Tout>(*x, *out_values, *out_vectors,
                                             infos, context);
    }
  }
};

template <typename DeviceContext, typename T, typename Tout>
class EigGradKernel : public framework::OpKernel<T> {  
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& L = const_cast<Tensor&>(*context.Input<Tensor>("OutValues"));
    auto& V = const_cast<Tensor&>(*context.Input<Tensor>("OutVectors"));
    auto& gL = const_cast<Tensor&>(
        *context.Input<Tensor>(framework::GradVarName("OutValues")));
    auto& gV = const_cast<Tensor&>(
        *context.Input<Tensor>(framework::GradVarName("OutVectors")));

    auto& x_grad = *context.Output<Tensor>(framework::GradVarName("X"));
    auto* x_grad_data =
        x_grad.mutable_data<Tout>(context.GetPlace());  

    auto& dims = V.dims();
    framework::DDim dim_origin = dims;
    int num_dims = dim_origin.size();
    int batch_count = BatchCount(V);
    const int order =
        dim_origin[num_dims - 1];  

    if (num_dims > 3) {
      L.Resize(framework::make_ddim({batch_count, order}));
      V.Resize(framework::make_ddim({batch_count, order, order}));
      gL.Resize(framework::make_ddim({batch_count, order}));
      gV.Resize(framework::make_ddim({batch_count, order, order}));
      x_grad.Resize(framework::make_ddim({batch_count, order, order}));
    }

    auto dito = math::DeviceIndependenceTensorOperations<DeviceContext,
                                                         /*Tin*/ Tout, Tout>(
        context);  

    Tensor trans_v = dito.Transpose(V);  
    Tensor Vh = dito.Conj(trans_v);  
    Tensor Lconj = dito.Conj(L);
    Tensor Econj =
        dito.SubBroadcast(dito.Unsqueeze(Lconj, -2), dito.Unsqueeze(Lconj, -1),
                          batch_count, order);
    
    Tensor VhgV = dito.Matmul(Vh, gV);  
    Tensor diag_real_VhgV;
    Tensor diag_VhgV;
    Tensor diag_tmp = dito.Real(VhgV);
    Tensor diag_res /*实*/ = dito.Diag(diag_tmp /*实*/, batch_count);
    Tensor diag_un /*实*/ = dito.Unsqueeze(diag_res /*实*/, -2);   
    auto numel = diag_un.numel();
    Tensor diag_un_com;
    auto* data_diag_un = diag_un.data<math::Real<Tout>>();
    auto* data_diag_un_com = diag_un_com.mutable_data<Tout>(
        diag_un.dims(), context.GetPlace(),
        static_cast<size_t>(numel * sizeof(Tout)));
    auto& dev_ctx = context.template device_context<DeviceContext>();
    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    math::RealToComplexFunctor<Tout> functor(data_diag_un, data_diag_un_com,
                                             numel);
    for_range(functor);
    Tensor res1 = dito.ElementwiseMul(diag_un_com /*复*/, V /*复*/);
    Tensor res2 = dito.Matmul(Vh, res1);
    Tensor result = dito.Sub(VhgV, res2);
    result.mutable_data<Tout>(/*dims*/ V.dims(), context.GetPlace());  // T
    result = dito.Div(result, Econj);
    result = dito.DiagFill(order, order, order, 0, gL, result);
    Tensor tmp = dito.Matmul(result, Vh);
    int m = Vh.dims()[Vh.dims().size() - 1];
    int k = tmp.dims()[tmp.dims().size() - 1];
    auto* matrix_data = Vh.data<Tout>();
    auto* rhs_data = tmp.data<Tout>();
    math::SolveLinearSystem<DeviceContext, Tout>(
        matrix_data, rhs_data, x_grad_data, m, k,
        batch_count);  // 需要循环batch处理

    if (num_dims > 3) {
      std::vector<int> dim_origin_vec = ExtendDims(dim_origin, batch_count);
      dim_origin_vec.push_back(order);
      dim_origin_vec.push_back(order);
      x_grad.Resize(framework::make_ddim(dim_origin_vec));
    }
  }
};

}  // namespace operators
}  // namespace paddle
