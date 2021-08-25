// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_MKLML
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#else
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif
#include "Eigen/Cholesky"
#include "Eigen/Core"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_sub_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/unsqueeze_op.h"

namespace paddle {
namespace operators {

template <typename T, typename ValueType>
inline void computeValues(char jobz, char uplo, int n, T* a, int lda,
                          ValueType* w, T* work, int lwork, ValueType* rwork,
                          int lrwork, int* iwork, int liwork, int* info);

template <>
inline void computeValues<paddle::platform::complex<double>, double>(
    char jobz, char uplo, int n, paddle::platform::complex<double>* a, int lda,
    double* w, paddle::platform::complex<double>* work, int lwork,
    double* rwork, int lrwork, int* iwork, int liwork, int* info) {
  zheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<double>*>(a), &lda, w,
          reinterpret_cast<std::complex<double>*>(work), &lwork, rwork, &lrwork,
          iwork, &liwork, info);
}

template <>
inline void computeValues<paddle::platform::complex<float>, float>(
    char jobz, char uplo, int n, paddle::platform::complex<float>* a, int lda,
    float* w, paddle::platform::complex<float>* work, int lwork, float* rwork,
    int lrwork, int* iwork, int liwork, int* info) {
  cheevd_(&jobz, &uplo, &n, reinterpret_cast<std::complex<float>*>(a), &lda, w,
          reinterpret_cast<std::complex<float>*>(work), &lwork, rwork, &lrwork,
          iwork, &liwork, info);
}

template <>
inline void computeValues<double, double>(char jobz, char uplo, int n,
                                          double* a, int lda, double* w,
                                          double* work, int lwork,
                                          double* rwork, int lrwork, int* iwork,
                                          int liwork, int* info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template <>
inline void computeValues<float, float>(char jobz, char uplo, int n, float* a,
                                        int lda, float* w, float* work,
                                        int lwork, float* rwork, int lrwork,
                                        int* iwork, int liwork, int* info) {
  (void)rwork;   // unused
  (void)lrwork;  // unused
  ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

using Tensor = framework::Tensor;

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename ValueType, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_var = ctx.Input<Tensor>("X");
    auto* output_w_var = ctx.Output<Tensor>("OutValue");
    auto* output_v_var = ctx.Output<Tensor>("OutVector");

    auto* output_value =
        output_w_var->mutable_data<ValueType>(ctx.GetPlace());  // eigenvalues
    auto* output_vector =
        output_v_var->mutable_data<T>(ctx.GetPlace());  // eigenvectors

    std::string lower = ctx.Attr<std::string>("UPLO");

    auto dims = input_var->dims();
    int dim_size = dims.size();
    int64_t batch_size = 1;
    for (int64_t i = 0; i < dim_size - 2; i++) {
      batch_size *= dims[i];
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    paddle::framework::TensorCopy(
        *input_var, input_var->place(), dev_ctx,
        output_v_var);  // copy input data to temp data

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    Tensor info_tensor;
    auto* infos_data = info_tensor.mutable_data<int>(
        framework::make_ddim({batch_size}), ctx.GetPlace());

    std::vector<int> axis(dim_size - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dim_size - 1, dim_size - 2});
    Tensor output_v_var_trans;
    output_v_var_trans.mutable_data<T>(dims, ctx.GetPlace());
    TransCompute<DeviceContext, T>(dim_size, dev_ctx, *output_v_var,
                                   &output_v_var_trans, axis);

    paddle::framework::TensorCopy(
        output_v_var_trans, output_v_var_trans.place(), dev_ctx, output_v_var);

    char uplo = (lower == "L") ? 'L' : 'U';
    char jobz = 'V';
    auto n = dims[dim_size - 1];
    auto lda = std::max<int64_t>(1, n);
    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    int iwork_query;
    ValueType rwork_query = static_cast<ValueType>(-1);

    T lwork_query = static_cast<T>(-1);

    computeValues<T, ValueType>(jobz, uplo, n, output_vector, lda, output_value,
                                &lwork_query, lwork, &rwork_query, lrwork,
                                &iwork_query, liwork, infos_data);

    lwork = std::max<int>(1, static_cast<int>(lwork_query));
    liwork = std::max<int>(1, iwork_query);

    Tensor rwork_tensor;
    ValueType* rwork_data = nullptr;
    // complex type
    if (framework::IsComplexType(input_var->type())) {
      lrwork = std::max<int>(1, static_cast<int>(rwork_query));
      rwork_data = rwork_tensor.mutable_data<ValueType>(
          framework::make_ddim({lrwork}), ctx.GetPlace());
    }

    Tensor iwork_tensor;
    auto* iwork_data = iwork_tensor.mutable_data<int>(
        framework::make_ddim({liwork}), ctx.GetPlace());

    Tensor work_tensor;
    auto* work_data = work_tensor.mutable_data<T>(framework::make_ddim({lwork}),
                                                  ctx.GetPlace());

    for (auto i = 0; i < batch_size; i++) {
      auto* vector_data = output_vector + i * vector_stride;
      auto* value_data = output_value + i * values_stride;
      int* info_ptr = &infos_data[i];
      computeValues<T, ValueType>(jobz, uplo, n, vector_data, lda, value_data,
                                  work_data, lwork, rwork_data, lrwork,
                                  iwork_data, liwork, info_ptr);

      // std::cout << "info_ptr: " << *info_ptr << std::endl;
      // PADDLE_ENFORCE_GT(*info_ptr, 0,
      //                   platform::errors::InvalidArgument(
      //                       "the [%d] argument had an illegal value",
      //                       *info_ptr));
      // PADDLE_ENFORCE_LT(*info_ptr, 0,
      //                   platform::errors::InvalidArgument(
      //         "if JOBZ = \'N\', [%d] off-diagonal elements of an intermediate
      //         tridiagonal form did not converge to zero;if JOBZ = \'V\', then
      //         the algorithm failed to compute an eigenvalue",
      //         *info_ptr));
    }
    TransCompute<DeviceContext, T>(dim_size, dev_ctx, *output_v_var,
                                   &output_v_var_trans, axis);

    paddle::framework::TensorCopy(
        output_v_var_trans, output_v_var_trans.place(), dev_ctx, output_v_var);
  }
};

template <typename T>
struct MatrixBandPartScaleEndFunctor {
  /*! Compared with MatrixBandPartFunctor, it scale up values at the end of
   * band. It can be used to fuse the following operations, which actually
   * output triangular with diagonal scaled up:
   * 1. dig = matrix_diag_part(middle)
   * 2. middle = matrix_set_diag(middle, diag * scalar)
   * 3. middle = matrix_band_part(middle, -1, 0)
   */
  MatrixBandPartScaleEndFunctor(const int m, const int n,
                                const int num_lower_diags,
                                const int num_upper_diags, const T* scale,
                                const T* input, T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        scale_(scale),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = input_[index];
    } else if (col == band_end - 1) {
      // std::cout << "scale: "<< scale_[index % m_] << "\t";
      output_[index] = scale_[index % m_];
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const T* scale_;
  const T* input_;
  T* output_;
};

template <typename DeviceContext, typename ValueType, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // std::cout << "backward>>>>>>>>>>>>>>>>>:" << std::endl;
    auto* x_grad = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad->mutable_data<T>(ctx.GetPlace());
    auto* output_w_var = ctx.Input<Tensor>("OutValue");
    auto* output_v_var = ctx.Input<Tensor>("OutVector");
    auto* output_w_grad = ctx.Input<Tensor>(framework::GradVarName("OutValue"));
    auto* output_v_grad =
        ctx.Input<Tensor>(framework::GradVarName("OutVector"));

    auto* output_w_grad_data = output_w_grad->data<ValueType>();
    // auto* output_v_grad_data = output_v_grad->data<T>();

    auto& dims = output_v_var->dims();
    int batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    const int m = dims[dims.size() - 1];
    int tensor_size = batch_size * m * m;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    std::vector<int> axis(dims.size() - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dims.size() - 1, dims.size() - 2});

    // //const auto Vh = V.conj().transpose(-2, -1);
    Tensor value_trans, result, result_trans, e_tensor, output_w_var_copy;
    value_trans.mutable_data<T>(dims, ctx.GetPlace());
    result_trans.mutable_data<T>(dims, ctx.GetPlace());
    auto* result_data = result.mutable_data<T>(dims, ctx.GetPlace());
    e_tensor.mutable_data<ValueType>(dims, ctx.GetPlace());
    output_w_var_copy.mutable_data<ValueType>(output_w_var->dims(),
                                              ctx.GetPlace());

    // std::cout << "dims size: " << dims.size() << std::endl;
    TransCompute<DeviceContext, T>(dims.size(), dev_ctx, *output_v_var,
                                   &value_trans, axis);
    // std::cout << "\n>>>>output_v_grad_data result: >>>>>>>>>>\n";
    // for(int i=0; i < tensor_size; i++){
    //   std::cout << output_v_grad_data[i] << "\t";
    // }
    // std::cout << "\n>>>>value_trans_data result: >>>>>>>>>>\n";
    // for(int i=0; i < tensor_size; i++){
    //   std::cout << value_trans_data[i] << "\t";
    // }

    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto no_trans_desc = math::CreateMatrixDescriptor(dims, 0, false);
    blas.MatMul(value_trans, no_trans_desc, *output_v_grad, no_trans_desc, T(1),
                &result, T(0));
    TransCompute<DeviceContext, T>(dims.size(), dev_ctx, result, &result_trans,
                                   axis);
    // std::cout << "\n>>>>result_trans_data result: >>>>>>>>>>\n";
    // for(int i=0; i < tensor_size; i++){
    //   std::cout << result_trans_data[i] << "\t";
    // }

    // std::cout << "\n>>>>matmul result: >>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto result_vector = EigenVector<T>::Flatten(result);
    auto result_trans_vector = EigenVector<T>::Flatten(result_trans);
    auto e_vector = EigenVector<ValueType>::Flatten(e_tensor);
    result_vector.device(place) =
        (result_vector - result_trans_vector) * static_cast<T>(0.5);
    // std::cout << "\n>>>>mul * 0.5>>>result: >>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }

    paddle::framework::TensorCopy(*output_w_var, output_w_var->place(), dev_ctx,
                                  &output_w_var_copy);

    // auto E = L.unsqueeze(-2) - L.unsqueeze(-1);
    framework::DDim out_dims_1;
    std::vector<int> dims_vec;
    dims_vec.insert(dims_vec.end(), {dims.size() - 2});
    out_dims_1 = UnsqueezeKernel<DeviceContext, T>::GetOutputShape(
        dims_vec, output_w_var_copy.dims());

    dims_vec.clear();
    framework::DDim out_dims_2;
    dims_vec.insert(dims_vec.end(), {dims.size() - 1});
    out_dims_2 = UnsqueezeKernel<DeviceContext, T>::GetOutputShape(
        dims_vec, output_w_var_copy.dims());

    Tensor xx = output_w_var_copy.Resize(out_dims_1);
    Tensor yy = output_w_var_copy.Resize(out_dims_2);
    // std::cout << "\n";
    // for(int i=0; i< out_dims_1.size(); i++){
    //   std::cout << out_dims_1[i] << "\t";
    // }
    // std::cout << "\n";
    // for(int i=0; i< out_dims_2.size(); i++){
    //   std::cout << out_dims_2[i] << "\t";
    // }

    // auto* xx_data = xx.data<ValueType>();
    // std::cout << "\n>>>>>>>>>>x_data>>>>>>>>>>>>>>>>>>\n";
    // std::cout << xx_data[0] << "\t" << xx_data[1] << "\t" << xx_data[2] <<
    // "\n";

    // auto* yy_data = yy.data<ValueType>();
    // std::cout << "\n>>>>>>>>>>y_data>>>>>>>>>>>>>>>>>>\n";
    // std::cout << yy_data[0] << "\t" << yy_data[1] << "\t" << yy_data[2] <<
    // "\n";
    // auto E = L.unsqueeze(-2) - L.unsqueeze(-1);

    if (batch_size > 1) {
      // Tensor xx = output_w_var_copy.Resize({batch_size,1,m});
      // Tensor yy = output_w_var_copy.Resize({batch_size,m,1});
      auto x_tensor = EigenTensor<ValueType, 3>::From(xx);
      auto y_tensor = EigenTensor<ValueType, 3>::From(yy);
      auto e_result = EigenTensor<ValueType, 3>::From(e_tensor);
      Eigen::DSizes<int, 3> a_bcast_dims(1, m, 1);
      Eigen::DSizes<int, 3> b_bcast_dims(1, 1, m);
      e_result.device(place) =
          x_tensor.broadcast(a_bcast_dims) - y_tensor.broadcast(b_bcast_dims);
    } else {
      // Tensor xx = output_w_var_copy.Resize({1,m});
      // Tensor yy = output_w_var_copy.Resize({m,1});
      auto x_tensor = EigenTensor<ValueType, 2>::From(xx);
      auto y_tensor = EigenTensor<ValueType, 2>::From(yy);
      auto e_result = EigenTensor<ValueType, 2>::From(e_tensor);
      Eigen::DSizes<int, 2> a_bcast_dims(m, 1);
      Eigen::DSizes<int, 2> b_bcast_dims(1, m);
      e_result.device(place) =
          x_tensor.broadcast(a_bcast_dims) - y_tensor.broadcast(b_bcast_dims);
    }
    // std::cout << "\n>>>>>>>E: >>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << e_data[i] << "\t";
    // }
    // std::cout << "\n>>>>div before>>>result: >>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }

    result_vector.device(place) = result_vector / e_vector;

    //  for(auto i=0; i<result.numel(); i++){
    //    result_data[i] /= static_cast<T>(sub_data[i]);
    //  }

    // std::cout << "\n>>>>div after>>>result: >>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }

    platform::ForRange<DeviceContext> for_range(dev_ctx, tensor_size);
    MatrixBandPartScaleEndFunctor<ValueType> matrix_band_part_scale_end_functor(
        m, m, /* num_lower_diags */ m, /* num_upper_diags */ 0,
        /* scale */ output_w_grad_data,
        reinterpret_cast<ValueType*>(result_data),
        reinterpret_cast<ValueType*>(result_data));
    for_range(matrix_band_part_scale_end_functor);
    // std::cout << "\ndiaglonal after:>>>>>>>>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }
    blas.MatMul(result, no_trans_desc, value_trans, no_trans_desc, T(1),
                &result, T(0));
    // std::cout << "\nmatmul1 after:>>>>>>>>>>>>>>>>\n";
    // for(int i=0; i< tensor_size; i++){
    //   std::cout << result_data[i] << "\t";
    // }
    blas.MatMul(*output_v_var, no_trans_desc, result, no_trans_desc, T(1),
                x_grad, T(0));
  }
};

}  // namespace operators
}  // namespace paddle
