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
extern "C" {
#include <Eigen/src/misc/lapacke.h>
}
// #include <complex.h>
// #define lapack_complex_float std::complex<float>
// #define lapack_complex_double std::complex<double>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/transpose_op.h"

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
  zheevd_(&jobz, &uplo, &n, reinterpret_cast<double _Complex*>(a), &lda, w,
          reinterpret_cast<double _Complex*>(work), &lwork, rwork, &lrwork,
          iwork, &liwork, info);
}

template <>
inline void computeValues<paddle::platform::complex<float>, float>(
    char jobz, char uplo, int n, paddle::platform::complex<float>* a, int lda,
    float* w, paddle::platform::complex<float>* work, int lwork, float* rwork,
    int lrwork, int* iwork, int liwork, int* info) {
  cheevd_(&jobz, &uplo, &n, reinterpret_cast<float _Complex*>(a), &lda, w,
          reinterpret_cast<float _Complex*>(work), &lwork, rwork, &lrwork,
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

template <typename DeviceContext, typename T, typename ValueType>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input_var = ctx.Input<Tensor>("X");
    auto* output_v_var = ctx.Output<Tensor>("OutValue");
    auto* output_w_var = ctx.Output<Tensor>("OutVector");

    auto* output_value =
        output_v_var->mutable_data<ValueType>(ctx.GetPlace());  // eigenvalues
    auto* output_vector =
        output_w_var->mutable_data<T>(ctx.GetPlace());  // eigenvectors

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
        output_w_var);  // copy input data to temp data

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    Tensor info_tensor;
    auto* infos_data = info_tensor.mutable_data<int>(
        framework::make_ddim({batch_size}), ctx.GetPlace());

    std::vector<int> axis(dim_size - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dim_size - 1, dim_size - 2});
    Tensor output_w_var_trans;
    output_w_var_trans.mutable_data<T>(dims, ctx.GetPlace());
    TransCompute<DeviceContext, T>(dim_size, dev_ctx, *output_w_var,
                                   &output_w_var_trans, axis);

    paddle::framework::TensorCopy(
        output_w_var_trans, output_w_var_trans.place(), dev_ctx, output_w_var);

    char uplo = (lower == "L") ? 'L' : 'U';
    char jobz = 'V';
    auto n = dims[dim_size - 1];
    auto lda = std::max<int64_t>(1, n);
    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    int iwork_query;
    ValueType rwork_query;
    T lwork_query;

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
    TransCompute<DeviceContext, T>(dim_size, dev_ctx, *output_w_var,
                                   &output_w_var_trans, axis);

    paddle::framework::TensorCopy(
        output_w_var_trans, output_w_var_trans.place(), dev_ctx, output_w_var);
  }
};

}  // namespace operators
}  // namespace paddle
