/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/eigh_op.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

template <typename T, typename ValueType>
void getBufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                   cublasFillMode_t uplo, int n, const T *A, int lda,
                   const ValueType *W, int *lwork);

template <>
void getBufferSize<float, float>(cusolverDnHandle_t handle,
                                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                 int n, const float *A, int lda, const float *W,
                                 int *lwork) {
  platform::dynload::cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda,
                                                 W, lwork);
}

template <>
void getBufferSize<double, double>(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n,
                                   const double *A, int lda, const double *W,
                                   int *lwork) {
  platform::dynload::cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda,
                                                 W, lwork);
}

template <>
void getBufferSize<paddle::platform::complex<float>, float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const paddle::platform::complex<float> *A, int lda, const float *W,
    int *lwork) {
  platform::dynload::cusolverDnCheevd_bufferSize(
      handle, jobz, uplo, n, reinterpret_cast<const cuComplex *>(A), lda, W,
      lwork);
}

template <>
void getBufferSize<paddle::platform::complex<double>, double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const paddle::platform::complex<double> *A, int lda, const double *W,
    int *lwork) {
  platform::dynload::cusolverDnZheevd_bufferSize(
      handle, jobz, uplo, n, reinterpret_cast<const cuDoubleComplex *>(A), lda,
      W, lwork);
}

template <typename T, typename ValueType>
void computeValues(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                   cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W,
                   T *work, int lwork, int *devInfo);

template <>
void computeValues<float, float>(cusolverDnHandle_t handle,
                                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                 int n, float *A, int lda, float *W,
                                 float *work, int lwork, int *devInfo) {
  platform::dynload::cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work,
                                      lwork, devInfo);
}

template <>
void computeValues<double, double>(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, double *A,
                                   int lda, double *W, double *work, int lwork,
                                   int *devInfo) {
  platform::dynload::cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work,
                                      lwork, devInfo);
}

template <>
void computeValues<paddle::platform::complex<float>, float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, paddle::platform::complex<float> *A, int lda, float *W,
    paddle::platform::complex<float> *work, int lwork, int *devInfo) {
  platform::dynload::cusolverDnCheevd(
      handle, jobz, uplo, n, reinterpret_cast<cuComplex *>(A), lda, W,
      reinterpret_cast<cuComplex *>(work), lwork, devInfo);
}

template <>
void computeValues<paddle::platform::complex<double>, double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, paddle::platform::complex<double> *A, int lda, double *W,
    paddle::platform::complex<double> *work, int lwork, int *devInfo) {
  platform::dynload::cusolverDnZheevd(
      handle, jobz, uplo, n, reinterpret_cast<cuDoubleComplex *>(A), lda, W,
      reinterpret_cast<cuDoubleComplex *>(work), lwork, devInfo);
}

using Tensor = framework::Tensor;

template <typename ValueType, typename T>
class EighGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    const auto *input_var = ctx.Input<Tensor>("X");
    auto *output_w_var = ctx.Output<Tensor>("OutValue");
    auto *output_v_var = ctx.Output<Tensor>("OutVector");
    std::string lower = ctx.Attr<std::string>("UPLO");

    auto &dims = input_var->dims();
    int dim_size = dims.size();
    int64_t batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    auto *out_value = output_w_var->mutable_data<ValueType>(ctx.GetPlace());
    auto *out_vector = output_v_var->mutable_data<T>(ctx.GetPlace());

    cublasFillMode_t uplo =
        (lower == "L") ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    int n = dims[dim_size - 1];
    int lda = std::max<int>(1, n);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];
    paddle::framework::TensorCopy(
        *input_var, input_var->place(), dev_ctx,
        output_v_var);  // copy input data to temp data
    std::vector<int> axis(dim_size - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dim_size - 1, dim_size - 2});
    Tensor output_v_var_trans;
    output_v_var_trans.mutable_data<T>(dims, ctx.GetPlace());
    TransCompute<platform::CUDADeviceContext, T>(
        dim_size, dev_ctx, *output_v_var, &output_v_var_trans, axis);
    paddle::framework::TensorCopy(
        output_v_var_trans, output_v_var_trans.place(), dev_ctx, output_v_var);

    int lwork = 0;
    T *d_work = NULL;

    int *info_ptr = NULL;
    cudaMalloc(reinterpret_cast<void **>(&info_ptr), sizeof(int));
#if CUDA_VERSION >= 9020 && !defined(_WIN32)

    // Evd_Buffer(dev_ctx, jobz, uplo, n, out_vector, lda, out_value, &lwork);
    getBufferSize<T, ValueType>(dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
                                out_vector, lda, out_value, &lwork);

    cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * lwork);

    for (auto i = 0; i < batch_size; i++) {
      auto vector_data = out_vector + i * vector_stride;
      auto value_data = out_value + i * values_stride;
      // Evd(dev_ctx, jobz, uplo, n, vector_data, lda, value_data, d_work,
      // lwork, info_ptr);
      auto handle = dev_ctx.cusolver_dn_handle();
      computeValues<T, ValueType>(handle, jobz, uplo, n, vector_data, lda,
                                  value_data, d_work, lwork, info_ptr);
    }
#endif
    // check the info
    std::vector<int> error_info;  // only for checking positive matrix
    error_info.resize(batch_size);

    memory::Copy(platform::CPUPlace(), error_info.data(),
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 info_ptr, sizeof(int) * batch_size, dev_ctx.stream());

    for (int i = 0; i < batch_size; ++i) {
      PADDLE_ENFORCE_EQ(
          error_info[i], 0,
          platform::errors::InvalidArgument(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              error_info[i]));
    }

    TransCompute<platform::CUDADeviceContext, T>(
        dim_size, dev_ctx, *output_v_var, &output_v_var_trans, axis);
    paddle::framework::TensorCopy(
        output_v_var_trans, output_v_var_trans.place(), dev_ctx, output_v_var);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    eigh, ops::EighGPUKernel<float, float>, ops::EighGPUKernel<double, double>,
    ops::EighGPUKernel<float, paddle::platform::complex<float>>,
    ops::EighGPUKernel<double, paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    eigh_grad,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext, float, float>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext, double, double>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext, float,
                        paddle::platform::complex<float>>,
    ops::EighGradKernel<paddle::platform::CUDADeviceContext, double,
                        paddle::platform::complex<double>>);

#endif  // not PADDLE_WITH_HIP
