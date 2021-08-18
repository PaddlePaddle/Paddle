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

// #include "paddle/fluid/framework/op_registry.h"
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
  cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
void getBufferSize<double, double>(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n,
                                   const double *A, int lda, const double *W,
                                   int *lwork) {
  cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
void getBufferSize<paddle::platform::complex<float>, float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const paddle::platform::complex<float> *A, int lda, const float *W,
    int *lwork) {
  cusolverDnCheevd_bufferSize(handle, jobz, uplo, n,
                              reinterpret_cast<const cuComplex *>(A), lda, W,
                              lwork);
}

template <>
void getBufferSize<paddle::platform::complex<double>, double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, const paddle::platform::complex<double> *A, int lda, const double *W,
    int *lwork) {
  cusolverDnZheevd_bufferSize(handle, jobz, uplo, n,
                              reinterpret_cast<const cuDoubleComplex *>(A), lda,
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
  cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

template <>
void computeValues<double, double>(cusolverDnHandle_t handle,
                                   cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, double *A,
                                   int lda, double *W, double *work, int lwork,
                                   int *devInfo) {
  cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

template <>
void computeValues<paddle::platform::complex<float>, float>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, paddle::platform::complex<float> *A, int lda, float *W,
    paddle::platform::complex<float> *work, int lwork, int *devInfo) {
  cusolverDnCheevd(handle, jobz, uplo, n, reinterpret_cast<cuComplex *>(A), lda,
                   W, reinterpret_cast<cuComplex *>(work), lwork, devInfo);
}

template <>
void computeValues<paddle::platform::complex<double>, double>(
    cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo,
    int n, paddle::platform::complex<double> *A, int lda, double *W,
    paddle::platform::complex<double> *work, int lwork, int *devInfo) {
  cusolverDnZheevd(handle, jobz, uplo, n,
                   reinterpret_cast<cuDoubleComplex *>(A), lda, W,
                   reinterpret_cast<cuDoubleComplex *>(work), lwork, devInfo);
}

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T, typename ValueType>
class EighGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    std::cout << "##########" << std::endl;
    const auto *input_var = ctx.Input<Tensor>("X");

    auto *output_w_var = ctx.Output<Tensor>("OutVector");
    auto *output_v_var = ctx.Output<Tensor>("OutValue");

    bool lower = ctx.Attr<bool>("UPLO");
    auto &dims = input_var->dims();
    int dim_size = dims.size();
    int64_t batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    std::cout << "batch_size: " << batch_size << std::endl;
    auto *out_vector = output_w_var->mutable_data<T>(ctx.GetPlace());
    auto *out_value = output_v_var->mutable_data<ValueType>(ctx.GetPlace());

    cublasFillMode_t uplo =
        lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    int n = dims[dim_size - 1];
    std::cout << "n: " << n << std::endl;
    int lda = std::max<int>(1, n);
    std::cout << "lda: " << lda << std::endl;
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto value_stride = dims[dim_size - 1];
    std::cout << "vector_stride: " << vector_stride << std::endl;
    std::cout << "value_stride: " << value_stride << std::endl;
    paddle::framework::TensorCopy(
        *input_var, input_var->place(), dev_ctx,
        output_w_var);  // copy input data to temp data

    int lwork = 0;
    T *d_work = NULL;
    // auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_count);
    // auto* info_ptr = reinterpret_cast<int*>(info->ptr());

    int *info_ptr = NULL;
    // cudaMalloc((void **)&info_ptr, sizeof(int));
    cudaMalloc(reinterpret_cast<void **>(&info_ptr), sizeof(int));

    getBufferSize<T, ValueType>(dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
                                out_vector, lda, out_value, &lwork);
    // std::cout << "lwork: " << lwork << std::endl;
    // printf("lwork: %d\t",lwork);
    // std::cout << "#######GPU" << std::endl;
    // cudaMalloc((void **)&d_work, sizeof(T) * lwork);
    cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(T) * lwork);
    for (auto i = 0; i < batch_size; i++) {
      auto vector_data = out_vector + i * vector_stride;
      auto value_data = out_value + i * value_stride;
      auto handle = dev_ctx.cusolver_dn_handle();
      computeValues<T, ValueType>(handle, jobz, uplo, n, vector_data, lda,
                                  value_data, d_work, lwork, info_ptr);
    }
    // std::cout << "##########info" << std::endl;
    // check the info
    // std::vector<int> error_info;
    // error_info.resize(batch_size);

    // memory::Copy(platform::CPUPlace(), error_info.data(),
    //              BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
    //              info_ptr, sizeof(int) * batch_size, dev_ctx.stream());

    // for (int i = 0; i < batch_size; ++i) {
    //   PADDLE_ENFORCE_EQ(error_info[i], 0,
    //                     platform::errors::PreconditionNotMet(
    //                         "For batch [%d]: U(%d, %d) is zero, singular U.",
    //                         i,
    //                         error_info[i], error_info[i]));
    // }
    // std::cout << ">>>>>>>>>>>>:" << std::endl;
    std::vector<int> axis(dim_size - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dim_size - 1, dim_size - 2});
    Tensor output_w_var_trans;
    output_w_var_trans.mutable_data<T>(dims, ctx.GetPlace());
    TransCompute<platform::CUDADeviceContext, T>(
        dim_size, dev_ctx, *output_w_var, &output_w_var_trans, axis);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    eigh, ops::EighGPUKernel<paddle::platform::CUDADeviceContext,
                             paddle::platform::complex<double>, double>,
    ops::EighGPUKernel<paddle::platform::CUDADeviceContext,
                       paddle::platform::complex<float>, float>,
    ops::EighGPUKernel<paddle::platform::CUDADeviceContext, double, double>,
    ops::EighGPUKernel<paddle::platform::CUDADeviceContext, float, float>);
