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
#include "paddle/fluid/operators/eigvalsh_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename ValueType, typename T>
class EigvalshGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto &input_var = *ctx.Input<Tensor>("X");
    auto &output_w_var = *ctx.Output<Tensor>("OutValue");
    auto &output_v_var = *ctx.Output<Tensor>("OutVector");
    std::string lower = ctx.Attr<std::string>("UPLO");
    auto &dims = input_var.dims();
    int dim_size = dims.size();
    int64_t batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    auto *out_value = output_w_var.mutable_data<ValueType>(ctx.GetPlace());
    auto *out_vector = output_v_var.mutable_data<T>(ctx.GetPlace());

    cublasFillMode_t uplo =
        (lower == "L") ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    int n = dims[dim_size - 1];
    int lda = std::max<int>(1, n);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    TensorCopy(input_var, ctx.GetPlace(), &output_v_var);
    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext, T,
                                                 ValueType>(ctx);
    Tensor output_v_var_trans;
    output_v_var_trans = dito.Transpose(output_v_var);
    TensorCopy(output_v_var_trans, ctx.GetPlace(), &output_v_var);

    int lwork = 0;
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_size);
    auto *info_ptr = reinterpret_cast<int *>(info->ptr());

    bool flag = (output_v_var.type() == framework::proto::VarType::FP32 &&
                 values_stride >= 32 && values_stride <= 512);
    syevjInfo_t syevj_params;

    if (flag) {
      platform::dynload::cusolverDnCreateSyevjInfo(&syevj_params);
      platform::dynload::cusolverDnSsyevj_bufferSize(
          dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
          reinterpret_cast<const float *>(out_vector), lda,
          reinterpret_cast<const float *>(out_value), &lwork, syevj_params);
    } else {
      EvdBuffer(dev_ctx.cusolver_dn_handle(), jobz, uplo, n, out_vector, lda,
                out_value, &lwork);
    }

    auto work = memory::Alloc(dev_ctx, sizeof(T) * lwork);
    auto *work_ptr = reinterpret_cast<T *>(work->ptr());

    for (auto i = 0; i < batch_size; i++) {
      auto vector_data = out_vector + i * vector_stride;
      auto value_data = out_value + i * values_stride;
      auto handle = dev_ctx.cusolver_dn_handle();
      if (flag) {
        platform::dynload::cusolverDnSsyevj(
            handle, jobz, uplo, n, reinterpret_cast<float *>(vector_data), lda,
            reinterpret_cast<float *>(value_data),
            reinterpret_cast<float *>(work_ptr), lwork, info_ptr, syevj_params);
      } else {
        Evd(handle, jobz, uplo, n, vector_data, lda, value_data, work_ptr,
            lwork, info_ptr);
      }
    }
    // check the info
    std::vector<int> error_info;
    error_info.resize(batch_size);

    memory::Copy(platform::CPUPlace(), error_info.data(),
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 info_ptr, sizeof(int) * batch_size, dev_ctx.stream());

    for (int i = 0; i < batch_size; ++i) {
      PADDLE_ENFORCE_EQ(
          error_info[i], 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              error_info[i]));
    }
    output_v_var_trans = dito.Transpose(output_v_var);
    TensorCopy(output_v_var_trans, ctx.GetPlace(), &output_v_var);
  }
  void EvdBuffer(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                 cublasFillMode_t uplo, int n, const T *A, int lda,
                 const ValueType *W, int *lwork) const;

  void Evd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
           cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W, T *work,
           int lwork, int *devInfo) const;
};

#define FUNC_WITH_TYPES(m)                                       \
  m(float, float, Ssy, float) m(double, double, Dsy, double)     \
      m(float, paddle::platform::complex<float>, Che, cuComplex) \
          m(double, paddle::platform::complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(ValueType, T, C, CastType)                          \
  template <>                                                                  \
  void EigvalshGPUKernel<ValueType, T>::EvdBuffer(                             \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                       \
      cublasFillMode_t uplo, int n, const T *A, int lda, const ValueType *W,   \
      int *lwork) const {                                                      \
    PADDLE_ENFORCE_CUDA_SUCCESS(                                               \
        platform::dynload::cusolverDn##C##evd_bufferSize(                      \
            handle, jobz, uplo, n, reinterpret_cast<const CastType *>(A), lda, \
            W, lwork));                                                        \
  }

FUNC_WITH_TYPES(EVDBUFFER_INSTANCE);

#define EVD_INSTANCE(ValueType, T, C, CastType)                           \
  template <>                                                             \
  void EigvalshGPUKernel<ValueType, T>::Evd(                              \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                  \
      cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W, T *work, \
      int lwork, int *devInfo) const {                                    \
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDn##C##evd(    \
        handle, jobz, uplo, n, reinterpret_cast<CastType *>(A), lda, W,   \
        reinterpret_cast<CastType *>(work), lwork, devInfo));             \
  }

FUNC_WITH_TYPES(EVD_INSTANCE);

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    eigvalsh, ops::EigvalshGPUKernel<float, float>,
    ops::EigvalshGPUKernel<double, double>,
    ops::EigvalshGPUKernel<float, paddle::platform::complex<float>>,
    ops::EigvalshGPUKernel<double, paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    eigvalsh_grad,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, float, float>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, double,
                            double>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, float,
                            paddle::platform::complex<float>>,
    ops::EigvalshGradKernel<paddle::platform::CUDADeviceContext, double,
                            paddle::platform::complex<double>>);

#endif  // not PADDLE_WITH_HIP
