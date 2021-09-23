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

#include "Eigen/Core"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/operators/svd_helper.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cusolver.h"
#endif  // PADDLE_WITH_CUDA

namespace paddle {
namespace operators {
namespace math {

inline int64_t GetBatchSize(framework::DDim dims) {
  int64_t batch_size = 1;
  auto dim_size = dims.size();
  for (int i = 0; i < dim_size - 2; i++) {
    batch_size *= dims[i];
  }
  return batch_size;
}

template <typename DeviceContext, typename ValueType, typename T>
struct MatrixEighFunctor {
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors);
};

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices, and uses the variable has_vectors to
// control whether to return the eigenvectors.
template <typename ValueType, typename T>
struct MatrixEighFunctor<platform::CPUDeviceContext, ValueType, T> {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    auto *out_value = eigen_values->mutable_data<ValueType>(ctx.GetPlace());

    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext, T>(
            ctx);
    *eigen_vectors = dito.Transpose(input);
    auto *out_vector = eigen_vectors->mutable_data<T>(ctx.GetPlace());

    auto dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    int values_stride = dims[dim_size - 1];
    char uplo = is_lower ? 'L' : 'U';
    char jobz = has_vectors ? 'V' : 'N';
    auto n = dims[dim_size - 1];
    auto lda = std::max<int64_t>(1, n);

    int lwork = -1;
    int lrwork = -1;
    int liwork = -1;
    int iwork_opt = -1;
    T lwork_opt = static_cast<T>(-1);
    ValueType rwork_opt = static_cast<ValueType>(-1);

    Tensor info_tensor;
    auto *infos_data = info_tensor.mutable_data<int>(
        framework::make_ddim({batch_size}), ctx.GetPlace());

    math::lapackEvd<T, ValueType>(jobz, uplo, n, out_vector, lda, out_value,
                                  &lwork_opt, lwork, &rwork_opt, lrwork,
                                  &iwork_opt, liwork, infos_data);

    lwork = std::max<int>(1, static_cast<int>(lwork_opt));
    liwork = std::max<int>(1, iwork_opt);

    Tensor rwork_tensor;
    ValueType *rwork_data = nullptr;

    // complex type
    if (framework::IsComplexType(eigen_vectors->type())) {
      lrwork = std::max<int>(1, static_cast<int>(rwork_opt));
      rwork_data = rwork_tensor.mutable_data<ValueType>(
          framework::make_ddim({lrwork}), ctx.GetPlace());
    }

    Tensor iwork_tensor, work_tensor;
    auto *iwork_data = iwork_tensor.mutable_data<int>(
        framework::make_ddim({liwork}), ctx.GetPlace());
    auto *work_data = work_tensor.mutable_data<T>(framework::make_ddim({lwork}),
                                                  ctx.GetPlace());

    for (auto i = 0; i < batch_size; i++) {
      auto *value_data = out_value + i * values_stride;
      auto *vector_data = out_vector + i * vector_stride;
      int *info_ptr = &infos_data[i];
      math::lapackEvd<T, ValueType>(jobz, uplo, n, vector_data, lda, value_data,
                                    work_data, lwork, rwork_data, lrwork,
                                    iwork_data, liwork, info_ptr);
      PADDLE_ENFORCE_EQ(
          *info_ptr, 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              *info_ptr));
    }
    if (has_vectors) {
      *eigen_vectors = dito.Transpose(*eigen_vectors);
    }
  }
};

#ifdef PADDLE_WITH_CUDA

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices on GPU, and uses the variable has_vectors
// to control whether to return the eigenvectors.
template <typename ValueType, typename T>
struct MatrixEighFunctor<platform::CUDADeviceContext, ValueType, T> {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    auto *out_value = eigen_values->mutable_data<ValueType>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(ctx);
    *eigen_vectors = dito.Transpose(input);
    auto *out_vector = eigen_vectors->mutable_data<T>(ctx.GetPlace());

    auto &dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    cublasFillMode_t uplo =
        is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cusolverEigMode_t jobz =
        has_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    int n = dims[dim_size - 1];
    int lda = std::max<int>(1, n);
    auto vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    auto values_stride = dims[dim_size - 1];

    int lwork = 0;
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_size);
    auto *info_ptr = reinterpret_cast<int *>(info->ptr());

    // When the input type is float32, and the feature value input dimension is
    // greater than or equal to [*,32,32]  and less than or equal to
    // [*,512,512], Syevj has better performance.
    bool use_syevj =
        (eigen_vectors->type() == framework::proto::VarType::FP32 &&
         values_stride >= 32 && values_stride <= 512);

    syevjInfo_t syevj_params;
    if (use_syevj) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnSsyevj_bufferSize(
              dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
              reinterpret_cast<const float *>(out_vector), lda,
              reinterpret_cast<const float *>(out_value), &lwork,
              syevj_params));
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
      if (use_syevj) {
        PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj(
            handle, jobz, uplo, n, reinterpret_cast<float *>(vector_data), lda,
            reinterpret_cast<float *>(value_data),
            reinterpret_cast<float *>(work_ptr), lwork, info_ptr,
            syevj_params));
      } else {
        Evd(handle, jobz, uplo, n, vector_data, lda, value_data, work_ptr,
            lwork, info_ptr);
      }
      int error_info;
      memory::Copy(platform::CPUPlace(), &error_info,
                   BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                   info_ptr, sizeof(int), dev_ctx.stream());
      PADDLE_ENFORCE_EQ(
          error_info, 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: the [%d] argument had an illegal value", i,
              error_info));
    }

    if (use_syevj) {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::cusolverDnDestroySyevjInfo(syevj_params));
    }

    if (has_vectors) {
      *eigen_vectors = dito.Transpose(*eigen_vectors);
    }
  }

  inline void EvdBuffer(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                        cublasFillMode_t uplo, int n, const T *A, int lda,
                        const ValueType *W, int *lwork) const;

  inline void Evd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W,
                  T *work, int lwork, int *devInfo) const;
};

#define FUNC_WITH_TYPES(m)                                       \
  m(float, float, Ssy, float) m(double, double, Dsy, double)     \
      m(float, paddle::platform::complex<float>, Che, cuComplex) \
          m(double, paddle::platform::complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(ValueType, T, C, CastType)                          \
  template <>                                                                  \
  inline void                                                                  \
  MatrixEighFunctor<platform::CUDADeviceContext, ValueType, T>::EvdBuffer(     \
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
  inline void                                                             \
  MatrixEighFunctor<platform::CUDADeviceContext, ValueType, T>::Evd(      \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                  \
      cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W, T *work, \
      int lwork, int *devInfo) const {                                    \
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDn##C##evd(    \
        handle, jobz, uplo, n, reinterpret_cast<CastType *>(A), lda, W,   \
        reinterpret_cast<CastType *>(work), lwork, devInfo));             \
  }

FUNC_WITH_TYPES(EVD_INSTANCE);

#undef FUNC_WITH_TYPES
#undef EVDBUFFER_INSTANCE
#undef EVD_INSTANCE

#endif  // PADDLE_WITH_CUDA

}  // namespace math
}  // namespace operators
}  // namespace paddle
