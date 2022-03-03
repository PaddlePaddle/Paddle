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

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
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

static void CheckEighResult(const int batch, const int info) {
  PADDLE_ENFORCE_LE(
      info, 0,
      platform::errors::PreconditionNotMet(
          "For batch [%d]: the [%d] off-diagonal elements of an intermediate"
          "tridiagonal form did not converge to zero",
          batch, info));
  PADDLE_ENFORCE_GE(
      info, 0, platform::errors::PreconditionNotMet(
                   "For batch [%d]: the [%d] argument had an illegal value",
                   batch, info));
}

template <typename DeviceContext, typename T>
struct MatrixEighFunctor {
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors);
};

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices, and uses the variable has_vectors to
// control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;
    auto *out_value = eigen_values->mutable_data<ValueType>(ctx.GetPlace());

    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext, T>(
            ctx);

    Tensor input_trans;
    // lapack is a column-major storge, transpose make the input to
    // have a continuous memory layout
    input_trans = dito.Transpose(input);
    auto *input_vector = input_trans.data<T>();

    auto dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    int values_stride = dims[dim_size - 1];
    char uplo = is_lower ? 'L' : 'U';
    char jobz = has_vectors ? 'V' : 'N';
    auto n = dims[dim_size - 1];
    auto lda = std::max<int64_t>(1, n);
    // if work = -1, it means that you need to use the lapack function to query
    // the optimal value
    int lwork = -1;      // The length of the array work
    int lrwork = -1;     // The dimension of the array rwork,rwork is REAL array
    int liwork = -1;     // The dimension of the array iwork
    int iwork_opt = -1;  // The optimal length of the array liwork
    T lwork_opt = static_cast<T>(-1);  // The optimal length of the array work
    ValueType rwork_opt =
        static_cast<ValueType>(-1);  // The optimal length of the array rwork

    int info = 0;
    // Call lapackEigh to get the optimal size of work data
    phi::funcs::lapackEigh<T, ValueType>(
        jobz, uplo, n, input_vector, lda, out_value, &lwork_opt, lwork,
        &rwork_opt, lrwork, &iwork_opt, liwork, &info);
    lwork = std::max<int>(1, static_cast<int>(lwork_opt));
    liwork = std::max<int>(1, iwork_opt);

    Tensor rwork_tensor;
    ValueType *rwork_data = nullptr;

    // complex type
    if (framework::IsComplexType(
            framework::TransToProtoVarType(input.dtype()))) {
      lrwork = std::max<int>(1, static_cast<int>(rwork_opt));
      rwork_data = rwork_tensor.mutable_data<ValueType>(
          phi::make_ddim({lrwork}), ctx.GetPlace());
    }
    Tensor iwork_tensor, work_tensor;
    auto *iwork_data = iwork_tensor.mutable_data<int>(phi::make_ddim({liwork}),
                                                      ctx.GetPlace());
    auto *work_data =
        work_tensor.mutable_data<T>(phi::make_ddim({lwork}), ctx.GetPlace());

    for (auto i = 0; i < batch_size; i++) {
      auto *value_data = out_value + i * values_stride;
      auto *input_data = input_vector + i * vector_stride;
      phi::funcs::lapackEigh<T, phi::dtype::Real<T>>(
          jobz, uplo, n, input_data, lda, value_data, work_data, lwork,
          rwork_data, lrwork, iwork_data, liwork, &info);
      CheckEighResult(i, info);
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              platform::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated, "
                                  "so the eigenvectors must be provided."));
      input_trans = dito.Transpose(input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }
};

#ifdef PADDLE_WITH_CUDA

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices on GPU, and uses the variable has_vectors
// to control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<platform::CUDADeviceContext, T> {
 public:
  void operator()(const framework::ExecutionContext &ctx, const Tensor &input,
                  Tensor *eigen_values, Tensor *eigen_vectors, bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;
    auto *out_value = eigen_values->mutable_data<ValueType>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(ctx);
    Tensor input_trans;
    input_trans = dito.Transpose(input);
    auto *input_vector = input_trans.data<T>();
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
    bool use_syevj = (framework::TransToProtoVarType(input.dtype()) ==
                          framework::proto::VarType::FP32 &&
                      values_stride >= 32 && values_stride <= 512);
    syevjInfo_t syevj_params;
    if (use_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSsyevj_bufferSize(
          dev_ctx.cusolver_dn_handle(), jobz, uplo, n,
          reinterpret_cast<const float *>(input_vector), lda,
          reinterpret_cast<const float *>(out_value), &lwork, syevj_params));
    } else {
      EvdBuffer(dev_ctx.cusolver_dn_handle(), jobz, uplo, n, input_vector, lda,
                out_value, &lwork);
    }
    auto work = memory::Alloc(dev_ctx, sizeof(T) * lwork);
    auto *work_ptr = reinterpret_cast<T *>(work->ptr());
    for (auto i = 0; i < batch_size; i++) {
      auto *input_data = input_vector + i * vector_stride;
      auto *value_data = out_value + i * values_stride;
      auto handle = dev_ctx.cusolver_dn_handle();
      if (use_syevj) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSsyevj(
            handle, jobz, uplo, n, reinterpret_cast<float *>(input_data), lda,
            reinterpret_cast<float *>(value_data),
            reinterpret_cast<float *>(work_ptr), lwork, info_ptr,
            syevj_params));
      } else {
        Evd(handle, jobz, uplo, n, input_data, lda, value_data, work_ptr, lwork,
            info_ptr);
      }
      int error_info = 0;
      memory::Copy(platform::CPUPlace(), &error_info, dev_ctx.GetPlace(),
                   info_ptr, sizeof(int), dev_ctx.stream());
      CheckEighResult(i, error_info);
    }

    if (use_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          platform::dynload::cusolverDnDestroySyevjInfo(syevj_params));
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              platform::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated,"
                                  "so the eigenvectors must be provided."));
      input_trans = dito.Transpose(input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }

  using ValueType = phi::dtype::Real<T>;
  inline void EvdBuffer(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                        cublasFillMode_t uplo, int n, const T *A, int lda,
                        const ValueType *W, int *lwork) const;

  inline void Evd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                  cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W,
                  T *work, int lwork, int *devInfo) const;
};

#define FUNC_WITH_TYPES(m)                                \
  m(float, Ssy, float) m(double, Dsy, double)             \
      m(paddle::platform::complex<float>, Che, cuComplex) \
          m(paddle::platform::complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(T, C, CastType)                                     \
  template <>                                                                  \
  inline void MatrixEighFunctor<platform::CUDADeviceContext, T>::EvdBuffer(    \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                       \
      cublasFillMode_t uplo, int n, const T *A, int lda, const ValueType *W,   \
      int *lwork) const {                                                      \
    PADDLE_ENFORCE_GPU_SUCCESS(                                                \
        platform::dynload::cusolverDn##C##evd_bufferSize(                      \
            handle, jobz, uplo, n, reinterpret_cast<const CastType *>(A), lda, \
            W, lwork));                                                        \
  }

FUNC_WITH_TYPES(EVDBUFFER_INSTANCE);

#define EVD_INSTANCE(T, C, CastType)                                      \
  template <>                                                             \
  inline void MatrixEighFunctor<platform::CUDADeviceContext, T>::Evd(     \
      cusolverDnHandle_t handle, cusolverEigMode_t jobz,                  \
      cublasFillMode_t uplo, int n, T *A, int lda, ValueType *W, T *work, \
      int lwork, int *devInfo) const {                                    \
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDn##C##evd(     \
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
