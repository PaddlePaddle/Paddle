// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cusolver.h"
#endif  // PADDLE_WITH_CUDA
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {
namespace funcs {

inline int64_t GetBatchSize(phi::DDim dims) {
  int64_t batch_size = 1;
  auto dim_size = dims.size();
  for (int i = 0; i < dim_size - 2; i++) {
    batch_size *= dims[i];
  }
  return batch_size;
}

static void CheckEighResult(const int batch, const int info) {
  PADDLE_ENFORCE_LE(
      info,
      0,
      phi::errors::PreconditionNotMet(
          "For batch [%d]: the [%d] off-diagonal elements of an intermediate"
          "tridiagonal form did not converge to zero",
          batch,
          info));
  PADDLE_ENFORCE_GE(
      info,
      0,
      phi::errors::PreconditionNotMet(
          "For batch [%d]: the [%d] argument had an illegal value",
          batch,
          info));
}

template <typename DeviceContext, typename T>
struct MatrixEighFunctor {
  void operator()(const DeviceContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors);
};

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices, and uses the variable has_vectors to
// control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<CPUContext, T> {
 public:
  void operator()(const CPUContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;
    ValueType *out_value = dev_ctx.template Alloc<ValueType>(eigen_values);

    DenseTensor input_trans;
    // lapack is a column-major storge, transpose make the input to
    // have a continuous memory layout
    input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input);
    T *input_vector = input_trans.data<T>();

    auto dims = input.dims();
    int dim_size = dims.size();
    int64_t batch_size = GetBatchSize(dims);

    int vector_stride = dims[dim_size - 1] * dims[dim_size - 2];
    int values_stride = dims[dim_size - 1];
    char uplo = is_lower ? 'L' : 'U';
    char jobz = has_vectors ? 'V' : 'N';
    int n = dims[dim_size - 1];
    int64_t lda = std::max<int64_t>(1, n);
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
    phi::funcs::lapackEigh<T, ValueType>(jobz,
                                         uplo,
                                         n,
                                         input_vector,
                                         lda,
                                         out_value,
                                         &lwork_opt,
                                         lwork,
                                         &rwork_opt,
                                         lrwork,
                                         &iwork_opt,
                                         liwork,
                                         &info);
    lwork = std::max<int>(1, static_cast<int>(lwork_opt));
    liwork = std::max<int>(1, iwork_opt);

    DenseTensor rwork_tensor;
    ValueType *rwork_data = nullptr;

    // complex type
    if (input.type() == phi::DataType::COMPLEX64 ||
        input.type() == phi::DataType::COMPLEX128) {
      lrwork = std::max<int>(1, static_cast<int>(rwork_opt));

      rwork_tensor.Resize(phi::make_ddim({lrwork}));
      rwork_data = dev_ctx.template Alloc<ValueType>(&rwork_tensor);
    }

    DenseTensor iwork_tensor, work_tensor;

    iwork_tensor.Resize(phi::make_ddim({liwork}));
    int *iwork_data = dev_ctx.template Alloc<int>(&iwork_tensor);

    work_tensor.Resize(phi::make_ddim({lwork}));
    T *work_data = dev_ctx.template Alloc<T>(&work_tensor);

    for (auto i = 0; i < batch_size; i++) {
      auto *value_data = out_value + i * values_stride;
      auto *input_data = input_vector + i * vector_stride;
      phi::funcs::lapackEigh<T, ValueType>(jobz,
                                           uplo,
                                           n,
                                           input_data,
                                           lda,
                                           value_data,
                                           work_data,
                                           lwork,
                                           rwork_data,
                                           lrwork,
                                           iwork_data,
                                           liwork,
                                           &info);
      CheckEighResult(i, info);
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              phi::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated, "
                                  "so the eigenvectors must be provided."));
      input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }
};

#ifdef PADDLE_WITH_CUDA

// Calculates the eigenvalues ​​and eigenvectors of Hermitian or real
// symmetric matrices on GPU, and uses the variable has_vectors
// to control whether to return the eigenvectors.
template <typename T>
struct MatrixEighFunctor<GPUContext, T> {
 public:
  void operator()(const GPUContext &dev_ctx,
                  const DenseTensor &input,
                  DenseTensor *eigen_values,
                  DenseTensor *eigen_vectors,
                  bool is_lower,
                  bool has_vectors) {
    using ValueType = phi::dtype::Real<T>;
    ValueType *out_value = dev_ctx.template Alloc<ValueType>(eigen_values);

    DenseTensor input_trans;
    input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input);
    T *input_vector = input_trans.data<T>();
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
    auto info = paddle::memory::Alloc(dev_ctx, sizeof(int) * batch_size);
    auto *info_ptr = reinterpret_cast<int *>(info->ptr());

    // When the input type is float32, and the feature value input dimension
    // is greater than or equal to [*,32,32]  and less than or equal to
    // [*,512,512], Syevj has better performance.
    bool use_syevj = (input.dtype() == phi::DataType::FLOAT32 &&
                      values_stride >= 32 && values_stride <= 512);
    syevjInfo_t syevj_params;
    if (use_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cusolverDnCreateSyevjInfo(&syevj_params));
      PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj_bufferSize(
          dev_ctx.cusolver_dn_handle(),
          jobz,
          uplo,
          n,
          reinterpret_cast<const float *>(input_vector),
          lda,
          reinterpret_cast<const float *>(out_value),
          &lwork,
          syevj_params));
    } else {
      EvdBuffer(dev_ctx.cusolver_dn_handle(),
                jobz,
                uplo,
                n,
                input_vector,
                lda,
                out_value,
                &lwork);
    }
    auto work = paddle::memory::Alloc(dev_ctx, sizeof(T) * lwork);
    auto *work_ptr = reinterpret_cast<T *>(work->ptr());
    for (auto i = 0; i < batch_size; i++) {
      auto *input_data = input_vector + i * vector_stride;
      auto *value_data = out_value + i * values_stride;
      auto handle = dev_ctx.cusolver_dn_handle();
      if (use_syevj) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            dynload::cusolverDnSsyevj(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      reinterpret_cast<float *>(input_data),
                                      lda,
                                      reinterpret_cast<float *>(value_data),
                                      reinterpret_cast<float *>(work_ptr),
                                      lwork,
                                      info_ptr,
                                      syevj_params));
      } else {
        Evd(handle,
            jobz,
            uplo,
            n,
            input_data,
            lda,
            value_data,
            work_ptr,
            lwork,
            info_ptr);
      }
      int error_info = 0;
      paddle::memory::Copy(phi::CPUPlace(),
                           &error_info,
                           dev_ctx.GetPlace(),
                           info_ptr,
                           sizeof(int),
                           dev_ctx.stream());
      CheckEighResult(i, error_info);
    }

    if (use_syevj) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          dynload::cusolverDnDestroySyevjInfo(syevj_params));
    }
    if (has_vectors) {
      PADDLE_ENFORCE_NOT_NULL(eigen_vectors,
                              phi::errors::InvalidArgument(
                                  "When has_vectors is true,"
                                  "the eigenvectors needs to be calculated,"
                                  "so the eigenvectors must be provided."));
      //   input_trans = dito.Transpose(input_trans);
      input_trans = phi::TransposeLast2Dim<T>(dev_ctx, input_trans);
      eigen_vectors->ShareDataWith(input_trans);
    }
  }

  using ValueType = phi::dtype::Real<T>;
  inline void EvdBuffer(cusolverDnHandle_t handle,
                        cusolverEigMode_t jobz,
                        cublasFillMode_t uplo,
                        int n,
                        const T *A,
                        int lda,
                        const ValueType *W,
                        int *lwork) const;

  inline void Evd(cusolverDnHandle_t handle,
                  cusolverEigMode_t jobz,
                  cublasFillMode_t uplo,
                  int n,
                  T *A,
                  int lda,
                  ValueType *W,
                  T *work,
                  int lwork,
                  int *devInfo) const;
};

using phi::dtype::complex;

#define FUNC_WITH_TYPES(m)                       \
  m(float, Ssy, float) m(double, Dsy, double) m( \
      complex<float>, Che, cuComplex) m(complex<double>, Zhe, cuDoubleComplex)

#define EVDBUFFER_INSTANCE(T, C, CastType)                             \
  template <>                                                          \
  inline void MatrixEighFunctor<GPUContext, T>::EvdBuffer(             \
      cusolverDnHandle_t handle,                                       \
      cusolverEigMode_t jobz,                                          \
      cublasFillMode_t uplo,                                           \
      int n,                                                           \
      const T *A,                                                      \
      int lda,                                                         \
      const ValueType *W,                                              \
      int *lwork) const {                                              \
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDn##C##evd_bufferSize( \
        handle,                                                        \
        jobz,                                                          \
        uplo,                                                          \
        n,                                                             \
        reinterpret_cast<const CastType *>(A),                         \
        lda,                                                           \
        W,                                                             \
        lwork));                                                       \
  }

FUNC_WITH_TYPES(EVDBUFFER_INSTANCE);

#define EVD_INSTANCE(T, C, CastType)                                           \
  template <>                                                                  \
  inline void MatrixEighFunctor<GPUContext, T>::Evd(cusolverDnHandle_t handle, \
                                                    cusolverEigMode_t jobz,    \
                                                    cublasFillMode_t uplo,     \
                                                    int n,                     \
                                                    T *A,                      \
                                                    int lda,                   \
                                                    ValueType *W,              \
                                                    T *work,                   \
                                                    int lwork,                 \
                                                    int *devInfo) const {      \
    PADDLE_ENFORCE_GPU_SUCCESS(                                                \
        dynload::cusolverDn##C##evd(handle,                                    \
                                    jobz,                                      \
                                    uplo,                                      \
                                    n,                                         \
                                    reinterpret_cast<CastType *>(A),           \
                                    lda,                                       \
                                    W,                                         \
                                    reinterpret_cast<CastType *>(work),        \
                                    lwork,                                     \
                                    devInfo));                                 \
  }

FUNC_WITH_TYPES(EVD_INSTANCE);

#undef FUNC_WITH_TYPES
#undef EVDBUFFER_INSTANCE
#undef EVD_INSTANCE

#endif  // PADDLE_WITH_CUDA

}  // namespace funcs
}  // namespace phi
