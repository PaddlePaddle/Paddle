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

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include "paddle/phi/kernels/matrix_rank_tol_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T>
static void GesvdjBatched(const phi::GPUContext& dev_ctx,
                          int batchSize,
                          int m,
                          int n,
                          int k,
                          T* A,
                          T* U,
                          T* V,
                          phi::dtype::Real<T>* S,
                          int* info,
                          int thin_UV = 1);

template <typename T>
void SyevjBatched(const phi::GPUContext& dev_ctx,
                  int batchSize,
                  int n,
                  T* A,
                  phi::dtype::Real<T>* W,
                  int* info);

template <>
void GesvdjBatched<float>(const phi::GPUContext& dev_ctx,
                          int batchSize,
                          int m,
                          int n,
                          int k,
                          float* A,
                          float* U,
                          float* V,
                          float* S,
                          int* info,
                          int thin_UV) {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnSgesvdj_bufferSize(handle,
                                            jobz,
                                            thin_UV,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            S,
                                            U,
                                            ldu,
                                            V,
                                            ldt,
                                            &lwork,
                                            gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(float),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgesvdj(handle,
                                                          jobz,
                                                          thin_UV,
                                                          m,
                                                          n,
                                                          A + stride_A * i,
                                                          lda,
                                                          S + k * i,
                                                          U + stride_U * i,
                                                          ldu,
                                                          V + stride_V * i,
                                                          ldt,
                                                          workspace_ptr,
                                                          lwork,
                                                          info,
                                                          gesvdj_params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatched<double>(const phi::GPUContext& dev_ctx,
                           int batchSize,
                           int m,
                           int n,
                           int k,
                           double* A,
                           double* U,
                           double* V,
                           double* S,
                           int* info,
                           int thin_UV) {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDgesvdj_bufferSize(handle,
                                            jobz,
                                            thin_UV,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            S,
                                            U,
                                            ldu,
                                            V,
                                            ldt,
                                            &lwork,
                                            gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(double),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgesvdj(handle,
                                                          jobz,
                                                          thin_UV,
                                                          m,
                                                          n,
                                                          A + stride_A * i,
                                                          lda,
                                                          S + k * i,
                                                          U + stride_U * i,
                                                          ldu,
                                                          V + stride_V * i,
                                                          ldt,
                                                          workspace_ptr,
                                                          lwork,
                                                          info,
                                                          gesvdj_params));
    // check the error info
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatched<phi::dtype::complex<float>>(const phi::GPUContext& dev_ctx,
                                               int batchSize,
                                               int m,
                                               int n,
                                               int k,
                                               phi::dtype::complex<float>* A,
                                               phi::dtype::complex<float>* U,
                                               phi::dtype::complex<float>* V,
                                               float* S,
                                               int* info,
                                               int thin_UV) {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCgesvdj_bufferSize(handle,
                                            jobz,
                                            thin_UV,
                                            m,
                                            n,
                                            reinterpret_cast<cuComplex*>(A),
                                            lda,
                                            S,
                                            reinterpret_cast<cuComplex*>(U),
                                            ldu,
                                            reinterpret_cast<cuComplex*>(V),
                                            ldt,
                                            &lwork,
                                            gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(cuComplex),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuComplex* workspace_ptr = reinterpret_cast<cuComplex*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgesvdj(
        handle,
        jobz,
        thin_UV,
        m,
        n,
        reinterpret_cast<cuComplex*>(A + stride_A * i),
        lda,
        S + k * i,
        reinterpret_cast<cuComplex*>(U + stride_U * i),
        ldu,
        reinterpret_cast<cuComplex*>(V + stride_V * i),
        ldt,
        workspace_ptr,
        lwork,
        info,
        gesvdj_params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatched<phi::dtype::complex<double>>(const phi::GPUContext& dev_ctx,
                                                int batchSize,
                                                int m,
                                                int n,
                                                int k,
                                                phi::dtype::complex<double>* A,
                                                phi::dtype::complex<double>* U,
                                                phi::dtype::complex<double>* V,
                                                double* S,
                                                int* info,
                                                int thin_UV) {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgesvdj_bufferSize(
      handle,
      jobz,
      thin_UV,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      S,
      reinterpret_cast<cuDoubleComplex*>(U),
      ldu,
      reinterpret_cast<cuDoubleComplex*>(V),
      ldt,
      &lwork,
      gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(cuDoubleComplex),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuDoubleComplex* workspace_ptr =
      reinterpret_cast<cuDoubleComplex*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgesvdj(
        handle,
        jobz,
        thin_UV,
        m,
        n,
        reinterpret_cast<cuDoubleComplex*>(A + stride_A * i),
        lda,
        S + k * i,
        reinterpret_cast<cuDoubleComplex*>(U + stride_U * i),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V + stride_V * i),
        ldt,
        workspace_ptr,
        lwork,
        info,
        gesvdj_params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void SyevjBatched<float>(const phi::GPUContext& dev_ctx,
                         int batchSize,
                         int n,
                         float* A,
                         float* W,
                         int* info) {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  // matrix is saved as column-major in cusolver.
  // numpy and torch use lower triangle to compute eigenvalues, so here use
  // upper triangle
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(float),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj(handle,
                                                         jobz,
                                                         uplo,
                                                         n,
                                                         A + stride_A * i,
                                                         lda,
                                                         W + n * i,
                                                         workspace_ptr,
                                                         lwork,
                                                         info,
                                                         params));

    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void SyevjBatched<double>(const phi::GPUContext& dev_ctx,
                          int batchSize,
                          int n,
                          double* A,
                          double* W,
                          int* info) {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  upper triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(double),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevj(handle,
                                                         jobz,
                                                         uplo,
                                                         n,
                                                         A + stride_A * i,
                                                         lda,
                                                         W + n * i,
                                                         workspace_ptr,
                                                         lwork,
                                                         info,
                                                         params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void SyevjBatched<phi::dtype::complex<float>>(const phi::GPUContext& dev_ctx,
                                              int batchSize,
                                              int n,
                                              phi::dtype::complex<float>* A,
                                              float* W,
                                              int* info) {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  upper triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCheevj_bufferSize(handle,
                                           jobz,
                                           uplo,
                                           n,
                                           reinterpret_cast<cuComplex*>(A),
                                           lda,
                                           W,
                                           &lwork,
                                           params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(cuComplex),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuComplex* workspace_ptr = reinterpret_cast<cuComplex*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCheevj(
        handle,
        jobz,
        uplo,
        n,
        reinterpret_cast<cuComplex*>(A + stride_A * i),
        lda,
        W + n * i,
        workspace_ptr,
        lwork,
        info,
        params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void SyevjBatched<phi::dtype::complex<double>>(const phi::GPUContext& dev_ctx,
                                               int batchSize,
                                               int n,
                                               phi::dtype::complex<double>* A,
                                               double* W,
                                               int* info) {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  upper triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZheevj_bufferSize(
      handle,
      jobz,
      uplo,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      W,
      &lwork,
      params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(cuDoubleComplex),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuDoubleComplex* workspace_ptr =
      reinterpret_cast<cuDoubleComplex*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZheevj(
        handle,
        jobz,
        uplo,
        n,
        reinterpret_cast<cuDoubleComplex*>(A + stride_A * i),
        lda,
        W + n * i,
        workspace_ptr,
        lwork,
        info,
        params));
    int error_info;
    memory_utils::Copy(phi::CPUPlace(),
                       &error_info,
                       dev_ctx.GetPlace(),
                       info,
                       sizeof(int),
                       dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <typename T, typename Context>
void MatrixRankTolKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& atol_tensor,
                         bool use_default_tol,
                         bool hermitian,
                         DenseTensor* out) {
  using RealType = phi::dtype::Real<T>;
  auto* x_data = x.data<T>();
  dev_ctx.template Alloc<int64_t>(out);

  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  PADDLE_ENFORCE_NE(rows,
                    0,
                    common::errors::InvalidArgument(
                        "The input Tensor x's shape[-2] should not "
                        "be 0, but received %s now.",
                        dim_x));
  PADDLE_ENFORCE_NE(cols,
                    0,
                    common::errors::InvalidArgument(
                        "The input Tensor x's shape[-1] should not "
                        "be 0, but received %s now.",
                        dim_x));
  if (x.numel() == 0) {
    std::vector<int64_t> out_dims_vec(dim_x.size() - 2);
    for (int i = 0; i < dim_x.size() - 2; ++i) {
      out_dims_vec[i] = dim_x[i];
    }
    out->Resize(phi::make_ddim(out_dims_vec));
    dev_ctx.template Alloc<int64_t>(out);
    return;
  }
  int k = std::min(rows, cols);
  auto numel = x.numel();
  int batches = numel / (rows * cols);

  RealType rtol_T = 0;
  if (use_default_tol) {
    rtol_T = std::numeric_limits<RealType>::epsilon() * std::max(rows, cols);
  }

  // Must Copy X once, because the gesvdj will destroy the content when exit.
  DenseTensor x_tmp;
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &x_tmp);
  auto info = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int) * batches,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* info_ptr = reinterpret_cast<int*>(info->ptr());

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<RealType>(&eigenvalue_tensor);

  if (hermitian) {
    SyevjBatched<T>(
        dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data, info_ptr);

    phi::AbsKernel<RealType, Context>(
        dev_ctx, eigenvalue_tensor, &eigenvalue_tensor);

  } else {
    DenseTensor U, VH;
    U.Resize(detail::GetUDDim(dim_x, k));
    VH.Resize(detail::GetVHDDim(dim_x, k));
    auto* u_data = dev_ctx.template Alloc<T>(&U);
    auto* vh_data = dev_ctx.template Alloc<T>(&VH);
    GesvdjBatched<T>(dev_ctx,
                     batches,
                     cols,
                     rows,
                     k,
                     x_tmp.data<T>(),
                     vh_data,
                     u_data,
                     eigenvalue_data,
                     info_ptr,
                     1);
  }

  DenseTensor max_eigenvalue_tensor;
  dev_ctx.template Alloc<RealType>(&max_eigenvalue_tensor);
  max_eigenvalue_tensor.Resize(detail::RemoveLastDim(eigenvalue_tensor.dims()));

  phi::MaxKernel<RealType, Context>(dev_ctx,
                                    eigenvalue_tensor,
                                    phi::IntArray({-1}),
                                    false,
                                    &max_eigenvalue_tensor);

  DenseTensor rtol_tensor = phi::Scale<RealType, Context>(
      dev_ctx, max_eigenvalue_tensor, rtol_T, 0.0f, false);

  DenseTensor atol_tensor_real;
  if (atol_tensor.dtype() == phi::DataType::COMPLEX64 ||
      atol_tensor.dtype() == phi::DataType::COMPLEX128) {
    atol_tensor_real = phi::Real<T, Context>(dev_ctx, atol_tensor);
  } else {
    atol_tensor_real = atol_tensor;
  }
  DenseTensor tol_tensor;
  tol_tensor.Resize(dim_out);
  dev_ctx.template Alloc<RealType>(&tol_tensor);

  funcs::ElementwiseCompute<GreaterElementFunctor<RealType>, RealType>(
      dev_ctx,
      atol_tensor_real,
      rtol_tensor,
      GreaterElementFunctor<RealType>(),
      &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<int64_t>(&compare_result);

  funcs::ElementwiseCompute<funcs::GreaterThanFunctor<RealType, int64_t>,
                            RealType,
                            int64_t>(
      dev_ctx,
      eigenvalue_tensor,
      tol_tensor,
      funcs::GreaterThanFunctor<RealType, int64_t>(),
      &compare_result);

  phi::SumKernel<int64_t>(dev_ctx,
                          compare_result,
                          std::vector<int64_t>{-1},
                          compare_result.dtype(),
                          false,
                          out);
}

template <typename T, typename Context>
void MatrixRankAtolRtolKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& atol,
                              const paddle::optional<DenseTensor>& rtol,
                              bool hermitian,
                              DenseTensor* out) {
  using RealType = phi::dtype::Real<T>;
  auto* x_data = x.data<T>();
  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  PADDLE_ENFORCE_NE(
      rows,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-2] should not "
                              "be 0, but received %s now.",
                              dim_x));
  PADDLE_ENFORCE_NE(
      cols,
      0,
      errors::InvalidArgument("The input Tensor x's shape[-1] should not "
                              "be 0, but received %s now.",
                              dim_x));
  if (x.numel() == 0) {
    std::vector<int64_t> out_dims_vec(dim_x.size() - 2);
    for (int i = 0; i < dim_x.size() - 2; ++i) {
      out_dims_vec[i] = dim_x[i];
    }
    out->Resize(phi::make_ddim(out_dims_vec));
    dev_ctx.template Alloc<int64_t>(out);
    return;
  }
  dev_ctx.template Alloc<int64_t>(out);
  int k = std::min(rows, cols);
  auto numel = x.numel();
  int batches = numel / (rows * cols);

  // Must Copy X once, because the gesvdj will destroy the content when exit.
  DenseTensor x_tmp;
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, &x_tmp);
  auto info = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      sizeof(int) * batches,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* info_ptr = reinterpret_cast<int*>(info->ptr());

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<RealType>(&eigenvalue_tensor);

  if (hermitian) {
    SyevjBatched<T>(
        dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data, info_ptr);

    phi::AbsKernel<RealType, Context>(
        dev_ctx, eigenvalue_tensor, &eigenvalue_tensor);

  } else {
    DenseTensor U, VH;
    U.Resize(detail::GetUDDim(dim_x, k));
    VH.Resize(detail::GetVHDDim(dim_x, k));
    auto* u_data = dev_ctx.template Alloc<T>(&U);
    auto* vh_data = dev_ctx.template Alloc<T>(&VH);
    GesvdjBatched<T>(dev_ctx,
                     batches,
                     cols,
                     rows,
                     k,
                     x_tmp.data<T>(),
                     vh_data,
                     u_data,
                     eigenvalue_data,
                     info_ptr,
                     1);
  }

  DenseTensor max_eigenvalue_tensor;
  dev_ctx.template Alloc<RealType>(&max_eigenvalue_tensor);
  max_eigenvalue_tensor.Resize(detail::RemoveLastDim(eigenvalue_tensor.dims()));

  phi::MaxKernel<RealType, Context>(dev_ctx,
                                    eigenvalue_tensor,
                                    phi::IntArray({-1}),
                                    false,
                                    &max_eigenvalue_tensor);

  DenseTensor atol_tensor;
  if (atol.dtype() == phi::DataType::COMPLEX64 ||
      atol.dtype() == phi::DataType::COMPLEX128) {
    atol_tensor = phi::Real<T, Context>(dev_ctx, atol);
  } else {
    atol_tensor = atol;
  }
  DenseTensor tol_tensor;
  tol_tensor.Resize(dim_out);
  dev_ctx.template Alloc<RealType>(&tol_tensor);

  if (rtol) {
    DenseTensor rtol_tensor = *rtol;
    if (rtol_tensor.dtype() == phi::DataType::COMPLEX64 ||
        rtol_tensor.dtype() == phi::DataType::COMPLEX128) {
      rtol_tensor = phi::Real<T, Context>(dev_ctx, *rtol);
    }
    DenseTensor tmp_rtol_tensor;
    tmp_rtol_tensor =
        phi::Multiply<RealType>(dev_ctx, rtol_tensor, max_eigenvalue_tensor);
    funcs::ElementwiseCompute<GreaterElementFunctor<RealType>, RealType>(
        dev_ctx,
        atol_tensor,
        tmp_rtol_tensor,
        GreaterElementFunctor<RealType>(),
        &tol_tensor);
  } else {
    // when `rtol` is specified to be None in py api
    // use rtol=eps*max(m, n) only if `atol` is passed with value 0.0, else use
    // rtol=0.0
    RealType rtol_T =
        std::numeric_limits<RealType>::epsilon() * std::max(rows, cols);

    DenseTensor default_rtol_tensor = phi::Scale<RealType, Context>(
        dev_ctx, max_eigenvalue_tensor, rtol_T, 0.0f, false);

    DenseTensor zero_tensor;
    zero_tensor = phi::FullLike<RealType, Context>(
        dev_ctx, default_rtol_tensor, static_cast<RealType>(0.0));

    DenseTensor atol_compare_result;
    atol_compare_result.Resize(default_rtol_tensor.dims());
    phi::EqualKernel<RealType, Context>(
        dev_ctx, atol_tensor, zero_tensor, &atol_compare_result);

    DenseTensor selected_rtol_tensor;
    selected_rtol_tensor.Resize(default_rtol_tensor.dims());
    phi::WhereKernel<RealType, Context>(dev_ctx,
                                        atol_compare_result,
                                        default_rtol_tensor,
                                        zero_tensor,
                                        &selected_rtol_tensor);
    funcs::ElementwiseCompute<GreaterElementFunctor<RealType>, RealType>(
        dev_ctx,
        atol_tensor,
        selected_rtol_tensor,
        GreaterElementFunctor<RealType>(),
        &tol_tensor);
  }

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<int64_t>(&compare_result);

  funcs::ElementwiseCompute<funcs::GreaterThanFunctor<RealType, int64_t>,
                            RealType,
                            int64_t>(
      dev_ctx,
      eigenvalue_tensor,
      tol_tensor,
      funcs::GreaterThanFunctor<RealType, int64_t>(),
      &compare_result);

  phi::SumKernel<int64_t>(dev_ctx,
                          compare_result,
                          std::vector<int64_t>{-1},
                          compare_result.dtype(),
                          false,
                          out);
}
}  // namespace phi

PD_REGISTER_KERNEL(matrix_rank_tol,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankTolKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

PD_REGISTER_KERNEL(matrix_rank_atol_rtol,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankAtolRtolKernel,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}

#endif  // not PADDLE_WITH_HIP
