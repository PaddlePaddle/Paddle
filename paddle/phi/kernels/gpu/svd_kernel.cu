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

#include "paddle/phi/kernels/svd_kernel.h"

#include <limits>

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

// Match PyTorch's gesvdj configuration for precision alignment.
// PyTorch sets tolerance to machine epsilon and max sweeps to 400
// (vs cuSOLVER defaults of epsilon*max(m,n) and 100).
constexpr int kGesvdjMaxSweeps = 400;

// cuSOLVER's gesvdjBatched only supports m,n <= 32.
// PyTorch uses gesvdjBatched for small matrices and gesvdj (looped) for larger.
constexpr int kGesvdjBatchedMaxDim = 32;

template <typename scalar_t>
static void ConfigureGesvdjParams(gesvdjInfo_t params) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnXgesvdjSetTolerance(
      params, std::numeric_limits<scalar_t>::epsilon()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnXgesvdjSetMaxSweeps(params, kGesvdjMaxSweeps));
}

template <typename scalar_t>
static void ConfigureGesvdjBatchedParams(gesvdjInfo_t params) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnXgesvdjSetTolerance(
      params, std::numeric_limits<scalar_t>::epsilon()));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnXgesvdjSetMaxSweeps(params, kGesvdjMaxSweeps));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnXgesvdjSetSortEig(params, 1));
}

// ============================================================================
// GesvdjLoop: non-batched gesvdj called in a loop (for m or n > 32)
// ============================================================================
template <class T>
static void GesvdjLoop(const GPUContext& dev_ctx,
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

template <>
void GesvdjLoop<float>(const GPUContext& dev_ctx,
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
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjParams<float>(gesvdj_params);
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
  for (int i = 0; i < batchSize; ++i) {
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
    memory_utils::Copy(CPUPlace(),
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
void GesvdjLoop<double>(const GPUContext& dev_ctx,
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
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjParams<double>(gesvdj_params);
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
    int error_info;
    memory_utils::Copy(CPUPlace(),
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
void GesvdjLoop<phi::complex64>(const GPUContext& dev_ctx,
                                int batchSize,
                                int m,
                                int n,
                                int k,
                                phi::complex64* A,
                                phi::complex64* U,
                                phi::complex64* V,
                                float* S,
                                int* info,
                                int thin_UV) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjParams<float>(gesvdj_params);
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
      lwork * sizeof(phi::complex64),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  phi::complex64* workspace_ptr =
      reinterpret_cast<phi::complex64*>(workspace->ptr());
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
        reinterpret_cast<float*>(S + k * i),
        reinterpret_cast<cuComplex*>(U + stride_U * i),
        ldu,
        reinterpret_cast<cuComplex*>(V + stride_V * i),
        ldt,
        reinterpret_cast<cuComplex*>(workspace_ptr),
        lwork,
        info,
        gesvdj_params));
    int error_info;
    memory_utils::Copy(CPUPlace(),
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
void GesvdjLoop<phi::complex128>(const GPUContext& dev_ctx,
                                 int batchSize,
                                 int m,
                                 int n,
                                 int k,
                                 phi::complex128* A,
                                 phi::complex128* U,
                                 phi::complex128* V,
                                 double* S,
                                 int* info,
                                 int thin_UV) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjParams<double>(gesvdj_params);
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
      lwork * sizeof(phi::complex128),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  phi::complex128* workspace_ptr =
      reinterpret_cast<phi::complex128*>(workspace->ptr());
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
        reinterpret_cast<double*>(S + k * i),
        reinterpret_cast<cuDoubleComplex*>(U + stride_U * i),
        ldu,
        reinterpret_cast<cuDoubleComplex*>(V + stride_V * i),
        ldt,
        reinterpret_cast<cuDoubleComplex*>(workspace_ptr),
        lwork,
        info,
        gesvdj_params));
    int error_info;
    memory_utils::Copy(CPUPlace(),
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

// ============================================================================
// GesvdjBatchedCuSOLVER: actual batched cuSOLVER call (for m,n <= 32)
// Matches PyTorch's dispatch logic for precision alignment.
// Note: gesvdjBatched always computes full U and V matrices.
// ============================================================================
template <class T>
static void GesvdjBatchedCuSOLVER(const GPUContext& dev_ctx,
                                  int batchSize,
                                  int m,
                                  int n,
                                  int k,
                                  T* A,
                                  T* U,
                                  T* V,
                                  phi::dtype::Real<T>* S,
                                  int* info);

template <>
void GesvdjBatchedCuSOLVER<float>(const GPUContext& dev_ctx,
                                  int batchSize,
                                  int m,
                                  int n,
                                  int k,
                                  float* A,
                                  float* U,
                                  float* V,
                                  float* S,
                                  int* info) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldv = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjBatchedParams<float>(gesvdj_params);
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnSgesvdjBatched_bufferSize(handle,
                                                   jobz,
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   S,
                                                   U,
                                                   ldu,
                                                   V,
                                                   ldv,
                                                   &lwork,
                                                   gesvdj_params,
                                                   batchSize));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(float),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgesvdjBatched(handle,
                                                               jobz,
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               S,
                                                               U,
                                                               ldu,
                                                               V,
                                                               ldv,
                                                               workspace_ptr,
                                                               lwork,
                                                               info,
                                                               gesvdj_params,
                                                               batchSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatchedCuSOLVER<double>(const GPUContext& dev_ctx,
                                   int batchSize,
                                   int m,
                                   int n,
                                   int k,
                                   double* A,
                                   double* U,
                                   double* V,
                                   double* S,
                                   int* info) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldv = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjBatchedParams<double>(gesvdj_params);
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDgesvdjBatched_bufferSize(handle,
                                                   jobz,
                                                   m,
                                                   n,
                                                   A,
                                                   lda,
                                                   S,
                                                   U,
                                                   ldu,
                                                   V,
                                                   ldv,
                                                   &lwork,
                                                   gesvdj_params,
                                                   batchSize));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(double),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgesvdjBatched(handle,
                                                               jobz,
                                                               m,
                                                               n,
                                                               A,
                                                               lda,
                                                               S,
                                                               U,
                                                               ldu,
                                                               V,
                                                               ldv,
                                                               workspace_ptr,
                                                               lwork,
                                                               info,
                                                               gesvdj_params,
                                                               batchSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatchedCuSOLVER<phi::complex64>(const GPUContext& dev_ctx,
                                           int batchSize,
                                           int m,
                                           int n,
                                           int k,
                                           phi::complex64* A,
                                           phi::complex64* U,
                                           phi::complex64* V,
                                           float* S,
                                           int* info) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldv = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjBatchedParams<float>(gesvdj_params);
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgesvdjBatched_bufferSize(
      handle,
      jobz,
      m,
      n,
      reinterpret_cast<cuComplex*>(A),
      lda,
      S,
      reinterpret_cast<cuComplex*>(U),
      ldu,
      reinterpret_cast<cuComplex*>(V),
      ldv,
      &lwork,
      gesvdj_params,
      batchSize));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(phi::complex64),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuComplex* workspace_ptr = reinterpret_cast<cuComplex*>(workspace->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCgesvdjBatched(handle,
                                        jobz,
                                        m,
                                        n,
                                        reinterpret_cast<cuComplex*>(A),
                                        lda,
                                        S,
                                        reinterpret_cast<cuComplex*>(U),
                                        ldu,
                                        reinterpret_cast<cuComplex*>(V),
                                        ldv,
                                        workspace_ptr,
                                        lwork,
                                        info,
                                        gesvdj_params,
                                        batchSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void GesvdjBatchedCuSOLVER<phi::complex128>(const GPUContext& dev_ctx,
                                            int batchSize,
                                            int m,
                                            int n,
                                            int k,
                                            phi::complex128* A,
                                            phi::complex128* U,
                                            phi::complex128* V,
                                            double* S,
                                            int* info) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldv = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  ConfigureGesvdjBatchedParams<double>(gesvdj_params);
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgesvdjBatched_bufferSize(
      handle,
      jobz,
      m,
      n,
      reinterpret_cast<cuDoubleComplex*>(A),
      lda,
      S,
      reinterpret_cast<cuDoubleComplex*>(U),
      ldu,
      reinterpret_cast<cuDoubleComplex*>(V),
      ldv,
      &lwork,
      gesvdj_params,
      batchSize));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(phi::complex128),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  cuDoubleComplex* workspace_ptr =
      reinterpret_cast<cuDoubleComplex*>(workspace->ptr());
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnZgesvdjBatched(handle,
                                        jobz,
                                        m,
                                        n,
                                        reinterpret_cast<cuDoubleComplex*>(A),
                                        lda,
                                        S,
                                        reinterpret_cast<cuDoubleComplex*>(U),
                                        ldu,
                                        reinterpret_cast<cuDoubleComplex*>(V),
                                        ldv,
                                        workspace_ptr,
                                        lwork,
                                        info,
                                        gesvdj_params,
                                        batchSize));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

// ============================================================================
// SvdKernel: main entry point
// Dispatches to GesvdjBatchedCuSOLVER for small matrices (m,n <= 32)
// and GesvdjLoop for larger matrices, matching PyTorch's dispatch logic.
// ============================================================================
template <typename T, typename Context>
void SvdKernel(const Context& dev_ctx,
               const DenseTensor& X,
               bool full_matrices,
               DenseTensor* U,
               DenseTensor* S,
               DenseTensor* VH) {
  if (X.numel() == 0) {
    dev_ctx.template Alloc<T>(U);
    dev_ctx.template Alloc<phi::dtype::Real<T>>(S);
    dev_ctx.template Alloc<T>(VH);
    return;
  }
  auto& dims = X.dims();
  int64_t batch_count64 = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count64 *= dims[i];
  }
  // TODO(large-tensor): cusolver batch_count not support int64
  PADDLE_ENFORCE_LE_INT_MAX(batch_count64, "batch_count");
  int batch_count = static_cast<int>(batch_count64);

  int rank = dims.size();
  int64_t m = dims[rank - 2];
  int64_t n = dims[rank - 1];
  // TODO(large-tensor): cusolver m/n not support int64
  PADDLE_ENFORCE_LE_INT_MAX(m, "m");
  PADDLE_ENFORCE_LE_INT_MAX(n, "n");
  int m_int = static_cast<int>(m);
  int n_int = static_cast<int>(n);

  auto* u_data = dev_ctx.template Alloc<T>(U);
  auto* vh_data = dev_ctx.template Alloc<T>(VH);
  auto* s_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(S);
  // NOTE:(@xiongkun03)
  // matrices are assumed to be stored in column-major order in cusolver
  // then view A as n x m and do A^T SVD, we can avoid transpose
  // Must Copy X once, because the gesvdj will change the origin input matrix
  DenseTensor x_tmp;
  Copy(dev_ctx, X, dev_ctx.GetPlace(), false, &x_tmp);
  auto info = Empty<int, Context>(dev_ctx, {batch_count});
  int* info_ptr = reinterpret_cast<int*>(info.data());

  // Note: we swap m and n because of column-major vs row-major layout
  // The actual matrix dimensions in cusolver are (n_int, m_int)
  int cusolver_m = n_int;  // cusolver sees n rows
  int cusolver_n = m_int;  // cusolver sees m columns
  int k = std::min(m_int, n_int);

  if (cusolver_m <= kGesvdjBatchedMaxDim &&
      cusolver_n <= kGesvdjBatchedMaxDim) {
    // Use cuSOLVER's actual batched API for small matrices.
    // gesvdjBatched always computes full U and V, so we need to handle
    // the thin_UV case ourselves.
    if (!full_matrices && m_int != n_int) {
      // Need full-size U/V buffers for the batched call, then copy thin
      // Allocate full-size temporary buffers
      // In cusolver's view: A is cusolver_m x cusolver_n
      // U is cusolver_m x cusolver_m, V is cusolver_n x cusolver_n
      int full_u_size = cusolver_m * cusolver_m;
      int full_v_size = cusolver_n * cusolver_n;
      auto u_buf = phi::memory_utils::Alloc(
          dev_ctx.GetPlace(),
          batch_count * full_u_size * sizeof(T),
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      auto v_buf = phi::memory_utils::Alloc(
          dev_ctx.GetPlace(),
          batch_count * full_v_size * sizeof(T),
          phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
      T* u_full = reinterpret_cast<T*>(u_buf->ptr());
      T* v_full = reinterpret_cast<T*>(v_buf->ptr());

      GesvdjBatchedCuSOLVER<T>(dev_ctx,
                               batch_count,
                               cusolver_m,
                               cusolver_n,
                               k,
                               dev_ctx.template Alloc<T>(&x_tmp),
                               u_full,
                               v_full,
                               s_data,
                               info_ptr);

      // Copy the thin parts back
      // vh_data corresponds to U in cusolver (it gets transposed later)
      // u_data corresponds to V in cusolver (it gets transposed later)
      // For thin SVD: we need k columns of U and k columns of V
      // cusolver U shape: cusolver_m x cusolver_m, we need cusolver_m x k
      // cusolver V shape: cusolver_n x cusolver_n, we need cusolver_n x k
      for (int b = 0; b < batch_count; ++b) {
        // Copy thin U (cusolver_m x k) from full U (cusolver_m x cusolver_m)
        // Both are column-major, so we can copy cusolver_m * k elements
        // (the first k columns are contiguous in memory)
        memory_utils::Copy(dev_ctx.GetPlace(),
                           vh_data + b * cusolver_m * k,
                           dev_ctx.GetPlace(),
                           u_full + b * full_u_size,
                           cusolver_m * k * sizeof(T),
                           dev_ctx.stream());
        // Copy thin V
        memory_utils::Copy(dev_ctx.GetPlace(),
                           u_data + b * cusolver_n * k,
                           dev_ctx.GetPlace(),
                           v_full + b * full_v_size,
                           cusolver_n * k * sizeof(T),
                           dev_ctx.stream());
      }
    } else {
      // full_matrices or m == n: output buffers match cuSOLVER expectation
      GesvdjBatchedCuSOLVER<T>(dev_ctx,
                               batch_count,
                               cusolver_m,
                               cusolver_n,
                               k,
                               dev_ctx.template Alloc<T>(&x_tmp),
                               vh_data,
                               u_data,
                               s_data,
                               info_ptr);
    }
  } else {
    // Large matrices: use non-batched gesvdj in a loop
    GesvdjLoop<T>(dev_ctx,
                  batch_count,
                  cusolver_m,
                  cusolver_n,
                  k,
                  dev_ctx.template Alloc<T>(&x_tmp),
                  vh_data,
                  u_data,
                  s_data,
                  info_ptr,
                  !full_matrices);
  }

  auto UT_dim = U->dims();
  std::swap(UT_dim[rank - 1], UT_dim[rank - 2]);  // Get the dim of UT_dim
  U->Resize(UT_dim);                              // U is entirely UT
  auto tmp_U = TransposeLast2Dim<T>(dev_ctx, Conj<T, Context>(dev_ctx, *U));
  U->ShareDataWith(tmp_U);  // U becomes UT, aka VT;
}
}  // namespace phi

PD_REGISTER_KERNEL(svd,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::SvdKernel,
                   float,
                   double,
                   phi::complex64,
                   phi::complex128) {}

#endif  // not PADDLE_WITH_HIP
