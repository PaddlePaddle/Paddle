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

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

template <class T>
static void GesvdjBatched(const phi::GPUContext& dev_ctx,
                          int batchSize,
                          int m,
                          int n,
                          int k,
                          T* A,
                          T* U,
                          T* V,
                          T* S,
                          int* info,
                          int thin_UV = 1);

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
  /* compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnSgesvdj_bufferSize(handle,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgesvdj(handle,
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
      phi::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
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
  /* compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      phi::dynload::cusolverDnDgesvdj_bufferSize(handle,
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgesvdj(handle,
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
      phi::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <typename T, typename Context>
void SvdKernel(const Context& dev_ctx,
               const DenseTensor& X,
               bool full_matrices,
               DenseTensor* U,
               DenseTensor* S,
               DenseTensor* VH) {
  auto& dims = X.dims();
  int batch_count = 1;
  for (int i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }
  int rank = dims.size();
  int m = dims[rank - 2];
  int n = dims[rank - 1];

  PADDLE_ENFORCE_LT(
      0,
      m,
      errors::InvalidArgument("The row of Input(X) should be greater than 0."));
  PADDLE_ENFORCE_LT(
      0,
      n,
      errors::InvalidArgument("The col of Input(X) should be greater than 0."));

  auto* u_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(U);
  auto* vh_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(VH);
  auto* s_data = dev_ctx.template Alloc<phi::dtype::Real<T>>(S);
  // NOTE:(@xiongkun03)
  // matrices are assumed to be stored in column-major order in cusolver
  // then view A as n x m and do A^T SVD, we can avoid transpose
  // Must Copy X once, because the gesvdj will change the origin input matrix
  DenseTensor x_tmp;
  Copy(dev_ctx, X, dev_ctx.GetPlace(), false, &x_tmp);
  auto info = Empty<int, Context>(dev_ctx, {batch_count});
  int* info_ptr = reinterpret_cast<int*>(info.data());

  GesvdjBatched<T>(dev_ctx,
                   batch_count,
                   n,
                   m,
                   std::min(m, n),
                   dev_ctx.template Alloc<T>(&x_tmp),
                   vh_data,
                   u_data,
                   s_data,
                   info_ptr,
                   !full_matrices);

  auto UT_dim = U->dims();
  std::swap(UT_dim[rank - 1], UT_dim[rank - 2]);  // Get the dim of UT_dim
  U->Resize(UT_dim);                              // U is entirely UT
  auto tmp_U = TransposeLast2Dim<T>(dev_ctx, *U);
  U->ShareDataWith(tmp_U);  // U becomse UT, aka VT;
}
}  // namespace phi

PD_REGISTER_KERNEL(svd,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::SvdKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
