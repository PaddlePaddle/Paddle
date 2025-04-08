// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/svdvals_kernel.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace phi {

template <class T>
static void GesvdjBatchedSvdvals(const phi::GPUContext& dev_ctx,
                                 int batchSize,
                                 int m,
                                 int n,
                                 int k,
                                 T* A,
                                 T* S,
                                 int* info,
                                 int thin_UV = 0  // only compute UV
);

template <>
void GesvdjBatchedSvdvals<float>(const phi::GPUContext& dev_ctx,
                                 int batchSize,
                                 int m,
                                 int n,
                                 int k,
                                 float* A,
                                 float* S,
                                 int* info,
                                 int thin_UV) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = 1;
  int ldv = 1;
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
                                                 nullptr,
                                                 ldu,
                                                 nullptr,
                                                 ldv,
                                                 &lwork,
                                                 gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(float),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgesvdj(handle,
                                                               jobz,
                                                               thin_UV,
                                                               m,
                                                               n,
                                                               A + stride_A * i,
                                                               lda,
                                                               S + k * i,
                                                               nullptr,
                                                               ldu,
                                                               nullptr,
                                                               ldv,
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
void GesvdjBatchedSvdvals<double>(const phi::GPUContext& dev_ctx,
                                  int batchSize,
                                  int m,
                                  int n,
                                  int k,
                                  double* A,
                                  double* S,
                                  int* info,
                                  int thin_UV) {
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = 1;
  int ldv = 1;
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
                                                 nullptr,
                                                 ldu,
                                                 nullptr,
                                                 ldv,
                                                 &lwork,
                                                 gesvdj_params));
  auto workspace = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(double),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgesvdj(handle,
                                                               jobz,
                                                               thin_UV,
                                                               m,
                                                               n,
                                                               A + stride_A * i,
                                                               lda,
                                                               S + k * i,
                                                               nullptr,
                                                               ldu,
                                                               nullptr,
                                                               ldv,
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
void SvdvalsKernel(const Context& dev_ctx,
                   const DenseTensor& X,
                   DenseTensor* S) {
  auto& dims = X.dims();
  int rows = static_cast<int>(dims[dims.size() - 2]);
  int cols = static_cast<int>(dims[dims.size() - 1]);
  PADDLE_ENFORCE_GT(
      rows,
      0,
      common::errors::InvalidArgument("Rows of X must be greater than 0."));
  PADDLE_ENFORCE_GT(
      cols,
      0,
      common::errors::InvalidArgument("Cols of X must be greater than 0."));
  int k = std::min(rows, cols);
  int batches = 1;
  for (int i = 0; i < dims.size() - 2; i++) batches *= dims[i];
  PADDLE_ENFORCE_GT(
      batches,
      0,
      common::errors::InvalidArgument("Batch size must be greater than 0."));

  DDim S_dims;
  if (dims.size() <= 2) {
    S_dims = {k};
  } else {
    S_dims = {batches, k};
  }
  S->Resize(S_dims);
  auto* S_out = dev_ctx.template Alloc<phi::dtype::Real<T>>(S);

  auto info = Empty<int, Context>(dev_ctx, {batches});
  int* info_ptr = reinterpret_cast<int*>(info.data());

  DenseTensor x_tmp;
  Copy(dev_ctx, X, dev_ctx.GetPlace(), false, &x_tmp);

  GesvdjBatchedSvdvals<T>(dev_ctx,
                          batches,
                          cols,
                          rows,
                          k,
                          dev_ctx.template Alloc<T>(&x_tmp),
                          S_out,
                          info_ptr,
                          0);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    svdvals, GPU, ALL_LAYOUT, phi::SvdvalsKernel, float, double) {}

#endif  // not PADDLE_WITH_HIP
