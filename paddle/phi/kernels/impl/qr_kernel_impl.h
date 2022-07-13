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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename DeviceContext, typename T>
void BatchedGeqrf(const DeviceContext& dev_ctx,
                  int batch_size,
                  int m,
                  int n,
                  T* a,
                  int lda,
                  T* tau,
                  int a_stride,
                  int tau_stride);

template <typename DeviceContext, typename T>
void BatchedOrgqr(const DeviceContext& dev_ctx,
                  int batch_size,
                  int m,
                  int n,
                  int k,
                  T* a,
                  int lda,
                  T* tau,
                  int a_stride,
                  int tau_stride);

template <>
void BatchedGeqrf<GPUContext, float>(const GPUContext& dev_ctx,
                                     int batch_size,
                                     int m,
                                     int n,
                                     float* a,
                                     int lda,
                                     float* tau,
                                     int a_stride,
                                     int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusolverDnSgeqrf_bufferSize(
          handle, m, n, a, lda, &lwork));

  DenseTensor* workspace = new DenseTensor();
  workspace->Resize(make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(workspace);

  DenseTensor* info = new DenseTensor();
  info->Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cusolverDnSgeqrf(handle,
                                                    m,
                                                    n,
                                                    a_working_ptr,
                                                    lda,
                                                    tau_working_ptr,
                                                    workspace_ptr,
                                                    lwork,
                                                    info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    paddle::memory::Copy(phi::CPUPlace(),
                         &info_h,
                         dev_ctx.GetPlace(),
                         info_d,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedGeqrf<GPUContext, double>(const GPUContext& dev_ctx,
                                      int batch_size,
                                      int m,
                                      int n,
                                      double* a,
                                      int lda,
                                      double* tau,
                                      int a_stride,
                                      int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusolverDnDgeqrf_bufferSize(
          handle, m, n, a, lda, &lwork));

  DenseTensor* workspace = new DenseTensor();
  workspace->Resize(make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(workspace);

  DenseTensor* info = new DenseTensor();
  info->Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cusolverDnDgeqrf(handle,
                                                    m,
                                                    n,
                                                    a_working_ptr,
                                                    lda,
                                                    tau_working_ptr,
                                                    workspace_ptr,
                                                    lwork,
                                                    info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    paddle::memory::Copy(phi::CPUPlace(),
                         &info_h,
                         dev_ctx.GetPlace(),
                         info_d,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, float>(const GPUContext& dev_ctx,
                                     int batch_size,
                                     int m,
                                     int n,
                                     int k,
                                     float* a,
                                     int lda,
                                     float* tau,
                                     int a_stride,
                                     int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusolverDnSorgqr_bufferSize(
          handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor* workspace = new DenseTensor();
  workspace->Resize(make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(workspace);

  DenseTensor* info = new DenseTensor();
  info->Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cusolverDnSorgqr(handle,
                                                    m,
                                                    n,
                                                    k,
                                                    a_working_ptr,
                                                    lda,
                                                    tau_working_ptr,
                                                    workspace_ptr,
                                                    lwork,
                                                    info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    paddle::memory::Copy(phi::CPUPlace(),
                         &info_h,
                         dev_ctx.GetPlace(),
                         info_d,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<GPUContext, double>(const GPUContext& dev_ctx,
                                      int batch_size,
                                      int m,
                                      int n,
                                      int k,
                                      double* a,
                                      int lda,
                                      double* tau,
                                      int a_stride,
                                      int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      paddle::platform::dynload::cusolverDnDorgqr_bufferSize(
          handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor* workspace = new DenseTensor();
  workspace->Resize(make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(workspace);

  DenseTensor* info = new DenseTensor();
  info->Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(
        paddle::platform::dynload::cusolverDnDorgqr(handle,
                                                    m,
                                                    n,
                                                    k,
                                                    a_working_ptr,
                                                    lda,
                                                    tau_working_ptr,
                                                    workspace_ptr,
                                                    lwork,
                                                    info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    paddle::memory::Copy(phi::CPUPlace(),
                         &info_h,
                         dev_ctx.GetPlace(),
                         info_d,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

}  // namespace phi
