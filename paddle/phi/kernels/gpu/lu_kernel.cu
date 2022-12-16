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

#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/impl/lu_kernel_impl.h"
#include "paddle/phi/kernels/lu_kernel.h"

namespace phi {

template <typename T>
void cusolver_bufferSize(const cusolverDnHandle_t& cusolverH,
                         int m,
                         int n,
                         T* d_A,
                         int lda,
                         int* lwork);
template <typename T>
void cusolver_getrf(const cusolverDnHandle_t& cusolverH,
                    int m,
                    int n,
                    T* d_A,
                    int lda,
                    T* d_work,
                    int* d_Ipiv,
                    int* d_info);

template <>
void cusolver_bufferSize<float>(const cusolverDnHandle_t& cusolverH,
                                int m,
                                int n,
                                float* d_A,
                                int lda,
                                int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnSgetrf_bufferSize(cusolverH, m, n, d_A, lda, lwork));
}

template <>
void cusolver_bufferSize<double>(const cusolverDnHandle_t& cusolverH,
                                 int m,
                                 int n,
                                 double* d_A,
                                 int lda,
                                 int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDgetrf_bufferSize(cusolverH, m, n, d_A, lda, lwork));
}

template <>
void cusolver_getrf<float>(const cusolverDnHandle_t& cusolverH,
                           int m,
                           int n,
                           float* d_A,
                           int lda,
                           float* d_work,
                           int* d_Ipiv,
                           int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgetrf(
      cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
}

template <>
void cusolver_getrf<double>(const cusolverDnHandle_t& cusolverH,
                            int m,
                            int n,
                            double* d_A,
                            int lda,
                            double* d_work,
                            int* d_Ipiv,
                            int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgetrf(
      cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
}

template <typename T, typename Context>
void lu_decomposed_kernel(const Context& dev_ctx,
                          int m,
                          int n,
                          T* d_A,
                          int lda,
                          int* d_Ipiv,
                          int* d_info) {
  /* step 1: get cusolver handle*/
  auto cusolverH = dev_ctx.cusolver_dn_handle();

  /* step 2: query working space of getrf */
  int lwork;
  cusolver_bufferSize(cusolverH, m, n, d_A, lda, &lwork);

  auto work_buff = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      lwork * sizeof(T),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  T* d_work = reinterpret_cast<T*>(work_buff->ptr());

  /* step 3: LU factorization */
  if (d_Ipiv) {
    cusolver_getrf(cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info);
  } else {
    cusolver_getrf(cusolverH, m, n, d_A, lda, d_work, NULL, d_info);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
}

template <typename T, typename Context>
void LUKernel(const Context& dev_ctx,
              const DenseTensor& x,
              bool pivot,
              DenseTensor* out,
              DenseTensor* pivots,
              DenseTensor* infos) {
#ifdef __HIPCC__
  const int64_t kMaxBlockDim = 256;
#else
  const int64_t kMaxBlockDim = 512;
#endif

  *out = Transpose2DTo6D<Context, T>(dev_ctx, x);

  auto outdims = out->dims();
  auto outrank = outdims.size();

  int m = static_cast<int>(outdims[outrank - 1]);
  int n = static_cast<int>(outdims[outrank - 2]);
  int lda = std::max(1, m);
  if (pivot) {
    auto ipiv_dims = phi::slice_ddim(outdims, 0, outrank - 1);
    ipiv_dims[outrank - 2] = std::min(m, n);
    pivots->Resize(ipiv_dims);
  }
  dev_ctx.template Alloc<int>(pivots);
  auto ipiv_data = pivots->data<int>();

  auto info_dims = phi::slice_ddim(outdims, 0, outrank - 2);
  if (info_dims.size() == 0) {
    info_dims = phi::make_ddim({1});
  }
  infos->Resize(info_dims);
  dev_ctx.template Alloc<int>(infos);
  auto info_data = infos->data<int>();

  auto batchsize = product(info_dims);
  batchsize = std::max(static_cast<int>(batchsize), 1);
  dev_ctx.template Alloc<T>(out);
  auto out_data = out->data<T>();
  for (int b = 0; b < batchsize; b++) {
    auto out_data_item = &out_data[b * m * n];
    int* info_data_item = &info_data[b];
    if (pivot) {
      auto ipiv_data_item = &ipiv_data[b * std::min(m, n)];
      lu_decomposed_kernel(
          dev_ctx, m, n, out_data_item, lda, ipiv_data_item, info_data_item);
    } else {
      lu_decomposed_kernel(
          dev_ctx, m, n, out_data_item, lda, NULL, info_data_item);
    }
  }
  *out = Transpose2DTo6D<Context, T>(dev_ctx, *out);
}

}  // namespace phi

PD_REGISTER_KERNEL(lu,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::LUKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
