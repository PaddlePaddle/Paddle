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

#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/common/memory_utils.h"
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
void cusolver_bufferSize<dtype::complex<float>>(
    const cusolverDnHandle_t& cusolverH,
    int m,
    int n,
    dtype::complex<float>* d_A,
    int lda,
    int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCgetrf_bufferSize(
      cusolverH, m, n, reinterpret_cast<cuComplex*>(d_A), lda, lwork));
}

template <>
void cusolver_bufferSize<dtype::complex<double>>(
    const cusolverDnHandle_t& cusolverH,
    int m,
    int n,
    dtype::complex<double>* d_A,
    int lda,
    int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnZgetrf_bufferSize(
      cusolverH, m, n, reinterpret_cast<cuDoubleComplex*>(d_A), lda, lwork));
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

template <>
void cusolver_getrf<dtype::complex<float>>(const cusolverDnHandle_t& cusolverH,
                                           int m,
                                           int n,
                                           dtype::complex<float>* d_A,
                                           int lda,
                                           dtype::complex<float>* d_work,
                                           int* d_Ipiv,
                                           int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCgetrf(cusolverH,
                                m,
                                n,
                                reinterpret_cast<cuComplex*>(d_A),
                                lda,
                                reinterpret_cast<cuComplex*>(d_work),
                                d_Ipiv,
                                d_info));
}

template <>
void cusolver_getrf<dtype::complex<double>>(const cusolverDnHandle_t& cusolverH,
                                            int m,
                                            int n,
                                            dtype::complex<double>* d_A,
                                            int lda,
                                            dtype::complex<double>* d_work,
                                            int* d_Ipiv,
                                            int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnZgetrf(cusolverH,
                                m,
                                n,
                                reinterpret_cast<cuDoubleComplex*>(d_A),
                                lda,
                                reinterpret_cast<cuDoubleComplex*>(d_work),
                                d_Ipiv,
                                d_info));
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

  auto work_buff = phi::memory_utils::Alloc(
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
  // big tensor currently not supported
  PADDLE_ENFORCE_GE(
      x.dims().size(),
      2,
      ::common::errors::PreconditionNotMet(
          "Invalid input x dimensionality: %d (expected ≥2)", x.dims().size()));
  if (x.numel() == 0) {
    phi::Full<int, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(infos->dims())),
                            static_cast<int>(0),
                            infos);
    phi::Full<int, Context>(dev_ctx,
                            phi::IntArray(common::vectorize(pivots->dims())),
                            static_cast<int>(0),
                            pivots);
    phi::Full<T, Context>(dev_ctx,
                          phi::IntArray(common::vectorize(out->dims())),
                          static_cast<T>(0),
                          out);
    return;
  }
  int64_t largest_matrix = (1LL << 31) - 1;
  int64_t last = x.dims()[x.dims().size() - 1],
          second_last = x.dims()[x.dims().size() - 2];
  int64_t matrix_size = last * second_last;
  PADDLE_ENFORCE_LE(matrix_size,
                    largest_matrix,
                    ::common::errors::PreconditionNotMet(
                        "Matrix size too large for LU decomposition. Maximum "
                        "allowed size is 2 ^ 31 - 1 elements, but got %lld",
                        matrix_size));

  const int64_t kMaxBlockDim = 512;

  *out = Transpose2DTo6D<Context, T>(dev_ctx, x);

  auto outdims = out->dims();
  auto outrank = outdims.size();

  int m = static_cast<int>(outdims[outrank - 1]);
  int n = static_cast<int>(outdims[outrank - 2]);
  int lda = std::max(1, m);
  if (pivot) {
    auto ipiv_dims = common::slice_ddim(outdims, 0, outrank - 1);
    ipiv_dims[outrank - 2] = std::min(m, n);
    pivots->Resize(ipiv_dims);
  }
  dev_ctx.template Alloc<int>(pivots);
  auto ipiv_data = pivots->data<int>();

  auto info_dims = common::slice_ddim(outdims, 0, outrank - 2);
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
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}

#endif  // not PADDLE_WITH_HIP
