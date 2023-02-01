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

#ifndef PADDLE_WITH_HIP  // HIP not support cusolver

#include <thrust/device_vector.h>
#include <algorithm>
#include <vector>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"
#include "paddle/phi/kernels/impl/qr_kernel_impl.h"
#include "paddle/phi/kernels/qr_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

namespace phi {

template <class T, class Context>
static DenseTensor Fill(const Context& ctx,
                        std::vector<int> shape,
                        float fill_value) {
  DenseTensor ret;
  ret.Resize(make_ddim(shape));
  ctx.template Alloc<T>(&ret);
  funcs::SetConstant<Context, T>()(ctx, &ret, T(fill_value));
  return ret;
}

template <typename T, typename Context>
void QrKernel(const Context& ctx,
              const DenseTensor& x,
              const std::string& mode,
              DenseTensor* q,
              DenseTensor* r) {
  bool compute_q;
  bool reduced_mode;
  std::tie(compute_q, reduced_mode) = phi::funcs::ParseQrMode(mode);
  auto numel = x.numel();
  PADDLE_ENFORCE_GT(
      numel, 0, errors::PreconditionNotMet("The input of QR is empty."));
  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  int m = x_dims[x_rank - 2];
  int n = x_dims[x_rank - 1];
  int min_mn = std::min(m, n);
  int k = reduced_mode ? min_mn : m;
  int batch_size = numel / (m * n);
  int qr_stride = m * n;
  int tau_stride = min_mn;

  if (compute_q) {
    ctx.template Alloc<phi::dtype::Real<T>>(
        q, batch_size * m * k * sizeof(phi::dtype::Real<T>));
  }
  ctx.template Alloc<phi::dtype::Real<T>>(
      r, batch_size * k * n * sizeof(phi::dtype::Real<T>));

  // Note: allocate temporary tensors because of lacking in-place operatios.
  // Prepare qr
  DenseTensor qr;
  ctx.template Alloc<phi::dtype::Real<T>>(
      &qr, size_t(batch_size * m * n * sizeof(phi::dtype::Real<T>)));
  // BatchedGeqrf performs computation in-place and 'qr' must be a copy of
  // input
  phi::Copy(ctx, x, ctx.GetPlace(), false, &qr);

  // Prepare tau
  auto tau_dims_vec = phi::vectorize<int>(x_dims);
  tau_dims_vec.pop_back();
  tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
  DenseTensor tau = Fill<T, Context>(ctx, tau_dims_vec, 0);

  // Transpose 'qr' to conform the column-major order
  auto tmp_qr = TransposeLast2Dim<T, Context>(ctx, qr);
  phi::Copy(ctx, tmp_qr, qr.place(), false, &qr);
  auto qr_data = ctx.template Alloc<phi::dtype::Real<T>>(&qr);
  auto tau_data = ctx.template Alloc<phi::dtype::Real<T>>(&tau);

  BatchedGeqrf<Context, T>(
      ctx, batch_size, m, n, qr_data, m, tau_data, qr_stride, tau_stride);

  if (reduced_mode) {
    auto trans_qr = TransposeLast2Dim<T, Context>(ctx, qr);
    auto sliced_qr = SliceKernel<T, Context>(
        ctx, trans_qr, {trans_qr.dims().size() - 2}, {0}, {min_mn}, {1}, {});
    auto tmp_r = TrilTriu<T, Context>(ctx, sliced_qr, 0, false);
    // Transpose 'tmp_r' to retore the original row-major order
    phi::Copy(ctx, tmp_r, r->place(), false, r);
  } else {
    auto trans_qr = TransposeLast2Dim<T, Context>(ctx, qr);
    auto tmp_r = TrilTriu<T, Context>(ctx, trans_qr, 0, false);
    // Transpose 'tmp_r' to retore the original row-major order
    phi::Copy(ctx, tmp_r, r->place(), false, r);
  }

  if (compute_q) {
    // Perform QRGQR for Q using the result from GEQRF
    // Transpose 'q' to retore the original row-major order
    if (reduced_mode) {
      BatchedOrgqr<Context, T>(ctx,
                               batch_size,
                               m,
                               min_mn,
                               min_mn,
                               qr_data,
                               m,
                               tau_data,
                               qr_stride,
                               tau_stride);
      auto trans_q = TransposeLast2Dim<T, Context>(ctx, qr);
      auto sliced_q = SliceKernel<T, Context>(
          ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {min_mn}, {1}, {});
      phi::Copy(ctx, sliced_q, q->place(), false, q);
    } else {
      if (m > n) {
        auto new_qr_dims_vec = phi::vectorize<int>(x_dims);
        new_qr_dims_vec[new_qr_dims_vec.size() - 1] = m;
        DenseTensor new_qr = Fill<T, Context>(ctx, new_qr_dims_vec, 0);
        auto new_qr_data = ctx.template Alloc<phi::dtype::Real<T>>(&new_qr);
        auto new_qr_stride = m * m;
        for (int i = 0; i < batch_size; ++i) {
          paddle::memory::Copy(ctx.GetPlace(),
                               (new_qr_data + i * new_qr_stride),
                               ctx.GetPlace(),
                               (qr_data + i * qr_stride),
                               qr_stride * sizeof(phi::dtype::Real<T>),
                               ctx.stream());
        }
        BatchedOrgqr<Context, T>(ctx,
                                 batch_size,
                                 m,
                                 m,
                                 min_mn,
                                 new_qr_data,
                                 m,
                                 tau_data,
                                 new_qr_stride,
                                 tau_stride);
        auto trans_q = TransposeLast2Dim<T, Context>(ctx, new_qr);
        phi::Copy(ctx, trans_q, q->place(), false, q);
      } else {
        BatchedOrgqr<Context, T>(ctx,
                                 batch_size,
                                 m,
                                 m,
                                 min_mn,
                                 qr_data,
                                 m,
                                 tau_data,
                                 qr_stride,
                                 tau_stride);
        auto trans_q = TransposeLast2Dim<T, Context>(ctx, qr);
        auto sliced_q = SliceKernel<T, Context>(
            ctx, trans_q, {trans_q.dims().size() - 1}, {0}, {m}, {1}, {});
        phi::Copy(ctx, sliced_q, q->place(), false, q);
      }
    }
  }
}

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
      phi::dynload::cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSgeqrf(handle,
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
      phi::dynload::cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDgeqrf(handle,
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
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(make_ddim({lwork}));
  float* workspace_ptr = dev_ctx.template Alloc<float>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnSorgqr(handle,
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
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));

  DenseTensor workspace = DenseTensor();
  workspace.Resize(make_ddim({lwork}));
  double* workspace_ptr = dev_ctx.template Alloc<double>(&workspace);

  DenseTensor info = DenseTensor();
  info.Resize(make_ddim({1}));
  int* info_d = dev_ctx.template Alloc<int>(&info);

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cusolverDnDorgqr(handle,
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

PD_REGISTER_KERNEL(qr,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::QrKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
