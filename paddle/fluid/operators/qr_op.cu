/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_WITH_HIP
// HIP not support cusolver

#include <thrust/device_vector.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/qr_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

// Reuse some helper functions from svd
#include "paddle/fluid/operators/svd_helper.h"

namespace paddle {
namespace operators {

template <typename T>
class QrGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool compute_q;
    bool reduced_mode;
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    const Tensor& x = *context.Input<Tensor>("X");
    Tensor& q = *context.Output<Tensor>("Q");
    Tensor& r = *context.Output<Tensor>("R");
    const std::string mode = context.Attr<std::string>("mode");
    std::tie(compute_q, reduced_mode) = _parse_qr_mode(mode);

    auto numel = x.numel();
    PADDLE_ENFORCE_GT(numel, 0, platform::errors::PreconditionNotMet(
                                    "The input of QR is empty."));
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
      q.mutable_data<math::Real<T>>(
          context.GetPlace(),
          size_t(batch_size * m * k * sizeof(math::Real<T>)));
    }
    r.mutable_data<math::Real<T>>(
        context.GetPlace(), size_t(batch_size * k * n * sizeof(math::Real<T>)));

    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(context);

    // Note: allocate temporary tensors because of lacking in-place operatios.
    // Prepare qr
    Tensor qr;
    qr.mutable_data<math::Real<T>>(
        context.GetPlace(), size_t(batch_size * m * n * sizeof(math::Real<T>)));
    // BatchedGeqrf performs computation in-place and 'qr' must be a copy of
    // input
    paddle::framework::TensorCopy(x, context.GetPlace(), &qr);

    // Prepare tau
    auto tau_dims_vec = framework::vectorize<int>(x_dims);
    tau_dims_vec.pop_back();
    tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
    Tensor tau = dito.Fill(tau_dims_vec, 0);

    // Transpose 'qr' to conform the column-major order
    auto tmp_qr = dito.Transpose(qr);
    framework::TensorCopy(tmp_qr, qr.place(), &qr);
    auto qr_data = qr.mutable_data<T>(context.GetPlace());
    auto tau_data = tau.mutable_data<T>(context.GetPlace());

    BatchedGeqrf<platform::CUDADeviceContext, T>(
        dev_ctx, batch_size, m, n, qr_data, m, tau_data, qr_stride, tau_stride);

    if (reduced_mode) {
      auto trans_qr = dito.Transpose(qr);
      auto sliced_qr = dito.Slice(trans_qr, {-2}, {0}, {min_mn});
      auto tmp_r = dito.TrilTriu(sliced_qr, 0, false);
      // Transpose 'tmp_r' to retore the original row-major order
      framework::TensorCopy(tmp_r, r.place(), &r);
    } else {
      auto trans_qr = dito.Transpose(qr);
      auto tmp_r = dito.TrilTriu(trans_qr, 0, false);
      // Transpose 'tmp_r' to retore the original row-major order
      framework::TensorCopy(tmp_r, r.place(), &r);
    }

    if (compute_q) {
      // Perform QRGQR for Q using the result from GEQRF
      // Transpose 'q' to retore the original row-major order
      if (reduced_mode) {
        BatchedOrgqr<platform::CUDADeviceContext, T>(
            dev_ctx, batch_size, m, min_mn, min_mn, qr_data, m, tau_data,
            qr_stride, tau_stride);
        auto trans_q = dito.Transpose(qr);
        auto sliced_q = dito.Slice(trans_q, {-1}, {0}, {min_mn});
        framework::TensorCopy(sliced_q, q.place(), &q);
      } else {
        if (m > n) {
          auto new_qr_dims_vec = framework::vectorize<int>(x_dims);
          new_qr_dims_vec[new_qr_dims_vec.size() - 1] = m;
          Tensor new_qr = dito.Fill(new_qr_dims_vec, 0);
          auto new_qr_data = new_qr.mutable_data<T>(context.GetPlace());
          auto new_qr_stride = m * m;
          for (int i = 0; i < batch_size; ++i) {
            memory::Copy(dev_ctx.GetPlace(), (new_qr_data + i * new_qr_stride),
                         dev_ctx.GetPlace(), (qr_data + i * qr_stride),
                         qr_stride * sizeof(math::Real<T>), dev_ctx.stream());
          }
          BatchedOrgqr<platform::CUDADeviceContext, T>(
              dev_ctx, batch_size, m, m, min_mn, new_qr_data, m, tau_data,
              new_qr_stride, tau_stride);
          auto trans_q = dito.Transpose(new_qr);
          framework::TensorCopy(trans_q, q.place(), &q);
        } else {
          BatchedOrgqr<platform::CUDADeviceContext, T>(
              dev_ctx, batch_size, m, m, min_mn, qr_data, m, tau_data,
              qr_stride, tau_stride);
          auto trans_q = dito.Transpose(qr);
          auto sliced_q = dito.Slice(trans_q, {-1}, {0}, {m});
          framework::TensorCopy(sliced_q, q.place(), &q);
        }
      }
    }
  }
};

template <>
void BatchedGeqrf<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& dev_ctx, int batch_size, int m, int n,
    float* a, int lda, float* tau, int a_stride, int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgeqrf_bufferSize(
      handle, m, n, a, lda, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgeqrf(
        handle, m, n, a_working_ptr, lda, tau_working_ptr, workspace_ptr, lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedGeqrf<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& dev_ctx, int batch_size, int m, int n,
    double* a, int lda, double* tau, int a_stride, int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgeqrf_bufferSize(
      handle, m, n, a, lda, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute geqrf
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgeqrf(
        handle, m, n, a_working_ptr, lda, tau_working_ptr, workspace_ptr, lwork,
        info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver geqrf is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& dev_ctx, int batch_size, int m, int n,
    int k, float* a, int lda, float* tau, int a_stride, int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSorgqr(
        handle, m, n, k, a_working_ptr, lda, tau_working_ptr, workspace_ptr,
        lwork, info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrgqr<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& dev_ctx, int batch_size, int m, int n,
    int k, double* a, int lda, double* tau, int a_stride, int tau_stride) {
  int lwork = 0;

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDorgqr_bufferSize(
      handle, m, n, k, a, lda, tau, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDorgqr(
        handle, m, n, k, a_working_ptr, lda, tau_working_ptr, workspace_ptr,
        lwork, info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(qr, ops::QrGPUKernel<float>, ops::QrGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    qr_grad, ops::QrGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::QrGradKernel<paddle::platform::CUDADeviceContext, double>);

#endif  // not PADDLE_WITH_HIP
