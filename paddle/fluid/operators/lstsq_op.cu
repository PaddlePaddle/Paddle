// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <vector>
#include "paddle/fluid/operators/lstsq_op.h"
#include "paddle/fluid/operators/qr_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename DeviceContext, typename T>
class LstsqCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor& x = *context.Input<Tensor>("X");
    const Tensor& y = *context.Input<Tensor>("Y");
    auto* solution = context.Output<Tensor>("Solution");

    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(context);
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    auto x_dims = x.dims();
    auto y_dims = y.dims();
    int dim_size = x_dims.size();
    int m = x_dims[dim_size - 2];
    int n = x_dims[dim_size - 1];
    int nrhs = y_dims[dim_size - 1];
    int min_mn = std::min(m, n);
    int max_mn = std::max(m, n);
    int k = min_mn;

    int x_stride = MatrixStride(x);
    int y_stride = MatrixStride(y);
    int tau_stride = min_mn;
    int batch_count = BatchCount(x);

    Tensor new_x, new_y;
    new_x.mutable_data<T>(context.GetPlace(),
                          size_t(batch_count * m * n * sizeof(T)));
    new_y.mutable_data<T>(context.GetPlace(),
                          size_t(batch_count * m * nrhs * sizeof(T)));
    framework::TensorCopy(x, context.GetPlace(), &new_x);
    framework::TensorCopy(y, context.GetPlace(), &new_y);

    // Prepare tau
    auto tau_dims_vec = framework::vectorize<int>(x_dims);
    tau_dims_vec.pop_back();
    tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
    Tensor tau = dito.Fill(tau_dims_vec, 0);
    auto tau_data = tau.mutable_data<T>(context.GetPlace());

    if (m >= n) {
      Tensor tmp_x = dito.Transpose(new_x);
      Tensor tmp_y = dito.Transpose(new_y);
      auto x_data = tmp_x.mutable_data<T>(context.GetPlace());
      auto y_data = tmp_y.mutable_data<T>(context.GetPlace());

      // step 1, compute QR factorization using geqrf
      BatchedGeqrf<DeviceContext, T>(dev_ctx, batch_count, m, n, x_data, m,
                                     tau_data, x_stride, tau_stride);

      // Step 2, Y <- Q^H Y
      BatchedOrmqr<DeviceContext, T>(dev_ctx, true, true, batch_count, m, n, k,
                                     x_data, x_stride, tau_data, tau_stride,
                                     y_data, y_stride);

      Tensor trans_r = dito.Transpose(tmp_x);
      Tensor slice_r = dito.Slice(trans_r, {-2}, {0}, {min_mn});
      Tensor res_r = dito.TrilTriu(slice_r, 0, false);

      Tensor trans_y = dito.Transpose(tmp_y);
      Tensor slice_y = dito.Slice(trans_y, {-2}, {0}, {min_mn});

      // Step 3, solve R X = Y
      triangular_solve<DeviceContext, T>(dev_ctx, res_r, slice_y, solution,
                                         true, false, false);
    } else {
      auto x_data = new_x.mutable_data<T>(context.GetPlace());
      auto y_data = new_y.mutable_data<T>(context.GetPlace());

      // step 1, compute QR factorization using geqrf
      BatchedGeqrf<DeviceContext, T>(dev_ctx, batch_count, n, m, x_data, n,
                                     tau_data, x_stride, tau_stride);

      // Step 2, solve R^H Z = Y
      Tensor trans_r = dito.Transpose(new_x);
      triangular_solve<DeviceContext, T>(dev_ctx, trans_r, new_y, solution,
                                         true, true, false);

      // Step 3, X <- Q Z
      BatchedOrgqr<DeviceContext, T>(dev_ctx, batch_count, n, n, min_mn, x_data,
                                     n, tau_data, x_stride, tau_stride);
      Tensor trans_q = dito.Transpose(new_x);
      Tensor slice_q = dito.Slice(trans_q, {-1}, {0}, {m});
      Tensor solu_tensor = dito.Matmul(slice_q, *solution, false, false);
      framework::TensorCopy(solu_tensor, solution->place(), solution);
    }
  }
};

template <>
void BatchedOrmqr<platform::CUDADeviceContext, float>(
    const platform::CUDADeviceContext& dev_ctx, bool left, bool transpose,
    int batch_size, int m, int n, int k, float* a, int a_stride, float* tau,
    int tau_stride, float* other, int other_stride) {
  int lwork = 0;
  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = std::max<int>(1, left ? m : n);
  int ldc = std::max<int>(1, m);

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSormqr_bufferSize(
      handle, side, trans, m, n, k, a, lda, tau, other, ldc, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    float* a_working_ptr = &a[i * a_stride];
    float* tau_working_ptr = &tau[i * tau_stride];
    float* other_working_ptr = &other[i * other_stride];
    // compute ormgr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSormqr(
        handle, side, trans, m, n, k, a_working_ptr, lda, tau_working_ptr,
        other_working_ptr, ldc, workspace_ptr, lwork, info_d));

    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver info is not zero but [%d]", i, info_h));
  }
}

template <>
void BatchedOrmqr<platform::CUDADeviceContext, double>(
    const platform::CUDADeviceContext& dev_ctx, bool left, bool transpose,
    int batch_size, int m, int n, int k, double* a, int a_stride, double* tau,
    int tau_stride, double* other, int other_stride) {
  int lwork = 0;
  auto side = left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
  auto trans = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = std::max<int>(1, left ? m : n);
  int ldc = std::max<int>(1, m);

  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDormqr_bufferSize(
      handle, side, trans, m, n, k, a, lda, tau, other, ldc, &lwork));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  auto info = memory::Alloc(dev_ctx, sizeof(int));
  int* info_d = reinterpret_cast<int*>(info->ptr());

  for (int i = 0; i < batch_size; ++i) {
    double* a_working_ptr = &a[i * a_stride];
    double* tau_working_ptr = &tau[i * tau_stride];
    double* other_working_ptr = &other[i * other_stride];
    // compute ormgr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDormqr(
        handle, side, trans, m, n, k, a_working_ptr, lda, tau_working_ptr,
        other_working_ptr, ldc, workspace_ptr, lwork, info_d));

    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h, dev_ctx.GetPlace(), info_d,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver info is not zero but [%d]", i, info_h));
  }
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    lstsq, ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, double>);

#endif  // not PADDLE_WITH_HIP
