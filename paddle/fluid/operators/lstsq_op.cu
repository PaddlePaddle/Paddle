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

#include <string>
#include <vector>
#include "paddle/fluid/operators/lstsq_op.h"
#include "paddle/fluid/operators/qr_op.h"

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

    // Tensor input_x_trans = dito.Transpose(x);
    // auto *x_vector = input_x_trans.data<T>();

    auto x_dims = x.dims();
    auto y_dims = y.dims();
    int dim_size = x_dims.size();
    int m = x_dims[dim_size - 2];
    int n = x_dims[dim_size - 1];
    int m_y = y_dims[dim_size - 2];
    int n_y = y_dims[dim_size - 1];
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
                          size_t(batch_count * m_y * n_y * sizeof(T)));
    TensorCopy(x, context.GetPlace(), &new_x);
    TensorCopy(y, context.GetPlace(), &new_y);

    auto tmp_x = dito.Transpose(new_x);
    auto tmp_y = dito.Transpose(new_y);
    framework::TensorCopy(tmp_x, new_x.place(), &new_x);
    framework::TensorCopy(tmp_y, new_y.place(), &new_y);

    // Prepare tau
    auto tau_dims_vec = framework::vectorize<int>(x_dims);
    tau_dims_vec.pop_back();
    tau_dims_vec[tau_dims_vec.size() - 1] = min_mn;
    Tensor tau = dito.Fill(tau_dims_vec, 0);

    auto x_data = new_x.mutable_data<T>(context.GetPlace());
    auto y_data = new_y.mutable_data<T>(context.GetPlace());
    auto tau_data = tau.mutable_data<T>(context.GetPlace());

    // step 1, compute QR factorization using geqrf
    BatchedGeqrf<T>(dev_ctx, batch_count, m, n, x_data, m, tau_data, x_stride,
                    tau_stride);

    // Step 2, B <- Q^H B
    BatchedOrmqr<T>(dev_ctx, true, true, batch_count, m, n, k, x_data, x_stride,
                    tau_data, tau_stride, y_data, y_stride);

    auto trans_r = dito.Transpose(new_x);
    auto trans_b = dito.Transpose(new_y);
    auto slice_r = dito.Slice(trans_r, {-2}, {0}, {min_mn});
    auto slice_b = dito.Slice(trans_b, {-2}, {0}, {min_mn});
    auto tmp_r = dito.TrilTriu(slice_r, 0, false);
    framework::TensorCopy(tmp_r, new_x.place(), &new_x);
    framework::TensorCopy(slice_b, new_y.place(), &new_y);

    // Step 3, solve R X = B
    triangular_solve<DeviceContext, T>(dev_ctx, new_x, new_y, solution, true,
                                       false, false);
  }
};

template <>
void BatchedOrmqr<float>(const platform::CUDADeviceContext& dev_ctx, bool left,
                         bool transpose, int batch_size, int m, int n, int k,
                         float* a, int a_stride, float* tau, int tau_stride,
                         float* other, int other_stride) {
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
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSormqr(
        handle, side, trans, m, n, k, a_working_ptr, lda, tau_working_ptr,
        other_working_ptr, ldc, workspace_ptr, lwork, info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 info_d, sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

template <>
void BatchedOrmqr<double>(const platform::CUDADeviceContext& dev_ctx, bool left,
                          bool transpose, int batch_size, int m, int n, int k,
                          double* a, int a_stride, double* tau, int tau_stride,
                          double* other, int other_stride) {
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
    // compute orggr
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDormqr(
        handle, side, trans, m, n, k, a_working_ptr, lda, tau_working_ptr,
        other_working_ptr, ldc, workspace_ptr, lwork, info_d));
    // Do we need synchronized here?
    // check the error info
    int info_h;
    memory::Copy(platform::CPUPlace(), &info_h,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()),
                 info_d, sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        info_h, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver QR is not zero. [%d]", i, info_h));
  }
}

}  // namespace operators
}  // namespace paddle

// using complex64 = paddle::platform::complex<float>;
// using complex128 = paddle::platform::complex<double>;

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    lstsq, ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, double>);  //,
// ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, complex64>,
// ops::LstsqCUDAKernel<paddle::platform::CUDADeviceContext, complex128>);