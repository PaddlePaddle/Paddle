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
#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/platform/dynload/cusolver.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
namespace detail {
DDim GetUDDim(const DDim& x_dim, int k) {
  auto x_vec = framework::vectorize(x_dim);
  x_vec[x_vec.size() - 1] = k;
  return framework::make_ddim(x_vec);
}

DDim GetVHDDim(const DDim& x_dim, int k) {
  auto x_vec = framework::vectorize(x_dim);
  x_vec[x_vec.size() - 2] = k;
  return framework::make_ddim(x_vec);
}
}  // namespace detail

template <typename T>
class MatrixRankGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<int64_t>(context.GetPlace());
    bool hermitian = context.Attr<bool>("hermitian");

    auto dim_x = x->dims();
    auto dim_out = out->dims();
    int rows = dim_x[dim_x.size() - 2];
    int cols = dim_x[dim_x.size() - 1];
    int k = std::min(rows, cols);
    auto numel = x->numel();
    int batches = numel / (rows * cols);

    bool use_default_tol = context.Attr<bool>("use_default_tol");
    const Tensor* atol_tensor = nullptr;
    Tensor temp_tensor;
    T rtol_T = 0;
    if (use_default_tol) {
      framework::TensorFromVector<T>(std::vector<T>{0},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
      rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    } else if (context.HasInput("TolTensor")) {
      atol_tensor = context.Input<Tensor>("TolTensor");
    } else {
      framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
    }

    // Must Copy X once, because the gesvdj will destory the content when exit.
    Tensor x_tmp;
    TensorCopy(*x, context.GetPlace(), &x_tmp);
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batches);
    int* info_ptr = reinterpret_cast<int*>(info->ptr());

    Tensor eigenvalue_tensor;
    auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
        detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
    if (hermitian) {
      SyevjBatched(dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data,
                   info_ptr);
      platform::ForRange<platform::CUDADeviceContext> for_range(
          dev_ctx, eigenvalue_tensor.numel());
      math::AbsFunctor<T> functor(eigenvalue_data, eigenvalue_data,
                                  eigenvalue_tensor.numel());
      for_range(functor);
    } else {
      Tensor U, VH;
      auto* u_data =
          U.mutable_data<T>(detail::GetUDDim(dim_x, k), context.GetPlace());
      auto* vh_data =
          VH.mutable_data<T>(detail::GetVHDDim(dim_x, k), context.GetPlace());
      GesvdjBatched(dev_ctx, batches, cols, rows, k, x_tmp.data<T>(), vh_data,
                    u_data, eigenvalue_data, info_ptr, 1);
    }

    auto dito_T =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(context);
    std::vector<int> max_eigenvalue_shape = framework::vectorize<int>(
        detail::RemoveLastDim(eigenvalue_tensor.dims()));
    Tensor max_eigenvalue_tensor =
        dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);
    Tensor temp_rtol_tensor;
    framework::TensorFromVector<T>(std::vector<T>{rtol_T},
                                   context.device_context(), &temp_rtol_tensor);
    Tensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);
    Tensor tol_tensor;
    tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
    ElementwiseComputeEx<GreaterElementFunctor<T>, platform::CUDADeviceContext,
                         T, T>(context, atol_tensor, &rtol_tensor, -1,
                               GreaterElementFunctor<T>(), &tol_tensor);

    tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

    Tensor compare_result;
    compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
                                         context.GetPlace());
    int axis = -1;
    ElementwiseComputeEx<GreaterThanFunctor<T>, platform::CUDADeviceContext, T,
                         int64_t>(context, &eigenvalue_tensor, &tol_tensor,
                                  axis, GreaterThanFunctor<T>(),
                                  &compare_result);
    auto dito_int =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 int64_t>(context);
    std::vector<int> result_shape = framework::vectorize<int>(dim_out);
    Tensor result = dito_int.ReduceSum(compare_result, result_shape);
    out->ShareDataWith(result);
  }

  void GesvdjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                     int m, int n, int k, T* A, T* U, T* V, T* S, int* info,
                     int thin_UV = 1) const;

  void SyevjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                    int n, T* A, T* W, int* info) const;
};

template <>
void MatrixRankGPUKernel<float>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, float* A, float* U, float* V, float* S, int* info,
    int thin_UV) const {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgesvdj(
        handle, jobz, thin_UV, m, n, A + stride_A * i, lda, S + k * i,
        U + stride_U * i, ldu, V + stride_V * i, ldt, workspace_ptr, lwork,
        info, gesvdj_params));
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void MatrixRankGPUKernel<double>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, double* A, double* U, double* V, double* S, int* info,
    int thin_UV) const {
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgesvdj(
        handle, jobz, thin_UV, m, n, A + stride_A * i, lda, S + k * i,
        U + stride_U * i, ldu, V + stride_V * i, ldt, workspace_ptr, lwork,
        info, gesvdj_params));
    // check the error info
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void MatrixRankGPUKernel<float>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n, float* A,
    float* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  // matrix is saved as column-major in cusolver.
  // numpy and torch use lower triangle to compute eigenvalues, so here use
  // upper triangle
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));

    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]", i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void MatrixRankGPUKernel<double>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n, double* A,
    double* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  upper triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]", i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(matrix_rank, ops::MatrixRankGPUKernel<float>,
                        ops::MatrixRankGPUKernel<double>);
#endif  // not PADDLE_WITH_HIP
