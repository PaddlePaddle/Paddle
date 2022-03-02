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

#include <algorithm>
#include <vector>
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/controlflow/compare_op.h"
#include "paddle/phi/kernels/impl/matrix_rank_kernel_impl.h"
// #include "paddle/fluid/operators/svd_helper.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/gpu/elementwise.h"

namespace phi {

template <typename T>
void GesvdjBatched(const phi::GPUContext& dev_ctx,
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

template <typename T>
void SyevjBatched(const phi::GPUContext& dev_ctx,
                  int batchSize,
                  int n,
                  T* A,
                  T* W,
                  int* info);

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
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnSgesvdj_bufferSize(handle,
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
  auto workspace = paddle::memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSgesvdj(handle,
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
    int error_info;
    paddle::memory::Copy(phi::CPUPlace(),
                         &error_info,
                         dev_ctx.GetPlace(),
                         info,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
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
  // do not compute singular vectors
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  gesvdjInfo_t gesvdj_params = NULL;
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDgesvdj_bufferSize(handle,
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
  auto workspace = paddle::memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDgesvdj(handle,
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
    paddle::memory::Copy(phi::CPUPlace(),
                         &error_info,
                         dev_ctx.GetPlace(),
                         info,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(
      dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void SyevjBatched<float>(const phi::GPUContext& dev_ctx,
                         int batchSize,
                         int n,
                         float* A,
                         float* W,
                         int* info) {
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
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = paddle::memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnSsyevj(handle,
                                                         jobz,
                                                         uplo,
                                                         n,
                                                         A + stride_A * i,
                                                         lda,
                                                         W + n * i,
                                                         workspace_ptr,
                                                         lwork,
                                                         info,
                                                         params));

    int error_info;
    paddle::memory::Copy(phi::CPUPlace(),
                         &error_info,
                         dev_ctx.GetPlace(),
                         info,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void SyevjBatched<double>(const phi::GPUContext& dev_ctx,
                          int batchSize,
                          int n,
                          double* A,
                          double* W,
                          int* info) {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  upper triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = paddle::memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDsyevj(handle,
                                                         jobz,
                                                         uplo,
                                                         n,
                                                         A + stride_A * i,
                                                         lda,
                                                         W + n * i,
                                                         workspace_ptr,
                                                         lwork,
                                                         info,
                                                         params));
    int error_info;
    paddle::memory::Copy(phi::CPUPlace(),
                         &error_info,
                         dev_ctx.GetPlace(),
                         info,
                         sizeof(int),
                         dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info,
        0,
        phi::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver eigenvalues is not zero. [%d]",
            i,
            error_info));
  }
  PADDLE_ENFORCE_GPU_SUCCESS(dynload::cusolverDnDestroySyevjInfo(params));
}

template <typename T, typename Context>
void MatrixRankKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const Scalar& atol_tensor,
                      bool hermitian,
                      bool use_default_tol,
                      float tol,
                      DenseTensor* out) {
  auto* x_data = x.data<T>();
  dev_ctx.template Alloc<int64_t>(out);

  auto dim_x = x.dims();
  auto dim_out = out->dims();
  int rows = dim_x[dim_x.size() - 2];
  int cols = dim_x[dim_x.size() - 1];
  int k = std::min(rows, cols);
  auto numel = x.numel();
  int batches = numel / (rows * cols);

  // const DenseTensor* atol_tensor = nullptr;
  // Scalar temp_tensor;
  DenseTensor atol_dense_tensor;
  T rtol_T = 0;
  if (use_default_tol) {
    // paddle::framework::TensorFromVector<T>(
    //     std::vector<T>{0}, dev_ctx, &temp_tensor);
    // const Scalar temp_tensor(0);
    // atol_tensor = temp_tensor;
    atol_dense_tensor = Full<T>(dev_ctx, {0}, atol_tensor);
    rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
  } else {
    // const Scalar temp_tensor(tol);
    atol_dense_tensor = Full<float>(dev_ctx, {tol}, atol_tensor);
    // paddle::framework::TensorFromVector<T>(std::vector<T>{tol},
    //        dev_ctx, &temp_tensor);
    // atol_tensor = temp_tensor;
  }

  // Must Copy X once, because the gesvdj will destory the content when exit.
  DenseTensor x_tmp;
  paddle::framework::TensorCopy(x, dev_ctx.GetPlace(), &x_tmp);
  auto info = paddle::memory::Alloc(dev_ctx, sizeof(int) * batches);
  int* info_ptr = reinterpret_cast<int*>(info->ptr());

  DenseTensor eigenvalue_tensor;
  eigenvalue_tensor.Resize(detail::GetEigenvalueDim(dim_x, k));
  auto* eigenvalue_data = dev_ctx.template Alloc<T>(&eigenvalue_tensor);
  // auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
  //   detail::GetEigenvalueDim(dim_x, k), context.GetPlace());
  if (hermitian) {
    SyevjBatched<T>(
        dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data, info_ptr);
    phi::funcs::ForRange<Context> for_range(dev_ctx, eigenvalue_tensor.numel());
    phi::funcs::AbsFunctor<T> functor(
        eigenvalue_data, eigenvalue_data, eigenvalue_tensor.numel());
    for_range(functor);
  } else {
    DenseTensor U, VH;
    U.Resize(detail::GetUDDim(dim_x, k));
    auto* u_data = dev_ctx.template Alloc<T>(&U);
    // auto* u_data =
    //     U.mutable_data<T>(detail::GetUDDim(dim_x, k), context.GetPlace());
    // auto* vh_data =
    //     VH.mutable_data<T>(detail::GetVHDDim(dim_x, k), context.GetPlace());
    VH.Resize(detail::GetVHDDim(dim_x, k));
    auto* vh_data = dev_ctx.template Alloc<T>(&VH);
    GesvdjBatched<T>(dev_ctx,
                     batches,
                     cols,
                     rows,
                     k,
                     x_tmp.data<T>(),
                     vh_data,
                     u_data,
                     eigenvalue_data,
                     info_ptr,
                     1);
  }

  auto dito_T = paddle::math::DeviceIndependenceTensorOperations<
      paddle::platform::CUDADeviceContext,
      T>(context);
  std::vector<int> max_eigenvalue_shape =
      phi::vectorize<int>(detail::RemoveLastDim(eigenvalue_tensor.dims()));
  DenseTensor max_eigenvalue_tensor =
      dito_T.ReduceMax(eigenvalue_tensor, max_eigenvalue_shape);
  DenseTensor temp_rtol_tensor;
  paddle::framework::TensorFromVector<T>(
      std::vector<T>{rtol_T}, dev_ctx, &temp_rtol_tensor);
  DenseTensor rtol_tensor = dito_T.Mul(temp_rtol_tensor, max_eigenvalue_tensor);
  DenseTensor tol_tensor;
  // tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
  tol_tensor.Resize(dim_out);
  dev_ctx.template Alloc<T>(tol_tensor);

  // ElementwiseComputeEx<GreaterElementFunctor<T>, phi::CUDADeviceContext, T,
  // T>(
  //     context,
  //     atol_tensor,
  //     &rtol_tensor,
  //     -1,
  //     GreaterElementFunctor<T>(),
  //     &tol_tensor);

  phi::ElementwiseCompute<GreaterElementFunctor<T>, T, T>(
      dev_ctx,
      atol_dense_tensor,
      rtol_tensor,
      -1,
      GreaterElementFunctor<T>(),
      &tol_tensor);

  tol_tensor.Resize(detail::NewAxisDim(tol_tensor.dims(), 1));

  DenseTensor compare_result;
  compare_result.Resize(detail::NewAxisDim(dim_out, k));
  dev_ctx.template Alloc<T>(&compare_result);
  // compare_result.mutable_data<int64_t>(detail::NewAxisDim(dim_out, k),
  //                                   context.GetPlace());
  int axis = -1;

  phi::ElementwiseCompute<paddle::operators::GreaterThanFunctor<T, int64_t>,
                          T,
                          int64_t>(
      dev_ctx,
      eigenvalue_tensor,
      tol_tensor,
      axis,
      paddle::operators::GreaterThanFunctor<T, int64_t>(),
      &compare_result);
  auto dito_int = math::DeviceIndependenceTensorOperations<
      paddle::platform::CUDADeviceContext,
      int64_t>(context);
  std::vector<int> result_shape = phi::vectorize<int>(dim_out);
  DenseTensor result = dito_int.ReduceSum(compare_result, result_shape);
  out->ShareDataWith(result);
}

}  // namespace phi

PD_REGISTER_KERNEL(matrix_rank,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::MatrixRankKernel,
                   float,
                   double) {}

#endif  // not PADDLE_WITH_HIP
