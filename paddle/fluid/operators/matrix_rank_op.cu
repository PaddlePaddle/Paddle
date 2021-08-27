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
#include "paddle/fluid/operators/cholesky_op.h"
#include "paddle/fluid/operators/elementwise/svd_helper.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/matrix_rank_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
class MatrixRankGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    // get input/output
    const Tensor* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    auto* out = context.Output<Tensor>("Out");
    out->mutable_data<int64_t>(context.GetPlace());
    bool hermitian = context.Attr<bool>("hermitian");
    // get shape
    auto dim_x = x->dims();
    auto dim_out = out->dims();
    // auto dim_atol_tensor = atol_tensor->dims();
    int rows = dim_x[dim_x.size() - 2];
    int cols = dim_x[dim_x.size() - 1];
    int k = std::min(rows, cols);
    auto numel = x->numel();
    int batches = numel / (rows * cols);
    // get tol
    bool use_default_tol = context.Attr<bool>("use_default_tol");
    const Tensor* atol_tensor = nullptr;
    Tensor temp_tensor;
    T rtol_T = 0;
    if (use_default_tol) {
      VLOG(3) << "not has tol";
      framework::TensorFromVector<T>(std::vector<T>{0},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
      rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    } else if (context.HasInput("TolTensor")) {
      VLOG(3) << "has toltensor";
      atol_tensor = context.Input<Tensor>("TolTensor");
    } else {
      VLOG(3) << "has tol";
      VLOG(3) << context.Attr<float>("tol");
      framework::TensorFromVector<T>(std::vector<T>{context.Attr<float>("tol")},
                                     context.device_context(), &temp_tensor);
      atol_tensor = &temp_tensor;
    }

    // Must Copy X once, because the gesvdj will destory the content when exit.
    Tensor x_tmp;
    TensorCopy(*x, context.GetPlace(), &x_tmp);
    // cusolver API use
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batches);
    int* info_ptr = reinterpret_cast<int*>(info->ptr());

    // compute eigenvalue/svd
    Tensor eigenvalue_tensor;
    auto* eigenvalue_data = eigenvalue_tensor.mutable_data<T>(
        EigenvalueDim(dim_x, k), context.GetPlace());
    if (hermitian) {
      // m == n
      VLOG(3) << "hermitian";
      SyevjBatched(dev_ctx, batches, rows, x_tmp.data<T>(), eigenvalue_data,
                   info_ptr);
      // compute abs(eigenvalues)
      platform::ForRange<platform::CUDADeviceContext> for_range(
          dev_ctx, eigenvalue_tensor.numel());
      math::AbsFunctor<T> functor(eigenvalue_data, eigenvalue_data,
                                  eigenvalue_tensor.numel());
      for_range(functor);
    } else {
      VLOG(3) << "not hermitian";
      Tensor U, VH;
      auto* u_data = U.mutable_data<T>(UDDim(dim_x, k), context.GetPlace());
      auto* vh_data = VH.mutable_data<T>(VHDDim(dim_x, k), context.GetPlace());
      GesvdjBatched(dev_ctx, batches, cols, rows, k, x_tmp.data<T>(), vh_data,
                    u_data, eigenvalue_data, info_ptr, 1);
    }

    VLOG(3) << "eigenvalue_tensor shape: " << eigenvalue_tensor.dims();
    std::vector<T> eigenvalue_vec(eigenvalue_tensor.numel());
    TensorToVector(eigenvalue_tensor, context.device_context(),
                   &eigenvalue_vec);
    for (int i = 0; i < eigenvalue_vec.size(); i++) {
      VLOG(3) << "eigenvalue_vec: " << eigenvalue_vec[i];
    }

    // compare atol(absolute tol) with rtol(relative tol)
    // T rtol_T = std::numeric_limits<T>::epsilon() * std::max(rows, cols);
    // if (hasTol) {
    //   rtol_T = 0;
    // }
    auto dito_T =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(context);
    std::vector<int> max_eigenvalue_shape =
        framework::vectorize<int>(RemoveLastDim(eigenvalue_tensor.dims()));
    Tensor max_eigenvalue_tensor =
        dito_T.reduce_max(eigenvalue_tensor, max_eigenvalue_shape);

    VLOG(3) << "max_eigenvalue_tensor shape: " << max_eigenvalue_tensor.dims();
    std::vector<T> max_eigenvalue_vec(max_eigenvalue_tensor.numel());
    TensorToVector(max_eigenvalue_tensor, context.device_context(),
                   &max_eigenvalue_vec);
    for (int i = 0; i < max_eigenvalue_vec.size(); i++) {
      VLOG(3) << "max_eigenvalue_vec: " << max_eigenvalue_vec[i];
    }

    Tensor temp_rtol_tensor;
    framework::TensorFromVector<T>(std::vector<T>{rtol_T},
                                   context.device_context(), &temp_rtol_tensor);
    // rtol_tensor.mutable_data<T>(max_eigenvalue_tensor.dims(),
    // context.GetPlace());
    Tensor rtol_tensor = dito_T.mul(temp_rtol_tensor, max_eigenvalue_tensor);

    Tensor tol_tensor;
    tol_tensor.mutable_data<T>(dim_out, context.GetPlace());
    ElementwiseComputeEx<GreaterElementFunctor<T>, platform::CUDADeviceContext,
                         T, T>(context, atol_tensor, &rtol_tensor, -1,
                               GreaterElementFunctor<T>(), &tol_tensor);
    tol_tensor.Resize(NewAxisDim(tol_tensor.dims(), 1));

    VLOG(3) << "tol_tensor shape: " << tol_tensor.dims();
    std::vector<T> tol_vec(tol_tensor.numel());
    TensorToVector(tol_tensor, context.device_context(), &tol_vec);
    for (int i = 0; i < tol_vec.size(); i++) {
      VLOG(3) << "tol_vec: " << tol_vec[i];
    }

    Tensor compare_result;
    compare_result.mutable_data<int64_t>(NewAxisDim(dim_out, k),
                                         context.GetPlace());
    int axis = -1;
    if (eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()) {
      VLOG(3) << "eigenvalue_tensor.dims().size() >= tol_tensor.dims().size()";
      ElementwiseComputeEx<CompareFunctor<T>, platform::CUDADeviceContext, T,
                           int64_t>(context, &eigenvalue_tensor, &tol_tensor,
                                    axis, CompareFunctor<T>(), &compare_result);
    } else {
      VLOG(3) << "eigenvalue_tensor.dims().size() < tol_tensor.dims().size()";
      ElementwiseComputeEx<InverseCompareFunctor<T>,
                           platform::CUDADeviceContext, T, int64_t>(
          context, &eigenvalue_tensor, &tol_tensor, axis,
          InverseCompareFunctor<T>(), &compare_result);
    }
    auto dito_int =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 int64_t>(context);
    std::vector<int> res_shape = framework::vectorize<int>(dim_out);
    Tensor res = dito_int.reduce_sum(compare_result, res_shape);
    out->ShareDataWith(res);
  }

  void GesvdjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                     int m, int n, int k, const T* cA, T* U, T* V, T* S,
                     int* info, int thin_UV = 1) const;

  void SyevjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                    int n, const T* cA, T* W, int* info) const;
};

template <>
void MatrixRankGPUKernel<float>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const float* cA, float* U, float* V, float* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_NOVECTOR; /* no compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  float* A = const_cast<float*>(cA);
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSgesvdj(
        handle, jobz, thin_UV, m, n, A + stride_A * i, lda, S + k * i,
        U + stride_U * i, ldu, V + stride_V * i, ldt, workspace_ptr, lwork,
        info, gesvdj_params));
    // platform::dynload::cusolverDnSgesvdj(
    //         handle, jobz, thin_UV, m, n, A+stride_A*i, lda, S+k*i,
    //         U+stride_U*i,
    //         ldu, V+stride_V*i, ldt, workspace_ptr, lwork, info,
    //         gesvdj_params);
    // std::cout << "info:" << *info << std::endl;
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
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void MatrixRankGPUKernel<double>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, const double* cA, double* U, double* V, double* S, int* info,
    int thin_UV) const {
  /* no compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_NOVECTOR; /* no compute singular vectors */
  gesvdjInfo_t gesvdj_params = NULL;
  double* A = const_cast<double*>(cA);
  int lda = m;
  int ldu = m;
  int ldt = n;
  int lwork = 0;
  auto handle = dev_ctx.cusolver_dn_handle();
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateGesvdjInfo(&gesvdj_params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDgesvdj_bufferSize(
      handle, jobz, thin_UV, m, n, A, lda, S, U, ldu, V, ldt, &lwork,
      gesvdj_params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());
  int stride_A = lda * n;
  int stride_U = ldu * (thin_UV ? k : m);
  int stride_V = ldt * (thin_UV ? k : n);
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDgesvdj(
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
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

template <>
void MatrixRankGPUKernel<float>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n,
    const float* cA, float* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  // Lower triangle of A is stored
  // cusolver中矩阵是column-major即转置的形式，numpy和torch中使用下三角来进行计算
  // 因为转置的缘故需要上三角来进行计算
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  float* A = const_cast<float*>(cA);
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnXsyevjSetMaxSweeps(params,
  // 15));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(float));
  float* workspace_ptr = reinterpret_cast<float*>(workspace->ptr());
  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnSsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));

    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}

template <>
void MatrixRankGPUKernel<double>::SyevjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int n,
    const double* cA, double* W, int* info) const {
  auto handle = dev_ctx.cusolver_dn_handle();
  // Compute eigenvalues only
  const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  //  Lower triangle of A is stored
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  double* A = const_cast<double*>(cA);
  int lda = n;
  int stride_A = lda * n;
  int lwork = 0;
  syevjInfo_t params = NULL;
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnCreateSyevjInfo(&params));
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDsyevj_bufferSize(
      handle, jobz, uplo, n, A, lda, W, &lwork, params));
  auto workspace = memory::Alloc(dev_ctx, lwork * sizeof(double));
  double* workspace_ptr = reinterpret_cast<double*>(workspace->ptr());

  for (int i = 0; i < batchSize; i++) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cusolverDnDsyevj(
        handle, jobz, uplo, n, A + stride_A * i, lda, W + n * i, workspace_ptr,
        lwork, info, params));
    int error_info;
    memory::Copy(platform::CPUPlace(), &error_info,
                 BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()), info,
                 sizeof(int), dev_ctx.stream());
    PADDLE_ENFORCE_EQ(
        error_info, 0,
        platform::errors::PreconditionNotMet(
            "For batch [%d]: CUSolver SVD is not zero. [%d]", i, error_info));
  }
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::cusolverDnDestroySyevjInfo(params));
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(matrix_rank, ops::MatrixRankGPUKernel<float>,
                        ops::MatrixRankGPUKernel<double>);
#endif  // not PADDLE_WITH_HIP
