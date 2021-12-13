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
#include "paddle/fluid/operators/svd_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

template <typename T>
class SvdGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    const Tensor* x = context.Input<Tensor>("X");
    Tensor* U = context.Output<Tensor>("U");
    Tensor* VH = context.Output<Tensor>("VH");
    Tensor* S = context.Output<Tensor>("S");
    const bool full_matrices = context.Attr<bool>("full_matrices");

    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    int rank = dims.size();
    int m = dims[rank - 2];
    int n = dims[rank - 1];

    auto* vh_data = VH->mutable_data<T>(context.GetPlace());
    auto* s_data = S->mutable_data<T>(context.GetPlace());
    auto* u_data = U->mutable_data<T>(context.GetPlace());
    // NOTE:(@xiongkun03)
    // matrices are assumed to be stored in column-major order in cusolver
    // then view A as n x m and do A^T SVD, we can avoid transpose
    // Must Copy X once, because the gesvdj will change the origin input matrix
    Tensor x_tmp;
    TensorCopy(*x, context.GetPlace(), &x_tmp);
    auto info = memory::Alloc(dev_ctx, sizeof(int) * batch_count);
    int* info_ptr = reinterpret_cast<int*>(info->ptr());

    GesvdjBatched(dev_ctx, batch_count, n, m, std::min(m, n),
                  x_tmp.mutable_data<T>(context.GetPlace()), vh_data, u_data,
                  s_data, info_ptr, !full_matrices);

    framework::DDim UT_dim = U->dims();
    std::swap(UT_dim[rank - 1], UT_dim[rank - 2]);  // Get the dim of UT_dim
    U->Resize(UT_dim);                              // U is entirely UT
    auto dito =
        math::DeviceIndependenceTensorOperations<platform::CUDADeviceContext,
                                                 T>(context);
    auto tmp_U = dito.Transpose(*U);
    U->ShareDataWith(tmp_U);  // U becomse UT, aka VT
  }
  void GesvdjBatched(const platform::CUDADeviceContext& dev_ctx, int batchSize,
                     int m, int n, int k, T* A, T* U, T* V, T* S, int* info,
                     int thin_UV = 1) const;
};

template <>
void SvdGPUKernel<float>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, float* A, float* U, float* V, float* S, int* info,
    int thin_UV) const {
  /* compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */
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
  for (int i = 0; i < batchSize; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgesvdj(
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
void SvdGPUKernel<double>::GesvdjBatched(
    const platform::CUDADeviceContext& dev_ctx, int batchSize, int m, int n,
    int k, double* A, double* U, double* V, double* S, int* info,
    int thin_UV) const {
  /* compute singular vectors */
  const cusolverEigMode_t jobz =
      CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(svd, ops::SvdGPUKernel<float>,
                        ops::SvdGPUKernel<double>);
REGISTER_OP_CUDA_KERNEL(
    svd_grad, ops::SvdGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SvdGradKernel<paddle::platform::CUDADeviceContext, double>);
#endif  // not PADDLE_WITH_HIP
