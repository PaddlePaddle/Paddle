/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/lu_op.h"
#include "paddle/fluid/platform/dynload/cusolver.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using CUDADeviceContext = paddle::platform::CUDADeviceContext;

template <typename T>
void cusolver_bufferSize(const cusolverDnHandle_t& cusolverH, int m, int n,
                         T* d_A, int lda, int* lwork);
template <typename T>
void cusolver_getrf(const cusolverDnHandle_t& cusolverH, int m, int n, T* d_A,
                    int lda, T* d_work, int* d_Ipiv, int* d_info);

template <>
void cusolver_bufferSize<float>(const cusolverDnHandle_t& cusolverH, int m,
                                int n, float* d_A, int lda, int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgetrf_bufferSize(
      cusolverH, m, n, d_A, lda, lwork));
}

template <>
void cusolver_bufferSize<double>(const cusolverDnHandle_t& cusolverH, int m,
                                 int n, double* d_A, int lda, int* lwork) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgetrf_bufferSize(
      cusolverH, m, n, d_A, lda, lwork));
}

template <>
void cusolver_getrf<float>(const cusolverDnHandle_t& cusolverH, int m, int n,
                           float* d_A, int lda, float* d_work, int* d_Ipiv,
                           int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnSgetrf(
      cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
}

template <>
void cusolver_getrf<double>(const cusolverDnHandle_t& cusolverH, int m, int n,
                            double* d_A, int lda, double* d_work, int* d_Ipiv,
                            int* d_info) {
  PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::cusolverDnDgetrf(
      cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info));
}

template <typename T>
void lu_decomposed_kernel(int m, int n, T* d_A, int lda, int* d_Ipiv,
                          int* d_info, const framework::ExecutionContext& ctx) {
  /* step 1: get cusolver handle*/
  auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
  auto cusolverH = dev_ctx.cusolver_dn_handle();

  /* step 2: query working space of getrf */
  int lwork;
  cusolver_bufferSize(cusolverH, m, n, d_A, lda, &lwork);

  auto work_buff = memory::Alloc(dev_ctx, lwork * sizeof(T));
  T* d_work = reinterpret_cast<T*>(work_buff->ptr());

  /* step 3: LU factorization */
  if (d_Ipiv) {
    cusolver_getrf(cusolverH, m, n, d_A, lda, d_work, d_Ipiv, d_info);
  } else {
    cusolver_getrf(cusolverH, m, n, d_A, lda, d_work, NULL, d_info);
  }
  PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceSynchronize());
}

template <typename T>
class LUCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef __HIPCC__
    const int64_t kMaxBlockDim = 256;
#else
    const int64_t kMaxBlockDim = 512;
#endif
    auto* xin = ctx.Input<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    auto* IpivT = ctx.Output<framework::Tensor>("Pivots");
    auto* InfoT = ctx.Output<framework::Tensor>("Infos");
    auto pivots = ctx.Attr<bool>("pivots");

    math::DeviceIndependenceTensorOperations<
        paddle::platform::CUDADeviceContext, T>
        helper(ctx);
    *out = helper.Transpose(*xin);

    auto outdims = out->dims();
    auto outrank = outdims.size();

    int m = static_cast<int>(outdims[outrank - 1]);
    int n = static_cast<int>(outdims[outrank - 2]);
    int lda = std::max(1, m);
    if (pivots) {
      auto ipiv_dims = phi::slice_ddim(outdims, 0, outrank - 1);
      ipiv_dims[outrank - 2] = std::min(m, n);
      IpivT->Resize(ipiv_dims);
    }
    auto ipiv_data = IpivT->mutable_data<int>(ctx.GetPlace());

    auto info_dims = phi::slice_ddim(outdims, 0, outrank - 2);
    if (info_dims.size() == 0) {
      info_dims = phi::make_ddim({1});
    }
    InfoT->Resize(info_dims);
    auto info_data = InfoT->mutable_data<int>(ctx.GetPlace());

    auto batchsize = product(info_dims);
    batchsize = std::max(static_cast<int>(batchsize), 1);
    auto out_data = out->mutable_data<T>(ctx.GetPlace());
    for (int b = 0; b < batchsize; b++) {
      auto out_data_item = &out_data[b * m * n];
      int* info_data_item = &info_data[b];
      if (pivots) {
        auto ipiv_data_item = &ipiv_data[b * std::min(m, n)];
        lu_decomposed_kernel(m, n, out_data_item, lda, ipiv_data_item,
                             info_data_item, ctx);
      } else {
        lu_decomposed_kernel(m, n, out_data_item, lda, NULL, info_data_item,
                             ctx);
      }
    }
    *out = helper.Transpose(*out);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(lu, ops::LUCUDAKernel<float>,
                        ops::LUCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(lu_grad,
                        ops::LUGradKernel<plat::CUDADeviceContext, float>,
                        ops::LUGradKernel<plat::CUDADeviceContext, double>);

#endif  // not PADDLE_WITH_HIP
