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

#include <cublas.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/batched_gemm_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
class BatchedGEMMCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *Y = ctx.Input<Tensor>("Y");
    int batch_count = ctx.Attr<int>("BatchCount");
    int mat_m = ctx.Attr<int>("Mat_M");
    int mat_n = ctx.Attr<int>("Mat_N");
    int mat_k = ctx.Attr<int>("Mat_K");
    auto *Out = ctx.Output<Tensor>("Out");

    auto x_dims = X->dims();
    auto y_dims = Y->dims();
    int x_numel = framework::product(x_dims);
    int y_numel = framework::product(y_dims);

    PADDLE_ENFORCE_EQ(
        x_numel, batch_count * mat_m * mat_k,
        platform::errors::OutOfRange("X of BatchedGEMM has error dims."));
    PADDLE_ENFORCE_EQ(
        y_numel, batch_count * mat_k * mat_n,
        platform::errors::OutOfRange("Y of BatchedGEMM has error dims."));

    // get data ptr
    const T *x_data = X->data<T>();
    const T *y_data = Y->data<T>();

    Out->mutable_data<T>(ctx.GetPlace());
    // initialize
    auto out_eigen = framework::EigenVector<T>::Flatten(*Out);
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

    // get data ptr
    T *out_data = Out->data<T>();

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    T alpha = 1;
    T beta = 0;
    int64_t strideA = mat_m * mat_k;
    int64_t strideB = mat_k * mat_n;

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    blas.BatchedGEMM(transA, transB, mat_m, mat_n, mat_k, alpha, x_data, y_data,
                     beta, out_data, batch_count, strideA, strideB);
  }
};

template <typename DeviceContext, typename T>
class BatchedGEMMGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *Y = ctx.Input<Tensor>("Y");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int batch_count = ctx.Attr<int>("BatchCount");
    int mat_m = ctx.Attr<int>("Mat_M");
    int mat_n = ctx.Attr<int>("Mat_N");
    int mat_k = ctx.Attr<int>("Mat_K");

    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();

    // initialize
    dx->mutable_data<T>(ctx.GetPlace());
    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
    dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));
    dy->mutable_data<T>(ctx.GetPlace());
    auto dy_eigen = framework::EigenVector<T>::Flatten(*dy);
    dy_eigen.device(place) = dy_eigen.constant(static_cast<T>(0));

    // get data ptr
    T *dx_data = dx->data<T>();
    T *dy_data = dy->data<T>();
    const T *x_data = X->data<T>();
    const T *y_data = Y->data<T>();
    const T *dout_data = dout->data<T>();

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    T alpha = 1;
    T beta = 0;
    // dx = dout_data * y^T
    blas.BatchedGEMM(CblasNoTrans, CblasTrans, mat_m, mat_k, mat_n, alpha,
                     dout_data, y_data, beta, dx_data, batch_count,
                     mat_m * mat_n, mat_k * mat_n);
    // dy = x^T * dout_data
    blas.BatchedGEMM(CblasTrans, CblasNoTrans, mat_k, mat_n, mat_m, alpha,
                     x_data, dout_data, beta, dy_data, batch_count,
                     mat_k * mat_m, mat_m * mat_n);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(batched_gemm, ops::BatchedGEMMCUDAKernel<GPUCtx, float>,
                        ops::BatchedGEMMCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(batched_gemm_grad,
                        ops::BatchedGEMMGradOpCUDAKernel<GPUCtx, float>,
                        ops::BatchedGEMMGradOpCUDAKernel<GPUCtx, double>);
