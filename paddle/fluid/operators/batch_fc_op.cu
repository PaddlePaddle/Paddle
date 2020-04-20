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
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/batch_fc_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {
using framework::Tensor;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void add_bias_with_relu_kernel(T* data, int slot_pairs_num,
                                          int batch_size, int out_dim,
                                          const T* bias, bool DoRelu) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * batch_size * out_dim) {
    int block_len = batch_size * out_dim;
    int slot_index = idx / block_len;
    int out_dim_index = (idx % block_len) % out_dim;
    T temp = data[idx] + bias[slot_index * out_dim + out_dim_index];
    if (DoRelu) {
      data[idx] = static_cast<int>(temp > 0) * temp;
    } else {
      data[idx] = temp;
    }
  }
}

template <typename T>
void add_bias_with_relu(cudaStream_t stream, T* data, int slot_pairs_num,
                        int batch_size, int out_dim, const T* bias,
                        bool DoRelu) {
  add_bias_with_relu_kernel<<<GET_BLOCKS(slot_pairs_num * batch_size * out_dim),
                              CUDA_NUM_THREADS, 0, stream>>>(
      data, slot_pairs_num, batch_size, out_dim, bias, DoRelu);
}

// template <typename T>
// __global__ void KeRelu2Grad(const T* y, const T* dy, const int num, T* dx) {
//  int gid = blockIdx.x * blockDim.x + threadIdx.x;
//  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
//    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
//  }
// }

template <typename DeviceContext, typename T>
class BatchFCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // X.dim = slot_pairs_num * batch_size * in_dim
    // W.dim = slot_pairs_num * in_dim * out_dim
    // b.dim = slot_pairs_num * out_dim
    // output.dim = slot_pairs_num * batch_size * out_dim
    // output = ReLU(X * W + b)
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    bool with_relu =
        (ctx.Attr<std::string>("activation_type") == "relu") ? true : false;
    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto bias_dims = bias->dims();
    auto slot_pairs_num = input_dims[0];
    auto batch_size = input_dims[1];
    auto in_dim = input_dims[2];
    auto out_dim = w_dims[2];

    // get data ptr
    const T* in_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();
    output->Resize({slot_pairs_num, batch_size, out_dim});
    T* out_data = output->mutable_data<T>(ctx.GetPlace());
    // initialize
    auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    T alpha = 1;
    T beta = 0;
    int64_t strideA = batch_size * in_dim;
    int64_t strideB = in_dim * out_dim;

    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
    blas.BatchedGEMM(transA, transB, batch_size, out_dim, in_dim, alpha,
                     in_data, w_data, beta, out_data, slot_pairs_num, strideA,
                     strideB);

    add_bias_with_relu<T>(ctx.cuda_device_context().stream(), out_data,
                          slot_pairs_num, batch_size, out_dim, bias_data,
                          with_relu);
  }
};

// template <typename DeviceContext, typename T>
// class BatchFCGradOpCUDAKernel : public framework::OpKernel<T> {
// public:
//  void Compute(const framework::ExecutionContext &ctx) const override {
//    auto *input = ctx.Input<Tensor>("Input");
//    auto *w = ctx.Input<Tensor>("W");
//    auto *bias = ctx.Input<Tensor>("Bias");
//    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
//    bool with_relu =
//        (ctx.Attr<std::string>("activation_type") == "relu") ? true : false;
//
//    auto *dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
//    auto *dw = ctx.Output<Tensor>(framework::GradVarName("W"));
//    auto *db = ctx.Output<Tensor>(framework::GradVarName("Bias"));
//
//    auto input_dims = input->dims();
//    auto w_dims = w->dims();
//    auto bias_dims = bias->dims();
//    auto slot_pairs_num = input_dims[0];
//    auto batch_size = input_dims[1];
//    auto in_dim = input_dims[2];
//    auto out_dim = w_dims[2];
//    auto stream = ctx.cuda_device_context().stream();
//    int device_id = platform::GetCurrentDeviceId();
//
//
//    T *dout_help_data;
//    auto param_help_size = slot_pairs_num * batch_size * out_dim * sizeof(T);
//    platform::RecordedCudaMalloc(reinterpret_cast<void **>(&dout_help_data),
//                                 param_help_size, device_id);
//    platform::GpuMemsetAsync(dout_help_data, 0, param_help_size, stream);
//
//    // get data ptr
//    const T *input_data = input->data<T>();
//    const T *w_data = w->data<T>();
//    const T *bias_data = bias->data<T>();
//    const T *dout_data = dout->data<T>();
//    T* dx_out = dx->data<T>();
//    T* dw_out = dw->data<T>();
//    T* db_out = db->data<T>();
//
//
//    KeRelu2Grad<T><<<GET_BLOCKS(slot_pairs_num * batch_size * out_dim),
//                            CUDA_NUM_THREADS, 0, stream>>>();
//
//
//    int batch_count = ctx.Attr<int>("BatchCount");
//    int mat_m = ctx.Attr<int>("Mat_M");
//    int mat_n = ctx.Attr<int>("Mat_N");
//    int mat_k = ctx.Attr<int>("Mat_K");
//
//    auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
//    auto *dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
//
//    auto &dev_ctx = ctx.template
//    device_context<platform::CUDADeviceContext>();
//    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
//                       .eigen_device();
//
//    // initialize
//    dx->mutable_data<T>(ctx.GetPlace());
//    auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
//    dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));
//    dy->mutable_data<T>(ctx.GetPlace());
//    auto dy_eigen = framework::EigenVector<T>::Flatten(*dy);
//    dy_eigen.device(place) = dy_eigen.constant(static_cast<T>(0));
//
//    // get data ptr
//    T *dx_data = dx->data<T>();
//    T *dy_data = dy->data<T>();
//    const T *x_data = X->data<T>();
//    const T *y_data = Y->data<T>();
//    const T *dout_data = dout->data<T>();
//
//    auto blas = math::GetBlas<platform::CUDADeviceContext, T>(dev_ctx);
//    T alpha = 1;
//    T beta = 0;
//    // dx = dout_data * y^T
//    blas.BatchFC(CblasNoTrans, CblasTrans, mat_m, mat_k, mat_n, alpha,
//                     dout_data, y_data, beta, dx_data, batch_count,
//                     mat_m * mat_n, mat_k * mat_n);
//    // dy = x^T * dout_data
//    blas.BatchFC(CblasTrans, CblasNoTrans, mat_k, mat_n, mat_m, alpha,
//                     x_data, dout_data, beta, dy_data, batch_count,
//                     mat_k * mat_m, mat_m * mat_n);
//  }
//};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(batch_fc, ops::BatchFCCUDAKernel<GPUCtx, float>,
                        ops::BatchFCCUDAKernel<GPUCtx, double>);

// REGISTER_OP_CUDA_KERNEL(batch_fc_grad,
//                        ops::BatchFCGradOpCUDAKernel<GPUCtx, float>,
//                        ops::BatchFCGradOpCUDAKernel<GPUCtx, double>);
