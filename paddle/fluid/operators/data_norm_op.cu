/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/operators/data_norm_op.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;
using platform::PADDLE_CUDA_NUM_THREADS;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T>
__global__ void kernel_data_norm_ff(int N, int C, const T *x, T *y,
                                    const T *mean, const T *scale) {
  CUDA_KERNEL_LOOP(i, N * C) {
    int col = i % C;
    y[i] = (x[i] - mean[col]) * scale[col];
  }
}

template <typename T>
__global__ void kernel_mean_scale(int C, const T *batch_size,
                                  const T *batch_sum, const T *batch_square_sum,
                                  T *mean, T *scale) {
  CUDA_KERNEL_LOOP(i, C) {
    mean[i] = batch_sum[i] / batch_size[i];
    scale[i] = sqrt(batch_size[i] / batch_square_sum[i]);
  }
}

template <typename T>
__global__ void kernel_data_norm_bp(int N, int C, const T *y_grad,
                                    const T *scale, T *x_grad) {
  CUDA_KERNEL_LOOP(i, N * C) { x_grad[i] = y_grad[i] * scale[i % C]; }
}

template <typename T>
__global__ void kernel_data_norm_bp_stat(int N, int C, const T *x_val,
                                         const T *means,
                                         const float squared_sum_epsilon,
                                         T *batch_size, T *batch_sum,
                                         T *batch_square_sum) {
  CUDA_KERNEL_LOOP(i, C) {
    T val_sum = 0;
    T square_sum = 0;
    for (int j = 0; j < N; j++) {
      val_sum += x_val[j * C + i];
      square_sum +=
          (x_val[j * C + i] - means[i]) * (x_val[j * C + i] - means[i]);
    }
    batch_size[i] = 1;
    batch_sum[i] = val_sum / N;
    batch_square_sum[i] = square_sum / N + squared_sum_epsilon;
  }
}

template <typename T>
__global__ void kernel_update_param(int C, const T *d_batch_size,
                                    const T *d_batch_sum,
                                    const T *d_batch_square_sum, T *batch_size,
                                    T *batch_sum, T *batch_square_sum,
                                    const float decay_rate) {
  CUDA_KERNEL_LOOP(i, C) {
    batch_size[i] = batch_size[i] * decay_rate + d_batch_size[i];
    batch_sum[i] = batch_sum[i] * decay_rate + d_batch_sum[i];
    batch_square_sum[i] =
        batch_square_sum[i] * decay_rate + d_batch_square_sum[i];
  }
}

template <typename T>
class DataNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    // Align with CPU version, but should we add this restriction?
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::PreconditionNotMet(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C = x_dims[1];
    const T *batch_size_in = ctx.Input<Tensor>("BatchSize")->data<T>();
    const T *batch_sum_in = ctx.Input<Tensor>("BatchSum")->data<T>();
    const T *batch_square_sum_in =
        ctx.Input<Tensor>("BatchSquareSum")->data<T>();
    auto *x_data = x->data<T>();

    // alloc memory
    T *y_data = ctx.Output<Tensor>("Y")->mutable_data<T>(ctx.GetPlace());
    T *mean_out_data =
        ctx.Output<Tensor>("Means")->mutable_data<T>(ctx.GetPlace());
    T *scale_out_data =
        ctx.Output<Tensor>("Scales")->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    kernel_mean_scale<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        C, batch_size_in, batch_sum_in, batch_square_sum_in, mean_out_data,
        scale_out_data);
    kernel_data_norm_ff<<<GET_BLOCKS(C * N), PADDLE_CUDA_NUM_THREADS, 0,
                          stream>>>(N, C, x_data, y_data, mean_out_data,
                                    scale_out_data);
  }
};

template <typename T>
class DataNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scales = ctx.Input<Tensor>("Scales");
    const auto *means = ctx.Input<Tensor>("Means");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float dr = ctx.Attr<float>("summary_decay_rate");
    const int sync_with_nccl = ctx.Attr<bool>("sync_with_nccl");

    const auto &x_dims = x->dims();
    // Align with CPU version, but should we add this restriction?
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::PreconditionNotMet(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C = x_dims[1];

    // init output
    Tensor *d_x = nullptr;
    if (ctx.HasOutput(framework::GradVarName("X"))) {
      d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    }
    T *d_batch_size = ctx.Output<Tensor>(framework::GradVarName("BatchSize"))
                          ->mutable_data<T>(ctx.GetPlace());
    T *d_batch_sum = ctx.Output<Tensor>(framework::GradVarName("BatchSum"))
                         ->mutable_data<T>(ctx.GetPlace());
    T *d_batch_square_sum =
        ctx.Output<Tensor>(framework::GradVarName("BatchSquareSum"))
            ->mutable_data<T>(ctx.GetPlace());

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    if (d_x != nullptr) {
      kernel_data_norm_bp<<<GET_BLOCKS(C * N), PADDLE_CUDA_NUM_THREADS, 0,
                            stream>>>(N, C, d_y->data<T>(), scales->data<T>(),
                                      d_x->mutable_data<T>(ctx.GetPlace()));
    }

    kernel_data_norm_bp_stat<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0,
                               stream>>>(N, C, x->data<T>(), means->data<T>(),
                                         epsilon, d_batch_size, d_batch_sum,
                                         d_batch_square_sum);

    if (sync_with_nccl) {
      auto comm = platform::NCCLCommContext::Instance().Get(0, ctx.GetPlace());
      cudaStream_t comm_stream = comm->stream();

      cudaError_t e_sync = cudaStreamSynchronize(stream);
      if (e_sync != 0) {
        LOG(FATAL) << "Fail to sync nccl stream: "
                   << cudaGetErrorString(e_sync);
      }
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          reinterpret_cast<const void *>(d_batch_size),
          reinterpret_cast<void *>(d_batch_size), C,
          platform::ToNCCLDataType(x->type()), ncclSum, comm->comm(),
          comm_stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          reinterpret_cast<const void *>(d_batch_sum),
          reinterpret_cast<void *>(d_batch_sum), C,
          platform::ToNCCLDataType(x->type()), ncclSum, comm->comm(),
          comm_stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
          reinterpret_cast<const void *>(d_batch_square_sum),
          reinterpret_cast<void *>(d_batch_square_sum), C,
          platform::ToNCCLDataType(x->type()), ncclSum, comm->comm(),
          comm_stream));
      e_sync = cudaStreamSynchronize(comm_stream);
      if (e_sync != 0) {
        LOG(FATAL) << "Fail to sync nccl stream: "
                   << cudaGetErrorString(e_sync);
      }
    }
    T *batch_size_data =
        ctx.Output<Tensor>("BatchSize")->mutable_data<T>(ctx.GetPlace());
    T *batch_sum_data =
        ctx.Output<Tensor>("BatchSum")->mutable_data<T>(ctx.GetPlace());
    T *batch_square_sum_data =
        ctx.Output<Tensor>("BatchSquareSum")->mutable_data<T>(ctx.GetPlace());
    kernel_update_param<<<GET_BLOCKS(C), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        C, d_batch_size, d_batch_sum, d_batch_square_sum, batch_size_data,
        batch_sum_data, batch_square_sum_data, dr);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    data_norm, ops::DataNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DataNormKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    data_norm_grad,
    ops::DataNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DataNormGradKernel<paddle::platform::CUDADeviceContext, double>);
