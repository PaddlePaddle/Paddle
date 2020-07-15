// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/gpu_launch_param_config.h"
// #include "paddle/fluid/platform/float16.h"
// #include "stdio.h"

namespace platform = paddle::platform;
namespace ops = paddle::operators;

namespace paddle {
namespace operators {

template <typename T>
__global__ void ExpCUDAKernel(const int N, const T* x, T* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out[idx] = exp(x[idx]);
    // printf("%d, %f, %f/n", idx,x[idx],out[idx]);
  }
}

template <typename T>
__global__ void SumCUDAKernel(const int n, const int d, const int in,
                              const int axis_dim, const T* exp_x, T* sum) {
  for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
    for (int idy = blockIdx.y * blockDim.x + threadIdx.x; idy < in;
         idy += gridDim.y * blockDim.x) {
      int dst_index = idx * in + idy;
      sum[dst_index] = 0;
#pragma unroll
      for (int k = 0; k < axis_dim; k++) {
        sum[dst_index] = sum[dst_index] + exp_x[idx * d + k * in + idy];
        //  printf("%d, %f, %d, %f/n", dst_index, sum[dst_index], idx * d + k *
        //  in + jdy, exp_x[idx * d + k * in + jdy]);
      }
    }
  }
}

template <typename T>
__global__ void softmaxCUDAKernel(const int n, const int d, const int in,
                                  const int axis_dim, const T* exp_x,
                                  const T* sum, T* out) {
  for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
    for (int idy = blockIdx.y * blockDim.x + threadIdx.x; idy < d;
         idy += gridDim.y * blockDim.x) {
      int dst_index = idx * d + idy;
      out[dst_index] = exp_x[dst_index] / sum[idx * in + idy % in];
      //       printf("%d, %d,%d, %f, %d, %f/n",idx, jdy,idx * d + jdy ,out[idx
      //       * d + jdy] ,idx * in + jdy % in,sum[idx * in+ jdy % in]);
    }
  }
}

template <typename T>
__global__ void DotCUDAKernel(const int n, const int d, const int in,
                              const int axis_dim, const T* dout, const T* out,
                              T* dot) {
  for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
    for (int idy = blockIdx.y * blockDim.x + threadIdx.x; idy < in;
         idy += gridDim.y * blockDim.x) {
      int dst_index = idx * in + idy;
      dot[dst_index] = 0;
#pragma unroll
      for (int k = 0; k < axis_dim; k++) {
        int src_index = idx * d + idy + k * in;
        dot[dst_index] = dot[dst_index] + dout[src_index] * out[src_index];
        // printf("%d, %d,%d,%d, %d,\n", idx, jdy, k, src_index, dst_index);
      }
    }
  }
}

template <typename T>
__global__ void softmaxgradientCUDAKernel(const int n, const int d,
                                          const int in, const int axis_dim,
                                          const T* dout, const T* out,
                                          const T* dot, T* dx) {
  for (int idx = blockIdx.x; idx < n; idx += gridDim.x) {
    for (int idy = blockIdx.y * blockDim.x + threadIdx.x; idy < d;
         idy += gridDim.y * blockDim.x) {
      int dst_index = idx * d + idy;

      dx[dst_index] =
          out[dst_index] * (dout[dst_index] - dot[idx * in + idy % in]);
      // printf("%d, %d,%d,%d, \n", idx, jdy, dst_index, idx * in + jdy % in);
    }
  }
}

template <typename T>
class SoftmaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* X = context.Input<Tensor>("X");
    auto* Out = context.Output<Tensor>("Out");
    auto x_dims = X->dims();
    const int rank = x_dims.size();
    int axis = CanonicalAxis(context.Attr<int>("axis"), rank);
    auto numel = X->numel();

    const int n = SizeToAxis(axis, x_dims);
    const int d = SizeFromAxis(axis, x_dims);

    const int axis_dim = x_dims[axis];
    int in = d / axis_dim;

    // LOG(INFO) << "numel: " << numel << " n: " << n << " d: " << d
    //          << " axis_dim: " << axis_dim << " in: " << in;

    auto* x_data = X->data<T>();

    framework::Tensor exp_x;
    exp_x.Resize({numel});
    auto* exp_x_data = exp_x.mutable_data<T>(context.GetPlace());

    auto stream = context.cuda_device_context().stream();
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto config = GetGpuLaunchConfig1D(dev_ctx, numel);
    ExpCUDAKernel<
        T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
        numel, x_data, exp_x_data);

    framework::Tensor sum_x;
    sum_x.Resize({n * axis_dim});
    auto* sum_x_data = sum_x.mutable_data<T>(context.GetPlace());
    dim3 block(std::min(512, in));
    dim3 grid(n, (in + block.x - 1) / block.x);
    SumCUDAKernel<T><<<grid, block, 0, stream>>>(n, d, in, axis_dim, exp_x_data,
                                                 sum_x_data);

    auto* out_data = Out->mutable_data<T>(context.GetPlace());
    dim3 block1(std::min(512, d));
    dim3 grid1(n, (d + block1.x - 1) / block1.x);
    softmaxCUDAKernel<T><<<grid1, block1, 0, stream>>>(
        n, d, in, axis_dim, exp_x_data, sum_x_data, out_data);
  }
};

template <typename T>
class SoftmaxGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* Out = context.Input<Tensor>("Out");
    auto* dOut = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dX = context.Output<Tensor>(framework::GradVarName("X"));
    auto dx_dims = dX->dims();
    const int rank = dx_dims.size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    const int n = SizeToAxis(axis, dx_dims);
    const int d = SizeFromAxis(axis, dx_dims);
    const int axis_dim = dx_dims[axis];
    const int in = d / axis_dim;
    // LOG(INFO) << " n: " << n << " d: " << d << " axis_dim: " << axis_dim
    //          << " in: " << in;

    auto* dx_data = dX->mutable_data<T>(context.GetPlace());
    // LOG(INFO) << "softmax grad1";
    auto* dout_data = dOut->data<T>();
    // LOG(INFO) << "softmax grad2";
    auto* out_data = Out->data<T>();
    // LOG(INFO) << "softmax grad3";
    auto stream = context.cuda_device_context().stream();

    // LOG(INFO) << "softmax grad4";
    // softmax_gradient_kernel<<<n, 128, 0, stream>>>(d, axis_dim, out_data,
    //                                               dout_data, dx_data);
    framework::Tensor dot;
    dot.Resize({n * axis_dim});
    auto* dot_data = dot.mutable_data<T>(context.GetPlace());
    dim3 block(std::min(512, in));
    dim3 grid(n, (in + block.x - 1) / block.x);
    DotCUDAKernel<T><<<grid, block, 0, stream>>>(n, d, in, axis_dim, dout_data,
                                                 out_data, dot_data);

    dim3 block1(std::min(512, d));
    dim3 grid1(n, (d + block1.x - 1) / block1.x);
    softmaxgradientCUDAKernel<T><<<grid1, block1, 0, stream>>>(
        n, d, in, axis_dim, dout_data, out_data, dot_data, dx_data);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    softmax, ops::SoftmaxKernel<platform::CUDADeviceContext, float>,
    ops::SoftmaxKernel<platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    softmax_grad, ops::SoftmaxGradKernel<platform::CUDADeviceContext, float>,
    ops::SoftmaxGradKernel<platform::CUDADeviceContext, double>);
