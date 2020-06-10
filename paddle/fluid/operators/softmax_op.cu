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
    for (int jdy = blockIdx.y * blockDim.x + threadIdx.x; jdy < in;
         jdy += gridDim.y * blockDim.x) {
      int dst_index = idx * in + jdy;
      sum[dst_index] = 0;
      for (int k = 0; k < axis_dim; k++) {
        sum[dst_index] = sum[dst_index] + exp_x[idx * d + k * in + jdy];
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
    for (int jdy = blockIdx.y * blockDim.x + threadIdx.x; jdy < d;
         jdy += gridDim.y * blockDim.x) {
      out[idx * d + jdy] = exp_x[idx * d + jdy] / sum[idx * in + jdy % in];
      //       printf("%d, %d,%d, %f, %d, %f/n",idx, jdy,idx * d + jdy ,out[idx
      //       * d + jdy] ,idx * in + jdy % in,sum[idx * in+ jdy % in]);
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
    int axis = context.Attr<int>("axis");
    auto numel = X->numel();
    auto x_dims = X->dims();
    const int rank = x_dims.size();

    if (axis < 0) {
      axis = axis + rank;
    }

    int n = 1;
    for (int i = 0; i < axis; i++) {
      n *= x_dims[i];
    }

    int d = 1;
    for (int i = axis; i < rank; i++) {
      d *= x_dims[i];
    }

    const int axis_dim = x_dims[axis];
    int in = d / axis_dim;

    LOG(INFO) << "numel: " << numel << " n: " << n << " d: " << d
              << " axis_dim: " << axis_dim << " in: " << in;

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
    sum_x.Resize({axis_dim});
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

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    softmax, ops::SoftmaxKernel<platform::CUDADeviceContext, float>,
    ops::SoftmaxKernel<platform::CUDADeviceContext, double>);
// REGISTER_OP_CUDA_KERNEL(
//    softmax_grad, ops::SoftmaxGradKernel<plat::CUDADeviceContext, float>,
//    ops::SoftmaxGradKernel<plat::CUDADeviceContext, double>,
//   ops::SoftmaxGradKernel<plat::CUDADeviceContext, plat::float16>);
