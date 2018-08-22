// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/device_vector.h>
#include "paddle/fluid/framework/array.h"
#include "paddle/fluid/operators/stack_op.h"

namespace paddle {
namespace operators {

template <typename T, typename VecXType>
__global__ void StackCUDAKernel(VecXType x, T* y, int total_num, int n,
                                int post) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < total_num) {
    int i = idx / (n * post);
    int which_x = idx / post - i * n;
    int x_index = i * post + idx % post;
    y[idx] = x[which_x][x_index];
  }
}

template <typename T, typename VecDxType>
__global__ void StackGradCUDAKernel(VecDxType dx, const T* dy, int total_num,
                                    int n, int post) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < total_num) {
    int i = idx / (n * post);
    int which_x = idx / post - i * n;
    int x_index = i * post + idx % post;
    dx[which_x][x_index] = dy[idx];
  }
}

struct GPUStackFunctor {
  template <typename DeviceContext, typename T>
  void operator()(const DeviceContext& ctx, const std::vector<const T*>& x,
                  T* y, int pre, int n, int post) const {
    int total_num = pre * post * n;
    int threads = 512;
    int grid = (total_num + threads - 1) / threads;

    constexpr auto kMaxThreshold = 16;
    if (n <= kMaxThreshold) {
      framework::Array<const T*, kMaxThreshold> arr;
      for (int i = 0; i < n; ++i) arr[i] = x[i];
      StackCUDAKernel<<<grid, threads, 0, ctx.stream()>>>(arr, y, total_num, n,
                                                          post);
    } else {
      VLOG(10) << "Stack more than " << kMaxThreshold
               << " tensors may be slow on GPU.";
      thrust::device_vector<const T*> dev_x(x);
      StackCUDAKernel<<<grid, threads, 0, ctx.stream()>>>(dev_x.data().get(), y,
                                                          total_num, n, post);
    }
  }
};

struct GPUStackGradFunctor {
  template <typename DeviceContext, typename T>
  void operator()(const DeviceContext& ctx, std::vector<T*>& dx,  // NOLINT
                  const T* dy, int pre, int n, int post) const {
    int total_num = pre * post * n;
    int threads = 512;
    int grid = (total_num + threads - 1) / threads;

    constexpr auto kMaxThreshold = 16;
    if (n <= kMaxThreshold) {
      framework::Array<T*, kMaxThreshold> arr;
      for (int i = 0; i < n; ++i) arr[i] = dx[i];
      StackGradCUDAKernel<<<grid, threads, 0, ctx.stream()>>>(
          arr, dy, total_num, n, post);
    } else {
      VLOG(10) << "Stack more than " << kMaxThreshold
               << " tensors may be slow on GPU.";
      thrust::device_vector<T*> dev_dx(dx);
      StackGradCUDAKernel<<<grid, threads, 0, ctx.stream()>>>(
          dev_dx.data().get(), dy, total_num, n, post);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    stack,
    ops::StackKernel<plat::CUDADeviceContext, float, ops::GPUStackFunctor>,
    ops::StackKernel<plat::CUDADeviceContext, double, ops::GPUStackFunctor>);

REGISTER_OP_CUDA_KERNEL(stack_grad,
                        ops::StackGradKernel<plat::CUDADeviceContext, float,
                                             ops::GPUStackGradFunctor>,
                        ops::StackGradKernel<plat::CUDADeviceContext, double,
                                             ops::GPUStackGradFunctor>);
