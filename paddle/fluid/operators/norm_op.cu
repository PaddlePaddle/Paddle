/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "cub/cub.cuh"
#include "paddle/fluid/operators/norm_op.h"

namespace paddle {
namespace operators {

__device__ __forceinline__ float square_root(float x) { return sqrtf(x); }

__device__ __forceinline__ double square_root(double x) { return sqrt(x); }

template <typename T, int BlockDim>
__global__ void Normalize(const T* x, const int pre,
                          const int axis_n,  // dim in axis
                          const int post, const T eps, T* y, T* out_norm) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);

    T sum = 0.0;
    __shared__ T norm;
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const T x_ij = x[base + j * post];
      sum += x_ij * x_ij;
    }
    T reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = square_root(reduce_result + eps);
      out_norm[i] = norm;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const int index = base + j * post;
      y[index] = x[index] / norm;
    }
  }
}

template <typename DeviceContext, typename T>
class NormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_y = ctx.Output<framework::Tensor>("Out");
    auto* out_norm = ctx.Output<framework::Tensor>("Norm");
    const T* x = in_x->data<T>();
    T* y = out_y->mutable_data<T>(ctx.GetPlace());
    T* norm = out_norm->mutable_data<T>(ctx.GetPlace());

    auto xdim = in_x->dims();
    auto ndim = out_norm->dims();
    int axis = ctx.Attr<int>("axis");
    T eps = static_cast<T>(ctx.Attr<float>("epsilon"));
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post);

    auto& dev_ctx = ctx.cuda_device_context();

    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid = std::min(max_blocks, pre * post);
    Normalize<T, block><<<grid, block, 0, dev_ctx.stream()>>>(x, pre, n, post,
                                                              eps, y, norm);
  }
};

template <typename T, int BlockDim>
__global__ void NormalizeGradient(const T* x, const T* x_norm, const T* y_grad,
                                  const int pre, const int axis_n,
                                  const int post, T* x_grad) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_sum;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    T sum = 0.0;
    __shared__ T row_sum;
    __shared__ T row_sqrt_norm;
    __shared__ T row_norm;

    auto base = (i / post) * post * axis_n + (i % post);

    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      sum += x[index] * y_grad[index];
    }
    T reduce_result = BlockReduce(temp_storage_sum).Sum(sum);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_sqrt_norm = x_norm[i];
      row_norm = row_sqrt_norm * row_sqrt_norm;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      const T x_ij = x[index];
      const T dy_ij = y_grad[index];
      x_grad[index] = (dy_ij - x_ij * row_sum / row_norm) / row_sqrt_norm;
    }
  }
}

template <typename DeviceContext, typename T, typename AttrType = T>
class NormGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* in_norm = ctx.Input<framework::Tensor>("Norm");
    auto* in_dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* out_dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    T* dx = out_dx->mutable_data<T>(ctx.GetPlace());
    const T* x = in_x->data<T>();
    const T* x_norm = in_norm->data<T>();
    const T* dy = in_dy->data<T>();

    auto xdim = in_x->dims();
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;
    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post);

    auto& dev_ctx = ctx.cuda_device_context();

    const int block = 512;
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid = std::min(max_blocks, pre * post);
    NormalizeGradient<T, block><<<grid, block, 0, dev_ctx.stream()>>>(
        x, x_norm, dy, pre, n, post, dx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;

REGISTER_OP_CUDA_KERNEL(norm, ops::NormCUDAKernel<CUDA, float>,
                        ops::NormCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(norm_grad, ops::NormGradCUDAKernel<CUDA, float>,
                        ops::NormGradCUDAKernel<CUDA, double>);
