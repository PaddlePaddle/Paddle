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
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/norm_op.h"
#include "paddle/fluid/platform/bfloat16.h"

namespace paddle {
namespace operators {

__device__ __forceinline__ platform::float16 square_root(platform::float16 x) {
  return static_cast<platform::float16>(sqrtf(static_cast<float>(x)));
}

__device__ __forceinline__ float square_root(float x) { return sqrtf(x); }

__device__ __forceinline__ double square_root(double x) { return sqrt(x); }

template <typename T, int BlockDim>
__global__ void Normalize(const T* x, const int pre,
                          const int axis_n,  // dim in axis
                          const int post, const T eps, T* y, T* out_norm) {
  using MT = typename details::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    int base = (i / post) * post * axis_n + (i % post);

    MT sum = 0.0;
    __shared__ MT norm;
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const MT x_ij = static_cast<MT>(x[base + j * post]);
      sum += x_ij * x_ij;
    }
    MT reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = square_root(reduce_result + static_cast<MT>(eps));
      out_norm[i] = static_cast<T>(norm);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      const int index = base + j * post;
      y[index] = static_cast<T>((static_cast<MT>(x[index]) / norm));
    }
  }
}

template <typename DeviceContext, typename T>
class NormCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_x = ctx.Input<framework::Tensor>("X");
    auto* out_y = ctx.Output<framework::Tensor>("Out");

    auto xdim = in_x->dims();
    int axis = ctx.Attr<int>("axis");
    if (axis < 0) axis = xdim.size() + axis;
    T eps = static_cast<T>(ctx.Attr<float>("epsilon"));

    bool is_test = ctx.Attr<bool>("is_test");

    framework::Tensor* out_norm;
    framework::Tensor out_norm_tmp;
    if (is_test) {
      auto out_dim = in_x->dims();
      out_dim[axis] = 1;
      out_norm = &out_norm_tmp;
      out_norm->Resize(out_dim);
    } else {
      out_norm = ctx.Output<framework::Tensor>("Norm");
    }

    const T* x = in_x->data<T>();
    T* y = out_y->mutable_data<T>(ctx.GetPlace());
    T* norm = out_norm->mutable_data<T>(ctx.GetPlace());

    int pre, n, post;
    GetDims(xdim, axis, &pre, &n, &post);

    auto& dev_ctx = ctx.cuda_device_context();
#ifdef __HIPCC__
    const int block = 256;
#else
    const int block = 512;
#endif
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
  using MT = typename details::MPTypeTrait<T>::Type;
  typedef cub::BlockReduce<MT, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_sum;
  int num = pre * post;
  for (int i = blockIdx.x; i < num; i += gridDim.x) {
    MT sum = 0.0;
    __shared__ MT row_sum;
    __shared__ MT row_sqrt_norm;
    __shared__ MT row_norm;

    auto base = (i / post) * post * axis_n + (i % post);

    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      sum += static_cast<MT>(x[index]) * static_cast<MT>(y_grad[index]);
    }
    MT reduce_result = BlockReduce(temp_storage_sum).Sum(sum);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_sqrt_norm = static_cast<MT>(x_norm[i]);
      row_norm = row_sqrt_norm * row_sqrt_norm;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < axis_n; j += blockDim.x) {
      int index = base + j * post;
      const MT x_ij = static_cast<MT>(x[index]);
      const MT dy_ij = static_cast<MT>(y_grad[index]);
      x_grad[index] =
          static_cast<T>((dy_ij - x_ij * row_sum / row_norm) / row_sqrt_norm);
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

#ifdef __HIPCC__
    const int block = 256;
#else
    const int block = 512;
#endif
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

REGISTER_OP_CUDA_KERNEL(norm,
                        ops::NormCUDAKernel<CUDA, paddle::platform::float16>,
                        ops::NormCUDAKernel<CUDA, float>,
                        ops::NormCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(
    norm_grad, ops::NormGradCUDAKernel<CUDA, paddle::platform::float16>,
    ops::NormGradCUDAKernel<CUDA, float>,
    ops::NormGradCUDAKernel<CUDA, double>);
