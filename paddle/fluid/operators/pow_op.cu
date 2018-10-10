/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include "paddle/fluid/operators/pow_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

static __device__ __forceinline__ platform::float16 real_pow(
    platform::float16 x, float factor) {
  return platform::float16(::powf(static_cast<float>(x), factor));
}
static __device__ __forceinline__ float real_pow(float x, float factor) {
  return powf(x, factor);
}
static __device__ __forceinline__ double real_pow(double x, float factor) {
  return pow(x, factor);
}

template <typename T>
__global__ void PowCudaKernel(const T* x, float factor, int64_t numel, T* out) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numel;
       index += blockDim.x * gridDim.x) {
    out[index] = real_pow(x[index], factor);
  }
}

template <typename T>
__global__ void PowGradCudaKernel(const T* x, const T* d_out, float factor,
                                  int64_t numel, T* d_x) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numel;
       index += blockDim.x * gridDim.x) {
    d_x[index] = d_out[index] * factor * real_pow(x[index], factor - 1);
  }
}

template <typename T>
struct PowFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& x, float factor,
                  framework::Tensor* out) const {
    const T* src_ptr = x.data<T>();
    T* dst_ptr = out->data<T>();
    int64_t numel = x.numel();
    PADDLE_ENFORCE_EQ(numel, out->numel());

    const int kThreadsPerBlock = 1024;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);
    int grid = std::min(static_cast<int>(numel), max_blocks);
    int threads = kThreadsPerBlock;

    PowCudaKernel<<<grid, threads, 0, context.stream()>>>(src_ptr, factor,
                                                          numel, dst_ptr);
  }
};

template <typename T>
struct PowGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& x, const framework::Tensor& d_out,
                  float factor, framework::Tensor* d_x) const {
    const T* x_ptr = x.data<T>();
    const T* d_out_ptr = d_out.data<T>();
    T* d_x_ptr = d_x->data<T>();
    int64_t numel = x.numel();
    PADDLE_ENFORCE_EQ(numel, d_out.numel());
    PADDLE_ENFORCE_EQ(numel, d_x->numel());

    const int kThreadsPerBlock = 1024;
    int max_threads = context.GetMaxPhysicalThreadCount();
    int max_blocks = std::max(max_threads / kThreadsPerBlock, 1);
    int grid = std::min(static_cast<int>(numel), max_blocks);
    int threads = kThreadsPerBlock;

    PowGradCudaKernel<<<grid, threads, 0, context.stream()>>>(
        x_ptr, d_out_ptr, factor, numel, d_x_ptr);
  }
};

template struct PowFunctor<platform::CUDADeviceContext, platform::float16>;
template struct PowFunctor<platform::CUDADeviceContext, float>;
template struct PowFunctor<platform::CUDADeviceContext, double>;

template struct PowGradFunctor<platform::CUDADeviceContext, float>;
template struct PowGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(pow, ops::PowKernel<plat::CUDADeviceContext, float>,
                        ops::PowKernel<plat::CUDADeviceContext, double>,
                        ops::PowKernel<plat::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(pow_grad,
                        ops::PowGradKernel<plat::CUDADeviceContext, float>,
                        ops::PowGradKernel<plat::CUDADeviceContext, double>);
