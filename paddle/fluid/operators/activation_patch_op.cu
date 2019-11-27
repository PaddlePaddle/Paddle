/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/activation_patch_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,            \
                                        grad_functor)                          \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type,                                                                \
      ops::ReluCUDAKernel<plat::CUDADeviceContext, float,                      \
                          ops::functor<plat::CUDADeviceContext, float>>,       \
      ops::ReluCUDAKernel<plat::CUDADeviceContext, double,                     \
                          ops::functor<plat::CUDADeviceContext, double>>,      \
      ops::ReluCUDAKernel<                                                     \
          plat::CUDADeviceContext, plat::float16,                              \
          ops::functor<plat::CUDADeviceContext, plat::float16>>);              \
  REGISTER_OP_CUDA_KERNEL(                                                     \
      act_type##_grad, ops::ReluGradCUDAKernel<                                \
                           plat::CUDADeviceContext, float,                     \
                           ops::grad_functor<plat::CUDADeviceContext, float>>, \
      ops::ReluGradCUDAKernel<                                                 \
          plat::CUDADeviceContext, double,                                     \
          ops::grad_functor<plat::CUDADeviceContext, double>>,                 \
      ops::ReluGradCUDAKernel<                                                 \
          plat::CUDADeviceContext, plat::float16,                              \
          ops::grad_functor<plat::CUDADeviceContext, plat::float16>>);

namespace paddle {
namespace operators {

template <typename T>
__global__ void KeRelu(const T* x, int num, T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<T>(0));
  }
}

template <typename T>
struct ReluFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx, const T* x,
                  int num, T* y) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu<T><<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
  }
};

template <typename T>
__global__ void KeReluGrad(const T* y, const T* dy, int num, T* dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > static_cast<T>(0) ? 1. : 0.);
  }
}

template <typename T>
struct ReluGradFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& dev_ctx, const T* y,
                  const T* dy, int num, T* dx) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeReluGrad<T><<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

#ifdef PADDLE_CUDA_FP16

inline DEVICE half2 half2_relu(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = max(a1, 0.0);
  float r2 = max(a2, 0.0);
  return __floats2half2_rn(r1, r2);
}

__global__ void KeRelufp16(const platform::float16* x, int num,
                           platform::float16* y) {
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n2 = num / 2;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  half2* y2 = reinterpret_cast<half2*>(y);
  for (int i = start; i < n2; i += stride) {
    y2[i] = half2_relu(x2[i]);
  }
  if (start == 0 && (num % 2)) {
    y[num - 1] = max(static_cast<float>(x[num - 1]), 0.0);
  }
}

template <>
struct ReluFunctor<platform::CUDADeviceContext, platform::float16> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const platform::float16* x, int num,
                  platform::float16* y) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelufp16<<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
  }
};

inline DEVICE half2 half2_relu_grad(const half2& y, const half2& dy) {
  float y1 = __low2float(y);
  float dy1 = __low2float(dy);
  float r1 = dy1 * (y1 > 0 ? 1. : 0.);

  float y2 = __high2float(y);
  float dy2 = __high2float(dy);
  float r2 = dy2 * (y2 > 0 ? 1. : 0.);

  return __floats2half2_rn(r1, r2);
}

__global__ void KeReluGradfp16(const platform::float16* y,
                               const platform::float16* dy, int num,
                               platform::float16* dx) {
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n2 = num / 2;

  const half2* y2 = reinterpret_cast<const half2*>(y);
  const half2* dy2 = reinterpret_cast<const half2*>(dy);
  half2* dx2 = reinterpret_cast<half2*>(dx);
  for (int i = start; i < n2; i += stride) {
    dx2[i] = half2_relu_grad(y2[i], dy2[i]);
  }

  if (start == 0 && (num % 2)) {
    dx[num - 1] = dy[num - 1] * (y[num - 1] > static_cast<plat::float16>(0)
                                     ? static_cast<plat::float16>(1.)
                                     : static_cast<plat::float16>(0.));
  }
}

template <>
struct ReluGradFunctor<platform::CUDADeviceContext, platform::float16> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const platform::float16* y, const platform::float16* dy,
                  int num, platform::float16* dx) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    KeReluGradfp16<<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

#endif

}  // namespace operators
}  // namespace paddle

REGISTER_ACTIVATION_CUDA_KERNEL(relu, Relu, ReluFunctor, ReluGradFunctor);
