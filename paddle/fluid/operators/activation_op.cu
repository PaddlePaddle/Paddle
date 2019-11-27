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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,         \
                                        grad_functor)                       \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type,                                                             \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<float>>,  \
      ops::ActivationKernel<plat::CUDADeviceContext, ops::functor<double>>, \
      ops::ActivationKernel<plat::CUDADeviceContext,                        \
                            ops::functor<plat::float16>>);                  \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type##_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,   \
                                                 ops::grad_functor<float>>, \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<double>>,                 \
      ops::ActivationGradKernel<plat::CUDADeviceContext,                    \
                                ops::grad_functor<plat::float16>>);

FOR_EACH_ACTIVATION_OP(REGISTER_ACTIVATION_CUDA_KERNEL);

/* ======================== leaky relu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(leaky_relu, LeakyRelu, LeakyReluFunctor,
                                LeakyReluGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    leaky_relu_grad_grad,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::LeakyReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<
        plat::CUDADeviceContext, ops::LeakyReluGradGradFunctor<plat::float16>>);
/* ========================================================================== */

REGISTER_OP_CUDA_KERNEL(
    relu_grad_grad,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<float>>,
    ops::ActivationDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<double>>,
    ops::ActivationDoubleGradKernel<plat::CUDADeviceContext,
                                    ops::ReluGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================   sqrt register  ============================= */
REGISTER_ACTIVATION_CUDA_KERNEL(sqrt, Sqrt, SqrtFunctor, SqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    sqrt_grad_grad,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<float>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<double>>,
    ops::SqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                              ops::SqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================  square register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(square, Square, SquareFunctor,
                                SquareGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    square_grad_grad,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<float>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<double>>,
    ops::SquareDoubleGradKernel<plat::CUDADeviceContext,
                                ops::SquareGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================   pow register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    pow, ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<float>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<double>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    pow_grad,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<float>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<double>>,
    ops::PowGradKernel<plat::CUDADeviceContext,
                       ops::PowGradFunctor<plat::float16>>);
/* ========================================================================== */

#define REGISTER_RELU_CUDA_KERNEL(act_type, op_name, functor, grad_functor)    \
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
struct ReluFunctor2<platform::CUDADeviceContext, T> {
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
struct ReluGradFunctor2<platform::CUDADeviceContext, T> {
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
struct ReluFunctor2<platform::CUDADeviceContext, platform::float16> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const platform::float16* x, int num,
                  platform::float16* y) const {
    int block = 512;
    int grid = (num + block - 1) / block;
    VLOG(10) << "ReluFunctor2 proc" << num;
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
struct ReluGradFunctor2<platform::CUDADeviceContext, platform::float16> {
  void operator()(const platform::CUDADeviceContext& dev_ctx,
                  const platform::float16* y, const platform::float16* dy,
                  int num, platform::float16* dx) const {
    int block = 512;
    VLOG(10) << "ReluGradFunctor2 proc" << num;
    int grid = (num + block - 1) / block;
    KeReluGradfp16<<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

#endif

}  // namespace operators
}  // namespace paddle

REGISTER_RELU_CUDA_KERNEL(relu, Relu, ReluFunctor2, ReluGradFunctor2);
