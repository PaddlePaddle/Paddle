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

namespace paddle {
namespace operators {

#define REGISTER_ACTIVATION_CUDA_KERNEL(act_type, op_name, functor,         \
                                        grad_functor)                       \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type, ops::ActivationPatchKernel<plat::CUDADeviceContext,         \
                                           ops::functor<float>>,            \
      ops::ActivationPatchKernel<plat::CUDADeviceContext,                   \
                                 ops::functor<double>>,                     \
      ops::ActivationPatchKernel<plat::CUDADeviceContext,                   \
                                 ops::functor<plat::float16>>);             \
  REGISTER_OP_CUDA_KERNEL(                                                  \
      act_type##_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,   \
                                                 ops::grad_functor<float>>, \
      ops::ActivationPatchGradKernel<plat::CUDADeviceContext,               \
                                     ops::grad_functor<double>>,            \
      ops::ActivationPatchGradKernel<plat::CUDADeviceContext,               \
                                     ops::grad_functor<plat::float16>>);

#ifdef PADDLE_CUDA_FP16

inline DEVICE half2 half2_relu(const half2& a) {
  float a1 = __low2float(a);
  float a2 = __high2float(a);
  float r1 = max(a1, 0.0);
  float r2 = max(a2, 0.0);
  return __floats2half2_rn(r1, r2);
}

template <>
__global__ void KeRelu<half>(const half* x, const int num, half* y) {
  int start = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n2 = num / 2;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  half2* y2 = reinterpret_cast<half2*>(y);
  for (int i = start; i < n2; i += stride) {
    y2[i] = half2_relu(x2[i]);
  }
  if (start == 0 && (num % 2)) {
    y[num - 1] = max(x[num - 1], 0.0);
  }
}

template <>
struct ReluFunctor<platform::CUDADeviceContext, platform::float16> {
  void operator()(const DeviceContext& dev_ctx, const T* x, int num,
                  T* y) const {
    int num = in_t->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu<half><<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
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

template <>
struct ReluFunctor<platform::CUDADeviceContext, platform::float16> {
  void operator(const platform::float16* y, const platform::float16* dy,
                const int num, platform::float16* dx) {
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
      dx[num - 1] = dy[num - 1] * (y[num - 1] > 0 ? 1. : 0.);
    }
  }
};

#endif

/* ===========================    relu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(relu, Relu, ReluFunctor, ReluGradFunctor);

}  // namespace operators
}  // namespace paddle
