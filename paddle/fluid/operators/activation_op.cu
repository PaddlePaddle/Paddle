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
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using float16 = paddle::platform::float16;

template <typename T, int N>
struct GetVecType;

template <typename T>
struct GetVecType<T, 1> {
  using type = T;
};

template <>
struct GetVecType<platform::float16, 2> {
  using type = __half2;
};

template <>
struct GetVecType<float, 4> {
  using type = float4;
};

template <typename T>
class BaseActivationGPUFunctor {
 public:
  using ELEMENT_TYPE = T;
};

template <typename Type>
class ReluGPUFunctor : public BaseActivationGPUFunctor<Type> {
 public:
  // for relu forward and backward when T is double
  template <typename T>
  __device__ __forceinline__ T compute(const T* a, const T* b) {
#ifdef __CUDA_ARCH__ >= 350 || HIP_VERSION >= 300
    return __ldg(a) > static_cast<T>(0) ? __ldg(b) : static_cast<T>(0);
#else
    return (*a) > static_cast<T>(0) ? (*b) : static_cast<T>(0);
#endif
  }

  // when num % vecsize != 0 this func will be used
  template <typename T>
  __device__ __forceinline__ T computeReminder(const T a, const T b) {
    return a > static_cast<T>(0) ? b : static_cast<T>(0);
  }
};

template <>
__device__ __forceinline__ float4
ReluGPUFunctor<float>::compute<float4>(const float4* a, const float4* b) {
  return make_float4((a->x > 0.0f) * (b->x), (a->y > 0.0f) * (b->y),
                     (a->z > 0.0f) * (b->z), (a->w > 0.0f) * (b->w));
}

template <>
__device__ __forceinline__ __half2
ReluGPUFunctor<float16>::compute<__half2>(const __half2* a, const __half2* b) {
#ifdef __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
  const half2 kzero = __float2half2_rn(0.0f);
  return __hmul2(__hgt2(__ldg(a), kzero), __ldg(b));
#else
  const float2 xx = __half22float2(*a);
  const float2 yy = __half22float2(*b);
  return __floats2half2_rn((xx.x > 0.0f) * static_cast<float>(yy.x),
                           (xx.y > 0.0f) * static_cast<float>(yy.y));
#endif
}

template <typename T, typename VecType, int VecSize, typename Functor>
__global__ void reluKernelCudaVec(const T* src1, const T* src2, T* dst, int num,
                                  Functor functor) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int loop = num / VecSize;
  int tail = num % VecSize;
  const VecType* in1 = reinterpret_cast<const VecType*>(src1);
  const VecType* in2 = reinterpret_cast<const VecType*>(src2);
  VecType* out = reinterpret_cast<VecType*>(dst);

  for (int i = idx; i < loop; i += stride) {
    out[i] = functor.compute((in1 + i), (in2 + i));
  }

  while (idx == loop && tail) {
    dst[num - tail] =
        functor.computeReminder(src1[num - tail], src2[num - tail]);
    --tail;
  }
}

template <typename DeviceContext, typename Functor, int VecSize>
class ActivationGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    auto& dev_ctx = context.template device_context<DeviceContext>();

    int num = in_x->numel();
    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>(dev_ctx.GetPlace(),
                                          static_cast<size_t>(num * sizeof(T)));

    int block = 512;
    int grid = (num + block - 1) / block;
    if (num > VecSize) {
      // update grid for float16 and float32
      grid = (num / VecSize + block - 1) / block;
    }

    Functor functor;
    using VecType = typename GetVecType<T, VecSize>::type;
    reluKernelCudaVec<T, VecType, VecSize, Functor><<<grid, block>>>(
        input_data, input_data, output_data, num, functor);
  }
};

template <typename DeviceContext, typename Functor, int VecSize>
class ActivationGradGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* out = ctx.Input<framework::Tensor>("Out");
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* out_data = out->data<T>();
    auto numel = d_out->numel();
    auto* dout_data = d_out->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        dev_ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));

    int block = 512;
    int grid = (numel + block - 1) / block;
    if (numel >= VecSize) {
      // update grid for float16 and float32
      grid = (numel / VecSize + block - 1) / block;
    }

    Functor functor;
    using VecType = typename GetVecType<T, VecSize>::type;
    reluKernelCudaVec<T, VecType, VecSize, Functor><<<grid, block>>>(
        out_data, dout_data, dx_data, numel, functor);
  }
};

}  // namespace operators
}  // namespace paddle

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

/* ======================== elu register  ============================ */
REGISTER_ACTIVATION_CUDA_KERNEL(elu, ELU, ELUFunctor, ELUGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    elu_grad_grad, ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::ELUGradGradFunctor<float>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<double>>,
    ops::ELUDoubleGradKernel<plat::CUDADeviceContext,
                             ops::ELUGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================    relu register  ============================ */
REGISTER_OP_CUDA_KERNEL(
    relu, ops::ActivationGPUKernel<paddle::platform::CUDADeviceContext,
                                   ops::ReluGPUFunctor<float>, 4>,
    ops::ActivationGPUKernel<paddle::platform::CUDADeviceContext,
                             ops::ReluGPUFunctor<double>, 1>,
    ops::ActivationGPUKernel<plat::CUDADeviceContext,
                             ops::ReluGPUFunctor<plat::float16>, 2>);

REGISTER_OP_CUDA_KERNEL(
    relu_grad, ops::ActivationGradGPUKernel<paddle::platform::CUDADeviceContext,
                                            ops::ReluGPUFunctor<float>, 4>,
    ops::ActivationGradGPUKernel<paddle::platform::CUDADeviceContext,
                                 ops::ReluGPUFunctor<double>, 1>,
    ops::ActivationGradGPUKernel<plat::CUDADeviceContext,
                                 ops::ReluGPUFunctor<plat::float16>, 2>);

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

/* ===========================   rsqrt register  =============================
 */
REGISTER_ACTIVATION_CUDA_KERNEL(rsqrt, Rsqrt, RsqrtFunctor, RsqrtGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    rsqrt_grad_grad,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<float>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<double>>,
    ops::RsqrtDoubleGradKernel<paddle::platform::CUDADeviceContext,
                               ops::RsqrtGradGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ===========================  square register  ============================ */
REGISTER_OP_CUDA_KERNEL(
    square,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<float>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::SquareFunctor<int64_t>>,
    ops::ActivationKernel<plat::CUDADeviceContext,
                          ops::SquareFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    square_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,
                                           ops::SquareGradFunctor<float>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<double>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<int>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<int64_t>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::SquareGradFunctor<plat::float16>>);

REGISTER_OP_CUDA_KERNEL(
    square_grad_grad,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<float>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<double>>,
    ops::SquareDoubleGradKernel<plat::CUDADeviceContext,
                                ops::SquareGradGradFunctor<plat::float16>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int>>,
    ops::SquareDoubleGradKernel<paddle::platform::CUDADeviceContext,
                                ops::SquareGradGradFunctor<int64_t>>);
/* ========================================================================== */

/* ==========================   pow register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    pow, ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<float>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<double>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<int64_t>>,
    ops::PowKernel<plat::CUDADeviceContext, ops::PowFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    pow_grad,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<float>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<double>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int>>,
    ops::PowGradKernel<plat::CUDADeviceContext, ops::PowGradFunctor<int64_t>>,
    ops::PowGradKernel<plat::CUDADeviceContext,
                       ops::PowGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================   exp register  ============================ */

REGISTER_OP_CUDA_KERNEL(
    exp, ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<float>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<double>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int>>,
    ops::ActivationKernel<plat::CUDADeviceContext, ops::ExpFunctor<int64_t>>,
    ops::ActivationKernel<plat::CUDADeviceContext,
                          ops::ExpFunctor<plat::float16>>);
REGISTER_OP_CUDA_KERNEL(
    exp_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,
                                        ops::ExpGradFunctor<float>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<double>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<int>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<int64_t>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ExpGradFunctor<plat::float16>>);
/* ========================================================================== */

/* ==========================  Log register ==================================*/
REGISTER_ACTIVATION_CUDA_KERNEL(log, Log, LogFunctor, LogGradFunctor);

REGISTER_OP_CUDA_KERNEL(
    log_grad_grad, ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                                            ops::LogGradGradFunctor<float>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<double>>,
    ops::LogDoubleGradKernel<plat::CUDADeviceContext,
                             ops::LogGradGradFunctor<plat::float16>>);
/* ========================================================================== */
