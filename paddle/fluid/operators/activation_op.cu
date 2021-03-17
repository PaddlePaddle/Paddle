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

template <typename T>
struct GetVecType;

template <>
struct GetVecType<double> {
  using type = double;
};

template <>
struct GetVecType<platform::float16> {
  using type = __half2;
};

template <>
struct GetVecType<float> {
  using type = float4;
};

template <typename T>
class BaseGPUFunctor {
 public:
  using ELEMENT_TYPE_ = T;
};

/* ========================================================================== */

/* ===========================    relu forward   ============================ */
template <typename Type>
class ReluGPUFuctor : public BaseGPUFunctor<Type> {
 public:
  // for relu forward and backward when T is double
  template <typename T>
  __device__ __forceinline__ T Compute(const T* a) {
#ifdef __CUDA_ARCH__ >= 350 || HIP_VERSION >= 300
    return __ldg(a) > static_cast<T>(0.0f) ? __ldg(a) : static_cast<T>(0.0f);
#else
    return (*a) > static_cast<T>(0.0f) ? (*a) : static_cast<T>(0.0f);
#endif
  }
  // when num % vecsize != 0 this func will be used
  template <typename T>
  __device__ __forceinline__ T ComputeReminder(const T a) {
    return a > static_cast<T>(0.0f) ? a : static_cast<T>(0.0f);
  }
};

template <>
template <>
__device__ __forceinline__ float4
ReluGPUFuctor<float>::Compute<float4>(const float4* a) {
  return make_float4((a->x > 0.0f) * (a->x), (a->y > 0.0f) * (a->y),
                     (a->z > 0.0f) * (a->z), (a->w > 0.0f) * (a->w));
}

template <>
template <>
__device__ __forceinline__ __half2
ReluGPUFuctor<float16>::Compute<__half2>(const __half2* a) {
#ifdef __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
  const half2 kzero = __float2half2_rn(0.0f);
  return __hmul2(__hgt2(__ldg(a), kzero), __ldg(a));
#else
  const float2 xx = __half22float2(*a);
  return __floats2half2_rn((xx.x > 0.0f) * static_cast<float>(xx.x),
                           (xx.y > 0.0f) * static_cast<float>(xx.y));
#endif
}
/* ========================================================================== */

/* ===========================    relu backward   ============================
 */

template <typename Type>
class ReluGradGPUFunctor : public BaseGPUFunctor<Type> {
 public:
  // for relu forward and backward when T is double
  template <typename T>
  __device__ __forceinline__ T Compute(const T* a, const T* b);

  // when num % vecsize != 0 this func will be used
  template <typename T>
  __device__ __forceinline__ T ComputeReminder(const T a, const T b) {
    return a > static_cast<T>(0) ? b : static_cast<T>(0);
  }

  static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <>
template <>
__device__ __forceinline__ double ReluGradGPUFunctor<double>::Compute<double>(
    const double* a, const double* b) {
#ifdef __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
  return __ldg(a) > static_cast<double>(0.0f) ? __ldg(b)
                                              : static_cast<double>(0.0f);
#else
  return (*a) > static_cast<double>(0.0f) ? (*b) : static_cast<double>(0.0f);
#endif
}

template <>
template <>
__device__ __forceinline__ float4
ReluGradGPUFunctor<float>::Compute<float4>(const float4* a, const float4* b) {
  return make_float4((a->x > 0.0f) * (b->x), (a->y > 0.0f) * (b->y),
                     (a->z > 0.0f) * (b->z), (a->w > 0.0f) * (b->w));
}

template <>
template <>
__device__ __forceinline__ __half2
ReluGradGPUFunctor<float16>::Compute<__half2>(const __half2* a,
                                              const __half2* b) {
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

/* ========================================================================== */

template <typename T, typename VecType, typename Functor>
__global__ void ActivationGradKernelVec(const T* forward_data, const T* dout,
                                        T* dx, int num, int vecsize,
                                        Functor functor) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int loop = num / vecsize;
  int tail = num % vecsize;
  const VecType* in_forward = reinterpret_cast<const VecType*>(forward_data);
  const VecType* in_dout = reinterpret_cast<const VecType*>(dout);
  VecType* out = reinterpret_cast<VecType*>(dx);

  for (int i = idx; i < loop; i += stride) {
    out[i] = functor.Compute((in_forward + i), (in_dout + i));
  }

  while (idx == loop && tail) {
    dx[num - tail] =
        functor.ComputeReminder(forward_data[num - tail], dout[num - tail]);
    --tail;
  }
}

template <typename T, typename VecType, typename Functor>
__global__ void ActivationKerenlVec(const T* src, T* dst, int num, int vecsize,
                                    Functor functor) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int loop = num / vecsize;
  int tail = num % vecsize;
  const VecType* in = reinterpret_cast<const VecType*>(src);
  VecType* out = reinterpret_cast<VecType*>(dst);

  for (int i = idx; i < loop; i += stride) {
    out[i] = functor.Compute((in + i));
  }

  while (idx == loop && tail) {
    dst[num - tail] = functor.ComputeReminder(src[num - tail]);
    --tail;
  }
}

template <typename DeviceContext, typename Functor>
class ActivationGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE_> {
 public:
  using T = typename Functor::ELEMENT_TYPE_;
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor* in_x = nullptr;
    framework::Tensor* out = nullptr;
    ExtractActivationTensor(context, &in_x, &out);
    auto& dev_ctx = context.template device_context<DeviceContext>();

    int num = in_x->numel();
    const T* input_data = in_x->data<T>();
    T* output_data = out->mutable_data<T>(dev_ctx.GetPlace(),
                                          static_cast<size_t>(num * sizeof(T)));

    using VecType = typename GetVecType<T>::type;
    int vecsize = sizeof(VecType) / sizeof(T);
    int block = 512;
    int grid = (num + block - 1) / block;
    if (num > vecsize) {
      // update grid for float16 and float32
      grid = (num / vecsize + block - 1) / block;
    }

    Functor functor;
    ActivationKerenlVec<T, VecType, Functor><<<grid, block>>>(
        input_data, output_data, num, vecsize, functor);
  }
};

template <typename DeviceContext, typename Functor>
class ActivationGradGPUKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE_> {
 public:
  using T = typename Functor::ELEMENT_TYPE_;
  void Compute(const framework::ExecutionContext& context) const override {
    const framework::Tensor *x, *out, *d_out;
    framework::Tensor* d_x = nullptr;
    x = out = d_out = nullptr;
    ExtractActivationGradTensor<Functor::FwdDeps()>(context, &x, &out, &d_out,
                                                    &d_x);
    auto numel = d_out->numel();
    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto* dx_data = d_x->mutable_data<T>(
        dev_ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));
    auto* dout_data = d_out->data<T>();

    auto* forward_data = dout_data;
    if (static_cast<int>(Functor::FwdDeps()) == static_cast<int>(kDepOut)) {
      forward_data = out->data<T>();
      // Only need forward output Out
    } else if (static_cast<int>(Functor::FwdDeps()) ==
               static_cast<int>(kDepX)) {
      // Only need forward input X
      forward_data = x->data<T>();
    }

    using VecType = typename GetVecType<T>::type;
    int vecsize = sizeof(VecType) / sizeof(T);
    int block = 512;
    int grid = (numel + block - 1) / block;
    if (numel >= vecsize) {
      // update grid for float16 and float32
      grid = (numel / vecsize + block - 1) / block;
    }

    Functor functor;
    ActivationGradKernelVec<T, VecType, Functor><<<grid, block>>>(
        forward_data, dout_data, dx_data, numel, vecsize, functor);
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
                                   ops::ReluGPUFuctor<float>>,
    ops::ActivationGPUKernel<paddle::platform::CUDADeviceContext,
                             ops::ReluGPUFuctor<double>>,
    ops::ActivationGPUKernel<plat::CUDADeviceContext,
                             ops::ReluGPUFuctor<plat::float16>>);

REGISTER_OP_CUDA_KERNEL(
    relu_grad, ops::ActivationGradGPUKernel<paddle::platform::CUDADeviceContext,
                                            ops::ReluGradGPUFunctor<float>>,
    ops::ActivationGradGPUKernel<paddle::platform::CUDADeviceContext,
                                 ops::ReluGradGPUFunctor<double>>,
    ops::ActivationGradGPUKernel<plat::CUDADeviceContext,
                                 ops::ReluGradGPUFunctor<plat::float16>>);

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
