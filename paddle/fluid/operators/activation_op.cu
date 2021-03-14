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
// result = a > 0 ? b : 0;
template <typename T>
__device__ __forceinline__ T reluFunc(const T* a, const T* b);

template <typename T, int N>
struct GetVecType;

template <typename T>
struct GetVecType<T, 1> {
  using type = T;
};
template <>
struct GetVecType<platform::float16, 2> {
  using type = half2;
};

template <>
struct GetVecType<float, 4> {
  using type = float4;
};
// for relu forward and backward
template <>
__device__ __forceinline__ double reluFunc<double>(const double* a,
                                                   const double* b) {
  return ((*a) > 0.0f) * (*b);
}

template <>
__device__ __forceinline__ float4 reluFunc<float4>(const float4* a,
                                                   const float4* b) {
  return make_float4((a->x > 0.0f) * (b->x), (a->y > 0.0f) * (b->y),
                     (a->z > 0.0f) * (b->z), (a->w > 0.0f) * (b->w));
}

template <>
__device__ __forceinline__ __half2 reluFunc<__half2>(const __half2* a,
                                                     const __half2* b) {
#if __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
  const half2 kzero = __float2half2_rn(0.0f);
  return __hmul2(__hgt2(__ldg(a), kzero), __ldg(b));
#else
  const float2 xx = __half22float2(*a);
  const float2 yy = __half22float2(*b);
  return __floats2half2_rn((xx.x > 0.0f) * static_cast<float>(yy.x),
                           (xx.y > 0.0f) * static_cast<float>(yy.y));
#endif
}

// relu forward kernel
template <typename T, typename VecType, int VECSIZE>
__global__ void reluKernelCudaVec(const T* in, T* out, int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int loop = num / VECSIZE;
  int tail = num % VECSIZE;
  const VecType* src = reinterpret_cast<const VecType*>(in);
  VecType* dst = reinterpret_cast<VecType*>(out);
  for (int i = idx; i < loop; i += stride) {
    dst[i] = reluFunc((src + i), (src + i));
  }
  T temp;
  T zero = (T)(0.0f);
  while (idx == loop && tail) {
    temp = in[num - tail];
    out[num - tail] = temp > zero ? temp : zero;
    --tail;
  }
}
// relu backward kernel
template <typename T, typename VecType, int VECSIZE>
__global__ void reluGradKernelVec(const T* out, const T* dout, T* dx, int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int loop = num / VECSIZE;
  int tail = num % VECSIZE;
  const VecType* out_v = reinterpret_cast<const VecType*>(out);
  const VecType* dout_v = reinterpret_cast<const VecType*>(dout);
  VecType* dx_out = reinterpret_cast<VecType*>(dx);
  for (int i = idx; i < loop; i += stride) {
    dx_out[i] = reluFunc((out_v + i), (dout_v + i));
  }
  T zero = (T)(0.0f);
  while (idx == loop && tail) {
    dx[num - tail] = out[num - tail] > zero ? dout[num - tail] : zero;
    --tail;
  }
}
template <typename T>
__global__ void reluGradFunc(const T* out, const T* dout, T* dx, int num) {
  const float4* out_f = reinterpret_cast<const float4*>(out);
  const float4* dout_f = reinterpret_cast<const float4*>(dout);
  float4* dx_f = reinterpret_cast<float4*>(dx);
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int loop = num / 4;
  int tail = num % 4;
  for (int t = i; t < loop; t += blockDim.x * gridDim.x) {
    dx_f[t].x = (out_f[t].x > 0.0f) * dout_f[t].x;
    dx_f[t].y = (out_f[t].y > 0.0f) * dout_f[t].y;
    dx_f[t].z = (out_f[t].z > 0.0f) * dout_f[t].z;
    dx_f[t].w = (out_f[t].w > 0.0f) * dout_f[t].w;
  }
  T zero = (T)(0.0f);
  while (i == loop && tail) {
    dx[num - tail] = out[num - tail] > zero ? dout[num - tail] : zero;
    --tail;
  }
}
template <typename T>
__global__ void reluGradFunc_half(const T* out, const T* dout, T* dx,
                                  size_t num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx * 2 >= num) return;
  int loop = num >> 1;
  int stride = blockDim.x * gridDim.x;
  const __half2* out_half = reinterpret_cast<const __half2*>(out);
  const __half2* dout_half = reinterpret_cast<const __half2*>(dout);
  __half2* dx_half = reinterpret_cast<__half2*>(dx);

  const half2 kzero = __float2half2_rn(0.0f);

  for (int i = idx; i < loop; i += stride) {
#if __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
    dx_half[i] =
        __hmul2(__hgt2(__ldg(out_half + i), kzero), __ldg(dout_half + i));
#else
    const float2 xx = __half22float2(out[i]);
    const float2 yy = __half22float2(dout[i]);
    dx_half[i] =
        __floats2half2_rn(xx.x > 0.0f ? static_cast<float>(yy.x) : 0.0f,
                          xx.y > 0.0f ? static_cast<float>(yy.y) : 0.0f);
#endif
  }
}

template <typename T, int VECSIZE>
struct ReluGPUFunctor : public BaseActivationFunctor<T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& in, framework::Tensor* out,
                  int num) {
    const T* input_data = in.data<T>();
    T* output_data = out->mutable_data<T>(context.GetPlace());
    using VecType = typename GetVecType<T, VECSIZE>::type;
    int block = 512;
    if (num > VECSIZE) {
      int grid = (num / VECSIZE + block - 1) / block;
      reluKernelCudaVec<T, VecType, VECSIZE><<<grid, block>>>(input_data,
                                                              output_data, num);
    } else {
      int grid1 = (num + block - 1) / block;
      reluKernelCudaVec<T, T, 1><<<grid1, block>>>(input_data, output_data,
                                                   num);
    }
  }
};
template <typename DeviceContext, typename Functor>
class ReluBaseKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int num = in_x->numel();
    Functor functor;
    functor(dev_ctx, *in_x, out, num);
  }
};

template <typename T, int VECSIZE>
struct ReluGradGPUFunctor : public BaseActivationFunctor<T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor* out, const framework::Tensor* d_out,
                  framework::Tensor* d_x) {
    auto numel = d_out->numel();
    auto* dout_data = d_out->data<T>();
    auto* out_data = out->data<T>();
    auto* dx_data = d_x->mutable_data<T>(
        context.GetPlace(), static_cast<size_t>(numel * sizeof(T)));
    using VecType = typename GetVecType<T, VECSIZE>::type;
    int block = 512;
    if (numel > VECSIZE) {
      int grid = (numel / VECSIZE + block - 1) / block;
      reluGradKernelVec<T, VecType, VECSIZE><<<grid, block>>>(
          out_data, dout_data, dx_data, numel);
    } else {
      int grid1 = (numel + block - 1) / block;
      reluGradKernelVec<T, T, 1><<<grid1, block>>>(out_data, dout_data, dx_data,
                                                   numel);
    }
  }
  // static constexpr ActBwdOpFwdDeps FwdDeps() { return kDepOut; }
};

template <typename DeviceContext, typename Functor>
class ReluGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  using T = typename Functor::ELEMENT_TYPE;
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* out = ctx.Input<framework::Tensor>("Out");
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    Functor functor;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    functor(dev_ctx, out, d_out, d_x);
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
    relu, ops::ReluBaseKernel<paddle::platform::CUDADeviceContext,
                              ops::ReluGPUFunctor<float, 4>>,
    ops::ReluBaseKernel<paddle::platform::CUDADeviceContext,
                        ops::ReluGPUFunctor<double, 1>>,
    ops::ReluBaseKernel<paddle::platform::CUDADeviceContext,
                        ops::ReluGPUFunctor<plat::float16, 2>>);

REGISTER_OP_CUDA_KERNEL(
    relu_grad, ops::ReluGradKernel<plat::CUDADeviceContext,
                                   ops::ReluGradGPUFunctor<float, 4>>,
    ops::ReluGradKernel<plat::CUDADeviceContext,
                        ops::ReluGradGPUFunctor<double, 1>>,
    ops::ReluGradKernel<plat::CUDADeviceContext,
                        ops::ReluGradGPUFunctor<plat::float16, 2>>);
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
