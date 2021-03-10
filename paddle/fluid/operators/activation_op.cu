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
template <typename T>
__global__ void reluKernelCudaHalf2(const T* in, T* out, int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int loop = num >> 1;
  int stride = blockDim.x * gridDim.x;
  const __half2* src = reinterpret_cast<const __half2*>(in);
  __half2* dst = reinterpret_cast<__half2*>(out);
  const half2 kzero = __float2half2_rn(0.0f);
  for (int i = idx; i < loop; i += stride) {
#if __CUDA_ARCH__ >= 530 || CUDA_VERSION >= 300
    dst[i] = __hmul2(__hgt2(__ldg(src + i), kzero), __ldg(src + i));
#else
    const float2 xx = __halfi22float2(src[i]);
    dst[i] = __floats2half2_rn(xx.x > 0.0f ? static_cast<float>(xx.x) : 0.0f,
                               xx.y > 0.0f ? static_cast<float>(xx.y) : 0.0f);
#endif
  }

  if (idx == loop && num % 2 == 1) {
    T zero = (T)(0.0f);
    out[num - 1] = in[num - 1] > zero ? in[num - 1] : zero;
  }
}
template <typename T>
__global__ void reluKernelCudaFloat4(const T* in, T* out, int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const float4* src = reinterpret_cast<const float4*>(in);
  float4* dst = reinterpret_cast<float4*>(out);
  float4 temp;
  int loop = num >> 2;
  for (int i = idx; i < loop; i += blockDim.x * gridDim.x) {
    temp = src[i];
    temp.x = max(temp.x, 0.0f);
    temp.y = max(temp.y, 0.0f);
    temp.z = max(temp.z, 0.0f);
    temp.w = max(temp.w, 0.0f);
    dst[i] = temp;
  }
  int tail = num % 4;
  while (idx == loop && tail) {
    temp.x = static_cast<float>(in[num - tail]);
    out[num - tail] = max(temp.x, 0.0f);
    --tail;
  }
}
template <typename T>
__global__ void reluKernelCudaDouble(const T* src, T* dst, int num) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  T zero = (T)(0.0f);

  for (int idx = i; idx < num; idx += gridDim.x * blockDim.x) {
    dst[i] = src[idx] > zero ? src[idx] : zero;
  }
}
template <typename T, int vec>
struct ReluGPUFunctor : public BaseActivationFunctor<T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& in, framework::Tensor* out,
                  int num) {
    const T* input_data = in.data<T>();
    T* output_data = out->mutable_data<T>(context.GetPlace());
    int block = 512;
    int grid = (num + block - 1) / block;
    int grid2 = (num / 2 + block - 1) / block;
    int grid4 = (num / 4 + block - 1) / block;
    switch (vec) {
      case 1:  // float16 -> half2
        reluKernelCudaHalf2<<<grid2, block>>>(input_data, output_data, num);
        break;
      case 2:  // float -> float4
        reluKernelCudaFloat4<<<grid4, block>>>(input_data, output_data, num);
        break;
      case 4:  // double -> double
        reluKernelCudaDouble<<<grid, block>>>(input_data, output_data, num);
        break;
      default:
        break;
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

template <typename DeviceContext, typename Functor>
class ReluGradKernel
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
                              ops::ReluGPUFunctor<float, 2>>,
    ops::ReluBaseKernel<paddle::platform::CUDADeviceContext,
                        ops::ReluGPUFunctor<double, 4>>,
    ops::ReluBaseKernel<paddle::platform::CUDADeviceContext,
                        ops::ReluGPUFunctor<plat::float16, 1>>);

REGISTER_OP_CUDA_KERNEL(
    relu_grad, ops::ActivationGradKernel<plat::CUDADeviceContext,
                                         ops::ReluGradFunctor<float>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ReluGradFunctor<double>>,
    ops::ActivationGradKernel<plat::CUDADeviceContext,
                              ops::ReluGradFunctor<plat::float16>>);
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
