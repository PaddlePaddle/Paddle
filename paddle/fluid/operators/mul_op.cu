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

#include <limits>
#include "paddle/fluid/operators/mul_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__device__ T from_float(float x);

template <>
__device__ __forceinline__ float from_float<float>(float x) {
  return x;
}

template <>
__device__ __forceinline__ int8_t from_float<int8_t>(float x) {
  x = fmaxf(x, std::numeric_limits<char>::min());
  x = fminf(x, std::numeric_limits<char>::max());
  return __float2int_rn(x);
}

__global__ void Fp32ToInt8Kernel(const int num, const float scale,
                                 const float* input, int8_t* output) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num) {
    output[index] = from_float<int8_t>(input[index] / scale);
  }
}

template <typename T>
__global__ void ScaleKernel(int count, const T* in_data, const T scale,
                            T* out_data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    out_data[tid] = scale * in_data[tid];
  }
}

template <>
struct QuantFp32ToInt8Functor<platform::CUDADeviceContext> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& in, const float scale,
                  framework::Tensor* out) {
    const float* input = in.data<float>();
    int numel = in.numel();
    int threads = 1024;
    int blocks = (numel + threads - 1) / threads;
    int8_t* output = out->mutable_data<int8_t>(context.GetPlace());
    Fp32ToInt8Kernel<<<blocks, threads, 0, context.stream()>>>(numel, scale,
                                                               input, output);
  }
  /*
  void operator()(const platform::CUDADeviceContext &context, const
  framework::Tensor& in,
                  const std::vector<float>& scale, framework::Tensor* out);
  void operator()(const platform::CUDADeviceContext &context, const
  framework::Tensor& in,
                  const framework::Tensor& scale, framework::Tensor* out);
  */
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(mul, ops::MulKernel<plat::CUDADeviceContext, float>,
                        ops::MulKernel<plat::CUDADeviceContext, double>,
                        ops::MulKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    mul_grad, ops::MulGradKernel<plat::CUDADeviceContext, float>,
    ops::MulGradKernel<plat::CUDADeviceContext, double>,
    ops::MulGradKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    mul_grad_grad,
    ops::MulDoubleGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MulDoubleGradKernel<paddle::platform::CUDADeviceContext, double>);
