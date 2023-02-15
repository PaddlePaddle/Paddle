// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/unpool_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T>
__global__ void KernelUnpool2dMaxGrad(const int nthreads,
                                      const T* input_data,
                                      const int* indices_data,
                                      const int input_height,
                                      const int input_width,
                                      const int channels,
                                      const T* output_data,
                                      const T* output_grad,
                                      const int output_height,
                                      const int output_width,
                                      T* input_grad) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_width / input_height) % channels;
    int n = linearIndex / input_width / input_height / channels;
    output_grad += (n * channels + c) * output_height * output_width;
    int maxind = indices_data[linearIndex];
    input_grad[linearIndex] = output_grad[maxind];
  }
}

template <typename T>
__global__ void KernelUnpool3dMaxGrad(const int nthreads,
                                      const T* input_data,
                                      const int* indices_data,
                                      const int input_depth,
                                      const int input_height,
                                      const int input_width,
                                      const int channels,
                                      const T* output_data,
                                      const T* output_grad,
                                      const int output_depth,
                                      const int output_height,
                                      const int output_width,
                                      T* input_grad) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_depth / input_width / input_height) % channels;
    int n = linearIndex / input_depth / input_width / input_height / channels;
    output_grad +=
        (n * channels + c) * output_depth * output_height * output_width;
    int maxind = indices_data[linearIndex];
    input_grad[linearIndex] = output_grad[maxind];
  }
}

template <typename T, typename Context>
class Unpool2dMaxGradFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  DenseTensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output.dims()[1];
    const int output_height = output.dims()[2];
    const int output_width = output.dims()[3];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
#ifdef __HIPCC__
    int threads = 256;
#else
    int threads = 1024;
#endif
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool2dMaxGrad<T>
        <<<grid, threads, 0, dev_ctx.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_grad_data,
                                                 output_height,
                                                 output_width,
                                                 input_grad_data);
  }
};

template <typename T, typename Context>
class Unpool3dMaxGradFunctor {
 public:
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& indices,
                  const DenseTensor& output,
                  const DenseTensor& output_grad,
                  DenseTensor* input_grad) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output.dims()[1];
    const int output_depth = output.dims()[2];
    const int output_height = output.dims()[3];
    const int output_width = output.dims()[4];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    const T* output_data = output.data<T>();
    const T* output_grad_data = output_grad.data<T>();
    T* input_grad_data = dev_ctx.template Alloc<T>(input_grad);
#ifdef __HIPCC__
    int threads = 256;
#else
    int threads = 1024;
#endif
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool3dMaxGrad<T>
        <<<grid, threads, 0, dev_ctx.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_grad_data,
                                                 output_depth,
                                                 output_height,
                                                 output_width,
                                                 input_grad_data);
  }
};

template <typename T, typename Context>
void UnpoolGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& indices,
                      const DenseTensor& out,
                      const DenseTensor& out_grad,
                      const std::vector<int>& ksize,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const IntArray& output_size,
                      const std::string& data_format,
                      DenseTensor* x_grad) {
  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, x_grad, static_cast<T>(0));
  Unpool2dMaxGradFunctor<T, Context> unpool2d_max_backward;
  unpool2d_max_backward(dev_ctx, x, indices, out, out_grad, x_grad);
}

template <typename T, typename Context>
void Unpool3dGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& indices,
                        const DenseTensor& out,
                        const DenseTensor& out_grad,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& paddings,
                        const std::vector<int>& output_size,
                        const std::string& data_format,
                        DenseTensor* x_grad) {
  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();
  phi::funcs::SetConstant<Context, T> zero;
  zero(dev_ctx, x_grad, static_cast<T>(0));
  Unpool3dMaxGradFunctor<T, Context> unpool3d_max_backward;
  unpool3d_max_backward(dev_ctx, x, indices, out, out_grad, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unpool_grad, GPU, ALL_LAYOUT, phi::UnpoolGradKernel, float, double) {}

PD_REGISTER_KERNEL(
    unpool3d_grad, GPU, ALL_LAYOUT, phi::Unpool3dGradKernel, float, double) {}
