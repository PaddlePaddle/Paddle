// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/funcs/math/unpooling.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace phi {
namespace math {
template <typename T>
__global__ void KernelUnpool2dMax(const int nthreads,
                                  const T* input_data,
                                  const int* indices_data,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_width / input_height) % channels;
    int n = linearIndex / input_width / input_height / channels;
    output_data += (n * channels + c) * output_height * output_width;
    int maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
  }
}

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
/*
 * All tensors are in NCHW format.
 */

template <typename T>
__global__ void KernelUnpool3dMax(const int nthreads,
                                  const T* input_data,
                                  const int* indices_data,
                                  const int input_depth,
                                  const int input_height,
                                  const int input_width,
                                  const int channels,
                                  T* output_data,
                                  const int output_depth,
                                  const int output_height,
                                  const int output_width) {
  CUDA_KERNEL_LOOP(linearIndex, nthreads) {
    int c = (linearIndex / input_depth / input_width / input_height) % channels;
    int n = linearIndex / input_depth / input_width / input_height / channels;
    output_data +=
        (n * channels + c) * output_depth * output_height * output_width;
    int maxind = indices_data[linearIndex];
    output_data[maxind] = input_data[linearIndex];
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
/*
 * All tensors are in NCDHW format.
 */

template <typename T>
class Unpool2dMaxFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& indices,
                  phi::DenseTensor* output) {
    const int batch_size = input.dims()[0];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = context.template Alloc<T>(output);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool2dMax<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_height,
                                                 output_width);
  }
};
/*
 * All tensors are in NCHW format.
 */
template <typename T>
class Unpool2dMaxGradFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& indices,
                  const phi::DenseTensor& output,
                  const phi::DenseTensor& output_grad,
                  phi::DenseTensor* input_grad) {
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
    T* input_grad_data = context.template Alloc<T>(input_grad);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool2dMaxGrad<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
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

template <typename T>
class Unpool3dMaxFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& indices,
                  phi::DenseTensor* output) {
    const int batch_size = input.dims()[0];
    const int input_depth = input.dims()[2];
    const int input_height = input.dims()[3];
    const int input_width = input.dims()[4];
    const int output_channels = output->dims()[1];
    const int output_depth = output->dims()[2];
    const int output_height = output->dims()[3];
    const int output_width = output->dims()[4];
    const T* input_data = input.data<T>();
    const int* indices_data = indices.data<int>();
    T* output_data = context.template Alloc<T>(output);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool3dMax<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
                                                 input_data,
                                                 indices_data,
                                                 input_depth,
                                                 input_height,
                                                 input_width,
                                                 output_channels,
                                                 output_data,
                                                 output_depth,
                                                 output_height,
                                                 output_width);
  }
};
/*
 * All tensors are in NCDHW format.
 */
template <typename T>
class Unpool3dMaxGradFunctor<phi::GPUContext, T> {
 public:
  void operator()(const phi::GPUContext& context,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& indices,
                  const phi::DenseTensor& output,
                  const phi::DenseTensor& output_grad,
                  phi::DenseTensor* input_grad) {
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
    T* input_grad_data = context.template Alloc<T>(input_grad);
    int threads = 1024;
    int grid = (input.numel() + threads - 1) / threads;
    KernelUnpool3dMaxGrad<T>
        <<<grid, threads, 0, context.stream()>>>(input.numel(),
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

template class Unpool2dMaxGradFunctor<phi::GPUContext, float>;
template class Unpool2dMaxGradFunctor<phi::GPUContext, double>;
template class Unpool2dMaxFunctor<phi::GPUContext, float>;
template class Unpool2dMaxFunctor<phi::GPUContext, double>;
template class Unpool3dMaxGradFunctor<phi::GPUContext, float>;
template class Unpool3dMaxGradFunctor<phi::GPUContext, double>;
template class Unpool3dMaxFunctor<phi::GPUContext, float>;
template class Unpool3dMaxFunctor<phi::GPUContext, double>;
}  // namespace math
}  // namespace phi
