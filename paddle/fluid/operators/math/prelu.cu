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

#include "paddle/fluid/operators/math/prelu.h"

namespace paddle {
namespace operators {
namespace math {

#define CUDA_NUM_THREADS 1024

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

inline static int PADDLE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void PReluChannelWiseKernel(const T *input, const T *alpha,
                                       T *output, size_t channel_num,
                                       size_t plane_size, size_t numel) {
  size_t index;
  CUDA_KERNEL_LOOP(index, numel) {
    size_t temp = index / plane_size;
    size_t channel_index = temp % channel_num;
    T scale = alpha[channel_index];
    T x = input[index];
    output[index] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
__global__ void PReluElementWiseKernel(const T *input, const T *alpha,
                                       T *output, size_t spatial_size,
                                       size_t numel) {
  size_t index;
  CUDA_KERNEL_LOOP(index, numel) {
    size_t element_index = index % spatial_size;
    T scale = alpha[element_index];
    T x = input[index];
    output[index] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
__global__ void PReluScalarKernel(const T *input, const T *alpha, T *output,
                                  size_t numel) {
  T scale = alpha[0];
  size_t index;
  CUDA_KERNEL_LOOP(index, numel) {
    T x = input[index];
    output[index] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
void PreluChannelWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = input_shape[1] * plane_size;
  size_t numel = input_shape[0] * spatial_size;
  PReluChannelWiseKernel<<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0,
                           stream>>>(input, alpha, output, input_shape[1],
                                     plane_size, numel);
}

template <typename T>
void PreluElementWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = input_shape[1] * plane_size;
  size_t numel = input_shape[0] * spatial_size;
  PReluElementWiseKernel<<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0,
                           stream>>>(input, alpha, output, spatial_size, numel);
}

template <typename T>
void PreluScalarDirectCUDAFunctor<T>::operator()(cudaStream_t stream,
                                                 const T *input, const T *alpha,
                                                 T *output,
                                                 std::vector<int> input_shape) {
  size_t plane_size = input_shape[2] * input_shape[3];
  size_t spatial_size = input_shape[1] * plane_size;
  size_t numel = input_shape[0] * spatial_size;
  PReluScalarKernel<<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, numel);
}

template class PreluChannelWiseDirectCUDAFunctor<float>;
template class PreluChannelWiseDirectCUDAFunctor<double>;

template class PreluElementWiseDirectCUDAFunctor<float>;
template class PreluElementWiseDirectCUDAFunctor<double>;

template class PreluScalarDirectCUDAFunctor<float>;
template class PreluScalarDirectCUDAFunctor<double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
