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

static const int CUDA_NUM_THREADS = 1024;
static const int CUDA_MAX_NUM_BLOCKS = 65535;
inline static int GET_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void PReluChannelWiseKernel(const T *input, const T *alpha,
                                       T *output, int channel,
                                       size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  T *out = output + offset;
  T scale = alpha[blockIdx.x % channel];

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    out[i] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
__global__ void PReluElementWiseKernel(const T *input, const T *alpha,
                                       T *output, size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  const T *scale = alpha + offset;
  T *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    out[i] = (x > 0) ? x : scale[i] * x;
  }
}

template <typename T>
__global__ void PReluScalarKernel(const T *input, const T *alpha, T *output,
                                  size_t spatial_size) {
  size_t offset = blockIdx.x * spatial_size;
  const T *in = input + offset;
  T scale = *alpha;
  T *out = output + offset;

  for (size_t i = threadIdx.x; i < spatial_size; i += blockDim.x) {
    T x = in[i];
    out[i] = (x > 0) ? x : scale * x;
  }
}

template <typename T>
static inline void PReluChannelWise(cudaStream_t stream, const T *input,
                                    const T *alpha, T *output,
                                    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluChannelWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, input_shape[1], spatial_size);
}

template <typename T>
static inline void PReluElementWise(cudaStream_t stream, const T *input,
                                    const T *alpha, T *output,
                                    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluElementWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
}

template <typename T>
static inline void PReluScalar(cudaStream_t stream, const T *input,
                               const T *alpha, T *output,
                               std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluScalarKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
}

template <typename T>
void PreluChannelWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluChannelWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, input_shape[1], spatial_size);
}

template <typename T>
void PreluElementWiseDirectCUDAFunctor<T>::operator()(
    cudaStream_t stream, const T *input, const T *alpha, T *output,
    std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluElementWiseKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
}

template <typename T>
void PreluScalarDirectCUDAFunctor<T>::operator()(cudaStream_t stream,
                                                 const T *input, const T *alpha,
                                                 T *output,
                                                 std::vector<int> input_shape) {
  size_t unroll = input_shape[0] * input_shape[1];
  size_t spatial_size = input_shape[2] * input_shape[3];
  CHECK_LT(unroll, CUDA_MAX_NUM_BLOCKS);
  PReluScalarKernel<<<unroll, CUDA_NUM_THREADS, 0, stream>>>(
      input, alpha, output, spatial_size);
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
