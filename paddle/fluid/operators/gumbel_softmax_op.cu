/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/gumbel_softmax_op.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;

template <typename T>
struct UniformCUDAGenerator {
  T min_, max_;
  unsigned int seed_;
  unsigned int offset_ = 0;
  HOSTDEVICE UniformCUDAGenerator(T min, T max, unsigned int seed)
      : min_(min), max_(max), seed_(seed) {}
  HOSTDEVICE UniformCUDAGenerator(T min, T max, unsigned int seed,
                                  unsigned int offset)
      : min_(min), max_(max), seed_(seed), offset_(offset) {}

  HOSTDEVICE T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(min_, max_);
    rng.discard(n + offset_);
    return dist(rng);
  }
};

template <typename T, size_t BlockDim>
__global__ void OneHotCUDAKernel(const int64_t height, const int64_t width,
                                 const int64_t size_out_axis, const T init,
                                 const T* in, T* out) {
  typedef cub::BlockReduce<KeyValuePair<int, T>, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int64_t idx = blockIdx.x; idx < height; idx += gridDim.x) {
    KeyValuePair<int, T> kv_pair = {-1, init};
    int h = idx / size_out_axis;
    int w = idx % size_out_axis;
    cub::ArgMax reducer;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      kv_pair = reducer(
          {k, in[h * width * size_out_axis + k * size_out_axis + w]}, kv_pair);
    }
    kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, reducer);
    if (threadIdx.x == 0) {
      int index = static_cast<int>(kv_pair.key);
      out[h * width * size_out_axis + index * size_out_axis + w] = 1;
    }
    __syncthreads();
  }
}

template <typename T>
struct OneHotGenerator<platform::CUDADeviceContext, T> {
  static void Transform(const platform::CUDADeviceContext& context,
                        const Tensor& X, Tensor* Out, int axis) {
    const int size_to_axis = SizeToAxis(axis, X.dims());
    const int size_from_axis = SizeFromAxis(axis, X.dims());
    const int size_out_axis = SizeOutAxis(axis, X.dims());
    constexpr int thread_size = 512;
    int64_t max_grid_dimx = context.GetCUDAMaxGridDimSize()[0];
    int64_t height = size_to_axis * size_out_axis;
    int block_size = height < max_grid_dimx ? height : max_grid_dimx;

    Tensor input_tensor;
    input_tensor.mutable_data<T>(Out->dims(), platform::CUDAPlace());
    paddle::framework::TensorCopy(*Out, context.GetPlace(), &input_tensor);
    pten::funcs::set_constant(context, Out, 0.0);
    OneHotCUDAKernel<
        T, thread_size><<<block_size, thread_size, 0, context.stream()>>>(
        height, size_from_axis / size_out_axis, size_out_axis,
        std::numeric_limits<T>::lowest(), input_tensor.data<T>(),
        Out->data<T>());
  }
};

template <typename T>
__global__ void AddGumbelNoiseCUDAKernel(const T* input_data, T* output_data,
                                         T* noise, const float temperature,
                                         int64_t n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int step = blockDim.x * gridDim.x;
  for (int64_t i = index; i < n; i += step) {
    T gumbel_noise = -log(-log(noise[i]));
    output_data[i] = (gumbel_noise + input_data[i]) / temperature;
  }
}

template <typename T>
struct GumbleNoiseGenerator<platform::CUDADeviceContext, T> {
  static void Transform(const platform::CUDADeviceContext& context,
                        const T* input_data, T* output_data, int size_to_axis,
                        int size_from_axis, const float temperature) {
    Tensor random_tensor;
    int64_t size = size_to_axis * size_from_axis;
    T* random_data =
        random_tensor.mutable_data<T>({size}, platform::CUDAPlace());
    thrust::counting_iterator<int64_t> index_sequence_begin(0);

    // generate gumbel noise
    int device_id = context.GetPlace().GetDeviceId();
    auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
    if (gen_cuda->GetIsInitPy()) {
      auto seed_offset = gen_cuda->IncrementOffset(1);
      int64_t gen_offset = size * seed_offset.second;
      thrust::transform(
          index_sequence_begin, index_sequence_begin + size,
          thrust::device_ptr<T>(random_data),
          UniformCUDAGenerator<T>(0.00001, 1, seed_offset.first, gen_offset));
    } else {
      const unsigned int seed = std::random_device()();
      thrust::transform(index_sequence_begin, index_sequence_begin + size,
                        thrust::device_ptr<T>(random_data),
                        UniformCUDAGenerator<T>(0.00001, 1, seed));
    }

    // add gumbel noise to X
    const int thread_size = 512;
    int64_t block_size = (size + thread_size) / thread_size;
    AddGumbelNoiseCUDAKernel<
        T><<<block_size, thread_size, 0, context.stream()>>>(
        input_data, output_data, random_data, temperature, size);
  }
};

#endif
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    gumbel_softmax, ops::GumbelSoftmaxKernel<plat::CUDADeviceContext, float>,
    ops::GumbelSoftmaxKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    gumbel_softmax_grad,
    ops::GumbelSoftmaxGradKernel<plat::CUDADeviceContext, float>,
    ops::GumbelSoftmaxGradKernel<plat::CUDADeviceContext, double>);
