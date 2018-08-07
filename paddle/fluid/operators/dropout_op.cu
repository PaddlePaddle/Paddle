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

#define EIGEN_USE_GPU
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, const size_t seed,
                                const uint32_t int_dropout_prob, const T *src,
                                MaskType *mask_data, T *dst) {
  constexpr uint32_t kUInt32Max = static_cast<uint32_t>(-1);
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  thrust::minstd_rand rng;
  thrust::uniform_int_distribution<uint32_t> dist(0, kUInt32Max - 1);
  rng.seed(HashCombine(seed, idx));
  rng.seed(dist(rng));

  if (idx < n) {
    if (dist(rng) < int_dropout_prob) {
      mask_data[idx] = static_cast<MaskType>(0);
      dst[idx] = static_cast<T>(0);
    } else {
      mask_data[idx] = static_cast<MaskType>(1);
      dst[idx] = src[idx];
    }
  }
}

struct GPUDropoutFunctor {
  template <typename T, typename MaskType>
  void operator()(const platform::CUDADeviceContext &ctx, const T *x_data,
                  T *y_data, MaskType *mask_data, size_t size,
                  float dropout_prob, size_t seed) {
    size_t threads = 512;
    size_t grid = (size + threads - 1) / threads;
    uint32_t int_dropout_prob = static_cast<uint32_t>(
        static_cast<double>(dropout_prob) * static_cast<uint32_t>(-1));

    RandomGenerator<<<grid, threads, 0, ctx.stream()>>>(
        size, seed, int_dropout_prob, x_data, mask_data, y_data);
  }
};

template <typename T>
using GPUDropoutKernel =
    DropoutKernel<platform::CUDADeviceContext, T, GPUDropoutFunctor>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(dropout, ops::GPUDropoutKernel<float>,
                        ops::GPUDropoutKernel<plat::float16>);
REGISTER_OP_CUDA_KERNEL(dropout_grad,
                        ops::DropoutGradKernel<plat::CUDADeviceContext, float>);
