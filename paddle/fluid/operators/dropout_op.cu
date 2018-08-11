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

#include "paddle/fluid/framework/philox_random.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void DropoutAllElementsKernel(const size_t n, MaskType *mask_data,
                                         T *dst) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    mask_data[idx] = static_cast<MaskType>(0);
    dst[idx] = static_cast<T>(0);
  }
}

template <typename T, typename MaskType>
__global__ void DropoutUsePhiloxRandomKernel(const size_t n, const size_t seed,
                                             const uint32_t int_dropout_prob,
                                             const T *src, MaskType *mask_data,
                                             T *dst) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t idx_beg = (idx << 2);
  if (idx_beg >= n) return;

  framework::random::PhiloxRandom rnd(seed);
  rnd.Skip(idx);
  auto ret = rnd();

#define PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(i)    \
  if (ret[i] < int_dropout_prob) {                 \
    mask_data[idx_beg] = static_cast<MaskType>(0); \
    dst[idx_beg] = static_cast<T>(0);              \
  } else {                                         \
    mask_data[idx_beg] = static_cast<MaskType>(1); \
    dst[idx_beg] = src[idx_beg];                   \
  }                                                \
  ++idx_beg

  if (idx_beg + 4 <= n) {  // For most cases, we unroll the loop
    PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(0);
    PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(1);
    PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(2);
    PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(3);
  } else {
    size_t left = n - idx_beg;
    for (size_t i = 0; i < left; ++i) {
      PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT(i);
    }
  }

#undef PRODUCE_PHILOX_RANDOM_DROPOUT_RESULT
}

struct GPUDropoutFunctor {
  template <typename T, typename MaskType>
  void operator()(const platform::CUDADeviceContext &ctx, const T *x_data,
                  T *y_data, MaskType *mask_data, size_t size,
                  float dropout_prob, size_t seed) {
    uint64_t int_dropout_prob = static_cast<uint64_t>(
        static_cast<double>(dropout_prob) * (static_cast<uint64_t>(1) << 32));

    size_t threads = 512;
    cudaStream_t stream = ctx.stream();
    if (int_dropout_prob >= (static_cast<uint64_t>(1) << 32)) {
      // dropout_prob == 1, set mask_data and y_data to 0
      size_t grid = (size + threads - 1) / threads;
      DropoutAllElementsKernel<T, MaskType><<<grid, threads, 0, stream>>>(
          size, mask_data, y_data);
    } else {
      // dropout_prob < 1, use PhiloxRandom engine
      // PhiloxRandom produces four uint32_t numbers once
      size_t size_div_4 = (size + 3) / 4;
      size_t grid = (size_div_4 + threads - 1) / threads;
      DropoutUsePhiloxRandomKernel<T, MaskType><<<grid, threads, 0, stream>>>(
          size, seed, static_cast<uint32_t>(int_dropout_prob), x_data,
          mask_data, y_data);
    }
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
