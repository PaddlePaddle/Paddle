/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <thrust/random.h>
#include <thrust/transform.h>
#include <string>
#include <vector>
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

#define CUDA_NUM_THREADS 1024

inline static int PADDLE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void RReluElementWiseKernel(const T *input,
                                       T *output,
                                       T *noise,
                                       const T mid_val,
                                       const float lower,
                                       const float upper,
                                       const bool is_test,
                                       const int seed_data,
                                       size_t numel) {
  CUDA_KERNEL_LOOP(index, numel) {
    T x = input[index];
    T zero = static_cast<T>(0);

    if (is_test) {
      if (x < zero) {
        output[index] = mid_val * x;
        noise[index] = mid_val;
      } else {
        output[index] = x;
        noise[index] = 1.0;
      }

    } else {
      if (x < zero) {
        thrust::minstd_rand rng;
        rng.seed(seed_data);
        thrust::uniform_real_distribution<float> dist(lower, upper);
        rng.discard(index);
        T scale = static_cast<T>(dist(rng));
        output[index] = scale * x;
        noise[index] = scale;
      } else {
        output[index] = x;
        noise[index] = 1.0;
      }
    }
  }
}

template <typename T>
class RReluElementWiseDirectCUDAFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T *input,
                  T *output,
                  T *noise,
                  const float lower,
                  const float upper,
                  const bool is_test,
                  const int seed_data,
                  size_t numel);
};

template <typename T>
void RReluElementWiseDirectCUDAFunctor<T>::operator()(gpuStream_t stream,
                                                      const T *input,
                                                      T *output,
                                                      T *noise,
                                                      const float lower,
                                                      const float upper,
                                                      const bool is_test,
                                                      const int seed_data,
                                                      size_t numel) {
  T mid_val = static_cast<T>((lower + upper) / 2.0);
  RReluElementWiseKernel<<<PADDLE_GET_BLOCKS(numel),
                           CUDA_NUM_THREADS,
                           0,
                           stream>>>(
      input, output, noise, mid_val, lower, upper, is_test, seed_data, numel);
}

template class RReluElementWiseDirectCUDAFunctor<float>;
template class RReluElementWiseDirectCUDAFunctor<double>;
template class RReluElementWiseDirectCUDAFunctor<phi::dtype::float16>;
}  // namespace phi
