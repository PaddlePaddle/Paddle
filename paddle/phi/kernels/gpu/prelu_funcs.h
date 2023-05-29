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

#include <string>
#include <vector>

#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

#define CUDA_NUM_THREADS 1024

inline static int PADDLE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
struct PReluChannelFirstWiseCUDAFunctor {
  const T* x_;
  const T* alpha_;
  size_t channel_num_;
  size_t plane_size_;
  int numel_;

  HOSTDEVICE inline PReluChannelFirstWiseCUDAFunctor(const T* x,
                                                     const T* alpha,
                                                     int numel,
                                                     size_t channel_num,
                                                     size_t plane_size)
      : x_(x),
        alpha_(alpha),
        numel_(numel),
        channel_num_(channel_num),
        plane_size_(plane_size) {}

  HOSTDEVICE inline T operator()(const unsigned int n) const {
    T zero = static_cast<T>(0);
    size_t temp = n / plane_size_;
    size_t channel_index = temp % channel_num_;
    T scale = alpha_[channel_index];
    T x = x_[n];
    return (x > zero) ? x : scale * x;
  }
};

template <typename T>
struct PReluChannelLastWiseCUDAFunctor {
  const T* x_;
  const T* alpha_;
  size_t channel_num_;

  HOSTDEVICE inline PReluChannelLastWiseCUDAFunctor(const T* x,
                                                    const T* alpha,
                                                    size_t channel_num)
      : x_(x), alpha_(alpha), channel_num_(channel_num) {}

  HOSTDEVICE inline T operator()(const unsigned int n) const {
    T zero = static_cast<T>(0);
    size_t channel_index = n % channel_num_;
    T scale = alpha_[channel_index];
    T x = x_[n];
    return (x > zero) ? x : scale * x;
  }
};

template <typename T>
struct PreluElementWiseDirectCUDAFunctor {
  const T* x_;
  const T* alpha_;
  size_t spatial_size_;

  HOSTDEVICE inline PreluElementWiseDirectCUDAFunctor(const T* x,
                                                      const T* alpha,
                                                      size_t spatial_size)
      : x_(x), alpha_(alpha), spatial_size_(spatial_size) {}

  HOSTDEVICE inline T operator()(const unsigned int n) const {
    T zero = static_cast<T>(0);
    size_t element_index = n % spatial_size_;
    T scale = alpha_[element_index];
    T x = x_[n];
    return (x > zero) ? x : scale * x;
  }
};

template <typename T>
struct PreluScalarDirectCUDAFunctor {
  const T* scalar_;
  HOSTDEVICE inline PreluScalarDirectCUDAFunctor(const T* scalar)
      : scalar_(scalar) {}
  HOSTDEVICE inline T operator()(const T x) const {
    T zero = static_cast<T>(0);
    return (x > zero) ? x : scalar_[0] * x;
  }
};

}  // namespace phi
