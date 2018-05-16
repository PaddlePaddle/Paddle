// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <random>
#include <type_traits>

#ifdef PADDLE_WITH_CUDA
#include <thrust/random.h>
#endif

#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

template <typename DeviceContext>
struct Random;

template <>
struct Random<CPUDeviceContext> {
  using Engine = std::minstd_rand;

  template <typename T>
  using UniformIntDist = std::uniform_int_distribution<T>;

  template <typename T>
  using UniformRealDist = std::uniform_real_distribution<T>;

  template <typename T>
  static typename std::enable_if<std::is_floating_point<T>::value,
                                 UniformRealDist<T>>::type
  UniformDist(T min, T max) {
    return UniformRealDist<T>(min, max);
  }

  template <typename T>
  static typename std::enable_if<!std::is_floating_point<T>::value,
                                 UniformIntDist<T>>::type
  UniformDist(T min, T max) {
    return UniformIntDist<T>(min, max);
  }

  template <typename T>
  using NormDist = std::normal_distribution<T>;
};

#ifdef PADDLE_WITH_CUDA
template <>
struct Random<CUDADeviceContext> {
  using Engine = thrust::minstd_rand;
  template <typename T>
  using UniformIntDist = thrust::uniform_int_distribution<T>;

  template <typename T>
  using UniformRealDist = thrust::uniform_real_distribution<T>;

  template <typename T>
  static typename std::enable_if<std::is_floating_point<T>::value,
                                 UniformRealDist<T>>::type
  UniformDist(T min, T max) {
    return UniformRealDist<T>(min, max);
  }

  template <typename T>
  static typename std::enable_if<!std::is_floating_point<T>::value,
                                 UniformIntDist<T>>::type
  UniformDist(T min, T max) {
    return UniformIntDist<T>(min, max);
  }

  template <typename T>
  using NormDist = thrust::normal_distribution<T>;
};
#endif

}  // namespace platform
}  // namespace paddle
