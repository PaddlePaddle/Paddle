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

#pragma once

#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0
#define EXP_MAX_INPUT 40.0

template <typename T>
inline T sigmoid(T x) {
  return 1. / (1. + exp(-x));
}

template <typename T>
inline T tanh(T x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_sigmoid(const int n, const T* x, T* y) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    T tmp = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = 1.0 / (1.0 + std::exp(-tmp));
  }
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_tanh(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = tanh<T>(x[i]);
  }
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_relu(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

template <>
inline void vec_relu<float, platform::jit::avx2>(const int n, const float* x,
                                                 float* y) {
  // TODO(TJ): complete me
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

template <>
inline void vec_relu<float, platform::jit::avx>(const int n, const float* x,
                                                float* y) {
  // TODO(TJ): complete me
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] > 0 ? x[i] : 0;
  }
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
