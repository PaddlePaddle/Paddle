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
#include <cmath>
#include <string>
#include "paddle/fluid/platform/cpu_info.h"
#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

namespace paddle {
namespace operators {
namespace math {

#define SIGMOID_THRESHOLD_MIN -40.0
#define SIGMOID_THRESHOLD_MAX 13.0

template <typename T>
inline void vec_exp(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
}

#ifdef PADDLE_WITH_MKLML
template <>
inline void vec_exp<float>(const int n, const float* x, float* y) {
  platform::dynload::vsExp(n, x, y);
}

template <>
inline void vec_exp<double>(const int n, const double* x, double* y) {
  platform::dynload::vdExp(n, x, y);
}
#endif

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_identity(const int n, const T* x, T* y) {
  // do nothing
  return;
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_sigmoid(const int n, const T* x, T* y) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  vec_exp<T>(n, y, y);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
inline void vec_tanh(const int n, const T* x, T* y) {
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  vec_exp<T>(n, y, y);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
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

template <typename T, platform::jit::cpu_isa_t isa = platform::jit::isa_any>
class VecActivations {
 public:
  std::function<void(const int, const T*, T*)> operator()(
      const std::string& type) {
    if (type == "sigmoid") {
      return vec_sigmoid<T, isa>;
    } else if (type == "relu") {
      return vec_relu<T, isa>;
    } else if (type == "tanh") {
      return vec_tanh<T, isa>;
    } else if (type == "identity" || type == "") {
      return vec_identity<T, isa>;
    }
    LOG(FATAL) << "Not support type: " << type;
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
