/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_GPU_FUNCTIONS_CUH_
#define HL_GPU_FUNCTIONS_CUH_

#include "hl_base.h"

namespace hppl {

template <typename T>
__device__ static T relu(const T a) {
  return a > 0.0f ? a : 0.0f;
}

template <>
__device__ static float sigmoid(const float a) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  float tmp = (a < min) ? min : ((a > max) ? max : a);
  return __fdividef(1.0f, 1.0f + __expf(-tmp));
}

template <>
__device__ static double sigmoid(const double a) {
  const double min = SIGMOID_THRESHOLD_MIN;
  const double max = SIGMOID_THRESHOLD_MAX;
  double tmp = (a < min) ? min : ((a > max) ? max : a);
  return 1.0 / (1.0 + exp(-tmp));
}

template <>
__device__ static float tanh(const float a) {
  return __fdividef(2.0f, (1.0f + __expf(-2.0f * a))) - 1.0f;
}

template <>
__device__ static double tanh(const double a) {
  return (2.0 / (1.0 + exp(-2.0 * a))) - 1.0;
}

template <typename T>
__device__ static T linear(const T a) {
  return a;
}

template <typename T>
__device__ static T relu(const T a, const T b) {
  return a * (b > 0.0f ? 1.0f : 0.0f);
}

template <typename T>
__device__ static T sigmoid(const T a, const T b) {
  return a * b * (1 - b);
}

template <typename T>
__device__ static T tanh(const T a, const T b) {
  return a * (1.0f - b * b);
}

template <typename T>
__device__ static T linear(const T a, const T b) {
  return a;
}

}  // namespace hppl

#endif  // HL_GPU_FUNCTIONS_CUH_
