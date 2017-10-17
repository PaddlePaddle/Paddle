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
namespace typef {

__device__ static float relu(const float a) { return a > 0.0f ? a : 0.0f; }

__device__ static float sigmoid(const float a) {
  const float min = SIGMOID_THRESHOLD_MIN;
  const float max = SIGMOID_THRESHOLD_MAX;
  float tmp = (a < min) ? min : ((a > max) ? max : a);
  return __fdividef(1.0f, 1.0f + __expf(-tmp));
}

__device__ static float tanh(const float a) {
  return __fdividef(2.0f, (1.0f + __expf(-2.0f * a))) - 1.0f;
}

__device__ static float linear(const float a) { return a; }

__device__ static float relu(const float a, const float b) {
  return a * (b > 0.0f ? 1.0f : 0.0f);
}

__device__ static float sigmoid(const float a, const float b) {
  return a * b * (1.0f - b);
}

__device__ static float tanh(const float a, const float b) {
  return a * (1.0f - b * b);
}

__device__ static float linear(const float a, const float b) { return a; }

}  // namespace typef

namespace typed {

__device__ static double relu(const double a) { return a > 0.0 ? a : 0.0; }

__device__ static double sigmoid(const double a) {
  const double min = SIGMOID_THRESHOLD_MIN;
  const double max = SIGMOID_THRESHOLD_MAX;
  double tmp = (a < min) ? min : ((a > max) ? max : a);
  return 1.0 / (1.0 + exp(-tmp));
}

__device__ static double tanh(const double a) {
  return (2.0 / (1.0 + exp(-2.0 * a))) - 1.0;
}

__device__ static double linear(const double a) { return a; }

__device__ static double relu(const double a, const double b) {
  return a * (b > 0.0 ? 1.0 : 0.0);
}

__device__ static double sigmoid(const double a, const double b) {
  return a * b * (1 - b);
}

__device__ static double tanh(const double a, const double b) {
  return a * (1.0 - b * b);
}

__device__ static double linear(const double a, const double b) { return a; }

}  // namespace typef

}  // namespace hppl

#endif  // HL_GPU_FUNCTIONS_CUH_
