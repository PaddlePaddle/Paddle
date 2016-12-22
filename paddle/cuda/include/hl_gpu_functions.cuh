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

  __device__ static real relu(const real a) {
    return a > 0.0f ? a : 0.0f;
  }

  __device__ static real sigmoid(const real a) {
    const real min = SIGMOID_THRESHOLD_MIN;
    const real max = SIGMOID_THRESHOLD_MAX;
    real tmp = (a < min) ? min : ((a > max) ? max : a);
#ifndef PADDLE_TYPE_DOUBLE
    return __fdividef(1.0f, 1.0f + __expf(-tmp));
#else
    return 1.0 / (1.0 + exp(-tmp));
#endif
  }

  __device__ static real tanh(const real a) {
#ifndef PADDLE_TYPE_DOUBLE
    return __fdividef(2.0f, (1.0f + __expf(-2.0f*a))) - 1.0f;
#else
    return (2.0 / (1.0 + exp(-2.0*a))) - 1.0;
#endif
  }

  __device__ static real linear(const real a) {
    return a;
  }

  __device__ static real relu(const real a, const real b) {
    return a * (b > 0.0f ? 1.0f : 0.0f);
  }

  __device__ static real sigmoid(const real a, const real b) {
    return a * b * (1 - b);
  }

  __device__ static real tanh(const real a, const real b) {
    return a * (1.0f - b * b);
  }

  __device__ static real linear(const real a, const real b) {
    return a;
  }

}  // namespace hppl

#endif  // HL_GPU_FUNCTIONS_CUH_
