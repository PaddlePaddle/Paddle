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

#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/platform/hostdevice.h"

#include <type_traits>

// TODO(guosheng): refine code style in gru_kernel
namespace paddle {
namespace operators {
namespace math {
namespace detail {

namespace forward {

template <typename T>
class gru_resetOutput {
 public:
  HOSTDEVICE void operator()(T &valueUpdateGate, T &valueResetGate, T &prevOut,
                             T &valueResetOutput, activation_mode_t actGate) {
    valueUpdateGate = activation(valueUpdateGate, actGate);
    valueResetGate = activation(valueResetGate, actGate);
    valueResetOutput = prevOut * valueResetGate;
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  HOSTDEVICE void operator()(__m256 &valueUpdateGate, __m256 &valueResetGate,
                             __m256 &prevOut, __m256 &valueResetOutput,
                             activation_mode_t actGate) {
    valueUpdateGate = activation(valueUpdateGate, actGate);
    valueResetGate = activation(valueResetGate, actGate);
    valueResetOutput = _mm256_mul_ps(prevOut, valueResetGate);
  }
#endif
#endif
};

template <typename T>
class gru_finalOutput {
 public:
  HOSTDEVICE void operator()(T &valueUpdateGate, T &valueFrameState, T &prevOut,
                             T &valueOutput, activation_mode_t actInput) {
    valueFrameState = activation(valueFrameState, actInput);
    valueOutput = prevOut - (valueUpdateGate * prevOut) +
                  (valueUpdateGate * valueFrameState);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  HOSTDEVICE void operator()(__m256 &valueUpdateGate, __m256 &valueFrameState,
                             __m256 &prevOut, __m256 &valueOutput,
                             activation_mode_t actInput) {
    valueFrameState = activation(valueFrameState, actInput);
    valueOutput = _mm256_add_ps(
        _mm256_sub_ps(prevOut, _mm256_mul_ps(valueUpdateGate, prevOut)),
        _mm256_mul_ps(valueUpdateGate, valueFrameState));
  }
#endif
#endif
};
}  // namespace forward

namespace backward {

template <typename T>
class gru_stateGrad {
 public:
  HOSTDEVICE void operator()(T &valueUpdateGate, T &gradUpdateGate,
                             T &valueFrameState, T &gradFrameState,
                             T &valuePrevOut, T &gradPrevOut, T &gradOutput,
                             activation_mode_t actInput) {
    gradUpdateGate = (gradOutput * valueFrameState);
    gradUpdateGate -= (gradOutput * valuePrevOut);
    gradPrevOut -= (gradOutput * valueUpdateGate);
    gradPrevOut += gradOutput;
    gradFrameState =
        activation(gradOutput * valueUpdateGate, valueFrameState, actInput);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  HOSTDEVICE void operator()(__m256 &valueUpdateGate, __m256 &gradUpdateGate,
                             __m256 &valueFrameState, __m256 &gradFrameState,
                             __m256 &valuePrevOut, __m256 &gradPrevOut,
                             __m256 &gradOutput, activation_mode_t actInput) {
    gradUpdateGate = _mm256_mul_ps(gradOutput, valueFrameState);
    gradUpdateGate =
        _mm256_sub_ps(gradUpdateGate, _mm256_mul_ps(gradOutput, valuePrevOut));
    gradPrevOut = _mm256_add_ps(
        _mm256_sub_ps(gradPrevOut, _mm256_mul_ps(gradOutput, valueUpdateGate)),
        gradOutput);
    gradFrameState = activation(_mm256_mul_ps(gradOutput, valueUpdateGate),
                                valueFrameState, actInput);
  }
#endif
#endif
};

template <typename T>
class gru_resetGrad {
 public:
  HOSTDEVICE void operator()(T &valueUpdateGate, T &gradUpdateGate,
                             T &valueResetGate, T &gradResetGate,
                             T &valuePrevOut, T &gradPrevOut,
                             T &gradResetOutput, activation_mode_t actGate) {
    gradResetGate = (gradResetOutput * valuePrevOut);
    gradPrevOut += (gradResetOutput * valueResetGate);
    gradUpdateGate = activation(gradUpdateGate, valueUpdateGate, actGate);
    gradResetGate = activation(gradResetGate, valueResetGate, actGate);
  }
#ifndef __NVCC__
#ifndef __AVX__
  static const bool avx = false;
#else
  static const bool avx = true;
  HOSTDEVICE void operator()(__m256 &valueUpdateGate, __m256 &gradUpdateGate,
                             __m256 &valueResetGate, __m256 &gradResetGate,
                             __m256 &valuePrevOut, __m256 &gradPrevOut,
                             __m256 &gradResetOutput,
                             activation_mode_t actGate) {
    gradResetGate = _mm256_mul_ps(gradResetOutput, valuePrevOut);
    gradPrevOut = _mm256_add_ps(gradPrevOut,
                                _mm256_mul_ps(gradResetOutput, valueResetGate));
    gradUpdateGate = activation(gradUpdateGate, valueUpdateGate, actGate);
    gradResetGate = activation(gradResetGate, valueResetGate, actGate);
  }
#endif
#endif
};

}  // namespace backward

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
