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

namespace paddle {
namespace operators {
namespace math {
namespace detail {

namespace forward {

template <class T>
class lstm {
 public:
  HOSTDEVICE void operator()(T &valueIn, T &valueIg, T &valueFg, T &valueOg,
                             T &prevState, T &state, T &stateAtv, T &output,
                             T &checkI, T &checkF, T &checkO,
                             activation_mode_t active_node,
                             activation_mode_t active_gate,
                             activation_mode_t active_state) {
    valueIn = activation(valueIn, active_node);
    valueIg = activation(valueIg + prevState * checkI, active_gate);
    valueFg = activation(valueFg + prevState * checkF, active_gate);
    state = valueIn * valueIg + prevState * valueFg;
    valueOg = activation(valueOg + state * checkO, active_gate);
    stateAtv = activation(state, active_state);
    output = valueOg * stateAtv;
  }
#ifndef __NVCC__
#ifndef __AVX__  // If not compiled with AVX instructs. Disable AVX by default
  static const bool avx = false;
#else
  // Only float support AVX optimization
  static const bool avx = std::is_same<T, float>::value;

  HOSTDEVICE void operator()(__m256 &valueIn, __m256 &valueIg, __m256 &valueFg,
                             __m256 &valueOg, __m256 &prevState, __m256 &state,
                             __m256 &stateAtv, __m256 &output, __m256 &checkI,
                             __m256 &checkF, __m256 &checkO,
                             activation_mode_t active_node,
                             activation_mode_t active_gate,
                             activation_mode_t active_state) {
    valueIn = activation(valueIn, active_node);
    valueIg = activation(
        _mm256_add_ps(valueIg, _mm256_mul_ps(prevState, checkI)), active_gate);
    valueFg = activation(
        _mm256_add_ps(valueFg, _mm256_mul_ps(prevState, checkF)), active_gate);
    state = _mm256_add_ps(_mm256_mul_ps(valueIn, valueIg),
                          _mm256_mul_ps(prevState, valueFg));
    valueOg = activation(_mm256_add_ps(valueOg, _mm256_mul_ps(state, checkO)),
                         active_gate);
    stateAtv = activation(state, active_state);
    output = _mm256_mul_ps(valueOg, stateAtv);
  }
#endif
#endif
};

}  // namespace forward

namespace backward {

template <class T>
class lstm {
 public:
  HOSTDEVICE void operator()(T &valueIn, T &valueIg, T &valueFg, T &valueOg,
                             T &gradIn, T &gradIg, T &gradFg, T &gradOg,
                             T &prevState, T &prevStateGrad, T &state,
                             T &stateGrad, T &stateAtv, T &outputGrad,
                             T &checkI, T &checkF, T &checkO, T &checkIGrad,
                             T &checkFGrad, T &checkOGrad,
                             activation_mode_t active_node,
                             activation_mode_t active_gate,
                             activation_mode_t active_state) {
    gradOg = activation(outputGrad * stateAtv, valueOg, active_gate);
    stateGrad += activation(outputGrad * valueOg, stateAtv, active_state) +
                 gradOg * checkO;
    gradIn = activation(stateGrad * valueIg, valueIn, active_node);
    gradIg = activation(stateGrad * valueIn, valueIg, active_gate);
    gradFg = activation(stateGrad * prevState, valueFg, active_gate);
    prevStateGrad = gradIg * checkI + gradFg * checkF + stateGrad * valueFg;
    checkIGrad = gradIg * prevState;
    checkFGrad = gradFg * prevState;
    checkOGrad = gradOg * state;
  }
#ifndef __NVCC__
#ifndef __AVX__  // If not compiled with AVX instructs. Disable AVX by default
  static const bool avx = false;
#else
  // Only float support AVX optimization
  static const bool avx = std::is_same<T, float>::value;
  HOSTDEVICE void operator()(
      __m256 &valueIn, __m256 &valueIg, __m256 &valueFg, __m256 &valueOg,
      __m256 &gradIn, __m256 &gradIg, __m256 &gradFg, __m256 &gradOg,
      __m256 &prevState, __m256 &prevStateGrad, __m256 &state,
      __m256 &stateGrad, __m256 &stateAtv, __m256 &outputGrad, __m256 &checkI,
      __m256 &checkF, __m256 &checkO, __m256 &checkIGrad, __m256 &checkFGrad,
      __m256 &checkOGrad, activation_mode_t active_node,
      activation_mode_t active_gate, activation_mode_t active_state) {
    gradOg =
        activation(_mm256_mul_ps(outputGrad, stateAtv), valueOg, active_gate);
    stateGrad = _mm256_add_ps(
        activation(_mm256_mul_ps(outputGrad, valueOg), stateAtv, active_state),
        stateGrad);
    stateGrad = _mm256_add_ps(_mm256_mul_ps(gradOg, checkO), stateGrad);
    gradIn =
        activation(_mm256_mul_ps(stateGrad, valueIg), valueIn, active_node);
    gradIg =
        activation(_mm256_mul_ps(stateGrad, valueIn), valueIg, active_gate);
    gradFg =
        activation(_mm256_mul_ps(stateGrad, prevState), valueFg, active_gate);
    prevStateGrad = _mm256_add_ps(_mm256_mul_ps(gradIg, checkI),
                                  _mm256_mul_ps(gradFg, checkF));
    prevStateGrad =
        _mm256_add_ps(_mm256_mul_ps(stateGrad, valueFg), prevStateGrad);
    checkIGrad = _mm256_mul_ps(gradIg, prevState);
    checkFGrad = _mm256_mul_ps(gradFg, prevState);
    checkOGrad = _mm256_mul_ps(gradOg, state);
  }
#endif
#endif
};

}  // namespace backward

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
