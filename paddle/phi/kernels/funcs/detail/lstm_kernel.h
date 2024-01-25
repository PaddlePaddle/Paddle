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
#include <type_traits>

#include "paddle/common/hostdevice.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"

namespace phi {
namespace funcs {
namespace detail {

namespace forward {

template <class T>
class lstm {
 public:
  HOSTDEVICE void operator()(T *value_in,
                             T *value_ig,
                             T *value_fg,
                             T *value_og,
                             T *prev_state,
                             T *state,
                             T *state_atv,
                             T *output,
                             T *checkI,
                             T *checkF,
                             T *checkO,
                             T *cell_clip,
                             ActivationType active_node,
                             ActivationType active_gate,
                             ActivationType active_state) {
    *value_in = activation(*value_in, active_node);
    *value_ig = activation(*value_ig + (*prev_state) * (*checkI), active_gate);
    *value_fg = activation(*value_fg + (*prev_state) * (*checkF), active_gate);
    *state = (*value_in) * (*value_ig) + (*prev_state) * (*value_fg);

    if (*cell_clip > 0.0) {
      if (*state < -1.0 * (*cell_clip)) {
        *state = -1.0 * (*cell_clip);
      }
      if (*state > *cell_clip) {
        *state = *cell_clip;
      }
    }
    *value_og = activation(*value_og + (*state) * (*checkO), active_gate);
    *state_atv = activation(*state, active_state);
    *output = (*value_og) * (*state_atv);
  }
#if !defined(__NVCC__) && !defined(__HIPCC___)  // @{ Group LSTM FWD
#ifndef __AVX__  // If not compiled with AVX instructs. Disable AVX by default
  static const bool avx = false;
#else
  // Only float support AVX optimization
  static const bool avx = std::is_same<T, float>::value;

  HOSTDEVICE void operator()(__m256 *value_in,
                             __m256 *value_ig,
                             __m256 *value_fg,
                             __m256 *value_og,
                             __m256 *prev_state,
                             __m256 *state,
                             __m256 *state_atv,
                             __m256 *output,
                             __m256 *checkI,
                             __m256 *checkF,
                             __m256 *checkO,
                             T *cell_clip,
                             ActivationType active_node,
                             ActivationType active_gate,
                             ActivationType active_state) {
    *value_in = activation(*value_in, active_node);
    *value_ig = activation(
        _mm256_add_ps(*value_ig, _mm256_mul_ps(*prev_state, *checkI)),
        active_gate);
    *value_fg = activation(
        _mm256_add_ps(*value_fg, _mm256_mul_ps(*prev_state, *checkF)),
        active_gate);
    *state = _mm256_add_ps(_mm256_mul_ps(*value_in, *value_ig),
                           _mm256_mul_ps(*prev_state, *value_fg));

    if (*cell_clip > 0.0f) {
      __m256 min = _mm256_set1_ps(0.0f - *cell_clip);
      __m256 max = _mm256_set1_ps(*cell_clip);
      *state = _mm256_min_ps(max, *state);
      *state = _mm256_max_ps(min, *state);
    }
    *value_og = activation(
        _mm256_add_ps(*value_og, _mm256_mul_ps(*state, *checkO)), active_gate);
    *state_atv = activation(*state, active_state);
    *output = _mm256_mul_ps(*value_og, *state_atv);
  }
#endif
#endif  // @} End Group LSTM FWD
};

}  // namespace forward

namespace backward {

template <class T>
class lstm {
 public:
  HOSTDEVICE void operator()(T *value_in,
                             T *value_ig,
                             T *value_fg,
                             T *value_og,
                             T *grad_in,
                             T *grad_ig,
                             T *grad_fg,
                             T *grad_og,
                             T *prev_state,
                             T *prev_state_grad,
                             T *state,
                             T *state_grad,
                             T *state_atv,
                             T *output_grad,
                             T *checkI,
                             T *checkF,
                             T *checkO,
                             T *checkIGrad,
                             T *checkFGrad,
                             T *checkOGrad,
                             T *cell_clip,
                             ActivationType active_node,
                             ActivationType active_gate,
                             ActivationType active_state) {
    *grad_og =
        activation((*output_grad) * (*state_atv), *value_og, active_gate);
    if (*cell_clip > 0.0f) {
      if (*state >= (*cell_clip) || *state <= (0.0f - (*cell_clip))) {
        *state_grad = 0.0f;
      } else {
        *state_grad +=
            activation((*output_grad) * (*value_og), *state_atv, active_state) +
            (*grad_og) * (*checkO);
      }
    } else {
      *state_grad +=
          activation((*output_grad) * (*value_og), *state_atv, active_state) +
          (*grad_og) * (*checkO);
    }

    *grad_in = activation((*state_grad) * (*value_ig), *value_in, active_node);
    *grad_ig = activation((*state_grad) * (*value_in), *value_ig, active_gate);
    *grad_fg =
        activation((*state_grad) * (*prev_state), *value_fg, active_gate);
    *prev_state_grad = (*grad_ig) * (*checkI) + (*grad_fg) * (*checkF) +
                       (*state_grad) * (*value_fg);
    *checkIGrad = (*grad_ig) * (*prev_state);
    *checkFGrad = (*grad_fg) * (*prev_state);
    *checkOGrad = (*grad_og) * (*state);
  }
#if !defined(__NVCC__) && !defined(__HIPCC___)  // @{ Group LSTM BWD
#ifndef __AVX__  // If not compiled with AVX instructs. Disable AVX by default
  static const bool avx = false;
#else
  // Only float support AVX optimization
  static const bool avx = std::is_same<T, float>::value;
  HOSTDEVICE void operator()(__m256 *value_in,
                             __m256 *value_ig,
                             __m256 *value_fg,
                             __m256 *value_og,
                             __m256 *grad_in,
                             __m256 *grad_ig,
                             __m256 *grad_fg,
                             __m256 *grad_og,
                             __m256 *prev_state,
                             __m256 *prev_state_grad,
                             __m256 *state,
                             __m256 *state_grad,
                             __m256 *state_atv,
                             __m256 *output_grad,
                             __m256 *checkI,
                             __m256 *checkF,
                             __m256 *checkO,
                             __m256 *checkIGrad,
                             __m256 *checkFGrad,
                             __m256 *checkOGrad,
                             T *cell_clip,
                             ActivationType active_node,
                             ActivationType active_gate,
                             ActivationType active_state) {
    *grad_og = activation(
        _mm256_mul_ps(*output_grad, *state_atv), *value_og, active_gate);
    if (*cell_clip > 0.0f) {
      T *state_ = reinterpret_cast<T *>(state);
      if (*state_ >= (*cell_clip) || *state_ <= (0.0f - (*cell_clip))) {
        *state_grad = _mm256_set1_ps(0.0f);
      } else {
        *state_grad =
            _mm256_add_ps(activation(_mm256_mul_ps(*output_grad, *value_og),
                                     *state_atv,
                                     active_state),
                          *state_grad);
        *state_grad =
            _mm256_add_ps(_mm256_mul_ps(*grad_og, *checkO), *state_grad);
      }
    }
    *grad_in = activation(
        _mm256_mul_ps(*state_grad, *value_ig), *value_in, active_node);
    *grad_ig = activation(
        _mm256_mul_ps(*state_grad, *value_in), *value_ig, active_gate);
    *grad_fg = activation(
        _mm256_mul_ps(*state_grad, *prev_state), *value_fg, active_gate);
    *prev_state_grad = _mm256_add_ps(_mm256_mul_ps(*grad_ig, *checkI),
                                     _mm256_mul_ps(*grad_fg, *checkF));
    *prev_state_grad =
        _mm256_add_ps(_mm256_mul_ps(*state_grad, *value_fg), *prev_state_grad);
    *checkIGrad = _mm256_mul_ps(*grad_ig, *prev_state);
    *checkFGrad = _mm256_mul_ps(*grad_fg, *prev_state);
    *checkOGrad = _mm256_mul_ps(*grad_og, *state);
  }
#endif
#endif  // @} End Group LSTM BWD
};

}  // namespace backward

}  // namespace detail
}  // namespace funcs
}  // namespace phi
