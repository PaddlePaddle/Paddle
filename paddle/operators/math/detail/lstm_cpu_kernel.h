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

#pragma once
#include <type_traits>
#include "paddle/operators/math/detail/activation_functions.h"
#include "paddle/operators/math/lstm_compute.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

#ifndef __NVCC__

template <class T, class Op>
void naive_lstm_forward_one_sequence(Op op, LstmMetaValue<T> value,
                                     int frameSize,
                                     activation_mode_t active_node,
                                     activation_mode_t active_gate,
                                     activation_mode_t active_state) {
  T rValueIn;
  T rValueIg;
  T rValueFg;
  T rValueOg;
  T rCheckI;
  T rCheckF;
  T rCheckO;
  T rState;
  T rPrevState = 0;
  T rStateAtv;
  T rOut;

  T *valueIn = value.gateValue;
  T *valueIg = value.gateValue + frameSize;
  T *valueFg = value.gateValue + frameSize * 2;
  T *valueOg = value.gateValue + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg ? value.checkIg[i] : 0;
    rCheckF = value.checkFg ? value.checkFg[i] : 0;
    rCheckO = value.checkOg ? value.checkOg[i] : 0;

    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn, rValueIg, rValueFg, rValueOg, rPrevState, rState, rStateAtv,
       rOut, rCheckI, rCheckF, rCheckO, active_node, active_gate, active_state);

    valueIn[i] = rValueIn;
    valueIg[i] = rValueIg;
    valueFg[i] = rValueFg;
    valueOg[i] = rValueOg;
    value.stateValue[i] = rState;
    value.stateActiveValue[i] = rStateAtv;
    value.outputValue[i] = rOut;
  }
}

template <class T, class Op>
void naive_lstm_backward_one_sequence(Op op, LstmMetaValue<T> value,
                                      LstmMetaGrad<T> grad, int frameSize,
                                      activation_mode_t active_node,
                                      activation_mode_t active_gate,
                                      activation_mode_t active_state) {
  T rValueIn;
  T rValueIg;
  T rValueFg;
  T rValueOg;
  T rGradIn;
  T rGradIg;
  T rGradFg;
  T rGradOg;
  T rPrevState = 0;
  T rPrevStateGrad;
  T rState;
  T rStateGrad;
  T rStateAtv;
  T rOutputGrad;
  T rCheckI;
  T rCheckF;
  T rCheckO;
  T rCheckIGrad;
  T rCheckFGrad;
  T rCheckOGrad;

  T *valueIn = value.gateValue;
  T *valueIg = value.gateValue + frameSize;
  T *valueFg = value.gateValue + frameSize * 2;
  T *valueOg = value.gateValue + frameSize * 3;
  T *gradIn = grad.gateGrad;
  T *gradIg = grad.gateGrad + frameSize;
  T *gradFg = grad.gateGrad + frameSize * 2;
  T *gradOg = grad.gateGrad + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg ? value.checkIg[i] : 0;
    rCheckF = value.checkFg ? value.checkFg[i] : 0;
    rCheckO = value.checkOg ? value.checkOg[i] : 0;
    rState = value.stateValue[i];
    rStateAtv = value.stateActiveValue[i];
    rOutputGrad = grad.outputGrad[i];
    rStateGrad = grad.stateGrad[i];
    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn, rValueIg, rValueFg, rValueOg, rGradIn, rGradIg, rGradFg,
       rGradOg, rPrevState, rPrevStateGrad, rState, rStateGrad, rStateAtv,
       rOutputGrad, rCheckI, rCheckF, rCheckO, rCheckIGrad, rCheckFGrad,
       rCheckOGrad, active_node, active_gate, active_state);

    gradIn[i] = rGradIn;
    gradIg[i] = rGradIg;
    gradFg[i] = rGradFg;
    gradOg[i] = rGradOg;
    grad.stateGrad[i] = rStateGrad;

    if (grad.prevStateGrad) grad.prevStateGrad[i] = rPrevStateGrad;
    if (value.prevStateValue) {
      if (grad.checkIgGrad) grad.checkIgGrad[i] += rCheckIGrad;
      if (grad.checkFgGrad) grad.checkFgGrad[i] += rCheckFGrad;
    }
    if (grad.checkOgGrad) grad.checkOgGrad[i] += rCheckOGrad;
  }
}

template <class T, class Op>
void avx_lstm_forward_one_sequence(Op op, LstmMetaValue<T> value, int frameSize,
                                   activation_mode_t active_node,
                                   activation_mode_t active_gate,
                                   activation_mode_t active_state) {
#ifdef __AVX__
  __m256 rValueIn;
  __m256 rValueIg;
  __m256 rValueFg;
  __m256 rValueOg;
  __m256 rCheckI = _mm256_set1_ps(0.0f);
  __m256 rCheckF = _mm256_set1_ps(0.0f);
  __m256 rCheckO = _mm256_set1_ps(0.0f);
  __m256 rState;
  __m256 rPrevState = _mm256_set1_ps(0.0f);
  __m256 rStateAtv;
  __m256 rOut;

  __m256 *valueIn = (__m256 *)value.gateValue;
  __m256 *valueIg = (__m256 *)(value.gateValue + frameSize);
  __m256 *valueFg = (__m256 *)(value.gateValue + frameSize * 2);
  __m256 *valueOg = (__m256 *)(value.gateValue + frameSize * 3);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    if (value.checkIg) {
      rCheckI = ((__m256 *)value.checkIg)[i];
      rCheckF = ((__m256 *)value.checkFg)[i];
      rCheckO = ((__m256 *)value.checkOg)[i];
    }

    if (value.prevStateValue) {
      rPrevState = ((__m256 *)value.prevStateValue)[i];
    }

    op(rValueIn, rValueIg, rValueFg, rValueOg, rPrevState, rState, rStateAtv,
       rOut, rCheckI, rCheckF, rCheckO, active_node, active_gate, active_state);

    valueIn[i] = rValueIn;
    valueIg[i] = rValueIg;
    valueFg[i] = rValueFg;
    valueOg[i] = rValueOg;
    ((__m256 *)value.stateValue)[i] = rState;
    ((__m256 *)value.stateActiveValue)[i] = rStateAtv;
    ((__m256 *)value.outputValue)[i] = rOut;
  }
#endif
}

template <class T, class Op>
void avx_lstm_backward_one_sequence(Op op, LstmMetaValue<T> value,
                                    LstmMetaGrad<T> grad, int frameSize,
                                    activation_mode_t active_node,
                                    activation_mode_t active_gate,
                                    activation_mode_t active_state) {
#ifdef __AVX__
  __m256 rValueIn;
  __m256 rValueIg;
  __m256 rValueFg;
  __m256 rValueOg;
  __m256 rGradIn;
  __m256 rGradIg;
  __m256 rGradFg;
  __m256 rGradOg;
  __m256 rPrevState = _mm256_set1_ps(0.0f);
  __m256 rPrevStateGrad;
  __m256 rStateGrad;
  __m256 rState;
  __m256 rStateAtv;
  __m256 rOutputGrad;
  __m256 rCheckI = _mm256_set1_ps(0.0f);
  __m256 rCheckF = _mm256_set1_ps(0.0f);
  __m256 rCheckO = _mm256_set1_ps(0.0f);
  __m256 rCheckIGrad;
  __m256 rCheckFGrad;
  __m256 rCheckOGrad;

  __m256 *valueIn = (__m256 *)value.gateValue;
  __m256 *valueIg = (__m256 *)(value.gateValue + frameSize);
  __m256 *valueFg = (__m256 *)(value.gateValue + frameSize * 2);
  __m256 *valueOg = (__m256 *)(value.gateValue + frameSize * 3);
  __m256 *gradIn = (__m256 *)grad.gateGrad;
  __m256 *gradIg = (__m256 *)(grad.gateGrad + frameSize);
  __m256 *gradFg = (__m256 *)(grad.gateGrad + frameSize * 2);
  __m256 *gradOg = (__m256 *)(grad.gateGrad + frameSize * 3);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    if (value.checkIg) {
      rCheckI = ((__m256 *)value.checkIg)[i];
      rCheckF = ((__m256 *)value.checkFg)[i];
      rCheckO = ((__m256 *)value.checkOg)[i];
    }
    rState = ((__m256 *)value.stateValue)[i];
    rStateAtv = ((__m256 *)value.stateActiveValue)[i];
    rOutputGrad = ((__m256 *)grad.outputGrad)[i];
    rStateGrad = ((__m256 *)grad.stateGrad)[i];
    if (value.prevStateValue) {
      rPrevState = ((__m256 *)value.prevStateValue)[i];
    }

    op(rValueIn, rValueIg, rValueFg, rValueOg, rGradIn, rGradIg, rGradFg,
       rGradOg, rPrevState, rPrevStateGrad, rState, rStateGrad, rStateAtv,
       rOutputGrad, rCheckI, rCheckF, rCheckO, rCheckIGrad, rCheckFGrad,
       rCheckOGrad, active_node, active_gate, active_state);

    gradIn[i] = rGradIn;
    gradIg[i] = rGradIg;
    gradFg[i] = rGradFg;
    gradOg[i] = rGradOg;
    ((__m256 *)grad.stateGrad)[i] = rStateGrad;

    if (grad.prevStateGrad) ((__m256 *)grad.prevStateGrad)[i] = rPrevStateGrad;
    if (value.prevStateValue) {
      if (grad.checkIgGrad) ((__m256 *)grad.checkIgGrad)[i] += rCheckIGrad;
      if (grad.checkFgGrad) ((__m256 *)grad.checkFgGrad)[i] += rCheckFGrad;
    }
    if (grad.checkOgGrad) ((__m256 *)grad.checkOgGrad)[i] += rCheckOGrad;
  }
#endif
}

template <class T, class Op>
void cpu_lstm_forward(Op op, LstmMetaValue<T> value, int frameSize,
                      activation_mode_t active_node,
                      activation_mode_t active_gate,
                      activation_mode_t active_state) {
  if (Op::avx && !(frameSize & (8 - 1)) && (std::is_same<T, float>::value)) {
    avx_lstm_forward_one_sequence<T>(op, value, frameSize, active_node,
                                     active_gate, active_state);
  } else {
    naive_lstm_forward_one_sequence<T>(op, value, frameSize, active_node,
                                       active_gate, active_state);
  }
}

template <class T, class Op>
void cpu_lstm_backward(Op op, LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                       int frameSize, activation_mode_t active_node,
                       activation_mode_t active_gate,
                       activation_mode_t active_state) {
  if (Op::avx && !(frameSize & (8 - 1)) && (std::is_same<T, float>::value)) {
    avx_lstm_backward_one_sequence<T>(op, value, grad, frameSize, active_node,
                                      active_gate, active_state);
  } else {
    naive_lstm_backward_one_sequence<T>(op, value, grad, frameSize, active_node,
                                        active_gate, active_state);
  }
}

#endif

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
