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
#include "paddle/operators/math/gru_compute.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

#ifndef __NVCC__

template <class OpResetOutput, typename T>
void hl_naive_gru_forward_reset_output(OpResetOutput opResetOutput,
                                       T *gateValue, T *resetOutputValue,
                                       T *prevOutputValue, int frameSize,
                                       activation_mode_t active_gate) {
  T rValueUpdateGate;
  T rValueResetGate;
  T rValueResetOutput;
  T rPrevOut = 0;
  T *updateGate = gateValue;
  T *resetGate = gateValue + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueResetGate = resetGate[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opResetOutput(rValueUpdateGate, rValueResetGate, rPrevOut,
                  rValueResetOutput, active_gate);

    updateGate[i] = rValueUpdateGate;
    resetGate[i] = rValueResetGate;
    resetOutputValue[i] = rValueResetOutput;
  }
}

template <class OpFinalOutput, typename T>
void hl_naive_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                       T *gateValue, T *prevOutputValue,
                                       T *outputValue, int frameSize,
                                       activation_mode_t active_node) {
  T rValueUpdateGate;
  T rValueFrameState;
  T rPrevOut = 0;
  T rOutput;
  T *updateGate = gateValue;
  T *frameState = gateValue + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rValueUpdateGate = updateGate[i];
    rValueFrameState = frameState[i];
    if (prevOutputValue) {
      rPrevOut = prevOutputValue[i];
    }

    opFinalOutput(rValueUpdateGate, rValueFrameState, rPrevOut, rOutput,
                  active_node);

    frameState[i] = rValueFrameState;
    outputValue[i] = rOutput;
  }
}

template <class OpResetOutput, typename T>
void hl_avx_gru_forward_reset_output(OpResetOutput opResetOutput, T *gateValue,
                                     T *resetOutputValue, T *prevOutputValue,
                                     int frameSize,
                                     activation_mode_t active_gate) {
#ifdef __AVX__
  __m256 rValueUpdateGate;
  __m256 rValueResetGate;
  __m256 rValueResetOutput;
  __m256 rPrevOut = _mm256_set1_ps(0.0f);
  __m256 *updateGate = (__m256 *)gateValue;
  __m256 *resetGate = (__m256 *)(gateValue + frameSize);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueUpdateGate = updateGate[i];
    rValueResetGate = resetGate[i];
    if (prevOutputValue) {
      rPrevOut = ((__m256 *)prevOutputValue)[i];
    }

    opResetOutput(rValueUpdateGate, rValueResetGate, rPrevOut,
                  rValueResetOutput, active_gate);

    updateGate[i] = rValueUpdateGate;
    resetGate[i] = rValueResetGate;
    ((__m256 *)resetOutputValue)[i] = rValueResetOutput;
  }
#endif
}

template <class OpFinalOutput, typename T>
void hl_avx_gru_forward_final_output(OpFinalOutput opFinalOutput, T *gateValue,
                                     T *prevOutputValue, T *outputValue,
                                     int frameSize,
                                     activation_mode_t active_node) {
#ifdef __AVX__
  __m256 rValueUpdateGate;
  __m256 rValueFrameState;
  __m256 rPrevOut = _mm256_set1_ps(0.0f);
  __m256 rOutput;
  __m256 *updateGate = (__m256 *)gateValue;
  __m256 *frameState = (__m256 *)(gateValue + frameSize * 2);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueUpdateGate = updateGate[i];
    rValueFrameState = frameState[i];
    if (prevOutputValue) {
      rPrevOut = ((__m256 *)prevOutputValue)[i];
    }

    opFinalOutput(rValueUpdateGate, rValueFrameState, rPrevOut, rOutput,
                  active_node);

    frameState[i] = rValueFrameState;
    ((__m256 *)outputValue)[i] = rOutput;
  }
#endif
}

template <class OpResetOutput, typename T>
inline void forward_reset_output(OpResetOutput opResetOutput,
                                 hl_gru_value<T> value, int frameSize,
                                 int batchSize, activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (OpResetOutput::avx && !(frameSize & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_forward_reset_output(
          opResetOutput, value.gateValue, value.resetOutputValue,
          value.prevOutValue, frameSize, active_gate);
    } else {
      hl_naive_gru_forward_reset_output(
          opResetOutput, value.gateValue, value.resetOutputValue,
          value.prevOutValue, frameSize, active_gate);
    }

    value.gateValue += frameSize * 3;
    value.resetOutputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template <class OpFinalOutput, typename T>
inline void forward_final_output(OpFinalOutput opFinalOutput,
                                 hl_gru_value<T> value, int frameSize,
                                 int batchSize, activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (OpFinalOutput::avx && !(frameSize & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_forward_final_output(opFinalOutput, value.gateValue,
                                      value.prevOutValue, value.outputValue,
                                      frameSize, active_node);
    } else {
      hl_naive_gru_forward_final_output(opFinalOutput, value.gateValue,
                                        value.prevOutValue, value.outputValue,
                                        frameSize, active_node);
    }

    value.gateValue += frameSize * 3;
    value.outputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template <class OpStateGrad, typename T>
void hl_naive_gru_backward_state_grad(OpStateGrad opStateGrad, T *gateValue,
                                      T *gateGrad, T *prevOutValue,
                                      T *prevOutGrad, T *outputGrad,
                                      int frameSize,
                                      activation_mode_t active_node) {
  T rUpdateGateValue;
  T rUpdateGateGrad;
  T rFrameStateValue;
  T rFrameStateGrad;
  T rOutGrad;
  T rPrevOutValue = 0;
  T rPrevOutGrad = 0;
  T *updateGateValue = gateValue;
  T *updateGateGrad = gateGrad;
  T *frameStateValue = gateValue + frameSize * 2;
  T *frameStateGrad = gateGrad + frameSize * 2;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rFrameStateValue = frameStateValue[i];
    rOutGrad = outputGrad[i];
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = prevOutGrad[i];
    }

    opStateGrad(rUpdateGateValue, rUpdateGateGrad, rFrameStateValue,
                rFrameStateGrad, rPrevOutValue, rPrevOutGrad, rOutGrad,
                active_node);

    updateGateGrad[i] = rUpdateGateGrad;
    frameStateGrad[i] = rFrameStateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

template <class OpResetGrad, typename T>
void hl_naive_gru_backward_reset_grad(OpResetGrad opResetGrad, T *gateValue,
                                      T *gateGrad, T *prevOutValue,
                                      T *prevOutGrad, T *resetOutputGrad,
                                      int frameSize,
                                      activation_mode_t active_gate) {
  T rUpdateGateValue;
  T rUpdateGateGrad;
  T rResetGateValue;
  T rResetGateGrad;
  T rResetOutputGrad = 0;
  T rPrevOutValue = 0;
  T rPrevOutGrad = 0;
  T *updateGateValue = gateValue;
  T *updateGateGrad = gateGrad;
  T *resetGateValue = gateValue + frameSize;
  T *resetGateGrad = gateGrad + frameSize;

  for (int i = 0; i < frameSize; i++) {
    rUpdateGateValue = updateGateValue[i];
    rUpdateGateGrad = updateGateGrad[i];
    rResetGateValue = resetGateValue[i];

    if (prevOutValue && prevOutGrad) {
      rResetOutputGrad = resetOutputGrad[i];
    }
    if (prevOutValue) {
      rPrevOutValue = prevOutValue[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = prevOutGrad[i];
    }

    opResetGrad(rUpdateGateValue, rUpdateGateGrad, rResetGateValue,
                rResetGateGrad, rPrevOutValue, rPrevOutGrad, rResetOutputGrad,
                active_gate);

    updateGateGrad[i] = rUpdateGateGrad;
    resetGateGrad[i] = rResetGateGrad;
    if (prevOutGrad) {
      prevOutGrad[i] = rPrevOutGrad;
    }
  }
}

template <class OpStateGrad, typename T>
void hl_avx_gru_backward_state_grad(OpStateGrad opStateGrad, T *gateValue,
                                    T *gateGrad, T *prevOutValue,
                                    T *prevOutGrad, T *outputGrad,
                                    int frameSize,
                                    activation_mode_t active_node) {
#ifdef __AVX__
  __m256 rUpdateGateValue;
  __m256 rUpdateGateGrad;
  __m256 rFrameStateValue;
  __m256 rFrameStateGrad;
  __m256 rOutGrad;
  __m256 rPrevOutValue = _mm256_set1_ps(0.0f);
  __m256 rPrevOutGrad = _mm256_set1_ps(0.0f);
  __m256 *updateGateValue = (__m256 *)gateValue;
  __m256 *updateGateGrad = (__m256 *)gateGrad;
  __m256 *frameStateValue = (__m256 *)(gateValue + frameSize * 2);
  __m256 *frameStateGrad = (__m256 *)(gateGrad + frameSize * 2);

  for (int i = 0; i < frameSize / 8; i++) {
    rUpdateGateValue = updateGateValue[i];
    rFrameStateValue = frameStateValue[i];
    rOutGrad = ((__m256 *)outputGrad)[i];
    if (prevOutValue) {
      rPrevOutValue = ((__m256 *)prevOutValue)[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = ((__m256 *)prevOutGrad)[i];
    }

    opStateGrad(rUpdateGateValue, rUpdateGateGrad, rFrameStateValue,
                rFrameStateGrad, rPrevOutValue, rPrevOutGrad, rOutGrad,
                active_node);

    updateGateGrad[i] = rUpdateGateGrad;
    frameStateGrad[i] = rFrameStateGrad;
    if (prevOutGrad) {
      ((__m256 *)prevOutGrad)[i] = rPrevOutGrad;
    }
  }
#endif
}

template <class OpResetGrad, typename T>
void hl_avx_gru_backward_reset_grad(OpResetGrad opResetGrad, T *gateValue,
                                    T *gateGrad, T *prevOutValue,
                                    T *prevOutGrad, T *resetOutputGrad,
                                    int frameSize,
                                    activation_mode_t active_gate) {
#ifdef __AVX__
  __m256 rUpdateGateValue;
  __m256 rUpdateGateGrad;
  __m256 rResetGateValue;
  __m256 rResetGateGrad;
  __m256 rResetOutputGrad = _mm256_set1_ps(0.0f);
  __m256 rPrevOutValue = _mm256_set1_ps(0.0f);
  __m256 rPrevOutGrad = _mm256_set1_ps(0.0f);
  __m256 *updateGateValue = (__m256 *)gateValue;
  __m256 *updateGateGrad = (__m256 *)gateGrad;
  __m256 *resetGateValue = (__m256 *)(gateValue + frameSize);
  __m256 *resetGateGrad = (__m256 *)(gateGrad + frameSize);

  for (int i = 0; i < frameSize / 8; i++) {
    rUpdateGateValue = updateGateValue[i];
    rUpdateGateGrad = updateGateGrad[i];
    rResetGateValue = resetGateValue[i];

    if (prevOutValue && prevOutGrad) {
      rResetOutputGrad = ((__m256 *)resetOutputGrad)[i];
    }
    if (prevOutValue) {
      rPrevOutValue = ((__m256 *)prevOutValue)[i];
    }
    if (prevOutGrad) {
      rPrevOutGrad = ((__m256 *)prevOutGrad)[i];
    }

    opResetGrad(rUpdateGateValue, rUpdateGateGrad, rResetGateValue,
                rResetGateGrad, rPrevOutValue, rPrevOutGrad, rResetOutputGrad,
                active_gate);

    updateGateGrad[i] = rUpdateGateGrad;
    resetGateGrad[i] = rResetGateGrad;
    if (prevOutGrad) {
      ((__m256 *)prevOutGrad)[i] = rPrevOutGrad;
    }
  }
#endif
}

template <class OpStateGrad, typename T>
inline void backward_state_grad(OpStateGrad opStateGrad, hl_gru_value<T> value,
                                hl_gru_grad<T> grad, int frameSize,
                                int batchSize, activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (OpStateGrad::avx && !(frameSize & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_backward_state_grad(
          opStateGrad, value.gateValue, grad.gateGrad, value.prevOutValue,
          grad.prevOutGrad, grad.outputGrad, frameSize, active_node);
    } else {
      hl_naive_gru_backward_state_grad(
          opStateGrad, value.gateValue, grad.gateGrad, value.prevOutValue,
          grad.prevOutGrad, grad.outputGrad, frameSize, active_node);
    }

    value.gateValue += frameSize * 3;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }

    grad.gateGrad += frameSize * 3;
    grad.outputGrad += frameSize;
    if (grad.prevOutGrad) {
      grad.prevOutGrad += frameSize;
    }
  }
}

template <class OpResetGrad, typename T>
inline void backward_reset_grad(OpResetGrad opResetGrad, hl_gru_value<T> value,
                                hl_gru_grad<T> grad, int frameSize,
                                int batchSize, activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (OpResetGrad::avx && !(frameSize & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_backward_reset_grad(
          opResetGrad, value.gateValue, grad.gateGrad, value.prevOutValue,
          grad.prevOutGrad, grad.resetOutputGrad, frameSize, active_gate);
    } else {
      hl_naive_gru_backward_reset_grad(
          opResetGrad, value.gateValue, grad.gateGrad, value.prevOutValue,
          grad.prevOutGrad, grad.resetOutputGrad, frameSize, active_gate);
    }

    value.gateValue += frameSize * 3;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }

    grad.gateGrad += frameSize * 3;
    grad.resetOutputGrad += frameSize;
    if (grad.prevOutGrad) {
      grad.prevOutGrad += frameSize;
    }
  }
}

#endif

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle
