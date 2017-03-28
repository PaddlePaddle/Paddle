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

#include "hl_base.h"
#include "paddle/math/MathFunctions.h"
#include "paddle/utils/CpuId.h"
#include "simd_recurrent.cuh"

#ifndef PADDLE_TYPE_DOUBLE
#define CBLAS_GEMM paddle::gemm<float>
#else
#define CBLAS_GEMM paddle::gemm<double>
#endif

template <class Op>
void hl_cpu_lstm_forward(Op op,
                         hl_lstm_value value,
                         int frameSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate,
                         hl_activation_mode_t active_state) {
  if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_forward_one_sequence(
        op, value, frameSize, active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_forward_one_sequence(
        op, value, frameSize, active_node, active_gate, active_state);
  }
}

template <class Op>
void hl_cpu_lstm_backward(Op op,
                          hl_lstm_value value,
                          hl_lstm_grad grad,
                          int frameSize,
                          hl_activation_mode_t active_node,
                          hl_activation_mode_t active_gate,
                          hl_activation_mode_t active_state) {
  if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_backward_one_sequence(
        op, value, grad, frameSize, active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_backward_one_sequence(
        op, value, grad, frameSize, active_node, active_gate, active_state);
  }
}

template <class OpResetOutput>
inline void forward_reset_output(OpResetOutput opResetOutput,
                                 hl_gru_value value,
                                 int frameSize,
                                 int batchSize,
                                 hl_activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_forward_reset_output(opResetOutput,
                                      value.gateValue,
                                      value.resetOutputValue,
                                      value.prevOutValue,
                                      frameSize,
                                      active_gate);
    } else {
      hl_naive_gru_forward_reset_output(opResetOutput,
                                        value.gateValue,
                                        value.resetOutputValue,
                                        value.prevOutValue,
                                        frameSize,
                                        active_gate);
    }

    value.gateValue += frameSize * 3;
    value.resetOutputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template <class OpFinalOutput>
inline void forward_final_output(OpFinalOutput opFinalOutput,
                                 hl_gru_value value,
                                 int frameSize,
                                 int batchSize,
                                 hl_activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_forward_final_output(opFinalOutput,
                                      value.gateValue,
                                      value.prevOutValue,
                                      value.outputValue,
                                      frameSize,
                                      active_node);
    } else {
      hl_naive_gru_forward_final_output(opFinalOutput,
                                        value.gateValue,
                                        value.prevOutValue,
                                        value.outputValue,
                                        frameSize,
                                        active_node);
    }

    value.gateValue += frameSize * 3;
    value.outputValue += frameSize;
    if (value.prevOutValue) {
      value.prevOutValue += frameSize;
    }
  }
}

template <class OpResetOutput, class OpFinalOutput>
void hl_cpu_gru_forward(OpResetOutput opResetOutput,
                        OpFinalOutput opFinalOutput,
                        hl_gru_value value,
                        int frameSize,
                        int batchSize,
                        hl_activation_mode_t active_node,
                        hl_activation_mode_t active_gate) {
  if (value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasNoTrans,
               batchSize,
               2 * frameSize,
               frameSize,
               1,
               value.prevOutValue,
               frameSize,
               value.gateWeight,
               frameSize * 2,
               1,
               value.gateValue,
               frameSize * 3);
  }

  forward_reset_output(opResetOutput, value, frameSize, batchSize, active_gate);

  if (value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasNoTrans,
               batchSize,
               frameSize,
               frameSize,
               1,
               value.resetOutputValue,
               frameSize,
               value.stateWeight,
               frameSize,
               1,
               value.gateValue + frameSize * 2,
               frameSize * 3);
  }

  forward_final_output(opFinalOutput, value, frameSize, batchSize, active_node);
}

template <class OpStateGrad>
inline void backward_state_grad(OpStateGrad opStateGrad,
                                hl_gru_value value,
                                hl_gru_grad grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_node) {
  for (int b = 0; b < batchSize; b++) {
    if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_backward_state_grad(opStateGrad,
                                     value.gateValue,
                                     grad.gateGrad,
                                     value.prevOutValue,
                                     grad.prevOutGrad,
                                     grad.outputGrad,
                                     frameSize,
                                     active_node);
    } else {
      hl_naive_gru_backward_state_grad(opStateGrad,
                                       value.gateValue,
                                       grad.gateGrad,
                                       value.prevOutValue,
                                       grad.prevOutGrad,
                                       grad.outputGrad,
                                       frameSize,
                                       active_node);
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

template <class OpResetGrad>
inline void backward_reset_grad(OpResetGrad opResetGrad,
                                hl_gru_value value,
                                hl_gru_grad grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_gate) {
  for (int b = 0; b < batchSize; b++) {
    if (paddle::HAS_AVX && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
      hl_avx_gru_backward_reset_grad(opResetGrad,
                                     value.gateValue,
                                     grad.gateGrad,
                                     value.prevOutValue,
                                     grad.prevOutGrad,
                                     grad.resetOutputGrad,
                                     frameSize,
                                     active_gate);
    } else {
      hl_naive_gru_backward_reset_grad(opResetGrad,
                                       value.gateValue,
                                       grad.gateGrad,
                                       value.prevOutValue,
                                       grad.prevOutGrad,
                                       grad.resetOutputGrad,
                                       frameSize,
                                       active_gate);
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

template <class OpStateGrad, class OpResetGrad>
void hl_cpu_gru_backward(OpStateGrad opStateGrad,
                         OpResetGrad opResetGrad,
                         hl_gru_value value,
                         hl_gru_grad grad,
                         int frameSize,
                         int batchSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate) {
  backward_state_grad(
      opStateGrad, value, grad, frameSize, batchSize, active_node);

  if (value.prevOutValue && grad.prevOutGrad) {
    CBLAS_GEMM(CblasNoTrans,
               CblasTrans,
               batchSize,
               frameSize,
               frameSize,
               1,
               grad.gateGrad + frameSize * 2,
               frameSize * 3,
               value.stateWeight,
               frameSize,
               0,
               grad.resetOutputGrad,
               frameSize);

    if (grad.stateWeightGrad) {
      CBLAS_GEMM(CblasTrans,
                 CblasNoTrans,
                 frameSize,
                 frameSize,
                 batchSize,
                 1,
                 value.resetOutputValue,
                 frameSize,
                 grad.gateGrad + frameSize * 2,
                 frameSize * 3,
                 1,
                 grad.stateWeightGrad,
                 frameSize);
    }
  }

  backward_reset_grad(
      opResetGrad, value, grad, frameSize, batchSize, active_gate);

  if (grad.prevOutGrad && value.prevOutValue) {
    CBLAS_GEMM(CblasNoTrans,
               CblasTrans,
               batchSize,
               frameSize,
               frameSize * 2,
               1,
               grad.gateGrad,
               frameSize * 3,
               value.gateWeight,
               frameSize * 2,
               1,
               grad.prevOutGrad,
               frameSize);

    if (grad.gateWeightGrad) {
      CBLAS_GEMM(CblasTrans,
                 CblasNoTrans,
                 frameSize,
                 frameSize * 2,
                 batchSize,
                 1,
                 value.prevOutValue,
                 frameSize,
                 grad.gateGrad,
                 frameSize * 3,
                 1,
                 grad.gateWeightGrad,
                 frameSize * 2);
    }
  }
}
