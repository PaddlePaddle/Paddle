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


#ifndef HL_CPU_LSTM_CUH_
#define HL_CPU_LSTM_CUH_

#ifndef __NVCC__

// using namespace hppl;

template<class Op>
void hl_naive_lstm_forward_one_sequence(Op op,
                                        hl_lstm_value value,
                                        int frameSize,
                                        hl_activation_mode_t active_node,
                                        hl_activation_mode_t active_gate,
                                        hl_activation_mode_t active_state) {
  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rCheckI;
  real rCheckF;
  real rCheckO;
  real rState;
  real rPrevState = 0;
  real rStateAtv;
  real rOut;

  real *valueIn = value.gateValue;
  real *valueIg = value.gateValue + frameSize;
  real *valueFg = value.gateValue + frameSize * 2;
  real *valueOg = value.gateValue + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg[i];
    rCheckF = value.checkFg[i];
    rCheckO = value.checkOg[i];

    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rPrevState,
       rState,
       rStateAtv,
       rOut,
       rCheckI,
       rCheckF,
       rCheckO,
       hppl::cpu::forward[active_node],
       hppl::cpu::forward[active_gate],
       hppl::cpu::forward[active_state]);

    valueIn[i] = rValueIn;
    valueIg[i] = rValueIg;
    valueFg[i] = rValueFg;
    valueOg[i] = rValueOg;
    value.stateValue[i] = rState;
    value.stateActiveValue[i] = rStateAtv;
    value.outputValue[i] = rOut;
  }
}

template<class Op>
void hl_naive_lstm_backward_one_sequence(Op op,
                                         hl_lstm_value value,
                                         hl_lstm_grad grad,
                                         int frameSize,
                                         hl_activation_mode_t active_node,
                                         hl_activation_mode_t active_gate,
                                         hl_activation_mode_t active_state) {
  real rValueIn;
  real rValueIg;
  real rValueFg;
  real rValueOg;
  real rGradIn;
  real rGradIg;
  real rGradFg;
  real rGradOg;
  real rPrevState = 0;
  real rPrevStateGrad;
  real rState;
  real rStateGrad;
  real rStateAtv;
  real rOutputGrad;
  real rCheckI;
  real rCheckF;
  real rCheckO;
  real rCheckIGrad;
  real rCheckFGrad;
  real rCheckOGrad;

  real *valueIn = value.gateValue;
  real *valueIg = value.gateValue + frameSize;
  real *valueFg = value.gateValue + frameSize * 2;
  real *valueOg = value.gateValue + frameSize * 3;
  real *gradIn = grad.gateGrad;
  real *gradIg = grad.gateGrad + frameSize;
  real *gradFg = grad.gateGrad + frameSize * 2;
  real *gradOg = grad.gateGrad + frameSize * 3;

  for (int i = 0; i < frameSize; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = value.checkIg[i];
    rCheckF = value.checkFg[i];
    rCheckO = value.checkOg[i];
    rState = value.stateValue[i];
    rStateAtv = value.stateActiveValue[i];
    rOutputGrad = grad.outputGrad[i];
    rStateGrad = grad.stateGrad[i];
    if (value.prevStateValue) {
      rPrevState = value.prevStateValue[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rGradIn,
       rGradIg,
       rGradFg,
       rGradOg,
       rPrevState,
       rPrevStateGrad,
       rState,
       rStateGrad,
       rStateAtv,
       rOutputGrad,
       rCheckI,
       rCheckF,
       rCheckO,
       rCheckIGrad,
       rCheckFGrad,
       rCheckOGrad,
       hppl::cpu::backward[active_node],
       hppl::cpu::backward[active_gate],
       hppl::cpu::backward[active_state]);

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

template<class Op>
void hl_avx_lstm_forward_one_sequence(Op op,
                                      hl_lstm_value value,
                                      int frameSize,
                                      hl_activation_mode_t active_node,
                                      hl_activation_mode_t active_gate,
                                      hl_activation_mode_t active_state) {
#ifdef __AVX__
  __m256 rValueIn;
  __m256 rValueIg;
  __m256 rValueFg;
  __m256 rValueOg;
  __m256 rCheckI;
  __m256 rCheckF;
  __m256 rCheckO;
  __m256 rState;
  __m256 rPrevState = _mm256_set1_ps(0.0f);
  __m256 rStateAtv;
  __m256 rOut;

  __m256 *valueIn = (__m256*)value.gateValue;
  __m256 *valueIg = (__m256*)(value.gateValue + frameSize);
  __m256 *valueFg = (__m256*)(value.gateValue + frameSize * 2);
  __m256 *valueOg = (__m256*)(value.gateValue + frameSize * 3);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = ((__m256*)value.checkIg)[i];
    rCheckF = ((__m256*)value.checkFg)[i];
    rCheckO = ((__m256*)value.checkOg)[i];

    if (value.prevStateValue) {
      rPrevState = ((__m256*)value.prevStateValue)[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rPrevState,
       rState,
       rStateAtv,
       rOut,
       rCheckI,
       rCheckF,
       rCheckO,
       hppl::avx::forward[active_node],
       hppl::avx::forward[active_gate],
       hppl::avx::forward[active_state]);

    valueIn[i] = rValueIn;
    valueIg[i] = rValueIg;
    valueFg[i] = rValueFg;
    valueOg[i] = rValueOg;
    ((__m256*)value.stateValue)[i] = rState;
    ((__m256*)value.stateActiveValue)[i] = rStateAtv;
    ((__m256*)value.outputValue)[i] = rOut;
  }
#endif
}

template<class Op>
void hl_avx_lstm_backward_one_sequence(Op op,
                                       hl_lstm_value value,
                                       hl_lstm_grad grad,
                                       int frameSize,
                                       hl_activation_mode_t active_node,
                                       hl_activation_mode_t active_gate,
                                       hl_activation_mode_t active_state) {
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
  __m256 rCheckI;
  __m256 rCheckF;
  __m256 rCheckO;
  __m256 rCheckIGrad;
  __m256 rCheckFGrad;
  __m256 rCheckOGrad;

  __m256 *valueIn = (__m256*)value.gateValue;
  __m256 *valueIg = (__m256*)(value.gateValue + frameSize);
  __m256 *valueFg = (__m256*)(value.gateValue + frameSize * 2);
  __m256 *valueOg = (__m256*)(value.gateValue + frameSize * 3);
  __m256 *gradIn = (__m256*)grad.gateGrad;
  __m256 *gradIg = (__m256*)(grad.gateGrad + frameSize);
  __m256 *gradFg = (__m256*)(grad.gateGrad + frameSize * 2);
  __m256 *gradOg = (__m256*)(grad.gateGrad + frameSize * 3);

  for (int i = 0; i < frameSize / 8; i++) {
    rValueIn = valueIn[i];
    rValueIg = valueIg[i];
    rValueFg = valueFg[i];
    rValueOg = valueOg[i];
    rCheckI = ((__m256*)value.checkIg)[i];
    rCheckF = ((__m256*)value.checkFg)[i];
    rCheckO = ((__m256*)value.checkOg)[i];
    rState = ((__m256*)value.stateValue)[i];
    rStateAtv = ((__m256*)value.stateActiveValue)[i];
    rOutputGrad = ((__m256*)grad.outputGrad)[i];
    rStateGrad = ((__m256*)grad.stateGrad)[i];
    if (value.prevStateValue) {
      rPrevState = ((__m256*)value.prevStateValue)[i];
    }

    op(rValueIn,
       rValueIg,
       rValueFg,
       rValueOg,
       rGradIn,
       rGradIg,
       rGradFg,
       rGradOg,
       rPrevState,
       rPrevStateGrad,
       rState,
       rStateGrad,
       rStateAtv,
       rOutputGrad,
       rCheckI,
       rCheckF,
       rCheckO,
       rCheckIGrad,
       rCheckFGrad,
       rCheckOGrad,
       hppl::avx::backward[active_node],
       hppl::avx::backward[active_gate],
       hppl::avx::backward[active_state]);

    gradIn[i] = rGradIn;
    gradIg[i] = rGradIg;
    gradFg[i] = rGradFg;
    gradOg[i] = rGradOg;
    ((__m256*)grad.stateGrad)[i] = rStateGrad;

    if (grad.prevStateGrad) ((__m256*)grad.prevStateGrad)[i] = rPrevStateGrad;
    if (value.prevStateValue) {
      if (grad.checkIgGrad) ((__m256*)grad.checkIgGrad)[i] += rCheckIGrad;
      if (grad.checkFgGrad) ((__m256*)grad.checkFgGrad)[i] += rCheckFGrad;
    }
    if (grad.checkOgGrad) ((__m256*)grad.checkOgGrad)[i] += rCheckOGrad;
  }
#endif
}

template<class Op>
void hl_cpu_lstm_forward(Op op,
                         hl_lstm_value value,
                         int frameSize,
                         hl_activation_mode_t active_node,
                         hl_activation_mode_t active_gate,
                         hl_activation_mode_t active_state) {
  if (Op::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_forward_one_sequence(op, value, frameSize,
        active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_forward_one_sequence(op, value, frameSize,
        active_node, active_gate, active_state);
  }
}

template<class Op>
void hl_cpu_lstm_backward(Op op,
                          hl_lstm_value value,
                          hl_lstm_grad grad,
                          int frameSize,
                          hl_activation_mode_t active_node,
                          hl_activation_mode_t active_gate,
                          hl_activation_mode_t active_state) {
  if (Op::avx && !(frameSize & (8 - 1)) && (sizeof(real) == 4)) {
    hl_avx_lstm_backward_one_sequence(op, value, grad, frameSize,
        active_node, active_gate, active_state);
  } else {
    hl_naive_lstm_backward_one_sequence(op, value, grad, frameSize,
        active_node, active_gate, active_state);
  }
}

#endif

#endif /* HL_CPU_LSTM_CUH_ */
