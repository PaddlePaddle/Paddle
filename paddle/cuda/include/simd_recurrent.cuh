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

#include "hl_base.h"

#ifdef __CUDA_ARCH__
#define DEVICE   __device__
#else
#define DEVICE
#endif

namespace hppl {

#define GENERIC_OPERATOR(__name__) \
class __name__ {                   \
public:                            \
template<typename ... Args>        \
DEVICE void operator()(Args...);   \
}

namespace forward {
  GENERIC_OPERATOR(gru_resetOutput);
  GENERIC_OPERATOR(gru_finalOutput);
  GENERIC_OPERATOR(lstm);
}  // namespace forward

namespace backward {
  GENERIC_OPERATOR(gru_stateGrad);
  GENERIC_OPERATOR(gru_resetGrad);
  GENERIC_OPERATOR(lstm);
}  // namespace backward

template <class Op>
void hl_naive_lstm_forward_one_sequence(Op op,
                                        hl_lstm_value value,
                                        int frameSize,
                                        hl_activation_mode_t active_node,
                                        hl_activation_mode_t active_gate,
                                        hl_activation_mode_t active_state);

template <class Op>
void hl_naive_lstm_backward_one_sequence(Op op,
                                         hl_lstm_value value,
                                         hl_lstm_grad grad,
                                         int frameSize,
                                         hl_activation_mode_t active_node,
                                         hl_activation_mode_t active_gate,
                                         hl_activation_mode_t active_state);

template <class OpResetOutput>
void hl_naive_gru_forward_reset_output(OpResetOutput opResetOutput,
                                       real *gateValue,
                                       real *resetOutputValue,
                                       real *prevOutputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_gate);

template <class OpFinalOutput>
void hl_naive_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                       real *gateValue,
                                       real *prevOutputValue,
                                       real *outputValue,
                                       int frameSize,
                                       hl_activation_mode_t active_node);

template <class OpStateGrad>
void hl_naive_gru_backward_state_grad(OpStateGrad opStateGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *outputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_node);

template <class OpResetGrad>
void hl_naive_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                      real *gateValue,
                                      real *gateGrad,
                                      real *prevOutValue,
                                      real *prevOutGrad,
                                      real *resetOutputGrad,
                                      int frameSize,
                                      hl_activation_mode_t active_gate);

template <class OpResetOutput>
void hl_avx_gru_forward_reset_output(OpResetOutput opResetOutput,
                                     real *gateValue,
                                     real *resetOutputValue,
                                     real *prevOutputValue,
                                     int frameSize,
                                     hl_activation_mode_t active_gate);

template <class OpFinalOutput>
void hl_avx_gru_forward_final_output(OpFinalOutput opFinalOutput,
                                     real *gateValue,
                                     real *prevOutputValue,
                                     real *outputValue,
                                     int frameSize,
                                     hl_activation_mode_t active_node);

template <class OpStateGrad>
void hl_avx_gru_backward_state_grad(OpStateGrad opStateGrad,
                                    real *gateValue,
                                    real *gateGrad,
                                    real *prevOutValue,
                                    real *prevOutGrad,
                                    real *outputGrad,
                                    int frameSize,
                                    hl_activation_mode_t active_node);

template <class OpResetGrad>
void hl_avx_gru_backward_reset_grad(OpResetGrad opResetGrad,
                                    real *gateValue,
                                    real *gateGrad,
                                    real *prevOutValue,
                                    real *prevOutGrad,
                                    real *resetOutputGrad,
                                    int frameSize,
                                    hl_activation_mode_t active_gate);
}  // namespace hppl
