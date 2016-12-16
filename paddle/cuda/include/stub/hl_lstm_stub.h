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

#ifndef HL_LSTM_STUB_H_
#define HL_LSTM_STUB_H_

#include "hl_lstm.h"

inline void hl_lstm_parallel_forward(real *gateValue,
                                     real *stateValue,
                                     real *preOutputValue,
                                     real *outputValue,
                                     real *checkIg,
                                     real *checkFg,
                                     real *checkOg,
                                     real *weight,
                                     const int *sequence,
                                     int frameSize,
                                     int numSequences,
                                     bool reversed,
                                     hl_activation_mode_t active_node,
                                     hl_activation_mode_t active_gate,
                                     hl_activation_mode_t active_state) {}

inline void hl_lstm_parallel_backward_data(real *gateValue,
                                           real *gateGrad,
                                           real *stateValue,
                                           real *stateGrad,
                                           real *preOutputValue,
                                           real *preOutputGrad,
                                           real *outputGrad,
                                           real *checkIg,
                                           real *checkIgGrad,
                                           real *checkFg,
                                           real *checkFgGrad,
                                           real *checkOg,
                                           real *checkOgGrad,
                                           real *weight,
                                           const int *sequence,
                                           int frameSize,
                                           int numSequences,
                                           bool reversed,
                                           hl_activation_mode_t active_node,
                                           hl_activation_mode_t active_gate,
                                           hl_activation_mode_t active_state) {}

inline void hl_lstm_parallel_backward_weight(real *weightGrad,
                                             real *outputValue,
                                             real *gateGrad,
                                             const int *sequence,
                                             int frameSize,
                                             int batchSize,
                                             int numSequences,
                                             bool reversed) {}

#endif  // HL_LSTM_STUB_H_
