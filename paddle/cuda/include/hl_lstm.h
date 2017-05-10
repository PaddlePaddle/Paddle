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

#ifndef HL_LSTM_H_
#define HL_LSTM_H_

#include "hl_base.h"

/**
 * @brief   Lstm sequence parallel forward.
 *
 * @param[in]   gateValue           input value.
 * @param[out]  stateValue          state value.
 * @param[out]  preOutputValue     prev output value.
 * @param[out]  outputValue         output value.
 * @param[in]   checkIg             bias.
 * @param[in]   checkFg             bias.
 * @param[in]   checkOg             bias.
 * @param[in]   weight              weight.
 * @param[in]   sequence            sequence index.
 * @param[in]   frameSize           frame size.
 * @param[in]   numSequences        number of sequences.
 * @param[in]   reversed            reverse.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 *
 *
 * @note    Only support frameSize = 32 or 64.
 */
extern void hl_lstm_parallel_forward(real *gateValue,
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
                                     hl_activation_mode_t active_state);

/**
 * @brief   Lstm sequence parallel backward data.
 *
 * @param[in]   gateValue           input value.
 * @param[out]  gateGrad            input gradient.
 * @param[in]   stateValue          state value.
 * @param[out]  stateGrad           state gradient.
 * @param[out]  preOutputValue     prev output value.
 * @param[out]  preOutputGrad      prev output gradient.
 * @param[in]   outputGrad          output gradient.
 * @param[in]   checkIg             bias.
 * @param[out]  checkIgGrad         bias gradient.
 * @param[in]   checkFg             bias.
 * @param[out]  checkFgGrad         bias gradient.
 * @param[in]   checkOg             bias.
 * @param[out]  checkOgGrad         bias gradient.
 * @param[in]   weight              weight.
 * @param[in]   sequence            sequence index.
 * @param[in]   frameSize           frame size.
 * @param[in]   numSequences        number of sequences.
 * @param[in]   reversed            reverse.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 *
 *
 * @note    Only support frameSize = 32 or 64.
 */
extern void hl_lstm_parallel_backward_data(real *gateValue,
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
                                           hl_activation_mode_t active_state);

/**
 * @brief   Lstm sequence parallel backward weight.
 *
 * @param[out]  weightGrad          weight gradient.
 * @param[in]   outputValue         output value.
 * @param[in]   gateGrad            gate gradient.
 * @param[in]   sequence            sequence index.
 * @param[in]   frameSize           frame size.
 * @param[in]   batchSize           batch size.
 * @param[in]   numSequences        number of sequences.
 * @param[in]   reversed            reverse.
 *
 */
extern void hl_lstm_parallel_backward_weight(real *weightGrad,
                                             real *outputValue,
                                             real *gateGrad,
                                             const int *sequence,
                                             int frameSize,
                                             int batchSize,
                                             int numSequences,
                                             bool reversed);

#endif /* HL_LSTM_H_ */
