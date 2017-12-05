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


#ifndef HL_RECURRENT_APPLY_CUH_
#define HL_RECURRENT_APPLY_CUH_

#include "hl_base.h"
#include "hl_activation_functions.h"
#include "hl_lstm_ops.cuh"
#include "hl_gpu_lstm.cuh"
#include "hl_cpu_lstm.cuh"
#include "hl_gru_ops.cuh"
#include "hl_gpu_gru.cuh"
#include "hl_cpu_gru.cuh"

/**
 * @brief   Cpu lstm forward one sequence.
 *
 * @param[in]   op                  hl_lstm_ops.cuh
 * @param[out]  value               hl_lstm_value type.
 * @param[in]   frameSize           frame size.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 */
template<class Op>
extern void hl_cpu_lstm_forward(Op op,
                                hl_lstm_value value,
                                int frameSize,
                                hl_activation_mode_t active_node,
                                hl_activation_mode_t active_gate,
                                hl_activation_mode_t active_state);

/**
 * @brief   Cpu lstm backward one sequence.
 *
 * @param[in]   op                  hl_lstm_ops.cuh
 * @param[in]   value               lstm value.
 * @param[out]  grad                output gradient.
 * @param[in]   frameSize           frame size.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 */
template<class Op>
extern void hl_cpu_lstm_backward(Op op,
                                 hl_lstm_value value,
                                 hl_lstm_grad grad,
                                 int frameSize,
                                 hl_activation_mode_t active_node,
                                 hl_activation_mode_t active_gate,
                                 hl_activation_mode_t active_state);

/**
 * @brief   Gpu lstm batch forward.
 *
 * @param[in]   op                  hl_lstm_ops.cuh
 * @param[out]  value               lstm value.
 * @param[in]   frameSize           frame size.
 * @param[in]   batchSize           size of current batch.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 */
template<class Op>
extern void hl_gpu_lstm_forward(Op op,
                                hl_lstm_value value,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_node,
                                hl_activation_mode_t active_gate,
                                hl_activation_mode_t active_state);

/**
 * @brief   Gpu lstm batch backward.
 *
 * @param[in]   op                  hl_lstm_ops.cuh
 * @param[out]  value               lstm value.
 * @param[out]  grad                lstm gradient.
 * @param[in]   frameSize           frame size.
 * @param[in]   batchSize           size of current batch.
 * @param[in]   active_node         active input type.
 * @param[in]   active_gate         active state type.
 * @param[in]   active_state        actvie gate type.
 */
template<class Op>
extern void hl_gpu_lstm_backward(Op op,
                                 hl_lstm_value value,
                                 hl_lstm_grad grad,
                                 int frameSize,
                                 int batchSize,
                                 hl_activation_mode_t active_node,
                                 hl_activation_mode_t active_gate,
                                 hl_activation_mode_t active_state);

/**
 * @brief   Cpu gru forward.
 *
 * @param[in]     opResetOutput   hl_gru_ops.cuh
 * @param[in]     opFinalOutput   hl_gru_ops.cuh
 * @param[in,out] value           gru value.
 * @param[in]     frameSize       frame length/size.
 * @param[in]     batchSize       size of current batch.
 * @param[in]     active_node     active input type.
 * @param[in]     active_gate     active state type.
 */
template<class OpResetOutput, class OpFinalOutput>
extern void hl_cpu_gru_forward(OpResetOutput opResetOutput,
                               OpFinalOutput opFinalOutput,
                               hl_gru_value value,
                               int frameSize,
                               int batchSize,
                               hl_activation_mode_t active_node,
                               hl_activation_mode_t active_gate);

/**
 * @brief   Cpu gru forward.
 *
 * @param[in]     opStateGrad     hl_gru_ops.cuh
 * @param[in]     opResetGrad     hl_gru_ops.cuh
 * @param[in]     value           gru value.
 * @param[in,out] grad            gru gradient.
 * @param[in]     frameSize       frame length/size.
 * @param[in]     batchSize       size of current batch.
 * @param[in]     active_node     active input type.
 * @param[in]     active_gate     active state type.
 */
template<class OpStateGrad, class OpResetGrad>
extern void hl_cpu_gru_backward(OpStateGrad opStateGrad,
                                OpResetGrad opResetGrad,
                                hl_gru_value value,
                                hl_gru_grad  grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_node,
                                hl_activation_mode_t active_gate);

/**
 * @brief   Gpu gru forward.
 *
 * @param[in]     opResetOutput   hl_gru_ops.cuh
 * @param[in]     opFinalOutput   hl_gru_ops.cuh
 * @param[in,out] value           gru value.
 * @param[in]     frameSize       frame length/size.
 * @param[in]     batchSize       size of current batch.
 * @param[in]     active_node     active input type.
 * @param[in]     active_gate     active state type.
 */
template<class OpResetOutput, class OpFinalOutput>
extern void hl_gpu_gru_forward(OpResetOutput opResetOutput,
                               OpFinalOutput opFinalOutput,
                               hl_gru_value value,
                               int frameSize,
                               int batchSize,
                               hl_activation_mode_t active_node,
                               hl_activation_mode_t active_gate);

/**
 * @brief   Gpu gru forward.
 *
 * @param[in]     opStateGrad     hl_gru_ops.cuh
 * @param[in]     opResetGrad     hl_gru_ops.cuh
 * @param[in]     value           gru value.
 * @param[in,out] grad            gru gradient.
 * @param[in]     frameSize       frame length/size.
 * @param[in]     batchSize       size of current batch.
 * @param[in]     active_node     active input type.
 * @param[in]     active_gate     active state type.
 */
template<class OpStateGrad, class OpResetGrad>
extern void hl_gpu_gru_backward(OpStateGrad opStateGrad,
                                OpResetGrad opResetGrad,
                                hl_gru_value value,
                                hl_gru_grad  grad,
                                int frameSize,
                                int batchSize,
                                hl_activation_mode_t active_node,
                                hl_activation_mode_t active_gate);

#endif /* HL_RECURRENT_APPLY_CUH_ */
