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

#include "GemmFunctor.h"
#include "hl_cpu_gru.cuh"

namespace paddle {

template <DeviceType Device, class T>
struct GruFunctor {
  template <class OpResetOutput, class OpFinalOutput>
  static void compute(OpResetOutput opResetOutput,
                      OpFinalOutput opFinalOutput,
                      hl_gru_value value,
                      int frameSize,
                      int batchSize,
                      hl_activation_mode_t active_node,
                      hl_activation_mode_t active_gate) {
#ifndef __NVCC__
    if (value.prevOutValue) {
      BlasGemm<Device, T>::compute(false,
                                   false,
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

    forward_reset_output(
        opResetOutput, value, frameSize, batchSize, active_gate);

    if (value.prevOutValue) {
      BlasGemm<Device, T>::compute(false,
                                   false,
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

    forward_final_output(
        opFinalOutput, value, frameSize, batchSize, active_node);
#endif
  }
};

template <DeviceType Device, class T>
struct GruGradFunctor {
  template <class OpStateGrad, class OpResetGrad>
  static void compute(OpStateGrad opStateGrad,
                      OpResetGrad opResetGrad,
                      hl_gru_value value,
                      hl_gru_grad grad,
                      int frameSize,
                      int batchSize,
                      hl_activation_mode_t active_node,
                      hl_activation_mode_t active_gate) {
#ifndef __NVCC__
    backward_state_grad(
        opStateGrad, value, grad, frameSize, batchSize, active_node);

    if (value.prevOutValue && grad.prevOutGrad) {
      BlasGemm<Device, T>::compute(false,
                                   true,
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
        BlasGemm<Device, T>::compute(true,
                                     false,
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
      BlasGemm<Device, T>::compute(false,
                                   true,
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
        BlasGemm<Device, T>::compute(true,
                                     false,
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
#endif
  }
};

}  // namespace paddle
