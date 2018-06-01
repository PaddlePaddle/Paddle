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

#include "LstmCompute.h"
#include "hl_recurrent_apply.cuh"
#include "paddle/utils/Util.h"

namespace paddle {

void LstmCompute::init(LayerConfig &config) {
  activeNode_ = hlActiveType(config.active_type());
  activeGate_ = hlActiveType(config.active_gate_type());
  activeState_ = hlActiveType(config.active_state_type());
}

template <>
void LstmCompute::forwardOneSequence<0>(hl_lstm_value value, int frameSize) {
  hl_cpu_lstm_forward(hppl::forward::lstm(),
                      value,
                      frameSize,
                      activeNode_,
                      activeGate_,
                      activeState_);
}

template <>
void LstmCompute::backwardOneSequence<0>(hl_lstm_value value,
                                         hl_lstm_grad grad,
                                         int frameSize) {
  hl_cpu_lstm_backward(hppl::backward::lstm(),
                       value,
                       grad,
                       frameSize,
                       activeNode_,
                       activeGate_,
                       activeState_);
}

template <>
void LstmCompute::forwardBatch<0>(hl_lstm_value value,
                                  int frameSize,
                                  int batchSize) {
  for (int b = 0; b < batchSize; b++) {
    forwardOneSequence<0>(value, frameSize);

    value.gateValue += frameSize * 4;
    value.stateValue += frameSize;
    value.stateActiveValue += frameSize;
    value.outputValue += frameSize;
    if (value.prevStateValue) {
      value.prevStateValue += frameSize;
    }
  }
}

template <>
void LstmCompute::backwardBatch<0>(hl_lstm_value value,
                                   hl_lstm_grad grad,
                                   int frameSize,
                                   int batchSize) {
  for (int b = 0; b < batchSize; b++) {
    backwardOneSequence<0>(value, grad, frameSize);

    value.gateValue += frameSize * 4;
    value.stateValue += frameSize;
    value.stateActiveValue += frameSize;
    value.outputValue += frameSize;
    if (value.prevStateValue) {
      value.prevStateValue += frameSize;
    }

    grad.gateGrad += frameSize * 4;
    grad.stateGrad += frameSize;
    grad.stateActiveGrad += frameSize;
    grad.outputGrad += frameSize;
    if (grad.prevStateGrad) {
      grad.prevStateGrad += frameSize;
    }
  }
}

}  // namespace paddle
