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


#include "LstmCompute.h"
#include "hl_recurrent_apply.cuh"

namespace paddle {

template <>
void LstmCompute::forwardBatch<1>(hl_lstm_value value, int frameSize,
                                 int batchSize) {
  hl_gpu_lstm_forward(hppl::forward::lstm(), value, frameSize,
                      batchSize, activeNode_, activeGate_,
                      activeState_);
}

template <>
void LstmCompute::backwardBatch<1>(hl_lstm_value value, hl_lstm_grad grad,
                                   int frameSize, int batchSize) {
  hl_gpu_lstm_backward(hppl::backward::lstm(), value, grad,
                       frameSize, batchSize, activeNode_,
                       activeGate_, activeState_);
}

template <>
void LstmCompute::forwardOneSequence<1>(hl_lstm_value value, int frameSize) {
  hl_gpu_lstm_forward(hppl::forward::lstm(), value,
                      frameSize, /* batchSize */ 1,
                      activeNode_, activeGate_, activeState_);
}

template <>
void LstmCompute::backwardOneSequence<1>(hl_lstm_value value, hl_lstm_grad grad,
                                         int frameSize) {
  hl_gpu_lstm_backward(hppl::backward::lstm(), value, grad,
                       frameSize, /* batchSize */ 1,
                       activeNode_, activeGate_, activeState_);
}

}  // namespace paddle
