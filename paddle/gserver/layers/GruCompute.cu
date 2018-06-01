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

#include "GruCompute.h"

#include "hl_recurrent_apply.cuh"

namespace paddle {

template <>
void GruCompute::forward<1>(hl_gru_value value, int frameSize, int batchSize) {
  hl_gpu_gru_forward(hppl::forward::gru_resetOutput(),
                     hppl::forward::gru_finalOutput(),
                     value,
                     frameSize,
                     batchSize,
                     activeNode_,
                     activeGate_);
}

template <>
void GruCompute::backward<1>(hl_gru_value value,
                             hl_gru_grad grad,
                             int frameSize,
                             int batchSize) {
  hl_gpu_gru_backward(hppl::backward::gru_stateGrad(),
                      hppl::backward::gru_resetGrad(),
                      value,
                      grad,
                      frameSize,
                      batchSize,
                      activeNode_,
                      activeGate_);
}

}  // namespace paddle
