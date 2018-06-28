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

#include "ModelConfig.pb.h"
#include "hl_gpu.h"
#include "paddle/utils/Common.h"

namespace paddle {

class LstmCompute {
 public:
  void init(LayerConfig &config);

  /**
   * LstmLayer batch compute API (forwardBatch, backwardBatch).
   * If use batch compute api, lstm value(and grad) need to be batch structure.
   * Compute order:
   *   forwardBatch:  for 0 <= id < numBatch
   *   backwardBatch:  for numBatch > id >= 0
   */
  template <bool useGpu>
  void forwardBatch(hl_lstm_value value, int frameSize, int batchSize);

  template <bool useGpu>
  void backwardBatch(hl_lstm_value value,
                     hl_lstm_grad grad,
                     int frameSize,
                     int batchSize);

  /**
   * LstmLayer sequence compute API (forwardOneSequence, backwardOneSequence).
   * Compute order(for each sequence):
   *   forwardOneSequence:
   *     if (!reversed) for 0 <= seqId < seqLength
   *     if (reversed)  for seqLength > seqId >= 0
   *   backwardOneSequence:
   *     if (!reversed) for seqLength > seqId >= 0
   *     if (reversed)  for 0 <= seqId < seqLength
   */
  template <bool useGpu>
  void forwardOneSequence(hl_lstm_value value, int frameSize);
  template <bool useGpu>
  void backwardOneSequence(hl_lstm_value value,
                           hl_lstm_grad grad,
                           int frameSize);

 public:
  hl_activation_mode_t activeNode_;
  hl_activation_mode_t activeGate_;
  hl_activation_mode_t activeState_;
};

}  // namespace paddle
