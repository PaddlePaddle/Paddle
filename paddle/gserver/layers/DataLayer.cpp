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

#include "DataLayer.h"

namespace paddle {

REGISTER_LAYER(data, DataLayer);

void DataLayer::copyDataToOutput(Argument& output) {
  if (output.deviceId == data_.deviceId) {
    output.value = data_.value;
    output.in = data_.in;
    output.grad = data_.grad;
    output.ids = data_.ids;
  } else {
    SetDevice device(output.deviceId);
    if (data_.value) {
      if (!output.value) {
        output.value = data_.value->clone(data_.value->getHeight(),
                                          data_.value->getWidth(),
                                          useGpu(output.deviceId));
      } else {
        output.value->resize(data_.value->getHeight(), data_.value->getWidth());
      }
      output.value->copyFrom(*data_.value);
    }
    if (data_.grad) {
      Matrix::resizeOrCreate(output.grad,
                             data_.grad->getHeight(),
                             data_.grad->getWidth(),
                             /* trans= */ false,
                             useGpu(output.deviceId));
    }
    if (data_.ids) {
      IVector::resizeOrCreate(
          output.ids, data_.ids->getSize(), useGpu(output.deviceId));
      output.ids->copyFrom(*data_.ids);
    }
  }
  if (config_.height() && config_.width()) {
    output.setFrameHeight(config_.height());
    output.setFrameWidth(config_.width());
  } else {
    output.setFrameHeight(data_.getFrameHeight());
    output.setFrameWidth(data_.getFrameWidth());
  }
  output.cpuSequenceDims = data_.cpuSequenceDims;
  output.sequenceStartPositions = data_.sequenceStartPositions;
  output.subSequenceStartPositions = data_.subSequenceStartPositions;
  output.strs = data_.strs;

  output.notifyValueReady();
}

}  // namespace paddle
