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

#include "MaxLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(max, MaxLayer);

void MaxLayer::forward(PassType passType) {
  SequencePoolLayer::forward(passType);

  IVector::resizeOrCreate(
      maxIndex_, newBatchSize_ * getSize(), useGpu(deviceId_));
  maxIndex_->zeroMem();

  MatrixPtr inputValue = getInputValue(0);
  MatrixPtr outputValue = getOutputValue();

  {
    REGISTER_TIMER_INFO("MaxLayerForward", getName().c_str());
    outputValue->maxSequenceForward(
        *inputValue, *startPositions_->getVector(useGpu_), *maxIndex_);
  }

  if (config_.output_max_index()) {
    // copy maxIndex_ to output
    outputValue->copyFrom(*maxIndex_);
  } else {
    /* add the bias-vector AFTER max operation */
    if (biases_.get() != NULL) {
      outputValue->addBias(*(biases_->getW()), 1);
    }
    /* activation */ { forwardActivation(); }
  }
}

void MaxLayer::backward(const UpdateCallback& callback) {
  CHECK(!config_.output_max_index())
      << "backward is not available when output_max_index is set";
  SequencePoolLayer::backward(callback);

  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  if (inputGrad) {
    REGISTER_TIMER_INFO("MaxLayerBackward", getName().c_str());
    inputGrad->maxSequenceBackward(
        *outputGrad, *(startPositions_->getVector(useGpu_)), *maxIndex_);
  }
}

}  // namespace paddle
