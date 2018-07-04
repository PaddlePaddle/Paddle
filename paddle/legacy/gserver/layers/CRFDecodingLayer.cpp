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

#include "CRFDecodingLayer.h"

namespace paddle {

REGISTER_LAYER(crf_decoding, CRFDecodingLayer);

bool CRFDecodingLayer::init(const LayerMap& layerMap,
                            const ParameterMap& parameterMap) {
  if (!CRFLayer::init(layerMap, parameterMap)) {
    return false;
  }
  crf_.reset(new LinearChainCRF(
      numClasses_, parameter_->getBuf(PARAMETER_VALUE)->getData()));
  return true;
}

void CRFDecodingLayer::forward(PassType passType) {
  Layer::forward(passType);

  CHECK(!useGpu_) << "GPU is not supported";

  const Argument& output = getInput(0);
  CHECK(output.sequenceStartPositions);

  size_t batchSize = output.getBatchSize();
  size_t numSequences = output.sequenceStartPositions->getSize() - 1;

  IVector::resizeOrCreate(output_.ids, batchSize, useGpu_);
  const int* starts = output.sequenceStartPositions->getData(false);
  CHECK_EQ(starts[numSequences], (int)batchSize);

  for (size_t i = 0; i < numSequences; ++i) {
    crf_->decode(output.value->getData() + numClasses_ * starts[i],
                 output_.ids->getData() + starts[i],
                 starts[i + 1] - starts[i]);
  }

  if (inputLayers_.size() == 2) {
    const Argument& label = getInput(1);
    resizeOutput(batchSize, 1);
    CHECK(label.ids);
    real* error = output_.value->getData();
    int* ids = label.ids->getData();
    int* result = output_.ids->getData();
    for (size_t i = 0; i < batchSize; ++i) {
      error[i] = ids[i] == result[i] ? 0 : 1;
    }
  }
}

void CRFDecodingLayer::backward(const UpdateCallback& callback) {
  parameter_->incUpdate(callback);
}

}  // namespace paddle
