/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "RotateLayer.h"

namespace paddle {

REGISTER_LAYER(rotate, RotateLayer);

bool RotateLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 1UL);
  sampleHeight_ = config_.height();
  return true;
}

void RotateLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr input = getInputValue(0);
  batchSize_ = input->getHeight();
  sampleSize_ = input->getWidth();
  sampleWidth_ = sampleSize_ / sampleHeight_;
  CHECK_EQ(sampleSize_ % sampleHeight_, 0);

  resizeOutput(batchSize_, sampleSize_);

  MatrixPtr outV = getOutputValue();

  for (int b = 0; b < batchSize_; b ++) {
    MatrixPtr inputSample
            = Matrix::create(input->getData() + b * sampleSize_,
                             sampleHeight_,
                             sampleWidth_,
                             false,
                             useGpu_);
    MatrixPtr outputSample
            = Matrix::create(outV->getData() + b * sampleSize_,
                             sampleWidth_,
                             sampleHeight_,
                             false,
                             useGpu_);
    inputSample->rotate(outputSample, false, true);
  }

  if (getInputGrad(0)) {
    zeroGrad();
  }
}

void RotateLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  MatrixPtr outputGrad = getOutputGrad();
  if (outputGrad == NULL) {
    return;
  }
  // the grad should be rotated in the reverse direction
  MatrixPtr preGrad = getInputGrad(0);

  for (int b = 0; b < batchSize_; b ++) {
    MatrixPtr inputSampleGrad
            = Matrix::create(preGrad->getData() + b * sampleSize_,
                             sampleHeight_,
                             sampleWidth_,
                             false,
                             useGpu_);
    MatrixPtr outputSampleGrad
            = Matrix::create(outputGrad->getData() + b * sampleSize_,
                             sampleWidth_,
                             sampleHeight_,
                             false,
                             useGpu_);
    MatrixPtr tmpGrad
            = Matrix::create(sampleHeight_, sampleWidth_, false, useGpu_);
    outputSampleGrad->rotate(tmpGrad, false, false);
    inputSampleGrad->add(*tmpGrad);
  }
}

}  // namespace paddle
