/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

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
  height_ = config_.height();
  width_ = config_.width();
  CHECK_GT(height_, 0);
  CHECK_GT(width_, 0);
  return true;
}

void RotateLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr input = getInputValue(0);
  batchSize_ = input->getHeight();
  size_ = input->getWidth();
  CHECK_GE(size_, height_ * width_);
  CHECK_EQ(size_ % (height_ * width_), 0)
      << "total size_ is not dividable by (height_ * width_), i.e., "
      << "channel number should be an integer";
  channels_ = size_ / (height_ * width_);

  resizeOutput(batchSize_, size_);

  MatrixPtr outV = getOutputValue();
  for (int b = 0; b < batchSize_; b++) {   // for each input feat map
    for (int c = 0; c < channels_; c++) {  // for each feat channel
      MatrixPtr inputSample =
          Matrix::create(input->getData() + b * size_ + c * height_ * width_,
                         height_,
                         width_,
                         false,
                         useGpu_);
      MatrixPtr outputSample =
          Matrix::create(outV->getData() + b * size_ + c * height_ * width_,
                         width_,
                         height_,
                         false,
                         useGpu_);
      inputSample->rotate(outputSample, false, true /* clock-wise */);
    }
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

  for (int b = 0; b < batchSize_; b++) {   // for each input feat map
    for (int c = 0; c < channels_; c++) {  // for each feat channel
      MatrixPtr inputSampleGrad =
          Matrix::create(preGrad->getData() + b * size_ + c * height_ * width_,
                         height_,
                         width_,
                         false,
                         useGpu_);
      MatrixPtr outputSampleGrad = Matrix::create(
          outputGrad->getData() + b * size_ + c * height_ * width_,
          width_,
          height_,
          false,
          useGpu_);
      MatrixPtr tmpGrad = nullptr;
      outputSampleGrad->rotate(tmpGrad, true, false /* anti clock-wise */);
      inputSampleGrad->add(*tmpGrad);
    }
  }
}

}  // namespace paddle
