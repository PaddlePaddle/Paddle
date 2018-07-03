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

#include "MaxPoolWithMaskLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

bool MaxPoolWithMaskLayer::init(const LayerMap& layerMap,
                                const ParameterMap& parameterMap) {
  PoolLayer::init(layerMap, parameterMap);
  setOutput("mask", &mask_);
  return true;
}

size_t MaxPoolWithMaskLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;

  outputY_ = outputSize(imgSizeY_,
                        sizeY_,
                        confPaddingY_,
                        strideY_,
                        /* caffeMode */ false);
  outputX_ = outputSize(imgSize_,
                        sizeX_,
                        confPadding_,
                        stride_,
                        /* caffeMode */ false);

  layerSize = outputX_ * outputY_ * channels_;
  getOutput().setFrameHeight(outputY_);
  getOutput().setFrameWidth(outputX_);

  return layerSize;
}

void MaxPoolWithMaskLayer::forward(PassType passType) {
  size_t size = getSize();
  MatrixPtr inputV = inputLayers_[0]->getOutputValue();
  int batchSize = inputV->getHeight();
  resetOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();
  CHECK_EQ(size, outV->getWidth());

  resetSpecifyOutput(mask_,
                     batchSize,
                     size,
                     /* isValueClean */ false,
                     /* isGradClean */ true);

  MatrixPtr maskV = mask_.value;
  outV->maxPoolForward(*inputV,
                       imgSizeY_,
                       imgSize_,
                       channels_,
                       sizeX_,
                       sizeY_,
                       strideY_,
                       stride_,
                       outputY_,
                       outputX_,
                       confPaddingY_,
                       confPadding_,
                       maskV);
}

void MaxPoolWithMaskLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  if (NULL == getInputGrad(0)) {
    return;
  }

  MatrixPtr outGrad = getOutputGrad();
  MatrixPtr inputV = inputLayers_[0]->getOutputValue();
  MatrixPtr outV = getOutputValue();
  MatrixPtr inputGrad = inputLayers_[0]->getOutputGrad();

  inputGrad->maxPoolBackward(*inputV,
                             imgSizeY_,
                             imgSize_,
                             *outGrad,
                             *outV,
                             sizeX_,
                             sizeY_,
                             strideY_,
                             stride_,
                             outputY_,
                             outputX_,
                             1,
                             1,
                             confPaddingY_,
                             confPadding_);
}

}  // namespace paddle
