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


#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "PoolProjectionLayer.h"

namespace paddle {

size_t PoolProjectionLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = imgSizeY_;
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = imgSize_;
  }

  outputH_ = outputSize(imgSizeH_, sizeY_, confPaddingY_, strideY_);
  outputW_ = outputSize(imgSizeW_, sizeX_, confPadding_, stride_);

  layerSize = outputH_ * outputW_ * channels_;

  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);
  return layerSize;
}

void MaxPoolProjectionLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one ROW */
  MatrixPtr input = getInputValue(0);
  int batchSize = input->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  outV->maxPoolForward(*input, imgSizeH_, imgSizeW_, channels_,
                       sizeX_, sizeY_, strideY_, stride_,
                       outputH_, outputW_, confPaddingY_, confPadding_);
}

void MaxPoolProjectionLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  if (NULL == getInputGrad(0)) {
    return;
  }

  /* Do derivation */
  MatrixPtr outGrad = getOutputGrad();
  MatrixPtr inputV = getInputValue(0);
  MatrixPtr outV = getOutputValue();
  MatrixPtr inputGrad = getInputGrad(0);

  inputGrad->maxPoolBackward(*inputV, imgSizeH_, imgSizeW_, *outGrad, *outV,
                             sizeX_, sizeY_,
                             strideY_, stride_, outputH_, outputW_, 1, 1,
                             confPaddingY_, confPadding_);
}

void AvgPoolProjectionLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one ROW */
  MatrixPtr input = getInputValue(0);
  int batchSize = input->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  outV->avgPoolForward(*input, imgSizeH_, imgSizeW_, channels_,
                       sizeX_, sizeY_, strideY_, stride_,
                       outputH_, outputW_, confPaddingY_, confPadding_);
}

void AvgPoolProjectionLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  if (NULL == getInputGrad(0)) {
    return;
  }
  /* Do derivation */
  MatrixPtr outputGrad = getOutputGrad();
  MatrixPtr inputGrad = getInputGrad(0);
  inputGrad->avgPoolBackward(*outputGrad, imgSizeH_, imgSizeW_,
                             sizeX_, sizeY_, strideY_, stride_,
                             outputH_, outputW_, 1, 1,
                             confPaddingY_, confPadding_);
}
}  // namespace paddle
