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

#include "NormProjectionLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {
size_t CMRProjectionNormLayer::getSize() {
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
  outputH_ = imgSizeH_;
  outputW_ = imgSizeW_;
  layerSize = outputH_ * outputW_ * channels_;

  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);
  return layerSize;
}

bool CMRProjectionNormLayer::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  ResponseNormLayer::init(layerMap, parameterMap);

  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  return true;
}

void CMRProjectionNormLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one row */
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  int batchSize = input->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  Matrix::resizeOrCreate(denoms_, batchSize, size, /* trans */ false, useGpu_);

  denoms_->zeroMem();

  outV->crossMapNormalFwd(
      *input, imgSizeH_, imgSizeW_, *denoms_, channels_, size_, scale_, pow_);
}

void CMRProjectionNormLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  if (NULL == inputLayers_[0]->getOutputGrad()) {
    return;
  }
  /* Do derivation */
  MatrixPtr preOutGrad = inputLayers_[0]->getOutputGrad();
  MatrixPtr localGrad = getOutputGrad();
  MatrixPtr localOutV = getOutputValue();
  MatrixPtr preOutV = inputLayers_[0]->getOutputValue();

  preOutGrad->crossMapNormalBwd(*localGrad,
                                *denoms_,
                                *preOutV,
                                *localOutV,
                                channels_,
                                imgSizeH_,
                                imgSizeW_,
                                size_,
                                scale_,
                                pow_);
}
}  // namespace paddle
