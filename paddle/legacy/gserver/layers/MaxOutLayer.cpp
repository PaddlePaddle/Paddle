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

#include "MaxOutLayer.h"
#include "hl_cnn.h"
#include "hl_gpu.h"

namespace paddle {

REGISTER_LAYER(maxout, MaxOutLayer);

size_t MaxOutLayer::getSize() {
  const MaxOutConfig& maxoutConf = config_.inputs(0).maxout_conf();
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = maxoutConf.image_conf().img_size_y();
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = maxoutConf.image_conf().img_size();
  }

  featLen_ = imgSizeH_ * imgSizeW_;
  size_t layerSize = featLen_ * outputChannels_;

  getOutput().setFrameHeight(imgSizeH_);
  getOutput().setFrameWidth(imgSizeW_);

  return layerSize;
}

bool MaxOutLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* the size of inputs for maxout-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const MaxOutConfig& conf = config_.inputs(0).maxout_conf();
  groups_ = conf.groups();
  channels_ = conf.image_conf().channels();
  CHECK_EQ(channels_ % groups_, 0UL);
  outputChannels_ = channels_ / groups_;

  return true;
}

void MaxOutLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one column */
  size_t batchSize = getInput(0).getBatchSize();
  size_t size = getSize();
  resetOutput(batchSize, size);
  MatrixPtr inputV = getInputValue(0);
  MatrixPtr outV = getOutputValue();

  IVector::resizeOrCreate(maxoutId_, size * batchSize, useGpu_);
  outV->maxoutForward(*inputV, *maxoutId_, outputChannels_, groups_);
}

void MaxOutLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  /* Do derivation */
  MatrixPtr inputG = getInputGrad(0);
  MatrixPtr outG = getOutputGrad();

  if (inputG) {
    inputG->maxoutBackward(*outG, *maxoutId_, outputChannels_, groups_);
  }
}

}  // namespace paddle
