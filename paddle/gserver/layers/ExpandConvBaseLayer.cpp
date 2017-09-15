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

#include "ExpandConvBaseLayer.h"

#include "paddle/utils/Logging.h"
namespace paddle {

bool ExpandConvBaseLayer::init(const LayerMap &layerMap,
                               const ParameterMap &parameterMap) {
  /* Initialize the basic convolutional parent class */
  ConvBaseLayer::init(layerMap, parameterMap);

  int index = 0;
  for (auto &inputConfig : config_.inputs()) {
    const ConvConfig &conf = inputConfig.conv_conf();
    /* Consistent caffe mode for multiple input */
    caffeMode_ = conf.caffe_mode();

    // create a new weight
    size_t height, width;
    height = filterPixels_[index] * filterChannels_[index];
    width = (!isDeconv_) ? numFilters_ : channels_[index];
    CHECK_EQ(parameters_[index]->getSize(), width * height);
    Weight *w = new Weight(height, width, parameters_[index]);
    weights_.emplace_back(w);
    index++;
  }
  if (biasParameter_.get()) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)numFilters_, biasParameter_->getSize());
      biases_ =
          std::unique_ptr<Weight>(new Weight(numFilters_, 1, biasParameter_));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(getSize(), 1, biasParameter_));
    }
  }
  getOutputSize();

  return true;
}

size_t ExpandConvBaseLayer::getOutputSize() {
  CHECK_NE(inputLayers_.size(), 0UL);
  size_t layerSize = ConvBaseLayer::calOutputSize();
  return layerSize;
}

}  // namespace paddle
