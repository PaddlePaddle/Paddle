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
#include "ConvBaseLayer.h"
namespace paddle {

bool ConvBaseLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  if (config_.type() == "exconv" || config_.type() == "cudnn_conv") {
    isDeconv_ = false;
  } else {
    isDeconv_ = true;
  }

  /* Initialize the convolutional layer parameter */
  numFilters_ = config_.num_filters();
  sharedBiases_ = config_.shared_biases();
  for (auto& inputConfig : config_.inputs()) {
    const ConvConfig& conf = inputConfig.conv_conf();
    padding_.push_back(conf.padding());
    stride_.push_back(conf.stride());
    filterSize_.push_back(conf.filter_size());
    paddingY_.push_back(conf.padding_y());
    strideY_.push_back(conf.stride_y());
    filterSizeY_.push_back(conf.filter_size_y());
    filterPixels_.push_back(filterSize_.back() * filterSizeY_.back());
    channels_.push_back(conf.channels());
    imgSizeH_.push_back(conf.img_size());
    imgSizeW_.push_back(conf.img_size());
    groups_.push_back(conf.groups());
    filterChannels_.push_back(conf.filter_channels());
    outputH_.push_back(conf.output_x());
    outputW_.push_back(conf.output_x());
  }

  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    height = filterPixels_[i] * filterChannels_[i];
    width = (!isDeconv_) ? numFilters_ : channels_[i];

    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  /* initialize the biases_ */
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

  // default caffe model
  caffeMode_ = true;

  return true;
}

size_t ConvBaseLayer::calOutputSize() {
  auto clearAndReserve = [this](IntV* vec) {
    vec->clear();
    vec->reserve(this->inputLayers_.size());
  };
  clearAndReserve(&imgSizeH_);
  clearAndReserve(&imgSizeW_);
  clearAndReserve(&outputH_);
  clearAndReserve(&outputW_);
  size_t layerSize = 0;

  if (!isDeconv_) {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      imgSizeH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
      imgSizeW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
      if (imgSizeH_[i] == 0)
        imgSizeH_[i] = config_.inputs(i).conv_conf().img_size();
      if (imgSizeW_[i] == 0)
        imgSizeW_[i] = config_.inputs(i).conv_conf().img_size();
      outputH_.push_back(
          outputSize(imgSizeH_[i], filterSizeY_[i], paddingY_[i], strideY_[i]));
      outputW_.push_back(
          outputSize(imgSizeW_[i], filterSize_[i], padding_[i], stride_[i]));
      CHECK_EQ(outputH_[i], outputH_[0]);
      CHECK_EQ(outputW_[i], outputW_[0]);
    }
    getOutput().setFrameHeight(outputH_[0]);
    getOutput().setFrameWidth(outputW_[0]);
    layerSize = outputH_[0] * outputW_[0] * size_t(numFilters_);
  } else {
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      outputH_.push_back(inputLayers_[i]->getOutput().getFrameHeight());
      outputW_.push_back(inputLayers_[i]->getOutput().getFrameWidth());
      if (outputH_[i] == 0)
        outputH_[i] = config_.inputs(i).conv_conf().output_x();
      if (outputW_[i] == 0)
        outputW_[i] = config_.inputs(i).conv_conf().output_x();
      imgSizeH_.push_back(
          imageSize(outputH_[i], filterSizeY_[i], paddingY_[i], strideY_[i]));
      imgSizeW_.push_back(
          imageSize(outputW_[i], filterSize_[i], padding_[i], stride_[i]));
      CHECK_EQ(imgSizeH_[i], imgSizeH_[0]);
      CHECK_EQ(imgSizeW_[i], imgSizeW_[0]);
    }
    getOutput().setFrameHeight(imgSizeH_[0]);
    getOutput().setFrameWidth(imgSizeW_[0]);
    layerSize = imgSizeH_[0] * imgSizeW_[0] * size_t(numFilters_);
  }

  return layerSize;
}

}  // namespace paddle
