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

#include "ConvBaseLayer.h"
#include "paddle/math/MathUtils.h"
#include "paddle/utils/Logging.h"
namespace paddle {

bool ConvBaseLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  isDeconv_ = (config_.type() == "exconv" || config_.type() == "cudnn_conv")
                  ? false
                  : true;

  /* Initialize the convolutional layer parameter */
  numFilters_ = config_.num_filters();
  sharedBiases_ = config_.shared_biases();
  for (auto& inputConfig : config_.inputs()) {
    const ConvConfig& conf = inputConfig.conv_conf();
    padding_.push_back(conf.padding());
    stride_.push_back(conf.stride());
    dilation_.push_back(conf.dilation());
    filterSize_.push_back(conf.filter_size());
    paddingY_.push_back(conf.padding_y());
    strideY_.push_back(conf.stride_y());
    dilationY_.push_back(conf.dilation_y());
    filterSizeY_.push_back(conf.filter_size_y());
    channels_.push_back(conf.channels());
    imgSizeH_.push_back(conf.has_img_size_y() ? conf.img_size_y()
                                              : conf.img_size());
    imgSizeW_.push_back(conf.img_size());
    groups_.push_back(conf.groups());
    filterChannels_.push_back(conf.filter_channels());
    outputH_.push_back(conf.has_output_y() ? conf.output_y() : conf.output_x());
    outputW_.push_back(conf.output_x());

    paddingZ_.push_back(conf.padding_z());
    strideZ_.push_back(conf.stride_z());
    filterSizeZ_.push_back(conf.filter_size_z());
    imgSizeD_.push_back(conf.img_size_z());
    outputD_.push_back(conf.output_z());
    filterPixels_.push_back(filterSize_.back() * filterSizeY_.back() *
                            filterSizeZ_.back());
  }

  CHECK(inputLayers_.size() == parameters_.size());

  // create new weights_ in derived class
  // create new biases_ in derived class

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

  auto setLayerSize = [&](IntV& inH, IntV& inW, IntV& outH, IntV& outW) {
    size_t filterSizeY;
    size_t filterSize;
    for (size_t i = 0; i < inputLayers_.size(); i++) {
      filterSizeY = (filterSizeY_[i] - 1) * dilationY_[i] + 1;
      filterSize = (filterSize_[i] - 1) * dilation_[i] + 1;
      inH.push_back(inputLayers_[i]->getOutput().getFrameHeight());
      inW.push_back(inputLayers_[i]->getOutput().getFrameWidth());
      const ConvConfig& conf = config_.inputs(i).conv_conf();
      if (isDeconv_) {
        if (inH[i] == 0)
          inH[i] = conf.has_output_y() ? conf.output_y() : conf.output_x();
        if (inW[i] == 0) inW[i] = conf.output_x();
        outH.push_back(imageSize(
            inH[i], filterSizeY, paddingY_[i], strideY_[i], caffeMode_));
        outW.push_back(
            imageSize(inW[i], filterSize, padding_[i], stride_[i], caffeMode_));
      } else {
        if (inH[i] == 0)
          inH[i] = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
        if (inW[i] == 0) inW[i] = conf.img_size();
        outH.push_back(outputSize(
            inH[i], filterSizeY, paddingY_[i], strideY_[i], caffeMode_));
        outW.push_back(outputSize(
            inW[i], filterSize, padding_[i], stride_[i], caffeMode_));
      }
      CHECK_EQ(outH[i], outH[0]);
      CHECK_EQ(outW[i], outW[0]);
    }
    getOutput().setFrameHeight(outH[0]);
    getOutput().setFrameWidth(outW[0]);
    layerSize = outH[0] * outW[0] * size_t(numFilters_);
  };

  setLayerSize(imgSizeH_, imgSizeW_, outputH_, outputW_);

  return layerSize;
}

}  // namespace paddle
