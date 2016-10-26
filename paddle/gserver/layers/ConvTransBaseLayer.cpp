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
#include "ConvTransBaseLayer.h"
namespace paddle {

bool ConvTransBaseLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* Initialize the convolutional layer parameter */
  /* Everything is the same as ConvBaseLayer.cpp except that the meaning of
   * num_filters and channel is switched.
   *
   * In the config, num_filters refer to the number of feature maps in the
   * output of convTransLayer, and channel refer to the number of feature maps
   * in the input of convTransLayer.
   *
   * However, within the convTrans class, the channel is related to the output
   * and num_filters is related to the input, so that it is consistent with the
   * settings in convLayer.
   * */
  channel_ = config_.num_filters();
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
    numFilters_.push_back(conf.channels());
    imgSize_.push_back(conf.img_size());
    imgPixels_.push_back(imgSize_.back() * imgSize_.back());
    groups_.push_back(conf.groups());
    filterChannels_.push_back(conf.filter_channels());
    outputX_.push_back(conf.output_x());
    outputs_.push_back(outputX_.back() * outputX_.back());
  }

  /* initialize the weightList */
  CHECK(inputLayers_.size() == parameters_.size());
  for (size_t i = 0; i < inputLayers_.size(); i++) {
    size_t height, width;
    height = filterPixels_[i] * filterChannels_[i];
    width = numFilters_[i];

    // create a new weight
    CHECK_EQ(parameters_[i]->getSize(), width * height);
    Weight* w = new Weight(height, width, parameters_[i]);
    weights_.emplace_back(w);
  }

  /* initialize the biases_ */
  if (biasParameter_.get() != NULL) {
    if (sharedBiases_) {
      CHECK_EQ((size_t)channel_, biasParameter_->getSize());
      biases_ =
          std::unique_ptr<Weight>(new Weight(channel_, 1, biasParameter_));
    } else {
      biases_ =
          std::unique_ptr<Weight>(new Weight(getSize(), 1, biasParameter_));
    }
  }

  // default caffe model
  caffeMode_ = true;

  return true;
}

}  // namespace paddle
