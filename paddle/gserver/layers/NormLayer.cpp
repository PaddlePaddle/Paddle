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

#include "NormLayer.h"
#include "NormProjectionLayer.h"
#include "paddle/utils/Logging.h"
namespace paddle {

REGISTER_LAYER_CREATE_FUNC(norm, &NormLayer::create);

Layer* NormLayer::create(const LayerConfig& config) {
  CHECK_EQ(config.inputs_size(), 1);
  const std::string& norm = config.inputs(0).norm_conf().norm_type();
  if (norm == "rnorm") {
    return new ResponseNormLayer(config);
  } else if (norm == "cmrnorm-projection") {
    return new CMRProjectionNormLayer(config);
  } else if (norm == "cross-channel-norm") {
    return new CrossChannelNormLayer(config);
  } else {
    LOG(FATAL) << "Unknown norm type: " << norm;
    return nullptr;
  }
}

bool ResponseNormLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  NormLayer::init(layerMap, parameterMap);

  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  const NormConfig& conf = config_.inputs(0).norm_conf();
  channels_ = conf.channels();
  size_ = conf.size();
  scale_ = conf.scale();
  pow_ = conf.pow();
  outputX_ = conf.output_x();
  imgSize_ = conf.img_size();
  denoms_ = NULL;

  outputY_ = conf.has_output_y() ? conf.output_y() : conf.output_x();
  imgSizeY_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
  return true;
}

}  // namespace paddle
