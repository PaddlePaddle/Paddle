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

#include "BatchNormBaseLayer.h"
#include "BatchNormalizationLayer.h"
#include "Layer.h"
#include "paddle/utils/Stat.h"
#ifdef PADDLE_WITH_CUDA
#include "CudnnBatchNormLayer.h"
#endif

namespace paddle {

bool BatchNormBaseLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  if (!Layer::init(layerMap, parameterMap)) return false;

  /* initialize the weightList */
  // first is Input in configure
  // other two is created in config_parser.py
  CHECK_EQ(inputLayers_.size(), 3U);
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  CHECK_EQ(inputLayers_.size(), size_t(config_.inputs_size()));
  const ImageConfig& conf = config_.inputs(0).image_conf();
  channels_ = conf.channels();
  calFeatureMapSize();

  if (config_.has_use_global_stats()) {
    useGlobalStats_ = config_.use_global_stats();
  }
  movingAvgFraction_ = config_.moving_average_fraction();
  epsilon_ = config_.epsilon();

  weight_.reset(new Weight(1, channels_, parameters_[0]));
  movingMean_.reset(new Weight(1, channels_, parameters_[1]));
  movingVar_.reset(new Weight(1, channels_, parameters_[2]));

  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, channels_, biasParameter_));
  }

  savedMean_ = Matrix::create(1, channels_, false, useGpu_);
  savedInvVar_ = Matrix::create(1, channels_, false, useGpu_);
  savedMean_->zeroMem();
  savedInvVar_->zeroMem();

  return true;
}

void BatchNormBaseLayer::calFeatureMapSize() {
  const ImageConfig& conf = config_.inputs(0).image_conf();
  imageH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imageW_ = inputLayers_[0]->getOutput().getFrameWidth();
  imageD_ = inputLayers_[0]->getOutput().getFrameDepth();

  if (0 == imageD_) imageD_ = conf.img_size_z();
  if (imageH_ == 0 && imageW_ == 0) {
    imageH_ = conf.has_img_size_y() ? conf.img_size_y() : conf.img_size();
    imageW_ = conf.img_size();
  } else {
    getOutput().setFrameHeight(imageH_);
    getOutput().setFrameWidth(imageW_);
    getOutput().setFrameDepth(imageD_);
  }
  imgPixels_ = imageH_ * imageW_ * imageD_;
}

}  // namespace paddle
