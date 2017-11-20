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

#include "UpsampleLayer.h"
#include "iostream"

namespace paddle {

REGISTER_LAYER(upsample, UpsampleLayer);

size_t UpsampleLayer::getOutputSize() {
  if (upsampleSize_ == 0) {
    upsampleSize_ = imgSize_ * scale_ - static_cast<int>(padOutX_);
    upsampleSizeY_ = imgSizeY_ * scaleY_ - static_cast<int>(padOutY_);
  }
  return upsampleSize_ * upsampleSizeY_ * channels_;
}

bool UpsampleLayer::init(const LayerMap& layerMap,
                         const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2U);
  CHECK_EQ(config_.inputs_size(), 2);
  const auto& conf = config_.inputs(0).upsample_conf();
  const auto& img_conf = conf.image_conf();

  imgSizeY_ =
      img_conf.has_img_size_y() ? img_conf.img_size_y() : img_conf.img_size();
  imgSize_ = img_conf.img_size();
  channels_ = img_conf.channels();

  CHECK((conf.has_upsample_size()) || (conf.has_scale()))
      << "scale or upsample_size is required.";

  if (conf.has_upsample_size()) {
    upsampleSize_ = conf.upsample_size();
    upsampleSizeY_ = upsampleSize_;
    if (conf.has_upsample_size_y()) {
      upsampleSizeY_ = conf.upsample_size_y();
    }
  } else {
    if (!conf.has_scale_y()) {
      scale_ = scaleY_ = conf.scale_y();
      CHECK_GT(static_cast<int>(scale_), 1);
    } else {
      scale_ = conf.scale();
      scaleY_ = conf.scale_y();
    }
    padOutX_ = conf.pad_out_x();
    padOutY_ = conf.pad_out_y();
    CHECK(!padOutX_ || scale_ == 2)
        << "Output height padding compensation requires scale_ == 2";
    CHECK(!padOutY_ || scaleY_ == 2)
        << "Output width padding compensation requires scaleY_ == 2";
    upsampleSize_ = upsampleSizeY_ = 0;
  }
  return true;
}

void UpsampleLayer::forward(PassType passType) {
  Layer::forward(passType);

  MatrixPtr input = getInputValue(0);
  MatrixPtr mask = inputLayers_[1]->getOutput("mask").value;

  size_t batchSize = input->getHeight();
  size_t outSize = getOutputSize();

  CHECK_EQ(input->getWidth(), mask->getWidth());
  CHECK_EQ(mask->getHeight(), batchSize);
  resetOutput(batchSize, outSize);

  MatrixPtr output = getOutputValue();
  output->upsampleForward(*input,
                          *mask,
                          imgSize_,
                          imgSizeY_,
                          channels_,
                          upsampleSize_,
                          upsampleSizeY_);
}

void UpsampleLayer::backward(const UpdateCallback& callback) {
  MatrixPtr mask = inputLayers_[1]->getOutput("mask").value;
  MatrixPtr inputGrad = getInputGrad(0);
  MatrixPtr outputGrad = getOutputGrad();
  inputGrad->upsampleBackward(*outputGrad,
                              *mask,
                              imgSize_,
                              imgSizeY_,
                              channels_,
                              upsampleSize_,
                              upsampleSizeY_);
}

}  // namespace paddle
