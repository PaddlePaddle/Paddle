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

#include "ScaleSubRegionLayer.h"
#include "paddle/utils/Stat.h"
namespace paddle {

REGISTER_LAYER(scale_sub_region, ScaleSubRegionLayer);

bool ScaleSubRegionLayer::init(const LayerMap& layerMap,
                               const ParameterMap& parameterMap) {
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(static_cast<int>(inputLayers_.size()), 2);
  auto& conf = config_.inputs(0).scale_sub_region_conf();
  value_ = conf.value();

  createFunction(forward_, "ScaleSubRegion", FuncConfig().set("value", value_));
  createFunction(
      backward_, "ScaleSubRegionGrad", FuncConfig().set("value", value_));

  return true;
}

void ScaleSubRegionLayer::forward(PassType passType) {
  Layer::forward(passType);
  auto in0 = getInput(0);
  imgH_ = in0.getFrameHeight();
  imgW_ = in0.getFrameWidth();
  if (imgH_ == 0 || imgW_ == 0) {
    auto& conf = config_.inputs(0).scale_sub_region_conf();
    imgH_ = conf.image_conf().img_size_y();
    imgW_ = conf.image_conf().img_size();
  }
  MatrixPtr imgV = in0.value;
  size_t batchSize = imgV->getHeight();
  size_t spatialSize = imgH_ * imgW_;
  channelsNum_ = imgV->getWidth() / spatialSize;
  shape_ = TensorShape({batchSize, channelsNum_, imgH_, imgW_});

  resetOutput(batchSize, imgV->getWidth());
  auto& out = getOutput();
  out.setFrameHeight(imgH_);
  out.setFrameWidth(imgW_);

  MatrixPtr indicesV = getInputValue(1);
  indicesShape_ = TensorShape({batchSize, 6});

  REGISTER_TIMER_INFO("ScaleSubRegionForward", getName().c_str());
  BufferArgs inArgs;
  BufferArgs outArgs;
  inArgs.addArg(*imgV, shape_);
  inArgs.addArg(*indicesV, indicesShape_);
  outArgs.addArg(*out.value, shape_, ASSIGN_TO);
  forward_[0]->calc(inArgs, outArgs);
}

void ScaleSubRegionLayer::backward(const UpdateCallback& callback) {
  REGISTER_TIMER_INFO("ScaleSubRegionBackward", getName().c_str());
  BufferArgs inArgs;
  BufferArgs outArgs;
  inArgs.addArg(*getOutputGrad(), shape_);
  inArgs.addArg(*getInputValue(1), indicesShape_);
  outArgs.addArg(*getInputGrad(0), shape_, ADD_TO);
  backward_[0]->calc(inArgs, outArgs);
}

}  // namespace paddle
