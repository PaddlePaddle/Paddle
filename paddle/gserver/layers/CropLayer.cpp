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

#include "CropLayer.h"
#include "paddle/utils/Stat.h"
namespace paddle {

REGISTER_LAYER(crop, CropLayer);

bool CropLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_LE(static_cast<int>(inputLayers_.size()), 2);
  CHECK_GE(static_cast<int>(inputLayers_.size()), 1);
  crop_axis_ = config_.axis();
  for (int i = 0; i < config_.offset_size(); i++) {
    crop_offsets_.push_back(config_.offset(i));
  }

  // 1. get input_0 shape
  auto& input0_img_conf = config_.inputs(0).image_conf();
  inDims_ = TensorShape({0,
                         input0_img_conf.channels(),
                         input0_img_conf.has_img_size_y()
                             ? input0_img_conf.img_size_y()
                             : input0_img_conf.img_size(),
                         input0_img_conf.img_size()});
  // 2. get target dims from config
  if (config_.inputs_size() == 1) {
    targetDims_ = TensorShape({config_.shape(0),
                               config_.shape(1),
                               config_.shape(2),
                               config_.shape(3)});
  } else {
    // 2. get input_1 shape
    auto& input1_img_conf = config_.inputs(1).image_conf();
    targetDims_ = TensorShape({0,
                               input1_img_conf.channels(),
                               input1_img_conf.has_img_size_y()
                                   ? input1_img_conf.img_size_y()
                                   : input1_img_conf.img_size(),
                               input1_img_conf.img_size()});
  }

  // 3. get final crop corner
  int dimSize = 4;
  crop_corner_ = {0, 0, 0, 0};
  for (int i = 0; i < dimSize; i++) {
    if (i >= crop_axis_) {
      if (crop_offsets_.size() > 1) {
        crop_corner_[i] = crop_offsets_[i - crop_axis_];
      } else {
        crop_corner_[i] = crop_offsets_[0];
      }
    }
  }

  outDims_ = TensorShape(4);

  createFunction(
      forward_, "Crop", FuncConfig().set("crop_corner", crop_corner_));
  createFunction(
      backward_, "CropGrad", FuncConfig().set("crop_corner", crop_corner_));

  return true;
}

void CropLayer::setOutDims() {
  MatrixPtr input = inputLayers_[1]->getOutputValue();
  size_t batchSize = input->getHeight();
  // get target dims from input_1
  if (config_.inputs_size() == 2) {
    targetDims_.setDim(0, batchSize);
    int ch = config_.inputs(0).image_conf().channels();
    if (ch != 0) targetDims_.setDim(1, ch);
    int h = inputLayers_[1]->getOutput().getFrameHeight();
    if (h != 0) targetDims_.setDim(2, h);
    int w = inputLayers_[1]->getOutput().getFrameWidth();
    if (w != 0) targetDims_.setDim(3, w);
  }
  // get final crop shape from target dims and crop axis
  std::vector<uint32_t> crop_shape;
  int dimSize = 4;
  for (int i = 0; i < dimSize; i++) {
    if (i >= crop_axis_) {
      crop_shape.push_back(targetDims_[i]);
    } else {
      crop_shape.push_back(inDims_[i]);
    }
  }

  outDims_.reshape(
      {crop_shape[0], crop_shape[1], crop_shape[2], crop_shape[3]});
  output_.setFrameHeight(crop_shape[2]);
  output_.setFrameWidth(crop_shape[3]);
}

void CropLayer::setInDims() {
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  inDims_.setDim(0, batchSize);
  int h = inputLayers_[0]->getOutput().getFrameHeight();
  if (h != 0) inDims_.setDim(2, h);
  int w = inputLayers_[0]->getOutput().getFrameWidth();
  if (w != 0) inDims_.setDim(3, w);
}

void CropLayer::forward(PassType passType) {
  Layer::forward(passType);
  setInDims();
  setOutDims();
  int size = outDims_[1] * outDims_[2] * outDims_[3];
  resetOutput(outDims_[0], size);
  MatrixPtr outV = getOutputValue();
  REGISTER_TIMER_INFO("CropForward", getName().c_str());

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), inDims_);
  outputs.addArg(*getOutputValue(), outDims_, ASSIGN_TO);
  forward_[0]->calc(inputs, outputs);
}

void CropLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  REGISTER_TIMER_INFO("CropBackward", getName().c_str());

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getOutputGrad(), outDims_);
  outputs.addArg(*getInputGrad(0), inDims_, ADD_TO);
  backward_[0]->calc(inputs, outputs);
}
}  // namespace paddle
