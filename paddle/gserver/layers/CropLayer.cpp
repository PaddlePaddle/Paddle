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

#include "CropLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(crop, CropLayer);

bool CropLayer::init(const LayerMap& layerMap,
                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  auto& crop_conf = config_.inputs(0).crop_conf();
  auto& img_conf = crop_conf.image_conf();
  CHECK_EQ(config_.inputs_size(), 1);
  inDims_ = TensorShape(
      {0,
       img_conf.channels(),
       img_conf.has_img_size_y() ? img_conf.img_size_y() : img_conf.img_size(),
       img_conf.img_size()});

  crop_corner_ = {crop_conf.crop_corner(0),
                  crop_conf.crop_corner(1),
                  crop_conf.crop_corner(2)};
  crop_shape_ = {crop_conf.crop_shape(0),
                 crop_conf.crop_shape(1),
                 crop_conf.crop_shape(2)};

  outDims_ = TensorShape(4);
  setOutDims(0);

  createFunction(forward_,
                 "Crop",
                 FuncConfig()
                     .set("crop_corner", crop_corner_)
                     .set("crop_shape", crop_shape_));
  createFunction(backward_,
                 "CropGrad",
                 FuncConfig()
                     .set("crop_corner", crop_corner_)
                     .set("crop_shape", crop_shape_));

  return true;
}

void CropLayer::setOutDims(const size_t batchSize) {
  outDims_.reshape({batchSize, crop_shape_[0], crop_shape_[1], crop_shape_[2]});
}

void CropLayer::setTensorDim(const size_t batchSize) {
  CHECK_EQ(static_cast<int>(inputLayers_.size()), 1);
  inDims_.setDim(0, batchSize);
  int h = inputLayers_[0]->getOutput().getFrameHeight();
  if (h != 0) inDims_.setDim(2, h);
  int w = inputLayers_[0]->getOutput().getFrameWidth();
  if (w != 0) inDims_.setDim(3, w);
  setOutDims(batchSize);
}

void CropLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  setTensorDim(batchSize);
  int size = outDims_[1] * outDims_[2] * outDims_[3];
  resetOutput(batchSize, size);
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
