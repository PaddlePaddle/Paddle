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

#include "PadLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(pad, PadLayer);

bool PadLayer::init(const LayerMap& layerMap,
                    const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  auto& pad_conf = config_.inputs(0).pad_conf();
  auto& img_conf = pad_conf.image_conf();
  CHECK_EQ(config_.inputs_size(), 1);
  inDims_ = TensorShape(
      {0,
       img_conf.channels(),
       img_conf.has_img_size_y() ? img_conf.img_size_y() : img_conf.img_size(),
       img_conf.img_size()});

  CHECK_EQ(2, pad_conf.pad_c_size());
  CHECK_EQ(2, pad_conf.pad_h_size());
  CHECK_EQ(2, pad_conf.pad_w_size());
  padc_ = {pad_conf.pad_c(0), pad_conf.pad_c(1)};
  padh_ = {pad_conf.pad_h(0), pad_conf.pad_h(1)};
  padw_ = {pad_conf.pad_w(0), pad_conf.pad_w(1)};

  outDims_ = TensorShape(4);
  setOutDims(0);

  createFunction(forward_,
                 "Pad",
                 FuncConfig()
                     .set("channel", padc_)
                     .set("height", padh_)
                     .set("width", padw_));
  createFunction(backward_,
                 "PadGrad",
                 FuncConfig()
                     .set("channel", padc_)
                     .set("height", padh_)
                     .set("width", padw_));

  return true;
}

void PadLayer::setOutDims(const size_t batchSize) {
  outDims_.reshape({batchSize,
                    inDims_[1] + padc_[0] + padc_[1],
                    inDims_[2] + padh_[0] + padh_[1],
                    inDims_[3] + padw_[0] + padw_[1]});
}

void PadLayer::setTensorDim(const size_t batchSize) {
  CHECK_EQ(static_cast<int>(inputLayers_.size()), 1);
  inDims_.setDim(0, batchSize);
  int h = inputLayers_[0]->getOutput().getFrameHeight();
  if (h != 0) inDims_.setDim(2, h);
  int w = inputLayers_[0]->getOutput().getFrameWidth();
  if (w != 0) inDims_.setDim(3, w);
  setOutDims(batchSize);
}

void PadLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  setTensorDim(batchSize);
  int size = outDims_[1] * outDims_[2] * outDims_[3];
  resetOutput(batchSize, size);
  MatrixPtr outV = getOutputValue();
  REGISTER_TIMER_INFO("PadForward", getName().c_str());

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), inDims_);
  outputs.addArg(*getOutputValue(), outDims_, ASSIGN_TO);
  forward_[0]->calc(inputs, outputs);
}

void PadLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  REGISTER_TIMER_INFO("PadBackward", getName().c_str());

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getOutputGrad(), outDims_);
  outputs.addArg(*getInputGrad(0), inDims_, ADD_TO);
  backward_[0]->calc(inputs, outputs);
}
}  // namespace paddle
