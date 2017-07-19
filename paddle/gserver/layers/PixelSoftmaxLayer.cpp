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

#include "PixelSoftmaxLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(pixel_softmax, PixelSoftmaxLayer);

bool PixelSoftmaxLayer::init(const LayerMap& layerMap,
                             const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  auto& img_conf = config_.inputs(0).image_conf();
  inH_ =
      img_conf.has_img_size_y() ? img_conf.img_size_y() : img_conf.img_size();
  inW_ = img_conf.img_size();
  inC_ = img_conf.channels();
  createFunction(forward_, "NCHW2NHWC", FuncConfig());
  createFunction(backward_, "NHWC2NCHW", FuncConfig());
  inDims_ = TensorShape({0, inH_, inW_, inC_});
  outDims_ = TensorShape({0, inC_, inH_, inW_});
  return true;
}

void PixelSoftmaxLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  // cout<<"useGpu:"<<useGpu(deviceId_)<<endl;
  Matrix::resizeOrCreate(
      tmpInput_, batchSize * inH_ * inW_, inC_, false, useGpu_);
  Matrix::resizeOrCreate(
      tmpOutput_, batchSize * inH_ * inW_, inC_, false, useGpu_);
  tmpOutput_->zeroMem();
  resetOutput(batchSize, inH_ * inW_ * inC_);
  inDims_.setDim(0, batchSize);
  outDims_.setDim(0, batchSize);

  // switch NCHW to NHWC
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), inDims_);
  outputs.addArg(*tmpInput_, outDims_);
  forward_[0]->calc(inputs, outputs);
  // softmax forward and save softmax result into tmpMatrix_
  tmpInput_->softmax(*tmpOutput_);

  // switch NHWC to NCHW
  BufferArgs inputs_1;
  BufferArgs outputs_1;
  inputs_1.addArg(*tmpOutput_, outDims_);
  outputs_1.addArg(*getOutputValue(), inDims_);
  backward_[0]->calc(inputs_1, outputs_1);
}

void PixelSoftmaxLayer::backward(const UpdateCallback& callback) {
  (void)callback;
  REGISTER_TIMER_INFO("PixelSoftmaxBackward", getName().c_str());

  // switch NCHW to NHWC
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getOutputGrad(), inDims_);
  outputs.addArg(*tmpInput_, outDims_);
  forward_[0]->calc(inputs, outputs);
  // softmax backward and save grad result into tmpOutput_
  tmpInput_->softmaxBackward(*tmpOutput_);

  // switch NHWC to NCHW
  BufferArgs inputs_1;
  BufferArgs outputs_1;
  inputs_1.addArg(*tmpInput_, outDims_);
  outputs_1.addArg(*getInputGrad(0), inDims_);
  backward_[0]->calc(inputs_1, outputs_1);
}
}  // namespace paddle
