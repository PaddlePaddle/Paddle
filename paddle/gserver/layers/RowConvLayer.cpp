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

#include "RowConvLayer.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(row_conv, RowConvLayer);

bool RowConvLayer::init(const LayerMap& layerMap,
                        const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  contexLength_ = config_.inputs(0).row_conv_conf().context_length();

  CHECK_EQ(inputLayers_.size(), 1UL);
  weight_.reset(new Weight(contexLength_, getSize(), parameters_[0]));
  createFunction(forward_, "RowConv", FuncConfig());
  createFunction(backward_, "RowConvGrad", FuncConfig());

  return true;
}

void RowConvLayer::forward(PassType passType) {
  Layer::forward(passType);
  MatrixPtr input = getInputValue(0);
  size_t height = input->getHeight();
  size_t width = input->getWidth();
  CHECK_EQ(width, getSize());
  resetOutput(height, width);

  const auto startPos = getInput(0).sequenceStartPositions->getVector(useGpu_);
  MatrixPtr w = weight_->getW();
  wDims_ = TensorShape({w->getHeight(), w->getWidth()});

  MatrixPtr outV = getOutputValue();
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), *startPos);
  inputs.addArg(*w, wDims_);
  outputs.addArg(*getOutputValue(), *startPos, ADD_TO);

  {
    REGISTER_TIMER_INFO("RowConvForward", getName().c_str());
    forward_[0]->calc(inputs, outputs);
  }

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void RowConvLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }

  const auto startPos = getInput(0).sequenceStartPositions->getVector(useGpu_);

  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getOutputGrad(), *startPos);
  inputs.addArg(*getInputValue(0), *startPos);
  inputs.addArg(*weight_->getW(), wDims_);

  MatrixPtr inGrad = getInputGrad(0);
  MatrixPtr wGrad = weight_->getWGrad();
  size_t h = getInputValue(0)->getHeight();
  size_t w = getInputValue(0)->getWidth();
  outputs.addArg(
      inGrad ? (*inGrad) : *(Matrix::create(nullptr, h, w, false, useGpu_)),
      *startPos,
      ADD_TO);
  outputs.addArg(
      wGrad ? (*wGrad)
            : *(Matrix::create(nullptr, contexLength_, w, false, useGpu_)),
      wDims_,
      ADD_TO);

  {
    REGISTER_TIMER_INFO("RowConvBackward", getName().c_str());
    backward_[0]->calc(inputs, outputs);
  }

  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
