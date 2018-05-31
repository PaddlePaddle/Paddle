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

#include "NormProjectionLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {
size_t CMRProjectionNormLayer::getSize() {
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t layerSize = 0;
  imgSizeH_ = inputLayers_[0]->getOutput().getFrameHeight();
  imgSizeW_ = inputLayers_[0]->getOutput().getFrameWidth();
  if (imgSizeH_ == 0) {
    imgSizeH_ = imgSizeY_;
  }
  if (imgSizeW_ == 0) {
    imgSizeW_ = imgSize_;
  }
  outputH_ = imgSizeH_;
  outputW_ = imgSizeW_;
  layerSize = outputH_ * outputW_ * channels_;

  getOutput().setFrameHeight(outputH_);
  getOutput().setFrameWidth(outputW_);
  return layerSize;
}

bool CMRProjectionNormLayer::init(const LayerMap& layerMap,
                                  const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  ResponseNormLayer::init(layerMap, parameterMap);

  /* the size of inputs for norm-layer is 1 */
  CHECK_EQ(config_.inputs_size(), 1);

  createFunction(
      forward_,
      "CrossMapNormal",
      FuncConfig().set("size", size_).set("scale", scale_).set("pow", pow_));
  createFunction(
      backward_,
      "CrossMapNormalGrad",
      FuncConfig().set("size", size_).set("scale", scale_).set("pow", pow_));

  return true;
}

void CMRProjectionNormLayer::forward(PassType passType) {
  Layer::forward(passType);
  /* malloc memory for the output_ if necessary */
  /* note: one sample correspond to one row */
  MatrixPtr input = inputLayers_[0]->getOutputValue();
  size_t batchSize = input->getHeight();
  int size = getSize();
  resetOutput(batchSize, size);

  Matrix::resizeOrCreate(denoms_, batchSize, size, /* trans */ false, useGpu_);

  shape_ = TensorShape({batchSize, channels_, imgSizeH_, imgSizeW_});

  // prepare forward arguments
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), shape_);
  outputs.addArg(*getOutputValue(), shape_, ASSIGN_TO);
  outputs.addArg(*denoms_, shape_, ASSIGN_TO);

  forward_[0]->calc(inputs, outputs);
}

void CMRProjectionNormLayer::backward(const UpdateCallback& callback) {
  (void)callback;

  if (NULL == getInputGrad(0)) {
    return;
  }

  // prepare backward arguments
  BufferArgs inputs;
  BufferArgs outputs;
  inputs.addArg(*getInputValue(0), shape_);
  inputs.addArg(*getOutputValue(), shape_);
  inputs.addArg(*getOutputGrad(), shape_);
  inputs.addArg(*denoms_, shape_);
  outputs.addArg(*getInputGrad(0), shape_, ADD_TO);

  backward_[0]->calc(inputs, outputs);
}
}  // namespace paddle
