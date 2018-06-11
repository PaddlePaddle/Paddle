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

#include "ParameterReluLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(prelu, ParameterReluLayer);

bool ParameterReluLayer::init(const LayerMap& layerMap,
                              const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);
  CHECK_EQ(inputLayers_.size(), 1UL);
  CHECK_EQ(inputLayers_.size(), parameters_.size());
  partialSum_ = config_.partial_sum();
  CHECK_GT(partialSum_, 0UL) << "partial_sum must be larger than zero.";
  CHECK(!(inputLayers_[0]->getSize() % partialSum_))
      << "Incorrect value for partialSum: " << partialSum_
      << " must divide input size: " << inputLayers_[0]->getSize();
  CHECK_EQ(getSize() / partialSum_, parameters_[0]->getSize());
  weight_ = std::unique_ptr<Weight>(new Weight(
      1UL, inputLayers_[0]->getSize() / partialSum_, parameters_[0]));
  return true;
}

void ParameterReluLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInput(0).getBatchSize();
  int size = getSize();
  reserveOutput(batchSize, size);
  MatrixPtr outV = getOutputValue();
  {
    REGISTER_TIMER_INFO("FwResetTimer", getName().c_str());
    outV->paramReluForward(*(getInput(0).value), *(weight_->getW()));
  }
}

void ParameterReluLayer::backward(const UpdateCallback& callback) {
  if (weight_->getWGrad()) {
    weight_->getWGrad()->paramReluBackwardW(*getOutputGrad(),
                                            *(getInputValue(0)));
  }

  MatrixPtr preGrad = getInputGrad(0);
  preGrad->paramReluBackwardDiff(
      *getOutputGrad(), *(getInputValue(0)), *(weight_->getW()));
  {
    REGISTER_TIMER_INFO("WeightUpdate", getName().c_str());
    weight_->getParameterPtr()->incUpdate(callback);
  }
}

}  // namespace paddle
