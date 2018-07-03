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

#include "CosSimLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(cos, CosSimLayer);

bool CosSimLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2LU);

  createFunction(forward_,
                 "CosSimForward",
                 FuncConfig().set("scale", (real)config_.cos_scale()));
  createFunction(backward_,
                 "CosSimBackward",
                 FuncConfig().set("scale", (real)config_.cos_scale()));

  return true;
}

void CosSimLayer::forward(PassType passType) {
  Layer::forward(passType);
  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(0)->getHeight();
  int size = getSize();
  CHECK_EQ(forward_.size(), 1UL) << "Only one forward function needed";

  {
    REGISTER_TIMER_INFO("CosFwResetTimer", getName().c_str());
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();
  /* activation */ {
    REGISTER_TIMER_INFO("CosFwAtvTimer", getName().c_str());
    MatrixPtr prevOut1 = getInputValue(0);
    MatrixPtr prevOut2 = getInputValue(1);

    CHECK(outV && prevOut1 && prevOut2);
    BufferArgs inputs;
    BufferArgs outputs;
    inputs.addArg(*prevOut1);
    inputs.addArg(*prevOut2);
    outputs.addArg(*outV, ASSIGN_TO);
    forward_[0]->calc(inputs, outputs);
  }
}

void CosSimLayer::backward(const UpdateCallback& callback) {
  /* activation */ {
    REGISTER_TIMER_INFO("CosBpAtvTimer", getName().c_str());
    CHECK_EQ(backward_.size(), 1UL) << "Only one backward function needed";

    const auto outG = this->getOutputGrad();
    const auto outV = this->getOutputValue();
    const auto inV1 = this->getInputValue(0);
    const auto inV2 = this->getInputValue(1);
    auto inG1 = this->getInputGrad(0);
    auto inG2 = this->getInputGrad(1);
    CHECK(outG && outV && inV1 && inV2 && inG1 && inG2);
    BufferArgs inputs;
    BufferArgs outputs;
    inputs.addArg(*outG);
    inputs.addArg(*outV);
    inputs.addArg(*inV1);
    inputs.addArg(*inV2);
    outputs.addArg(*inG1, ADD_TO);
    outputs.addArg(*inG2, ADD_TO);

    backward_[0]->calc(inputs, outputs);
  }
}

}  // namespace paddle
