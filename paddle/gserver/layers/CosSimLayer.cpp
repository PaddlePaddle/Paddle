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
  CHECK_EQ(forward_.size(), 1) << "Only one forward function needed";

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
    forward_[0]->calc(
        {Tensor(prevOut1->getData(),
                Dims{prevOut1->getHeight(), prevOut1->getWidth()}),
         Tensor(prevOut2->getData(),
                Dims{prevOut2->getHeight(), prevOut2->getWidth()})},
        {Tensor(outV->getData(), Dims{outV->getHeight(), outV->getWidth()})},
        {});
  }
}

void CosSimLayer::backward(const UpdateCallback& callback) {
  /* activation */ {
    REGISTER_TIMER_INFO("CosBpAtvTimer", getName().c_str());
    CHECK_EQ(backward_.size(), 1) << "Only one backward function needed";

    const auto outG = this->getOutputGrad();
    const auto outV = this->getOutputValue();
    const auto inV1 = this->getInputValue(0);
    const auto inV2 = this->getInputValue(1);
    auto inG1 = this->getInputGrad(0);
    auto inG2 = this->getInputGrad(1);
    CHECK(outG && outV && inV1 && inV2 && inG1 && inG2);
    backward_[0]->calc(
        {Tensor(outV->getData(), Dims{outV->getHeight(), outV->getWidth()}),
         Tensor(inV1->getData(), Dims{inV1->getHeight(), inV1->getWidth()}),
         Tensor(inV2->getData(), Dims{inV2->getHeight(), inV2->getWidth()}),
         Tensor(inG1->getData(), Dims{inG1->getHeight(), inG1->getWidth()}),
         Tensor(inG2->getData(), Dims{inG2->getHeight(), inG2->getWidth()})},
        {Tensor(outG->getData(), Dims{outG->getHeight(), outG->getWidth()})},
        {});
  }
}

}  // namespace paddle
