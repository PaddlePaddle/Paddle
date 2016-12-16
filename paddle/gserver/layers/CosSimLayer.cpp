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
  return true;
}

void CosSimLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(0)->getHeight();
  int size = getSize();

  {
    REGISTER_TIMER_INFO("CosFwResetTimer", getName().c_str());
    reserveOutput(batchSize, size);
  }

  MatrixPtr outV = getOutputValue();

  /* activation */ {
    REGISTER_TIMER_INFO("CosFwAtvTimer", getName().c_str());
    MatrixPtr prevOut1 = getInputValue(0);
    MatrixPtr prevOut2 = getInputValue(1);
    outV->cosSim(*prevOut1, *prevOut2, config_.cos_scale());
  }
}

void CosSimLayer::backward(const UpdateCallback& callback) {
  /* activation */ {
    REGISTER_TIMER_INFO("CosBpAtvTimer", getName().c_str());
    MatrixPtr outG = this->getOutputGrad();

    outG->cosSimDerivative(*this->getOutputValue(),
                           *getInputValue(0),
                           *getInputValue(1),
                           *getInputGrad(0),
                           *getInputGrad(1),
                           config_.cos_scale());
  }
}

}  // namespace paddle
