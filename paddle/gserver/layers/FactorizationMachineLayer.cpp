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

#include "FactorizationMachineLayer.h"
#include <algorithm>
#include <vector>
#include "paddle/math/SparseMatrix.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(factorization_machine, FactorizationMachineLayer);

bool FactorizationMachineLayer::init(const LayerMap& layerMap,
                                     const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  factorSize_ = config_.factor_size();

  /* initialize the latentVectors_ */
  CHECK_EQ(inputLayers_.size(), 1UL);
  size_t height = inputLayers_[0]->getSize();
  latentVectors_.reset(new Weight(height, factorSize_, parameters_[0]));

  return true;
}

void FactorizationMachineLayer::forward(PassType passType) {
  Layer::forward(passType);

  auto input = getInput(0);

  int batchSize = input.getBatchSize();
  int size = getSize();
  reserveOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();

  /* activation */ {
    REGISTER_TIMER_INFO("FwAtvTimer", getName().c_str());
    forwardActivation();
  }
}

void FactorizationMachineLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ {
    REGISTER_TIMER_INFO("BpAvtTimer", getName().c_str());
    backwardActivation();
  }
}

}  // namespace paddle
