/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#include "InputProjLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/Matrix.h"

namespace paddle {

REGISTER_LAYER(input_proj, InputProjLayer);

bool InputProjLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2LU);
  return true;
}

void InputProjLayer::forward(PassType passType) {
  Layer::forward(passType);
  // get inputs
  IVectorPtr ids = getInputLabel(*inputLayers_[0]);
  MatrixPtr input_matrix = getInputValue(1);
  // reshape input matrix and output matrix
  int r = getSize();
  int h = input_matrix->getWidth()/r;
  input_matrix->reshape(h, r);
  {
      REGISTER_TIMER_INFO("InputProjFwResetTimer", getName().c_str());
      reserveOutput(1, getSize());
  }
  MatrixPtr outV = getOutputValue();
  outV->zeroMem();
  // select rows
  outV->selectRows(*input_matrix, *ids);
  // reshape input matrix back
  input_matrix->reshape(1, r*h);
}

void InputProjLayer::backward(const UpdateCallback& callback) {
     IVectorPtr ids = getInputLabel(*inputLayers_[0]);
     MatrixPtr outG = getOutputGrad();
     MatrixPtr inG = getInputGrad(1);
     int r = getSize();
     int h = inG->getWidth()/r;
     inG->reshape(h, r);
     // add outG to inG's row
     outG->addToRows(*inG, *ids);
     inG->reshape(1, r*h);
}
}  // namespace paddle
