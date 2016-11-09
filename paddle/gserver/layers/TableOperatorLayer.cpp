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


#include "TableOperatorLayer.h"
#include "paddle/utils/Logging.h"
#include "paddle/utils/Stat.h"
#include "paddle/math/Matrix.h"

namespace paddle {

REGISTER_LAYER(table_operator, TableOperatorLayer);

bool TableOperatorLayer::init(const LayerMap& layerMap,
                       const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  CHECK_EQ(inputLayers_.size(), 2LU);
  return true;
}

void TableOperatorLayer::forward(PassType passType) {
  Layer::forward(passType);
  // get inputs
  IVectorPtr ids = getInputLabel(*inputLayers_[0]);
  MatrixPtr inputMatrix = getInputValue(1);
  // reshape input matrix and output matrix
  int batch_size = getInput(1).getBatchSize();
  int r = getSize();
  int h = inputMatrix->getWidth()/r;
  inputMatrix->reshape(batch_size*h, r);
  {
      REGISTER_TIMER_INFO("TableOperatorFwResetTimer", getName().c_str());
      reserveOutput(batch_size, getSize());
  }
  MatrixPtr outV = getOutputValue();
  outV->zeroMem();
  for (int i = 0; i < batch_size; i++) {
      ids->setElement(i, ids->getElement(i) + i*r);
  }
  // select rows
  outV->selectRows(*inputMatrix, *ids);
  // reshape input matrix back
  inputMatrix->reshape(batch_size, r*h);
  for (int i = 0; i < batch_size; i++) {
      ids->setElement(i, ids->getElement(i) - i*r);
  }
}

void TableOperatorLayer::backward(const UpdateCallback& callback) {
     IVectorPtr ids = getInputLabel(*inputLayers_[0]);
     MatrixPtr outG = getOutputGrad();
     MatrixPtr inG = getInputGrad(1);
     int batch_size = getInput(1).getBatchSize();
     int r = getSize();
     int h = inG->getWidth()/r;
     //
     inG->reshape(batch_size*h, r);
     for (int i = 0; i < batch_size; i++) {
         ids->setElement(i, ids->getElement(i) + i*r);
     }
     // add outG to inG's row
     outG->addToRows(*inG, *ids);
     //
     inG->reshape(batch_size, r*h);
     for (int i = 0; i < batch_size; i++) {
         ids->setElement(i, ids->getElement(i) - i*r);
     }
}
}  // namespace paddle
