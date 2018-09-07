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

#include "AddtoLayer.h"

#include "paddle/utils/Logging.h"

#include "paddle/utils/Stat.h"

namespace paddle {

REGISTER_LAYER(addto, AddtoLayer);

bool AddtoLayer::init(const LayerMap& layerMap,
                      const ParameterMap& parameterMap) {
  /* Initialize the basic parent class */
  Layer::init(layerMap, parameterMap);

  /* initialize biases_ */
  if (biasParameter_.get() != NULL) {
    biases_ = std::unique_ptr<Weight>(new Weight(1, getSize(), biasParameter_));
  }

  return true;
}

void AddtoLayer::forward(PassType passType) {
  Layer::forward(passType);

  /* malloc memory for the output_ if necessary */
  int batchSize = getInputValue(0)->getHeight();
  int size = getSize();

  reserveOutput(batchSize, size);

  MatrixPtr outV = getOutputValue();
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    MatrixPtr input = getInputValue(i);
    i == 0 ? outV->assign(*input) : outV->add(*input);
  }
  /* add the bias-vector */
  if (biases_.get() != NULL) {
    outV->addBias(*(biases_->getW()), 1);
  }

  /* activation */ { forwardActivation(); }
}

void AddtoLayer::backward(const UpdateCallback& callback) {
  /* Do derivation */ { backwardActivation(); }

  if (biases_ && biases_->getWGrad()) {
    biases_->getWGrad()->collectBias(*getOutputGrad(), 1);

    /* Increasing the number of gradient */
    biases_->getParameterPtr()->incUpdate(callback);
  }

  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    /* Calculate the input layers error */
    MatrixPtr preGrad = getInputGrad(i);
    if (NULL != preGrad) {
      preGrad->add(*getOutputGrad());
    }
  }
}

}  // namespace paddle
